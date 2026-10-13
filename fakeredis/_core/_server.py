from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

from fakeredis._core._database import Database
from fakeredis._typing import ServerType, VersionType
from fakeredis.model import AccessControlList

LOGGER = logging.getLogger("fakeredis")

# Selectable resources wait_for_quiescence can wait for -- the same things a test cannot otherwise see at teardown.
BLOCKED_CONDITION = "blocked"  # a client parked in a blocking command (BLPOP, XREAD BLOCK, ...)
SUBSCRIPTIONS_CONDITION = "subscriptions"  # a client holding pub/sub subscriptions
PENDING_REPLIES_CONDITION = "pending_replies"  # replies queued on a socket but not yet read by its client
QUIESCENCE_CONDITIONS = frozenset({BLOCKED_CONDITION, SUBSCRIPTIONS_CONDITION, PENDING_REPLIES_CONDITION})
# The waiter re-checks on an interval: no event fires when a client reads its reply, so the wait is poll based.
_DEFAULT_POLL_INTERVAL = 0.01


def _normalize_conditions(conditions: Iterable[str] | None) -> frozenset[str]:
    """Validate selected wait conditions and return them as a frozenset.

    None means all conditions. A single string is not accepted (it would iterate into one-letter "conditions").
    """
    if conditions is None:
        return QUIESCENCE_CONDITIONS
    normalized = frozenset(conditions)
    unknown = normalized - QUIESCENCE_CONDITIONS
    if unknown:
        raise ValueError(f"Unknown quiescence condition(s): {sorted(unknown)}; valid: {sorted(QUIESCENCE_CONDITIONS)}")
    if not normalized:
        raise ValueError("At least one quiescence condition must be selected")
    return normalized


@dataclass(frozen=True)
class ClientDiagnostics:
    """Read-only per-connection state at snapshot time."""

    client_id: int
    db: int
    blocked: bool
    paused: bool
    command: str | None  # last command run; when blocked, this is the blocking command
    subscriptions: int  # total channel + pattern + shard subscriptions held by this client
    pending_replies: int  # replies queued for this client but not yet read


@dataclass(frozen=True)
class ServerDiagnostics:
    """Read-only snapshot of the resources that keep a FakeServer busy."""

    clients: tuple[ClientDiagnostics, ...]
    channels: dict[str, int]  # channel -> subscriber count
    patterns: dict[str, int]  # pattern -> subscriber count
    shard_channels: dict[str, int]  # shard channel -> subscriber count

    @property
    def client_count(self) -> int:
        return len(self.clients)

    @property
    def blocked_count(self) -> int:
        return sum(client.blocked for client in self.clients)

    @property
    def paused_count(self) -> int:
        return sum(client.paused for client in self.clients)

    @property
    def subscription_count(self) -> int:
        return sum(client.subscriptions for client in self.clients)

    @property
    def pending_replies_count(self) -> int:
        return sum(client.pending_replies for client in self.clients)

    def is_quiescent(self, conditions: Iterable[str] | None = None) -> bool:
        """Whether none of the selected resources are busy.

        With conditions=None every resource is checked; pass a subset of QUIESCENCE_CONDITIONS to wait on only
        some resources.
        """
        normalized = _normalize_conditions(conditions)
        return not (
            (BLOCKED_CONDITION in normalized and self.blocked_count)
            or (SUBSCRIPTIONS_CONDITION in normalized and self.subscription_count)
            or (PENDING_REPLIES_CONDITION in normalized and self.pending_replies_count)
        )

    def format_summary(self) -> str:
        """Human-readable summary for failure output: which client holds what, and on which channels."""
        lines = [
            (
                f"{self.client_count} client(s): {self.blocked_count} blocked, "
                f"{self.subscription_count} subscription(s), {self.pending_replies_count} pending reply(ies)"
            )
        ]
        for client in self.clients:
            details = []
            if client.blocked:
                details.append(f"blocked({client.command or 'unknown command'})")
            if client.paused:
                details.append("paused")
            if client.subscriptions:
                details.append(f"{client.subscriptions} subscription(s)")
            if client.pending_replies:
                details.append(f"{client.pending_replies} pending reply(ies)")
            lines.append(f"  client id={client.client_id} db={client.db}: {', '.join(details) or 'idle'}")
        for label, registry in (
            ("channels", self.channels),
            ("patterns", self.patterns),
            ("shard channels", self.shard_channels),
        ):
            if registry:
                entries = ", ".join(f"{name}({count})" for name, count in sorted(registry.items()))
                lines.append(f"  {label}: {entries}")
        return "\n".join(lines)


class QuiescenceTimeoutError(TimeoutError):
    """Raised when the server still holds selected resources once the timeout expires.

    The final read-only snapshot is available as `diagnostics`, and str() already renders it, so a failing test
    shows what was blocked, subscribed or left unread without any extra work.
    """

    def __init__(self, diagnostics: ServerDiagnostics, conditions: Iterable[str], timeout: float | None) -> None:
        self.diagnostics = diagnostics
        self.conditions = _normalize_conditions(conditions)
        self.timeout = timeout
        super().__init__(
            f"FakeServer did not become quiescent within {timeout}s "
            f"(waiting for: {', '.join(sorted(self.conditions))})\n{diagnostics.format_summary()}"
        )


def _create_version(v: tuple[int, ...] | int | str) -> VersionType:
    if isinstance(v, tuple):
        return v
    if isinstance(v, int):
        return (v,)
    if isinstance(v, str):
        v_split = v.split(".")
        return tuple(int(x) for x in v_split)
    raise ValueError(f"Unsupported version: {v}")


class FakeServer:
    _servers_map: ClassVar[dict[str, FakeServer]] = {}

    def __init__(
        self,
        version: VersionType = (8,),
        server_type: ServerType = "redis",
        config: dict[bytes, bytes] | None = None,
    ) -> None:
        """Initialize a new FakeServer instance.
        :param version: The version of the server (e.g. 6, 7.4, "7.4.1", can also be a tuple)
        :param server_type: The type of server (redis, dragonfly, valkey, kividb)
        :param config: A dictionary of configuration options.

        Configuration options:
        - `requirepass`: The password required to authenticate to the server.
        - `aclfile`: The path to the ACL file.
        """
        self.lock = threading.Lock()
        self.dbs: dict[int, Database] = defaultdict(lambda: Database(self.lock))
        # Maps channel/pattern to a weak set of sockets
        self.script_cache: dict[bytes, bytes] = {}  # Maps SHA1 to the script source
        self.subscribers: dict[bytes, weakref.WeakSet[Any]] = defaultdict(weakref.WeakSet)
        self.psubscribers: dict[bytes, weakref.WeakSet[Any]] = defaultdict(weakref.WeakSet)
        self.ssubscribers: dict[bytes, weakref.WeakSet[Any]] = defaultdict(weakref.WeakSet)
        self.lastsave: int = int(time.time())
        self.connected = True
        # List of weakrefs to sockets that are being closed lazily
        self.sockets: list[Any] = []
        self.closed_sockets: list[Any] = []
        self.version: VersionType = _create_version(version)
        if server_type not in ("redis", "dragonfly", "valkey", "kividb"):
            raise ValueError(f"Unsupported server type: {server_type}")
        self.server_type: ServerType = server_type
        self.config: dict[bytes, bytes] = config or {}
        self.acl: AccessControlList = AccessControlList()
        self.clients: dict[str, dict[str, Any]] = {}
        self._next_client_id = 1
        # CLIENT PAUSE state. Recorded so CLIENT PAUSE/UNPAUSE validate and round-trip, but command processing is never
        # actually suspended (see CLIENT PAUSE docs).
        self.pause_until: float = 0.0
        self.pause_mode: bytes = b"all"
        # Where each recent SCAN-family cursor resumes: (the iteration's sorted snapshot, the position in it), keyed by
        # (scan state, cursor). See `_scan`.
        self.scan_cursors: OrderedDict[tuple[Any, int], tuple[list[Any], int]] = OrderedDict()

    SCAN_CURSORS_LIMIT: ClassVar[int] = 1024

    def remember_scan_cursor(self, cursor_key: tuple[Any, int], resume_at: tuple[list[Any], int]) -> None:
        """Record where a SCAN-family cursor resumes, forgetting the oldest cursor once too many are held."""
        self.scan_cursors[cursor_key] = resume_at
        self.scan_cursors.move_to_end(cursor_key)
        while len(self.scan_cursors) > self.SCAN_CURSORS_LIMIT:
            self.scan_cursors.popitem(last=False)

    def forget_scan_snapshot(self, snapshot: list[Any]) -> None:
        """Drop the cursors of an iteration that has finished, so its snapshot does not outlive it."""
        finished = [cursor_key for cursor_key, (held, _) in self.scan_cursors.items() if held is snapshot]
        for cursor_key in finished:
            del self.scan_cursors[cursor_key]

    def get_next_client_id(self) -> int:
        with self.lock:
            client_id = self._next_client_id
            self._next_client_id += 1
        return client_id

    def diagnostics(self) -> ServerDiagnostics:
        """Return a read-only snapshot of the server's in-flight resources.

        Reads only: no commands are run, no queued replies are consumed, and no keys, subscriptions or other
        Redis state are changed. Safe to call while clients are blocked or mid-command, from any thread.
        """
        with self.lock:
            clients: list[ClientDiagnostics] = []
            # close() can mutate self.sockets at any time (it may run during garbage collection, without the lock),
            # so iterate a copy and tolerate fields already nulled.
            for sock in list(self.sockets):
                responses = getattr(sock, "responses", None)
                info = getattr(sock, "_client_info", None)
                clients.append(
                    ClientDiagnostics(
                        client_id=int(info["id"]) if info is not None else -1,
                        db=int(getattr(sock, "_db_num", 0) or 0),
                        blocked=bool(getattr(sock, "_blocked", False)),
                        paused=bool(getattr(sock, "_paused", False)),
                        command=getattr(sock, "_last_command", None),
                        subscriptions=int(getattr(sock, "_pubsub", 0) or 0),
                        pending_replies=(
                            int(responses.qsize()) if responses is not None and hasattr(responses, "qsize") else 0
                        ),
                    )
                )

            def _channel_counts(registry: dict[bytes, weakref.WeakSet[Any]]) -> dict[str, int]:
                return {
                    channel.decode() if isinstance(channel, bytes) else str(channel): len(socks)
                    for channel, socks in registry.items()
                }

            channels = _channel_counts(self.subscribers)
            patterns = _channel_counts(self.psubscribers)
            shard_channels = _channel_counts(self.ssubscribers)
        return ServerDiagnostics(tuple(clients), channels, patterns, shard_channels)

    def wait_for_quiescence(
        self,
        timeout: float | None = None,
        conditions: Iterable[str] | None = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> ServerDiagnostics:
        """Block until none of the selected resources are busy, or raise QuiescenceTimeoutError.

        This is the synchronous variant: never call it on an event-loop thread while async clients are active,
        because blocking the loop stops the very tasks that would make those clients settle -- use
        wait_for_quiescence_async there instead.

        :param timeout: Maximum seconds to wait. None waits indefinitely.
        :param conditions: Subset of QUIESCENCE_CONDITIONS to wait for (default: all of them).
        :param poll_interval: Seconds between diagnostics checks.
        :returns: The final read-only ServerDiagnostics snapshot.
        """
        normalized = _normalize_conditions(conditions)
        snapshot = self.diagnostics()
        if snapshot.is_quiescent(normalized):
            return snapshot
        deadline = time.monotonic() + timeout if timeout is not None else None
        # Backstop for frozen clocks (freezegun/time-machine): cap the number of polls so a frozen time.monotonic
        # cannot turn the timeout into an endless wait.
        max_polls = None if timeout is None else max(1, math.ceil(timeout / poll_interval))
        polls = 0
        while True:
            time.sleep(poll_interval)
            snapshot = self.diagnostics()
            if snapshot.is_quiescent(normalized):
                return snapshot
            polls += 1
            if (deadline is not None and time.monotonic() >= deadline) or (
                max_polls is not None and polls >= max_polls
            ):
                raise QuiescenceTimeoutError(snapshot, normalized, timeout)

    async def wait_for_quiescence_async(
        self,
        timeout: float | None = None,
        conditions: Iterable[str] | None = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> ServerDiagnostics:
        """Asynchronous variant of wait_for_quiescence: yields control so blocked clients can settle.

        :param timeout: Maximum seconds to wait. None waits indefinitely.
        :param conditions: Subset of QUIESCENCE_CONDITIONS to wait for (default: all of them).
        :param poll_interval: Seconds between diagnostics checks.
        :returns: The final read-only ServerDiagnostics snapshot.
        """
        normalized = _normalize_conditions(conditions)
        snapshot = self.diagnostics()
        if snapshot.is_quiescent(normalized):
            return snapshot
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout if timeout is not None else None
        max_polls = None if timeout is None else max(1, math.ceil(timeout / poll_interval))
        polls = 0
        while True:
            await asyncio.sleep(poll_interval)
            snapshot = self.diagnostics()
            if snapshot.is_quiescent(normalized):
                return snapshot
            polls += 1
            if (deadline is not None and loop.time() >= deadline) or (max_polls is not None and polls >= max_polls):
                raise QuiescenceTimeoutError(snapshot, normalized, timeout)

    @staticmethod
    def get_server(key: str, version: VersionType, server_type: ServerType) -> FakeServer:
        if key not in FakeServer._servers_map:
            FakeServer._servers_map[key] = FakeServer(version=version, server_type=server_type)
        return FakeServer._servers_map[key]
