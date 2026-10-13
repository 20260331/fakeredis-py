from __future__ import annotations

import logging
import threading
import time
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Collection
from typing import Any, ClassVar

from fakeredis._core._database import Database
from fakeredis._typing import ServerType, VersionType
from fakeredis.model import AccessControlList

LOGGER = logging.getLogger("fakeredis")


class CommandBarrier:
    """Test-only barrier parking a top-level command after validation, before it touches the database.

    Concurrency tests that rely on ``sleep`` cannot reliably reproduce a specific command interleave. An armed
    barrier parks the next matching command at the dispatch seam: arity/ACL checks have passed, but the command has
    not yet read or written anything. The test then drives other clients against the same server and only releases
    the parked command afterwards.

    The barrier's condition shares the server lock, so a parked command releases that lock while it waits: other
    clients keep running, and transactions stay atomic because the wait only happens at top-level dispatch --
    commands queued in MULTI/EXEC and commands issued from scripts never reach it. The barrier is one-shot: it
    disarms itself before parking, so the matching commands the other clients send meanwhile run normally instead
    of piling up behind the parked one.
    """

    def __init__(self, server: FakeServer) -> None:
        self._server = server
        # Shares the server lock: wait() atomically frees it and re-acquires it on release.
        self._condition = threading.Condition(server.lock)
        self._armed_command: str | None = None
        self._armed_key: bytes | None = None
        self._released = False
        # Set once a command has parked, so the test thread knows the interleave point is reached.
        self.reached = threading.Event()

    def arm(self, command: str | bytes, key: bytes | str | None = None) -> None:
        """Arm the barrier for the next top-level ``command`` (command name, case-insensitive).

        When ``key`` is given, the command must reference that key (by a key argument) to be parked. Replaces any
        previously armed barrier. A parked command (if any) is left parked; use :meth:`clear` to release it.
        """
        command_name = command.decode().lower() if isinstance(command, bytes) else command.lower()
        armed_key = key.encode() if isinstance(key, str) else key
        with self._condition:
            self._armed_command = command_name
            self._armed_key = armed_key
            self._released = False
            self.reached.clear()

    def clear(self) -> None:
        """Disarm the barrier, releasing a parked command if one is currently waiting."""
        with self._condition:
            self._armed_command = None
            self._armed_key = None
            self._released = True
            self._condition.notify_all()
            self.reached.clear()

    def wait_reached(self, timeout: float | None = 5.0) -> bool:
        """Wait until a command has parked at the barrier. Returns False if ``timeout`` elapses first."""
        return self.reached.wait(timeout)

    def matches(self, command_name: str) -> bool:
        """Whether the barrier is armed for ``command_name``. Cheap pre-check so the dispatch hot path pays nothing
        (and skips collecting the command's key arguments) while no barrier is armed."""
        return self._armed_command == command_name

    def release(self) -> None:
        """Release the parked command. Safe to call when nothing is parked."""
        with self._condition:
            self._released = True
            self._condition.notify_all()

    def park_if_armed(self, command_name: str, key_args: Collection[bytes]) -> None:
        """Block the calling command when the barrier is armed for it. Callers must hold the server lock.

        ``key_args`` are the command's key arguments, used for the optional key filter. The lock is released while
        waiting and re-acquired before returning, with the databases' clocks refreshed so expiry checks do not use
        the pre-wait timestamp.
        """
        if not self.matches(command_name):
            return
        if self._armed_key is not None and self._armed_key not in key_args:
            return
        # Disarm before parking: matching commands from the other clients involved in the interleave must run.
        self._armed_command = None
        self._armed_key = None
        self._released = False
        self.reached.set()
        while not self._released:
            self._condition.wait()
        now = time.time()
        for db in self._server.dbs.values():
            db.time = now


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
        # Test-only interleave hook: parks an armed top-level command between validation and database access. Unused
        # unless a test arms it.
        self.command_barrier = CommandBarrier(self)

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

    @staticmethod
    def get_server(key: str, version: VersionType, server_type: ServerType) -> FakeServer:
        if key not in FakeServer._servers_map:
            FakeServer._servers_map[key] = FakeServer(version=version, server_type=server_type)
        return FakeServer._servers_map[key]
