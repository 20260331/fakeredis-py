from __future__ import annotations

import copy
import logging
import threading
import time
import weakref
from collections import OrderedDict, defaultdict
from typing import Any, ClassVar

from fakeredis._core._database import Database
from fakeredis._typing import ServerType, VersionType
from fakeredis.model import AccessControlList, Hash

LOGGER = logging.getLogger("fakeredis")


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

    def fork(self) -> FakeServer:
        """Create an independent server from an atomic snapshot of this one.

        The copy starts with the same data (every key, its TTLs, stream and time-series state) and configuration
        (server version/type, config options, ACL users and cached Lua scripts). It shares nothing mutable with the
        origin, so writes on either side never reach the other.

        Connection-bound state is deliberately not copied: the forked server has no sockets, pub/sub subscribers,
        WATCHes, blocked clients or in-flight SCAN cursors. Its client ids start again from 1 and CLIENT PAUSE does
        not carry over.
        """
        with self.lock:
            new_server = FakeServer(version=self.version, server_type=self.server_type, config=dict(self.config))
            new_server.lastsave = self.lastsave
            new_server.script_cache = dict(self.script_cache)
            # ACL users and rules are configuration; the denial log is per-connection history and starts empty.
            new_server.acl = copy.deepcopy(self.acl)
            new_server.acl.reset_log()
            for db_num, old_db in self.dbs.items():
                new_db = old_db.snapshot(new_server.lock)
                for item in new_db._dict.values():
                    # Pending hash-field expiry notifications belong to commands already served on the origin; with
                    # no subscribers crossing the fork they must not be replayed here.
                    if isinstance(item.value, Hash):
                        item.value._expired_fields.clear()
                new_server.dbs[db_num] = new_db
        return new_server

    @staticmethod
    def get_server(key: str, version: VersionType, server_type: ServerType) -> FakeServer:
        if key not in FakeServer._servers_map:
            FakeServer._servers_map[key] = FakeServer(version=version, server_type=server_type)
        return FakeServer._servers_map[key]
