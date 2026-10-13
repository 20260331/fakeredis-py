from __future__ import annotations

import threading
from dataclasses import dataclass

# A fault with this phase answers with an error reply before the command is executed, so the request has no effect and
# a retry cannot apply it twice.
BEFORE_EXECUTION = "before_execution"
# A fault with this phase executes the command (its writes are committed) but discards the reply and drops the
# connection, leaving the client unable to tell whether the request landed -- the ambiguous-write case.
AFTER_EXECUTION = "after_execution"

# Default error reply for BEFORE_EXECUTION plans. The "ERR " prefix is what redis-py maps to its generic ResponseError;
# pass e.g. error="TRYAGAIN ..." to make the client raise a retry-specific error class.
DEFAULT_ERROR_MESSAGE = "ERR fakeredis fault injected before command execution"

# Wildcard matching every command name ("*").
ANY_COMMAND = "*"


def _normalize_command(command: str | bytes) -> str:
    cmd = command.decode("utf-8") if isinstance(command, bytes) else command
    return cmd.strip().lower()


def _to_bytes(value: str | bytes) -> bytes:
    return value.encode("utf-8") if isinstance(value, str) else value


@dataclass(frozen=True)
class FaultPlan:
    """A single-shot fault to fire on one command, matched by client, database and command name.

    :param command: Command name, matched case-insensitively (``"set"``/``b"SET"``); use ``"*"`` to match any command.
        This is the name as sent on the wire, not the client method -- redis-py sends e.g. ``INCR`` as ``INCRBY key 1``,
        and subcommands are matched with their full name (``"client setname"``).
    :param db: Database number the client must be selected to; ``None`` matches every database.
    :param client_id: ``CLIENT ID`` of the client the fault is restricted to; ``None`` matches any client.
    :param client_name: Name set with ``CLIENT SETNAME`` the fault is restricted to; ``None`` matches any client.
    :param phase: ``BEFORE_EXECUTION`` to return an error reply without running the command, or ``AFTER_EXECUTION``
        to run the command and then discard the reply and close the connection.
    :param error: The error reply text for a ``BEFORE_EXECUTION`` fault (its prefix, e.g. ``ERR`` or ``TRYAGAIN``,
        decides the exception class redis-py raises).

    A plan fires at most once and is removed when it does. Only commands executed directly are matched: commands
    queued while a ``MULTI`` is open run at ``EXEC`` time and are never matched.
    """

    command: str | bytes
    db: int | None = None
    client_id: int | None = None
    client_name: str | bytes | None = None
    phase: str = BEFORE_EXECUTION
    error: str | bytes = DEFAULT_ERROR_MESSAGE

    def __post_init__(self) -> None:
        if self.phase not in (BEFORE_EXECUTION, AFTER_EXECUTION):
            raise ValueError(f"Invalid fault phase: {self.phase!r}")
        if not _normalize_command(self.command):
            raise ValueError("Fault plan requires a command name (or '*' to match every command)")
        object.__setattr__(self, "command", _normalize_command(self.command))
        if self.client_name is not None:
            object.__setattr__(self, "client_name", _to_bytes(self.client_name))
        object.__setattr__(self, "error", _to_bytes(self.error))

    def matches(self, *, command: str, db: int, client_id: int, client_name: bytes) -> bool:
        if self.command != ANY_COMMAND and self.command != command:
            return False
        if self.db is not None and self.db != db:
            return False
        if self.client_id is not None and self.client_id != client_id:
            return False
        return self.client_name is None or self.client_name == client_name

    @property
    def error_message(self) -> str:
        """The error reply text (SimpleError and the RESP encoder take ``str``)."""
        return _to_bytes(self.error).decode("utf-8")


class FaultPlanBook:
    """The single-shot fault plans armed on a :class:`FakeServer`, shared by every socket of that server."""

    def __init__(self) -> None:
        self._plans: list[FaultPlan] = []
        self._lock = threading.Lock()

    def arm(self, plan: FaultPlan) -> FaultPlan:
        """Schedule ``plan``; it fires on the first command it matches, in arming order."""
        with self._lock:
            self._plans.append(plan)
        return plan

    def clear(self) -> None:
        """Drop every armed but not yet fired plan."""
        with self._lock:
            self._plans.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._plans)

    def consume(self, *, command: str, db: int, client_id: int, client_name: bytes) -> FaultPlan | None:
        """Remove and return the first plan matching the command about to execute, or ``None``."""
        with self._lock:
            for index, plan in enumerate(self._plans):
                if plan.matches(command=command, db=db, client_id=client_id, client_name=client_name):
                    return self._plans.pop(index)
        return None
