"""Server-side command expectation scopes.

Asserting only the final keyspace state cannot tell whether an application issued extra commands, talked to the
wrong database or skipped a transaction it was supposed to use. This module observes commands on a
:class:`~fakeredis.FakeServer` -- the point every sync/async connection, pipeline and MULTI/EXEC block flows
through -- and offers two composable scopes:

* :func:`record_commands` records every well-formed command clients issue, together with the database it ran on
  and its raw result.
* :func:`expect_commands` additionally declares the exact ordered command sequence. Any mismatch is refused as a
  Redis error *before* the command executes (inside a MULTI the command fails to queue, so EXEC aborts and the
  transaction has no side effects); leaving a scope while expectations are unconsumed raises ``AssertionError``.

Scopes attach to a ``FakeServer`` and stack: nested/overlapping scopes each observe every command, so a recording
scope can wrap a strict one.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Union, cast

from fakeredis import _msgs as msgs
from fakeredis._core import FakeServer
from fakeredis._helpers import SimpleError
from fakeredis._typing import Self

__all__ = [
    "ANY",
    "CommandExpectation",
    "CommandExpectations",
    "RecordedCommand",
    "expect_commands",
    "record_commands",
]


class _AnyArgs:
    """Singleton matching any argument (or any whole argument tuple)."""

    _instance: _AnyArgs | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cast(Self, cls._instance)

    def __repr__(self) -> str:
        return "ANY"

    def __bool__(self) -> bool:
        return True


ANY = _AnyArgs()

_ArgMatcher = Union[bytes, str, int, float, bool, _AnyArgs, Callable[[bytes], bool]]
_ArgsSpec = Union[_AnyArgs, Callable[[tuple[bytes, ...]], bool], tuple[_ArgMatcher, ...]]
_CommandPredicate = Callable[[str, tuple[bytes, ...], int], bool]
ExpectationSpec = Union[str, tuple[Any, ...], list[Any], "CommandExpectation", _CommandPredicate]


def _encode_arg(value: _ArgMatcher) -> object:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bool):
        return str(value).lower().encode("utf-8")
    if isinstance(value, (int, float)):
        return str(value).encode("utf-8")
    return value


def _render_argv(name: str, args: tuple[bytes, ...]) -> str:
    rendered = name.upper()
    if args:
        rendered += " " + " ".join(arg.decode("utf-8", errors="replace") for arg in args)
    return rendered


@dataclass
class RecordedCommand:
    """One command observed by a scope, in issue order."""

    name: str
    args: tuple[bytes, ...]
    db: int
    result: Any = None
    #: Issued while a MULTI was queueing; ``result`` is the QUEUED reply until EXEC runs it.
    queued: bool = False
    #: The queued command was actually executed by EXEC and ``result`` was replaced with its real reply.
    executed: bool = False
    #: The final result is a Redis error (or, for an async blocking command, the decoded client error).
    error: bool = False
    #: Refused by a strict expectation (possibly another active scope's); the command produced no side effects.
    rejected: bool = False
    #: Async blocking command whose reply has not arrived yet; ``result`` is filled in when it does.
    pending: bool = False

    @property
    def command(self) -> tuple[Any, ...]:
        """The command name followed by its raw byte arguments."""
        return (self.name, *self.args)

    def _set_result(self, result: Any) -> None:
        self.pending = False
        self.result = result
        self.error = isinstance(result, SimpleError)

    def _mark_pending_blocking(self) -> None:
        self.pending = True

    def _set_transaction_result(self, result: Any) -> None:
        self.result = result
        self.queued = True
        self.executed = True
        self.pending = False
        self.error = isinstance(result, SimpleError)

    def _set_blocking_result(self, result: Any) -> None:
        # On the async path the reply has already been decoded, so errors are client-side exception instances.
        self.result = result
        self.pending = False
        self.error = isinstance(result, BaseException)


class CommandExpectation:
    """Matches one expected command by name, arguments and database.

    :param name: command name, case-insensitive, including the subcommand form (``"script load"``).
    :param args: ``ANY`` (the default) matches any arguments; a tuple matches the exact argument count with each
        element either an exact value (``bytes``/``str``/number, encoded as the client would), ``ANY`` or a
        ``bytes -> bool`` predicate; a callable matches the whole argument tuple.
    :param db: database index the command must be issued on; ``ANY`` matches any database.
    """

    _predicate: _CommandPredicate | None

    def __init__(self, name: str, *, args: _ArgsSpec = ANY, db: int | _AnyArgs = ANY) -> None:
        self.name = name.lower()
        self.args = args
        self.db = db
        self._predicate = None

    @classmethod
    def predicate(cls, predicate: _CommandPredicate) -> CommandExpectation:
        """Build an expectation from a ``(name, args, db) -> bool`` predicate."""
        expectation = cls.__new__(cls)
        expectation.name = ""
        expectation.args = ANY
        expectation.db = ANY
        expectation._predicate = predicate
        return expectation

    def matches(self, name: str, args: tuple[bytes, ...], db: int) -> bool:
        if self._predicate is not None:
            return bool(self._predicate(name, args, db))
        if name != self.name:
            return False
        if not isinstance(self.db, _AnyArgs) and db != self.db:
            return False
        if isinstance(self.args, _AnyArgs):
            return True
        if callable(self.args):
            return bool(self.args(args))
        if len(args) != len(self.args):
            return False
        return all(_match_arg(expected, actual) for expected, actual in zip(self.args, args))

    def describe(self) -> str:
        if self._predicate is not None:
            return getattr(self._predicate, "__name__", repr(self._predicate))
        rendered = self.name.upper()
        if not isinstance(self.args, _AnyArgs):
            if callable(self.args):
                rendered += " " + getattr(self.args, "__name__", repr(self.args))
            else:
                rendered += " " + " ".join(_describe_arg(arg) for arg in self.args)
        if not isinstance(self.db, _AnyArgs):
            rendered += f" (db {self.db})"
        return rendered


def _match_arg(expected: _ArgMatcher, actual: bytes) -> bool:
    if isinstance(expected, _AnyArgs):
        return True
    if callable(expected):
        return bool(expected(actual))
    return _encode_arg(expected) == actual


def _describe_arg(arg: _ArgMatcher) -> str:
    if isinstance(arg, _AnyArgs):
        return "*"
    if callable(arg):
        return getattr(arg, "__name__", repr(arg))
    encoded = _encode_arg(arg)
    return encoded.decode("utf-8", errors="replace") if isinstance(encoded, bytes) else str(encoded)


def _coerce_expectation(spec: ExpectationSpec) -> CommandExpectation:
    if isinstance(spec, CommandExpectation):
        return spec
    if isinstance(spec, str):
        return CommandExpectation(spec)
    if callable(spec):
        return CommandExpectation.predicate(spec)
    if isinstance(spec, (tuple, list)) and spec and isinstance(spec[0], str):
        return CommandExpectation(spec[0], args=tuple(spec[1:]))
    raise TypeError(f"Cannot interpret {spec!r} as a command expectation")


class CommandExpectations:
    """A composable server-side command scope.

    Prefer the :func:`record_commands` and :func:`expect_commands` factories. Instantiate directly for custom
    combinations, e.g. a scope that both records and enforces. While active, every command issued on the server
    is offered to the scope before it runs.
    """

    def __init__(
        self,
        server: FakeServer,
        *expectations: ExpectationSpec,
        strict: bool = False,
        name: str | None = None,
    ) -> None:
        self._server = server
        self._strict = strict
        self._name = name
        self._expectations: list[CommandExpectation] = [_coerce_expectation(spec) for spec in expectations]
        self._position = 0
        self.recorded: list[RecordedCommand] = []
        self._lock = threading.RLock()
        # Set once this scope itself refused a command; the mismatch was then surfaced to the client, so exiting
        # the block does not pile an AssertionError about unmet expectations on top.
        self._had_rejection = False

    @property
    def strict(self) -> bool:
        return self._strict

    @property
    def records(self) -> list[RecordedCommand]:
        """All observed commands, in issue order."""
        with self._lock:
            return list(self.recorded)

    @property
    def names(self) -> list[str]:
        """Lower-cased names of the observed commands, in issue order."""
        return [record.name for record in self.records]

    @property
    def pending(self) -> list[CommandExpectation]:
        """Expectations no command has matched yet."""
        with self._lock:
            return list(self._expectations[self._position :])

    def add_expectations(self, *expectations: ExpectationSpec) -> CommandExpectations:
        """Append further ordered expectations; useful for dynamically composed strict scopes."""
        with self._lock:
            self._expectations.extend(_coerce_expectation(spec) for spec in expectations)
        return self

    def verify(self) -> None:
        """Assert every expectation was matched. Called automatically when a strict scope exits cleanly."""
        with self._lock:
            pending = list(self._expectations[self._position :])
        if pending:
            rendered = ", ".join(expectation.describe() for expectation in pending)
            label = f" in {self._name}" if self._name else ""
            raise AssertionError(
                f"fakeredis command expectations{label} not satisfied: {len(pending)} expected command(s) "
                f"never issued: {rendered}"
            )

    def __enter__(self) -> Self:
        self._server.command_scopes.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        with self._lock:
            if self in self._server.command_scopes:
                self._server.command_scopes.remove(self)
        if self._strict and exc_type is None and not self._had_rejection:
            self.verify()
        return False

    # -- hooks invoked by the command dispatch layer, always with the server lock held for ``_observe`` --

    def _observe(
        self, name: str, args: tuple[bytes, ...], db: int, queued: bool
    ) -> tuple[RecordedCommand, SimpleError | None]:
        with self._lock:
            record = RecordedCommand(name=name, args=args, db=db, queued=queued)
            self.recorded.append(record)
            rejection: SimpleError | None = None
            if self._strict:
                if self._position < len(self._expectations):
                    expectation = self._expectations[self._position]
                    if expectation.matches(name, args, db):
                        self._position += 1
                    else:
                        rejection = SimpleError(
                            msgs.EXPECTATION_VIOLATION_MSG.format(
                                got=_render_argv(name, args),
                                db=db,
                                expected=expectation.describe(),
                            )
                        )
                        self._had_rejection = True
                else:
                    rejection = SimpleError(msgs.EXPECTATION_EXHAUSTED_MSG.format(got=_render_argv(name, args), db=db))
                    self._had_rejection = True
            return record, rejection


def _server_of(target: FakeServer | Any) -> FakeServer:
    if isinstance(target, FakeServer):
        return target
    pool = getattr(target, "connection_pool", None)
    server = getattr(pool, "connection_kwargs", {}).get("server")
    if isinstance(server, FakeServer):
        return server
    raise TypeError(
        "command expectation scopes need a FakeServer: pass the server, or a client created with "
        "FakeRedis(server=server)/FakeAsyncRedis(server=server)"
    )


def record_commands(target: FakeServer | Any) -> CommandExpectations:
    """Create a recording scope.

    :param target: a :class:`~fakeredis.FakeServer`, or a client bound to one via ``server=``.
    :returns: a context manager; inside the ``with`` block, ``.records`` holds the observed commands.
    """
    return CommandExpectations(_server_of(target), strict=False, name="record_commands")


def expect_commands(target: FakeServer | Any, *expectations: ExpectationSpec) -> CommandExpectations:
    """Create a strict scope expecting the given commands in issue order.

    Each expectation is one of: a command name (``"set"``), a ``(name, *args)`` tuple (values encoded as the
    client would; ``ANY`` matches a single argument), a :class:`CommandExpectation`, or a
    ``(name, args, db) -> bool`` predicate.

    Mismatching commands are refused with a Redis error before they run (inside MULTI they fail to queue, so
    EXEC aborts without side effects); if not every expectation is matched when the block exits,
    ``AssertionError`` is raised.
    """
    return CommandExpectations(_server_of(target), *expectations, strict=True, name="expect_commands")


# -----------------------------------------------------------------------------------------------------------
# Dispatch entry points used by BaseFakeSocket / the transaction mixin; not part of the public API.
# -----------------------------------------------------------------------------------------------------------


def begin_command(
    scopes: list[CommandExpectations],
    name: str,
    args: tuple[Any, ...],
    db: int,
    queued: bool,
) -> tuple[SimpleError | None, list[RecordedCommand]]:
    """Offer a command to every active scope before it executes.

    Returns the first rejection (if any) and the per-scope records created for the command. Every scope
    observes the command even when a sibling scope rejects it.
    """
    if not scopes:
        return None, []
    byte_args = tuple(args)
    records: list[RecordedCommand] = []
    rejection: SimpleError | None = None
    for scope in scopes:
        record, scope_rejection = scope._observe(name, byte_args, db, queued)
        records.append(record)
        if scope_rejection is not None and rejection is None:
            rejection = scope_rejection
    if rejection is not None:
        for record in records:
            record.rejected = True
    return rejection, records


def finish_command(records: list[RecordedCommand], result: Any, delayed: bool = False) -> None:
    """Attach the raw result of a dispatched command to its records.

    ``delayed`` marks an async blocking command whose reply arrives later; the socket resolves those records
    via :func:`resolve_blocking_reply`. Other commands that carry no reply (pub/sub, CLIENT REPLY) record
    ``None``.
    """
    for record in records:
        if delayed:
            record._mark_pending_blocking()
        else:
            record._set_result(result)


def resolve_transaction_item(records: list[RecordedCommand], result: Any) -> None:
    """Attach the real result of a queued command run by EXEC."""
    for record in records:
        record._set_transaction_result(result)


def resolve_blocking_reply(records: list[RecordedCommand], result: Any) -> None:
    """Attach the delayed reply of an async blocking command."""
    for record in records:
        record._set_blocking_result(result)
