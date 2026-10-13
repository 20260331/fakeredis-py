from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest
import redis

import fakeredis

pytestmark = [pytest.mark.fake, pytest.mark.timeout(10)]


@pytest.fixture
def clients() -> tuple[fakeredis.FakeServer, Any, Any]:
    server = fakeredis.FakeServer()
    r1 = fakeredis.FakeRedis(server=server)
    r2 = fakeredis.FakeRedis(server=server)
    yield server, r1, r2
    server.command_barrier.clear()
    r1.close()
    r2.close()


def _start(client: Any, func: Callable[[Any], Any]) -> tuple[threading.Thread, dict[str, Any]]:
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["result"] = func(client)
        except Exception as exc:  # reported back on join instead of escaping the thread
            outcome["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    return thread, outcome


def test_parks_after_validation_before_the_write(clients: tuple[fakeredis.FakeServer, Any, Any]) -> None:
    server, r1, r2 = clients
    server.command_barrier.arm("set", b"foo")

    thread, outcome = _start(r1, lambda c: c.set("foo", "a"))
    try:
        assert server.command_barrier.wait_reached(5)
        # The parked command passed validation but has not written anything.
        assert r2.get("foo") is None
        # Another client completes a full command meanwhile: the parked thread is not holding the server lock.
        assert r2.set("foo", "b") is True
    finally:
        server.command_barrier.release()
    thread.join()

    assert "error" not in outcome
    assert outcome["result"] is True
    # The parked command resumes after the other client's write, so the ordering is deterministic.
    assert r2.get("foo") == b"a"


def test_parked_read_observes_the_other_clients_write(clients: tuple[fakeredis.FakeServer, Any, Any]) -> None:
    server, r1, r2 = clients
    r1.set("foo", "before")
    server.command_barrier.arm("get", b"foo")

    thread, outcome = _start(r1, lambda c: c.get("foo"))
    try:
        assert server.command_barrier.wait_reached(5)
        assert r2.set("foo", "after") is True
    finally:
        server.command_barrier.release()
    thread.join()

    assert "error" not in outcome
    # The parked GET reads the database only after being released, so it sees the interleaved write.
    assert outcome["result"] == b"after"


def test_one_shot_matching_command_from_other_client_runs(clients: tuple[fakeredis.FakeServer, Any, Any]) -> None:
    server, r1, r2 = clients
    server.command_barrier.arm("set", b"foo")

    thread, _ = _start(r1, lambda c: c.set("foo", "a"))
    try:
        assert server.command_barrier.wait_reached(5)
        # The barrier disarmed itself on park: the other client's matching SET must not park behind the first one.
        assert r2.set("foo", "b") is True
        assert r2.set("foo", "c") is True
    finally:
        server.command_barrier.release()
    thread.join()


def test_key_filter_only_parks_commands_on_the_armed_key(clients: tuple[fakeredis.FakeServer, Any, Any]) -> None:
    server, r1, _ = clients
    server.command_barrier.arm("set", "other")  # str key is accepted too

    assert r1.set("foo", "a") is True
    assert server.command_barrier.wait_reached(0.05) is False

    thread, outcome = _start(r1, lambda c: c.set("other", "x"))
    try:
        assert server.command_barrier.wait_reached(5)
    finally:
        server.command_barrier.release()
    thread.join()
    assert "error" not in outcome
    assert r1.get("other") == b"x"


def test_bytes_command_name_is_accepted(clients: tuple[fakeredis.FakeServer, Any, Any]) -> None:
    server, r1, _ = clients
    server.command_barrier.arm(b"SET", b"foo")

    thread, _ = _start(r1, lambda c: c.set("foo", "a"))
    try:
        assert server.command_barrier.wait_reached(5)
    finally:
        server.command_barrier.release()
    thread.join()


def test_invalid_arity_does_not_park_and_keeps_barrier_armed(
    clients: tuple[fakeredis.FakeServer, Any, Any],
) -> None:
    server, r1, _ = clients
    server.command_barrier.arm("set")

    with pytest.raises(redis.ResponseError):
        r1.execute_command("SET")  # SET needs a key; the error comes back instead of parking
    assert server.command_barrier.wait_reached(0.05) is False

    thread, outcome = _start(r1, lambda c: c.set("foo", "a"))
    try:
        # Still armed: the first valid SET after the failed one parks.
        assert server.command_barrier.wait_reached(5)
    finally:
        server.command_barrier.release()
    thread.join()
    assert "error" not in outcome


def test_queued_transaction_commands_are_not_interrupted(
    clients: tuple[fakeredis.FakeServer, Any, Any],
) -> None:
    server, r1, r2 = clients
    server.command_barrier.arm("set", b"foo")

    pipe = r1.pipeline(transaction=True)
    pipe.set("foo", "a")
    pipe.set("bar", "b")
    # Queued commands bypass the barrier; otherwise this would block while the same thread waits for the barrier.
    assert pipe.execute() == [True, True]

    assert r2.get("foo") == b"a"
    assert r2.get("bar") == b"b"
    assert server.command_barrier.wait_reached(0.05) is False


def test_exec_barrier_parks_before_the_transaction_runs_anything(
    clients: tuple[fakeredis.FakeServer, Any, Any],
) -> None:
    server, r1, r2 = clients
    r1.set("foo", "old")
    server.command_barrier.arm("exec")

    def run_transaction(client: Any) -> list[Any]:
        pipe = client.pipeline(transaction=True)
        pipe.set("foo", "new")
        return pipe.execute()

    thread, outcome = _start(r1, run_transaction)
    try:
        assert server.command_barrier.wait_reached(5)
        # EXEC parked before running the queue, so the transaction's write is not visible yet.
        assert r2.get("foo") == b"old"
    finally:
        server.command_barrier.release()
    thread.join()

    assert "error" not in outcome
    assert outcome["result"] == [True]
    assert r2.get("foo") == b"new"


def test_script_issued_commands_are_not_interrupted(clients: tuple[fakeredis.FakeServer, Any, Any]) -> None:
    pytest.importorskip("lupa")
    server, r1, r2 = clients
    server.command_barrier.arm("set", b"foo")

    # EVAL is the top-level command (it does not match "set"); the redis.call inside runs through _run_command and
    # bypasses the barrier so script atomicity is preserved.
    r1.eval("redis.call('set', KEYS[1], ARGV[1])", 1, "foo", "from-script")

    assert r2.get("foo") == b"from-script"
    assert server.command_barrier.wait_reached(0.05) is False


def test_clear_releases_a_parked_command(clients: tuple[fakeredis.FakeServer, Any, Any]) -> None:
    server, r1, _ = clients
    r1.set("foo", "kept")
    server.command_barrier.arm("get", b"foo")

    thread, outcome = _start(r1, lambda c: c.get("foo"))
    assert server.command_barrier.wait_reached(5)

    server.command_barrier.clear()
    thread.join()
    assert "error" not in outcome
    assert outcome["result"] == b"kept"
    # clear() also disarmed: subsequent commands run without stopping.
    assert r1.get("foo") == b"kept"


def test_wait_reached_times_out_when_nothing_parks(clients: tuple[fakeredis.FakeServer, Any, Any]) -> None:
    server, _, _ = clients
    server.command_barrier.arm("del", b"never-sent")
    assert server.command_barrier.wait_reached(0.05) is False


def test_barrier_can_be_rearmed_for_another_round(clients: tuple[fakeredis.FakeServer, Any, Any]) -> None:
    server, r1, r2 = clients
    for value in ("a", "b"):
        server.command_barrier.arm("set", b"foo")
        thread, outcome = _start(r1, lambda c, v=value: c.set("foo", v))
        try:
            assert server.command_barrier.wait_reached(5)
        finally:
            server.command_barrier.release()
        thread.join()
        assert "error" not in outcome
        assert r2.get("foo") == value.encode()
