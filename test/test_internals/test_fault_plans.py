"""Server-level single-shot fault plans: in-process sync and async clients.

The two fault shapes model the two outcomes application retry code must tell apart:
* BEFORE_EXECUTION -- the request never ran: the client gets an error reply and the write is absent, so a retry is safe.
* AFTER_EXECUTION -- the write is committed but the reply is discarded and the connection dropped, so the client sees a
  ConnectionError and a retry may apply the write twice.
"""

from __future__ import annotations

import pytest
import redis

import fakeredis
from fakeredis import AFTER_EXECUTION, BEFORE_EXECUTION, FakeServer, FaultPlan, aioredis

pytestmark = [pytest.mark.fake]


# ------------------------------------------------------------------------------------------------- sync, in-process


def test_before_execution_error_skips_the_command_and_retry_succeeds():
    server = FakeServer()
    r = fakeredis.FakeStrictRedis(server=server)
    server.fault_before_execution("SET")

    with pytest.raises(redis.ResponseError) as exc_info:
        r.set("key", "value")
    assert "fault" in str(exc_info.value).lower()

    # The request never executed: no side effect...
    assert r.get("key") is None
    # ...the plan is spent...
    assert len(server.fault_plans) == 0
    # ...and the connection still works, so the retry lands the write exactly once.
    assert r.set("key", "value") is True
    assert r.get("key") == b"value"
    r.close()


def test_before_execution_custom_error_prefix():
    server = FakeServer()
    r = fakeredis.FakeStrictRedis(server=server)
    server.fault_before_execution("set", error="TRYAGAIN fake server is busy")

    with pytest.raises(redis.ResponseError) as exc_info:
        r.set("key", "value")
    # The TRYAGAIN prefix is what redis-py maps to its TryAgainError (the message itself drops the prefix).
    assert type(exc_info.value) is redis.exceptions.TryAgainError
    assert "fake server is busy" in str(exc_info.value)
    assert r.get("key") is None
    r.close()


def test_after_execution_commits_the_write_then_drops_the_connection():
    server = FakeServer()
    r = fakeredis.FakeStrictRedis(server=server)
    # An independent client observes committed state without touching r's broken connection.
    observer = fakeredis.FakeStrictRedis(server=server)
    server.fault_after_execution("SET")

    with pytest.raises(redis.ConnectionError):
        r.set("key", "value")

    # The write landed even though the caller saw a connection error...
    assert observer.get("key") == b"value"
    # ...the plan is spent, so the (reconnecting) retry goes through normally.
    assert r.set("key", "retry") is True
    assert observer.get("key") == b"retry"
    r.close()
    observer.close()


def test_after_execution_makes_a_retry_apply_the_write_twice():
    """The ambiguous-write case that retry guards are built for: INCRBY committed twice."""
    server = FakeServer()
    r = fakeredis.FakeStrictRedis(server=server)
    # redis-py encodes INCR as "INCRBY key 1", so the plan uses the wire command name.
    server.fault_after_execution("INCRBY")

    with pytest.raises(redis.ConnectionError):
        r.incr("counter")
    # The caller cannot know the first INCR committed...
    assert r.get("counter") == b"1"
    # ...so its (correct-looking) retry increments a second time.
    assert r.incr("counter") == 2
    r.close()


def test_fault_matches_only_the_armed_database():
    server = FakeServer()
    r_db0 = fakeredis.FakeStrictRedis(server=server, db=0)
    r_db1 = fakeredis.FakeStrictRedis(server=server, db=1)
    server.fault_before_execution("SET", db=1)

    assert r_db0.set("key", "in-db0") is True
    with pytest.raises(redis.ResponseError):
        r_db1.set("key", "in-db1")
    assert r_db1.get("key") is None
    r_db0.close()
    r_db1.close()


def test_fault_matches_only_the_armed_client_id():
    server = FakeServer()
    victim = fakeredis.FakeStrictRedis(server=server)
    other = fakeredis.FakeStrictRedis(server=server)
    victim_id = victim.client_id()
    assert other.client_id() != victim_id
    server.fault_before_execution("SET", client_id=victim_id)

    assert other.set("other-key", "other") is True
    with pytest.raises(redis.ResponseError):
        victim.set("victim-key", "victim")
    # The victim's write never executed; the other client's write (different key) committed normally.
    assert victim.get("victim-key") is None
    assert victim.get("other-key") == b"other"
    victim.close()
    other.close()


def test_fault_matches_only_the_armed_client_name():
    server = FakeServer()
    victim = fakeredis.FakeStrictRedis(server=server, client_name="victim")
    other = fakeredis.FakeStrictRedis(server=server, client_name="other")
    # Establish both connections so CLIENT SETNAME has run before the fault is armed.
    victim.ping()
    other.ping()
    server.fault_after_execution("SET", client_name="victim")

    with pytest.raises(redis.ConnectionError):
        victim.set("key", "victim")
    assert other.set("key", "other") is True
    victim.close()
    other.close()


def test_wildcard_command_matches_any_command():
    server = FakeServer()
    r = fakeredis.FakeStrictRedis(server=server)
    # Connect first: a wildcard plan armed before connecting would fire on the handshake (HELLO/CLIENT SETINFO).
    r.ping()
    server.fault_before_execution("*")

    with pytest.raises(redis.ResponseError):
        r.incr("counter")
    assert r.get("counter") is None
    r.close()


def test_faults_fire_once_in_arming_order():
    server = FakeServer()
    r = fakeredis.FakeStrictRedis(server=server)
    server.fault_before_execution("SET", error="ERR first")
    server.fault_before_execution("SET", error="ERR second")

    with pytest.raises(redis.ResponseError, match="first"):
        r.set("key", "a")
    with pytest.raises(redis.ResponseError, match="second"):
        r.set("key", "b")
    # Both plans consumed: later commands run normally.
    assert r.set("key", "c") is True
    assert r.get("key") == b"c"
    r.close()


def test_clear_faults_cancels_armed_plans():
    server = FakeServer()
    r = fakeredis.FakeStrictRedis(server=server)
    server.fault_before_execution("SET")
    server.fault_after_execution("INCRBY")
    server.clear_faults()

    assert r.set("key", "value") is True
    assert r.incr("counter") == 1
    r.close()


def test_queued_commands_in_multi_are_not_matched():
    """Faults fire on direct execution; a command queued in MULTI runs at EXEC and bypasses the plan."""
    server = FakeServer()
    r = fakeredis.FakeStrictRedis(server=server)
    server.fault_before_execution("SET")

    pipe = r.pipeline()
    pipe.set("key", "value")
    assert pipe.execute() == [True]
    assert r.get("key") == b"value"
    # The plan is still armed and fires on the next direct SET.
    with pytest.raises(redis.ResponseError):
        r.set("other", "x")
    r.close()


def test_invalid_fault_plan_is_rejected():
    with pytest.raises(ValueError):
        FaultPlan(command="SET", phase="not-a-phase")
    with pytest.raises(ValueError):
        FaultPlan(command="")


# ------------------------------------------------------------------------------------------------ async, in-process


@pytest.mark.asyncio
async def test_async_before_execution_error_skips_the_command():
    server = FakeServer()
    r = aioredis.FakeRedis(server=server)
    server.fault_before_execution("SET")

    with pytest.raises(redis.ResponseError):
        await r.set("key", "value")
    assert await r.get("key") is None
    assert await r.set("key", "value") is True
    assert await r.get("key") == b"value"
    await r.aclose()


@pytest.mark.asyncio
async def test_async_after_execution_commits_then_drops_connection():
    server = FakeServer()
    r = aioredis.FakeRedis(server=server)
    observer = aioredis.FakeRedis(server=server)
    server.fault_after_execution("SET")

    with pytest.raises(redis.asyncio.ConnectionError):
        await r.set("key", "value")
    assert await observer.get("key") == b"value"
    assert await r.set("key", "retry") is True
    assert await observer.get("key") == b"retry"
    await r.aclose()
    await observer.aclose()


@pytest.mark.asyncio
async def test_async_fault_phase_constants_are_exported():
    assert BEFORE_EXECUTION == "before_execution"
    assert AFTER_EXECUTION == "after_execution"
