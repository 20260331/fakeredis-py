import asyncio
import contextlib
import threading
import time

import pytest

from fakeredis import FakeServer, FakeStrictRedis, QuiescenceTimeoutError
from fakeredis._core import (
    BLOCKED_CONDITION,
    PENDING_REPLIES_CONDITION,
    SUBSCRIPTIONS_CONDITION,
)
from fakeredis.aioredis import FakeRedis
from test import testtools

pytestmark = [pytest.mark.fake]


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


# --- synchronous ----------------------------------------------------------


def test_diagnostics_idle():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    assert r.ping()
    snap = server.diagnostics()
    assert snap.client_count >= 1
    assert (snap.blocked_count, snap.subscription_count, snap.pending_replies_count) == (0, 0, 0)
    assert snap.is_quiescent()


def test_diagnostics_blocked_sync():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    result: dict = {}
    thread = threading.Thread(target=lambda: result.update(v=r.blpop("list", timeout=5)))
    thread.start()
    _wait_until(lambda: server.diagnostics().blocked_count == 1)
    snap = server.diagnostics()
    blocked = [client for client in snap.clients if client.blocked]
    assert len(blocked) == 1
    assert blocked[0].command == "BLPOP"
    assert not snap.is_quiescent()

    r.rpush("list", "x")
    final = server.wait_for_quiescence(timeout=2)
    assert final.is_quiescent()
    thread.join()
    assert result["v"] == (b"list", b"x")


def test_wait_for_quiescence_timeout_summary():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    result: dict = {}
    thread = threading.Thread(target=lambda: result.update(v=r.blpop("list", timeout=5)))
    thread.start()
    _wait_until(lambda: server.diagnostics().blocked_count == 1)
    try:
        with pytest.raises(QuiescenceTimeoutError) as exc_info:
            server.wait_for_quiescence(timeout=0.2, poll_interval=0.02)
        message = str(exc_info.value)
        assert "did not become quiescent" in message
        assert "BLPOP" in message
        assert "id=" in message
        assert exc_info.value.diagnostics.blocked_count == 1
    finally:
        r.rpush("list", "x")
        thread.join()


def test_pending_replies_and_condition_selection():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    conn = testtools.pool_get_connection(r.connection_pool)
    conn.send_command("SET", "k", "v")

    snap = server.diagnostics()
    assert snap.pending_replies_count == 1
    assert snap.subscription_count == 0
    assert not snap.is_quiescent()
    # A pending reply is irrelevant when only blocked commands/subscriptions are waited for.
    assert snap.is_quiescent([BLOCKED_CONDITION, SUBSCRIPTIONS_CONDITION])

    assert conn.read_response() == b"OK"
    r.connection_pool.release(conn)
    assert server.wait_for_quiescence(timeout=1).is_quiescent()


def test_subscriptions_detected():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    pubsub = r.pubsub()
    pubsub.subscribe("ch")
    pubsub.get_message(timeout=1)  # consume the subscribe confirmation
    try:
        snap = server.diagnostics()
        assert snap.subscription_count == 1
        assert snap.channels.get("ch") == 1
        assert snap.pending_replies_count == 0
        assert not snap.is_quiescent()
        assert snap.is_quiescent([BLOCKED_CONDITION, PENDING_REPLIES_CONDITION])
    finally:
        pubsub.close()
    assert server.wait_for_quiescence(timeout=1).is_quiescent()


def test_diagnostics_is_read_only():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    r.set("k", "v")
    conn = testtools.pool_get_connection(r.connection_pool)
    conn.send_command("GET", "k")

    snap1 = server.diagnostics()
    snap2 = server.diagnostics()
    # Repeated snapshots do not consume the reply.
    assert snap1.pending_replies_count == snap2.pending_replies_count == 1
    # No Redis state changes: the key is intact and the reply is still waiting.
    assert r.get("k") == b"v"

    assert conn.read_response() == b"v"
    r.connection_pool.release(conn)


def test_invalid_conditions():
    server = FakeServer()
    with pytest.raises(ValueError, match="Unknown quiescence"):
        server.diagnostics().is_quiescent(["bogus"])
    with pytest.raises(ValueError, match="At least one"):
        server.wait_for_quiescence(timeout=1, conditions=[])


# --- asynchronous ---------------------------------------------------------


@pytest.mark.asyncio
async def test_async_blocked_detected_and_settles():
    server = FakeServer()
    r = FakeRedis(server=server)
    async with r.client() as conn:
        task = asyncio.get_running_loop().create_task(conn.blpop("missing", timeout=5))
        await asyncio.sleep(0.3)
        snap = server.diagnostics()
        assert snap.blocked_count == 1
        assert snap.clients[0].command == "BLPOP"

        await r.rpush("missing", "x")
        final = await server.wait_for_quiescence_async(timeout=2)
        assert final.is_quiescent()
        assert await task == (b"missing", b"x")
    await r.connection_pool.disconnect()


@pytest.mark.asyncio
async def test_async_wait_timeout_summary():
    server = FakeServer()
    r = FakeRedis(server=server)
    async with r.client() as conn:
        task = asyncio.get_running_loop().create_task(conn.blpop("missing", timeout=5))
        await asyncio.sleep(0.3)
        try:
            with pytest.raises(QuiescenceTimeoutError) as exc_info:
                await server.wait_for_quiescence_async(timeout=0.2, poll_interval=0.02)
            message = str(exc_info.value)
            assert "did not become quiescent" in message
            assert "BLPOP" in message
            assert exc_info.value.diagnostics.blocked_count == 1
        finally:
            task.cancel()
            # CancelledError is a BaseException, so suppress(Exception) would not catch it.
            with contextlib.suppress(asyncio.CancelledError):
                await task
    await r.connection_pool.disconnect()


@pytest.mark.asyncio
async def test_async_subscriptions_detected():
    server = FakeServer()
    r = FakeRedis(server=server)
    pubsub = r.pubsub()
    await pubsub.subscribe("ch")
    assert await pubsub.get_message(timeout=1) is not None  # consume the subscribe confirmation
    try:
        snap = server.diagnostics()
        assert snap.subscription_count == 1
        assert snap.channels.get("ch") == 1
        assert snap.pending_replies_count == 0
        assert not snap.is_quiescent()
        assert snap.is_quiescent([BLOCKED_CONDITION, PENDING_REPLIES_CONDITION])
    finally:
        await pubsub.unsubscribe("ch")
        while await pubsub.get_message(timeout=0.1) is not None:
            pass
        close = getattr(pubsub, "aclose", pubsub.close)
        await close()
    await server.wait_for_quiescence_async(timeout=1)
    await r.connection_pool.disconnect()


@pytest.mark.asyncio
async def test_async_pending_replies_detected():
    server = FakeServer()
    r = FakeRedis(server=server)
    conn = await testtools.pool_get_connection(r.connection_pool)
    await conn.send_command("SET", "k", "v")

    snap = server.diagnostics()
    assert snap.pending_replies_count == 1
    assert await conn.read_response() == b"OK"

    await server.wait_for_quiescence_async(timeout=1)
    await r.connection_pool.disconnect()
