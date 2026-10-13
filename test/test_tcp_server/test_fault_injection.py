"""Server-level single-shot fault plans against TcpFakeServer over real TCP sockets.

Over TCP the two fault shapes are exactly what a client sees against a real server:
* BEFORE_EXECUTION -- an error reply arrives on the open connection and the write is absent;
* AFTER_EXECUTION -- the write commits, no reply is sent and the server closes the connection, so the client's read
  ends with a ConnectionError.
"""

from __future__ import annotations

import time
from threading import Thread
from typing import Iterator

import pytest
import redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry

from fakeredis import FakeServer
from fakeredis._clients._tcp_server import TcpFakeServer

pytestmark = [pytest.mark.tcp_server]

# redis-py 8 retries once by default when a connection drops mid-command; the tests need to observe the raw failure and
# perform the retry themselves, so built-in retries are disabled.
_NO_RETRY = Retry(ExponentialBackoff(), 0)


@pytest.fixture
def tcp_fake_server() -> Iterator[tuple[TcpFakeServer, tuple[str, int]]]:
    server = TcpFakeServer(("127.0.0.1", 0))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    try:
        yield server, ("127.0.0.1", server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _client(address: tuple[str, int], **kwargs: object) -> redis.Redis:
    kwargs.setdefault("retry", _NO_RETRY)
    return redis.Redis(host=address[0], port=address[1], **kwargs)  # type: ignore[arg-type]


def test_before_execution_error_over_tcp(tcp_fake_server: tuple[TcpFakeServer, tuple[str, int]]):
    server, address = tcp_fake_server
    fake_server: FakeServer = server.fake_server
    fake_server.fault_before_execution("SET")

    with _client(address) as r:
        with pytest.raises(redis.ResponseError):
            r.set("key", "value")
        # Nothing was executed...
        assert r.get("key") is None
        # ...the fault was single-shot, so the retry succeeds (the client reconnects transparently if the server
        # closed the TCP link after delivering the error reply).
        assert r.set("key", "value") is True
        assert r.get("key") == b"value"


def test_after_execution_drops_reply_and_connection_over_tcp(
    tcp_fake_server: tuple[TcpFakeServer, tuple[str, int]],
):
    server, address = tcp_fake_server
    server.fake_server.fault_after_execution("SET")

    with _client(address) as r, pytest.raises(redis.ConnectionError):
        r.set("key", "value")

    # A fresh connection sees the committed write: the server applied it even though the first client got no reply.
    with _client(address) as r:
        assert r.get("key") == b"value"
        # The plan fired only once: further writes answer normally.
        assert r.set("key", "again") is True
        assert r.get("key") == b"again"


def test_after_execution_retried_write_applies_twice_over_tcp(
    tcp_fake_server: tuple[TcpFakeServer, tuple[str, int]],
):
    """Ambiguous write over TCP: a retried INCR is applied twice by the server."""
    server, address = tcp_fake_server
    # redis-py encodes INCR as "INCRBY key 1", so the plan uses the wire command name.
    server.fake_server.fault_after_execution("INCRBY")

    with _client(address) as r, pytest.raises(redis.ConnectionError):
        r.incr("counter")

    # The application did not get the reply of the first INCR; its retry on a new connection increments again.
    with _client(address) as r:
        assert r.get("counter") == b"1"
        assert r.incr("counter") == 2


def test_fault_scoped_by_database_over_tcp(tcp_fake_server: tuple[TcpFakeServer, tuple[str, int]]):
    server, address = tcp_fake_server
    server.fake_server.fault_before_execution("SET", db=1)

    with _client(address, db=0) as r0, _client(address, db=1) as r1:
        assert r0.set("key", "db0") is True
        with pytest.raises(redis.ResponseError):
            r1.set("key", "db1")
        assert r1.get("key") is None
        assert r0.get("key") == b"db0"


def test_fault_scoped_by_client_name_over_tcp(tcp_fake_server: tuple[TcpFakeServer, tuple[str, int]]):
    server, address = tcp_fake_server
    server.fake_server.fault_after_execution("SET", client_name="victim")

    with _client(address, client_name="victim") as victim, _client(address, client_name="bystander") as other:
        victim.ping()
        other.ping()
        with pytest.raises(redis.ConnectionError):
            victim.set("key", "victim")
        assert other.set("key", "other") is True

    with _client(address) as inspector:
        assert inspector.get("key") == b"other"


def test_server_threads_and_clients_are_cleaned_after_drop(
    tcp_fake_server: tuple[TcpFakeServer, tuple[str, int]],
):
    server, address = tcp_fake_server
    server.fake_server.fault_after_execution("SET")

    with _client(address) as r, pytest.raises(redis.ConnectionError):
        r.set("key", "value")

    deadline = time.time() + 5.0
    while time.time() < deadline and server.clients:
        time.sleep(0.02)
    assert server.clients == {}
    assert server.fake_server.sockets == []
