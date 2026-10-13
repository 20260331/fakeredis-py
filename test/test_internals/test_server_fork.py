"""Tests for FakeServer.fork(): an independent server built from one atomic in-process snapshot.

Forked servers copy data and configuration, but never connection-bound state: sockets, subscriptions, WATCHes,
blocked clients or transient cursors all start empty.
"""

import threading
import time

import pytest
import redis

from fakeredis import FakeServer, FakeStrictRedis
from fakeredis.model import TimeSeries


def _wait_for(pubsub, predicate, timeout=1.0, ignore_subscribe_messages=True):
    deadline = time.time() + timeout
    while time.time() < deadline:
        message = pubsub.get_message(ignore_subscribe_messages=ignore_subscribe_messages, timeout=0.05)
        if message is not None and predicate(message):
            return message
    return None


def test_fork_copies_data_of_every_type_independently():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    r.set("string", "value")
    r.lpush("list", "a", "b")
    r.sadd("set", "m1", "m2")
    r.zadd("zset", {"m1": 1, "m2": 2.5})
    r.hset("hash", mapping={"f1": "v1", "f2": "v2"})
    r.xadd("stream", {"f": "1"})
    r.xadd("stream", {"f": "2"})

    fork = server.fork()
    rf = FakeStrictRedis(server=fork)
    assert rf.get("string") == b"value"
    assert rf.lrange("list", 0, -1) == [b"b", b"a"]
    assert rf.smembers("set") == {b"m1", b"m2"}
    assert rf.zrange("zset", 0, -1, withscores=True) == [(b"m1", 1.0), (b"m2", 2.5)]
    assert rf.hgetall("hash") == {b"f1": b"v1", b"f2": b"v2"}
    assert [entry[1] for entry in rf.xrange("stream")] == [{b"f": b"1"}, {b"f": b"2"}]

    # Writes on the fork never reach the origin.
    rf.set("string", "frozen")
    rf.delete("set")
    rf.hset("hash", "f3", "v3")
    assert r.get("string") == b"value"
    assert r.smembers("set") == {b"m1", b"m2"}
    assert r.hgetall("hash") == {b"f1": b"v1", b"f2": b"v2"}

    # And writes after the fork on the origin never reach it.
    r.set("string", "changed")
    r.sadd("set", "m3")
    assert rf.get("string") == b"frozen"
    assert rf.smembers("set") == set()


def test_fork_preserves_ttls_and_keeps_an_independent_expiry_heap():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    r.set("long", "v", ex=100)
    r.set("short", "v", px=50)

    fork = server.fork()
    rf = FakeStrictRedis(server=fork)
    assert 80_000 <= rf.pttl("long") <= 100_000
    assert 0 < rf.pttl("short") <= 50

    # The inherited short-TTL key expires on the fork through its rebuilt expiry heap.
    time.sleep(0.1)
    assert rf.get("short") is None

    # A TTL scheduled after the fork also expires, proving new heap entries work on the copy.
    rf.set("fresh", "v", px=50)
    time.sleep(0.1)
    assert rf.get("fresh") is None
    # Removing a TTL on the fork leaves the origin's TTL untouched.
    rf.persist("long")
    assert rf.pttl("long") == -1
    assert 80_000 <= r.pttl("long") <= 100_000


def test_fork_copies_every_database():
    server = FakeServer()
    r0 = FakeStrictRedis(server=server, db=0)
    r1 = FakeStrictRedis(server=server, db=1)
    r0.set("in-db-0", "x")
    r1.set("in-db-1", "y")

    fork = server.fork()
    assert FakeStrictRedis(server=fork, db=0).get("in-db-0") == b"x"
    assert FakeStrictRedis(server=fork, db=1).get("in-db-1") == b"y"
    assert FakeStrictRedis(server=fork, db=2).dbsize() == 0

    FakeStrictRedis(server=fork, db=0).flushdb()
    assert r0.get("in-db-0") == b"x"


def test_fork_copies_stream_groups_consumers_and_pel():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    r.xadd("events", {"f": "1"})
    r.xgroup_create("events", "g", id="0")
    r.xreadgroup("g", "c1", {"events": ">"}, count=1)
    assert r.xpending("events", "g")["pending"] == 1

    fork = server.fork()
    rf = FakeStrictRedis(server=fork)
    assert rf.xpending("events", "g")["pending"] == 1
    consumers = rf.xinfo_consumers("events", "g")
    assert [c["name"] for c in consumers] == [b"c1"]
    # Acknowledging on the fork only settles the copied PEL.
    pending_id = rf.xpending_range("events", "g", min="-", max="+", count=1)[0]["message_id"]
    assert rf.xack("events", "g", pending_id) == 1
    assert rf.xpending("events", "g")["pending"] == 0
    assert r.xpending("events", "g")["pending"] == 1

    # The copied group has already consumed the stream, and new entries stay on the fork.
    assert rf.xreadgroup("g", "c2", {"events": ">"}, count=10) == []
    rf.xadd("events", {"f": "2"})
    assert rf.xlen("events") == 2
    assert r.xlen("events") == 1


def test_fork_has_no_connections_or_subscriptions():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    r.set("k", "v")  # force one command so the socket exists
    pubsub = r.pubsub()
    pubsub.subscribe("news")
    assert _wait_for(pubsub, lambda m: m["type"] == "subscribe", ignore_subscribe_messages=False) is not None
    assert len(server.subscribers[b"news"]) == 1

    fork = server.fork()
    assert fork.sockets == []
    assert len(fork.subscribers[b"news"]) == 0

    rf = FakeStrictRedis(server=fork)
    # Publishing on the fork must not reach the origin's subscriber, nor the other way around by mistake.
    assert rf.publish("news", "from-fork") == 0
    assert _wait_for(pubsub, lambda m: m["data"] == b"from-fork", timeout=0.2) is None
    assert r.publish("news", "from-origin") == 1
    assert _wait_for(pubsub, lambda m: m["data"] == b"from-origin") is not None
    pubsub.close()


def test_fork_has_no_watches():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    other = FakeStrictRedis(server=server)
    r.set("watched", "0")

    pipe = r.pipeline()
    pipe.watch("watched")
    fork = server.fork()
    for db in fork.dbs.values():
        assert not any(db._watches.values())

    # The origin's watch survives the fork; only a write through the origin breaks it.
    other.set("watched", "1")
    pipe.multi()
    pipe.get("watched")
    with pytest.raises(redis.WatchError):
        pipe.execute()

    # Writes on the fork never had the power to invalidate it (separate watch sets).
    FakeStrictRedis(server=fork).set("other-watched", "x")


def test_fork_does_not_share_blocked_clients():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    popped = []

    def block():
        popped.append(r.blpop("queue", timeout=5))

    thread = threading.Thread(target=block)
    thread.start()
    time.sleep(0.3)  # let the client park in BLPOP

    fork = server.fork()
    rf = FakeStrictRedis(server=fork)
    # A blocking pop on the fork simply times out: it shares neither the waiting client nor the wake-up.
    assert rf.blpop("queue", timeout=0.5) is None
    # A push with the same key name on the fork does not wake the origin's client.
    rf.lpush("queue", "from-fork")
    time.sleep(0.2)
    assert thread.is_alive()
    # The origin still wakes exactly once from its own push.
    assert r.lpush("queue", "from-origin") == 1
    thread.join()
    assert popped == [(b"queue", b"from-origin")]
    assert rf.lpop("queue") == b"from-fork"


def test_fork_copies_config_and_acl_but_not_the_denial_log():
    server = FakeServer(config={b"requirepass": b"unused"})
    r = FakeStrictRedis(server=server)
    r.acl_setuser(
        "alice",
        enabled=True,
        reset=True,
        passwords=["+secret"],
        keys=["*"],
        channels=["*"],
        categories=["-@all"],
        commands=["+ping"],
    )

    fork = server.fork()
    assert fork.config == {b"requirepass": b"unused"}
    assert fork.config is not server.config
    alice = FakeStrictRedis(server=fork)
    # redis-py 8 sends credentials through HELLO, which the fake server does not authenticate, so log in with AUTH;
    # the single-threaded pool keeps every command on that authenticated connection.
    assert alice.auth(username="alice", password="secret") is True
    assert alice.ping()
    assert len(fork.acl._log) == 0
    with pytest.raises(redis.exceptions.ResponseError):
        alice.set("x", "y")
    assert len(fork.acl._log) == 1
    assert len(server.acl._log) == 0

    # ACL edits after the fork stay on their own server: the origin loosens and then drops the user,
    # while the fork keeps its copied restriction and user.
    r.acl_setuser("alice", categories=["+@all"])
    with pytest.raises(redis.exceptions.ResponseError):
        alice.set("x", "y")
    assert r.acl_deluser("alice") == b"OK"
    assert alice.ping()

    # Forking again snapshots the fork's own ACL state, not the origin's.
    fork2 = fork.fork()
    assert b"alice" in fork2.acl.get_users()
    assert b"alice" not in server.acl.get_users()
    assert fork2.acl._log == []


def test_fork_copies_cached_lua_scripts():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    sha = r.script_load("return 42")

    fork = server.fork()
    rf = FakeStrictRedis(server=fork)
    assert rf.evalsha(sha, 0) == 42

    r.script_flush()
    assert rf.evalsha(sha, 0) == 42
    with pytest.raises(redis.exceptions.ResponseError):
        FakeStrictRedis(server=server).evalsha(sha, 0)


def test_fork_restarts_client_ids_and_does_not_carry_pause_state():
    server = FakeServer()
    FakeStrictRedis(server=server).ping()
    fork = server.fork()
    rf = FakeStrictRedis(server=fork)
    assert rf.client_info()["id"] == 1
    assert fork.pause_until == 0.0
    assert fork.clients == {}


def test_fork_copies_timeseries_compaction_rules_and_reconnects_their_database():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    r.ts().create("src")
    r.ts().create("dst")
    r.ts().createrule("src", "dst", aggregation_type="sum", bucket_size_msec=10)
    r.ts().add("src", 1, 5)
    r.ts().add("src", 2, 7)
    # Closing the first 10ms bucket compacts it into the destination.
    r.ts().add("src", 11, 3)

    fork = server.fork()
    src_copy = fork.dbs[0][b"src"].value
    dst_copy = fork.dbs[0][b"dst"].value
    assert isinstance(src_copy, TimeSeries)
    # The copied series points at the copied database, not the origin's.
    assert src_copy._db is fork.dbs[0]
    assert src_copy.rules[0].dest_key is dst_copy

    rf = FakeStrictRedis(server=fork)
    assert rf.ts().range("dst", 0, 100) == [(0, 12.0)]
    # A sample on the fork closes the fork's open bucket and compacts into the fork's destination only.
    rf.ts().add("src", 21, 4)
    assert rf.ts().range("dst", 0, 100) == [(0, 12.0), (10, 3.0)]
    assert r.ts().range("dst", 0, 100) == [(0, 12.0)]


def test_fork_copies_bloom_filters():
    pytest.importorskip("probables")
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    assert r.execute_command("BF.ADD", "bf", "member1") == 1
    assert r.execute_command("BF.ADD", "bf", "member1") == 0

    fork = server.fork()
    rf = FakeStrictRedis(server=fork)
    assert rf.execute_command("BF.EXISTS", "bf", "member1") == 1
    assert rf.execute_command("BF.EXISTS", "bf", "member2") == 0
    rf.execute_command("BF.ADD", "bf", "member2")
    assert r.execute_command("BF.EXISTS", "bf", "member2") == 0


def test_fork_copies_json_documents():
    pytest.importorskip("jsonpath_ng")
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    r.json().set("doc", "$", {"a": [1, 2], "b": {"c": 3}})

    fork = server.fork()
    rf = FakeStrictRedis(server=fork)
    assert rf.json().get("doc") == {"a": [1, 2], "b": {"c": 3}}
    rf.json().set("doc", "$.a[0]", 99)
    assert r.json().get("doc", "$.a[0]") == [1]


def test_fork_snapshot_is_atomic_against_transactions():
    server = FakeServer()
    r = FakeStrictRedis(server=server)
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            pipe = r.pipeline(transaction=True)
            pipe.set("a", i)
            pipe.set("b", i)
            pipe.execute()
            i += 1

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(50):
            fork = server.fork()
            rf = FakeStrictRedis(server=fork)
            a, b = rf.mget("a", "b")
            assert a == b  # a snapshot never splits a MULTI/EXEC pair
    finally:
        stop.set()
        thread.join()
