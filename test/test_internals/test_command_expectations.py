"""Tests for server-side command expectation scopes (fakeredis._expectations).

These assert the things final-state checks cannot: extra commands issued, the database a command ran on,
transaction wrappers being present, and strict rejection happening before any side effect -- across plain
clients, pipelines, MULTI/EXEC and the async client.
"""

from __future__ import annotations

import pytest
import redis

import fakeredis
from fakeredis import (
    ANY,
    CommandExpectation,
    FakeAsyncRedis,
    FakeRedis,
    FakeServer,
    expect_commands,
    record_commands,
)
from fakeredis._helpers import SimpleString

pytestmark = pytest.mark.fake


def _client(server: FakeServer | None = None) -> FakeRedis:
    client = FakeRedis(server=server) if server is not None else FakeRedis()
    client.ping()  # warm the connection so handshake traffic stays outside the scopes under test
    return client


# ------------------------------------------------------------------------------------------------------------------
# Recording mode
# ------------------------------------------------------------------------------------------------------------------


def test_record_command_sequence_args_db_and_results():
    server = FakeServer()
    r = _client(server)
    with record_commands(server) as scope:
        assert r.set("k", "v") is True
        assert r.get("k") == b"v"
        assert r.delete("k") == 1

    records = scope.records
    assert scope.names == ["set", "get", "del"]
    assert records[0].args == (b"k", b"v")
    assert records[0].db == 0
    assert isinstance(records[0].result, SimpleString) and records[0].result.value == b"OK"
    assert records[1].result == b"v"
    assert records[2].result == 1
    assert not any(record.error or record.rejected for record in records)
    assert records[0].command == ("set", b"k", b"v")


def test_record_sees_extra_commands_that_leave_no_final_trace():
    server = FakeServer()
    r = _client(server)
    with record_commands(server) as scope:
        r.set("k", "v")
        r.delete("k")  # final state looks untouched, but the command was still issued

    assert scope.names == ["set", "del"]
    assert [record.args[0] for record in scope.records] == [b"k", b"k"]


def test_record_tracks_selected_database():
    server = FakeServer()
    r = _client(server)
    r.set("k", "db0")
    try:
        with record_commands(server) as scope:
            r.select(1)
            r.set("k", "db1")

        select_record, set_record = scope.records
        assert select_record.name == "select"
        assert select_record.db == 0
        assert select_record.args == (b"1",)
        assert set_record.db == 1
        assert set_record.args == (b"k", b"db1")
    finally:
        r.select(0)


def test_record_marks_error_replies():
    server = FakeServer()
    r = _client(server)
    r.set("k", "v")
    with record_commands(server) as scope, pytest.raises(redis.ResponseError):
        r.lpop("k")  # WRONGTYPE against a string key

    (record,) = scope.records
    assert record.name == "lpop"
    assert record.error
    assert not record.rejected


def test_scope_attaches_to_client_bound_to_server():
    server = FakeServer()
    r = _client(server)
    with record_commands(r) as scope:
        r.set("k", "v")
    assert scope.names == ["set"]


def test_scope_needs_a_server():
    r = redis.Redis()  # a real client, no fake server
    with pytest.raises(TypeError):
        record_commands(r)


# ------------------------------------------------------------------------------------------------------------------
# Strict mode
# ------------------------------------------------------------------------------------------------------------------


def test_strict_happy_path_with_tuple_and_any():
    server = FakeServer()
    r = _client(server)
    with expect_commands(server, "set", ("get", b"k"), ("set", "k", ANY)):
        r.set("k", "v")
        assert r.get("k") == b"v"
        r.set("k", "whatever")


def test_strict_rejects_unexpected_command_before_side_effect():
    server = FakeServer()
    r = _client(server)
    with expect_commands(server, "get"):
        with pytest.raises(redis.RedisError) as exc_info:
            r.set("k", "v")
        assert "expectation violated" in str(exc_info.value)
    # The SET never ran: key absent and a follow-up command behaves normally.
    assert r.exists("k") == 0


def test_strict_argument_mismatch_does_not_execute():
    server = FakeServer()
    r = _client(server)
    with expect_commands(server, ("set", b"k", b"other")), pytest.raises(redis.ResponseError):
        r.set("k", "v")
    assert r.exists("k") == 0


def test_strict_database_mismatch_does_not_execute():
    server = FakeServer()
    r = _client(server)
    r.select(1)
    try:
        with expect_commands(server, CommandExpectation("set", db=0)), pytest.raises(redis.ResponseError):
            r.set("k", "v")  # runs on db 1, expectation demands db 0
        assert r.exists("k") == 0
    finally:
        r.select(0)


def test_strict_extra_command_is_refused():
    server = FakeServer()
    r = _client(server)
    with expect_commands(server, "set"):
        r.set("k", "v")
        with pytest.raises(redis.ResponseError):
            r.get("k")  # no further commands expected


def test_strict_missing_commands_raise_on_exit():
    server = FakeServer()
    r = _client(server)
    with pytest.raises(AssertionError, match="never issued"), expect_commands(server, "set", "get"):
        r.set("k", "v")


def test_strict_predicate_expectation():
    server = FakeServer()
    r = _client(server)
    with expect_commands(
        server,
        lambda name, args, db: name == "set" and args[0] == b"k" and db == 0,
    ):
        r.set("k", "v")


def test_strict_argument_predicate():
    server = FakeServer()
    r = _client(server)
    with expect_commands(server, ("set", "k", lambda value: value.isdigit())):
        with pytest.raises(redis.ResponseError):
            r.set("k", "not-a-number")
        r.set("k", "42")
    assert r.get("k") == b"42"


def test_strict_does_not_mask_unrelated_exception_and_skips_verify():
    server = FakeServer()
    with pytest.raises(RuntimeError, match="boom"), expect_commands(server, "get"):
        raise RuntimeError("boom")


def test_rejected_command_is_recorded_by_sibling_recording_scope():
    server = FakeServer()
    r = _client(server)
    with record_commands(server) as recorder, expect_commands(server, "get"), pytest.raises(redis.ResponseError):
        r.set("k", "v")
    (record,) = recorder.records
    assert record.name == "set"
    assert record.rejected
    assert record.error
    assert r.exists("k") == 0


# ------------------------------------------------------------------------------------------------------------------
# Pipelines and transactions
# ------------------------------------------------------------------------------------------------------------------


def test_non_transactional_pipeline_sequence():
    server = FakeServer()
    r = _client(server)
    with expect_commands(server, "set", "get", "del"):
        pipe = r.pipeline(transaction=False)
        pipe.set("k", "v")
        pipe.get("k")
        pipe.delete("k")
        assert pipe.execute() == [True, b"v", 1]
    assert r.exists("k") == 0


def test_transactional_pipeline_requires_multi_and_exec():
    server = FakeServer()
    r = _client(server)
    with expect_commands(server, "multi", "set", "exec"):
        pipe = r.pipeline(transaction=True)
        pipe.set("k", "v")
        assert pipe.execute() == [True]
    assert r.get("k") == b"v"


def test_bare_command_is_rejected_when_transaction_was_expected():
    server = FakeServer()
    r = _client(server)
    with expect_commands(server, "multi", "set", "exec"), pytest.raises(redis.ResponseError):
        r.set("k", "v")  # bypassed MULTI/EXEC entirely
    assert r.exists("k") == 0


def test_transaction_mismatch_fails_to_queue_and_aborts_without_side_effects():
    server = FakeServer()
    r = _client(server)
    with pytest.raises(redis.ResponseError), expect_commands(server, "multi", "get", "exec"):
        pipe = r.pipeline(transaction=True)
        pipe.set("k", "v")
        pipe.execute()
    assert r.exists("k") == 0
    # The connection is usable again and subsequent scopes behave normally.
    with expect_commands(server, "ping"):
        assert r.ping() is True


def test_recorded_transaction_results_are_filled_in_at_exec():
    server = FakeServer()
    r = _client(server)
    with record_commands(server) as scope:
        pipe = r.pipeline(transaction=True)
        pipe.set("k", "v")
        pipe.get("k")
        assert pipe.execute() == [True, b"v"]

    assert scope.names == ["multi", "set", "get", "exec"]
    multi_record, set_record, get_record, exec_record = scope.records
    assert not multi_record.queued
    assert set_record.queued and set_record.executed
    assert isinstance(set_record.result, SimpleString) and set_record.result.value == b"OK"
    assert get_record.queued and get_record.executed and get_record.result == b"v"
    assert exec_record.name == "exec"
    assert [item.value if isinstance(item, SimpleString) else item for item in exec_record.result] == [
        b"OK",
        b"v",
    ]


def test_discard_leaves_queued_records_unexecuted():
    server = FakeServer()
    r = _client(server)
    with record_commands(server) as scope:
        r.execute_command("MULTI")
        r.execute_command("SET", "k", "v")
        r.execute_command("DISCARD")

    multi_record, set_record, discard_record = scope.records
    assert multi_record.name == "multi"
    assert set_record.queued and not set_record.executed
    assert isinstance(set_record.result, SimpleString) and set_record.result.value == b"QUEUED"
    assert discard_record.name == "discard"
    assert r.exists("k") == 0


def test_watch_failed_exec_leaves_queued_commands_unexecuted():
    server = FakeServer()
    r = _client(server)
    other = _client(server)
    other.set("k", "v")
    with record_commands(server) as scope:
        pipe = r.pipeline(transaction=True)
        pipe.watch("k")
        other.set("k", "changed-by-other")
        pipe.multi()
        pipe.set("k", "mine")
        with pytest.raises(redis.WatchError):
            pipe.execute()

    queued_set = [record for record in scope.records if record.queued]
    assert len(queued_set) == 1
    assert not queued_set[0].executed
    assert r.get("k") == b"changed-by-other"


# ------------------------------------------------------------------------------------------------------------------
# Composition and multiple connections
# ------------------------------------------------------------------------------------------------------------------


def test_nested_scopes_compose():
    server = FakeServer()
    r = _client(server)
    with record_commands(server) as outer:
        with expect_commands(server, "set"):
            r.set("k", "v")
        r.get("k")  # inner strict scope is gone; the outer recorder still sees it

    assert outer.names == ["set", "get"]


def test_all_connections_sharing_a_server_are_observed():
    server = FakeServer()
    r1 = _client(server)
    r2 = _client(server)
    with record_commands(server) as scope:
        r1.set("a", "1")
        r2.set("b", "2")
    assert [record.args[0] for record in scope.records] == [b"a", b"b"]


# ------------------------------------------------------------------------------------------------------------------
# Async client
# ------------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_record_and_strict():
    server = FakeServer()
    r = FakeAsyncRedis(server=server)
    await r.ping()
    with record_commands(server) as scope:
        await r.set("k", "v")
        assert await r.get("k") == b"v"
    assert scope.names == ["set", "get"]
    assert scope.records[1].result == b"v"

    with expect_commands(server, "set", "get"):
        await r.set("k", "w")
        assert await r.get("k") == b"w"


@pytest.mark.asyncio
async def test_async_strict_rejects_before_side_effect():
    server = FakeServer()
    r = FakeAsyncRedis(server=server)
    await r.ping()
    with expect_commands(server, "get"), pytest.raises(redis.RedisError):
        await r.set("k", "v")
    assert await r.exists("k") == 0


@pytest.mark.asyncio
async def test_async_transaction_sequence_and_results():
    server = FakeServer()
    r = FakeAsyncRedis(server=server)
    await r.ping()
    with record_commands(server) as scope:
        pipe = r.pipeline(transaction=True)
        pipe.set("k", "v")
        pipe.get("k")
        assert await pipe.execute() == [True, b"v"]
    assert scope.names == ["multi", "set", "get", "exec"]
    assert scope.records[1].executed
    assert scope.records[2].result == b"v"


@pytest.mark.asyncio
async def test_async_blocking_command_result_is_filled_when_reply_arrives():
    server = FakeServer()
    r = FakeAsyncRedis(server=server)
    await r.ping()
    with record_commands(server) as scope:
        # Times out immediately with a nil reply; the reply is produced asynchronously after dispatch returns.
        assert await r.blpop("missing", timeout=0.1) is None

    (record,) = scope.records
    assert record.name == "blpop"
    assert record.pending is False
    assert record.result is None
    assert not record.error


def test_public_api_surface():
    assert {
        "record_commands",
        "expect_commands",
        "CommandExpectations",
        "CommandExpectation",
        "RecordedCommand",
        "ANY",
    } <= set(fakeredis.__all__)
