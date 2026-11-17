"""同号重传确认修复：逐块核验长度/SHA-256，以及删除/同号重建与慢重传交错。

覆盖的关键不变量：
1. 已发布接收的重传只凭 Content-Length 头不再放行——实际读取整包，
   同长度异内容必须 422，原件/接收状态/额度均不变；
2. 字节与该次已发布原包完全一致才沿用成功结果；
3. 成功结果必须对应本次实际核验的**有效**接收：另一实例先删除则重传 404，
   绝不允许用删除前读到的旧记录应答成功；
4. 同号重建（删除后重新声明）期间的重传按当前在途接收走完整发布；
5. 同一接收串行化期间，其他接收仍可正常入湖。
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import declare_payload, make_parquet
from app import main as main_mod
from app.errors import ContentMismatch, ReceiptNotFound
from app.service import _verify_published_retransmit, upload_package
from app.storage import blob_path


def _decl(client, receipt_id, data, **kw):
    return client.put(
        f"/receipts/{receipt_id}/declaration",
        json=declare_payload(receipt_id, data, **kw),
    )


def _publish(client, rid, data):
    r = client.post(f"/receipts/{rid}/content", content=data,
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200, r.text
    return r


def _same_length_other_content(data: bytes) -> bytes:
    """构造与原包等长但 SHA-256 不同的另一份 Parquet 字节。"""
    table = pq.read_table(pa.BufferReader(data))
    col = table.column("salinity").to_pylist()
    col[0] = col[0] + 1.5
    patched = table.set_column(
        table.schema.get_field_index("salinity"),
        "salinity",
        pa.array(col, pa.float64()),
    )
    sink = pa.BufferOutputStream()
    pq.write_table(patched, sink)
    other = sink.getvalue().to_pybytes()
    if len(other) < len(data):  # 压缩偶发差异时补齐到完全等长
        other = other + b"\x00" * (len(data) - len(other))
    other = other[: len(data)]
    assert len(other) == len(data)
    assert hashlib.sha256(other).hexdigest() != hashlib.sha256(data).hexdigest()
    return other


# ---------------------------------------------------------------------------
# 1. 同长度异内容重传：明确拒绝，原件/状态/额度不变
# ---------------------------------------------------------------------------

def test_same_length_different_content_rejected(client, tmp_path, helpers, settings):
    data = make_parquet(tmp_path / "a.parquet")
    rid = "RCV-RT-01"
    _decl(client, rid, data)
    _publish(client, rid, data)
    sha = helpers.sha256_of(data)
    blob = pathlib.Path(blob_path(settings, sha))
    quota_before = client.get("/quota").json()
    receipt_before = client.get(f"/receipts/{rid}").json()

    other = _same_length_other_content(data)

    # 不发 Content-Length（分块传输场景）：头信息完全帮不上忙，也必须拒绝
    r = client.post(
        f"/receipts/{rid}/content",
        content=other,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "content_mismatch"
    assert "摘要不符" in r.json()["error"]["message"]

    # 原件、接收状态、额度均不变
    assert blob.read_bytes() == data
    receipt_after = client.get(f"/receipts/{rid}").json()
    assert receipt_after["state"] == "PUBLISHED"
    assert receipt_after["blob_sha256"] == receipt_before["blob_sha256"]
    assert receipt_after["bytes_received"] == receipt_before["bytes_received"]
    assert receipt_after["actual_sha256"] == receipt_before["actual_sha256"]
    assert client.get("/quota").json() == quota_before
    assert client.get(f"/receipts/{rid}/download").content == data
    # 仅诊断字段 last_error 记录拒绝原因
    assert "摘要不符" in receipt_after["last_error"]


def test_length_mismatch_retransmit_rejected_even_without_header(client, tmp_path, helpers):
    data = make_parquet(tmp_path / "a.parquet")
    rid = "RCV-RT-02"
    _decl(client, rid, data)
    _publish(client, rid, data)

    # 权威判断以实际收齐的字节为准，不依赖 Content-Length 头
    r = client.post(f"/receipts/{rid}/content", content=data + b"\x00",
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "content_mismatch"
    assert client.get(f"/receipts/{rid}").json()["state"] == "PUBLISHED"


def test_exact_same_bytes_retransmit_confirmed(client, tmp_path, helpers):
    data = make_parquet(tmp_path / "a.parquet")
    rid = "RCV-RT-03"
    _decl(client, rid, data)
    _publish(client, rid, data)

    for i in range(3):
        r = client.post(f"/receipts/{rid}/content", content=data)
        assert r.status_code == 200, f"第 {i} 次重传失败"
        out = r.json()
        assert out["state"] == "PUBLISHED"
        assert out["actual_sha256"] == helpers.sha256_of(data)
        assert out["bytes_received"] == len(data)
    assert client.get(f"/receipts/{rid}/download").content == data


def test_verify_helper_rejects_wrong_digest_directly(
    database, tmp_path, helpers, settings
):
    """直接核验单元：同长度异摘要 422 且不改状态；记录不存在 → 404。"""
    data = make_parquet(tmp_path / "a.parquet")
    rid = "RCV-RT-04"
    sha = helpers.sha256_of(data)
    blob = pathlib.Path(blob_path(settings, sha))
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    with database.transaction() as conn:
        conn.execute(
            "INSERT INTO receipts(receipt_id, station_id, obs_date, declared_size, "
            "declared_sha256, state, bytes_received, actual_sha256, blob_sha256) "
            "VALUES (?, 'BUOY-001', '2026-11-15', ?, ?, 'PUBLISHED', ?, ?, ?)",
            (rid, len(data), sha, len(data), sha, sha),
        )

    before = dict(database.query_one("SELECT * FROM receipts WHERE receipt_id = ?", (rid,)))
    with pytest.raises(ContentMismatch):
        _verify_published_retransmit(
            settings, database, rid,
            total_bytes=len(data), digest="b" * 64,
        )
    after = dict(database.query_one("SELECT * FROM receipts WHERE receipt_id = ?", (rid,)))
    # 除 last_error / updated_at 外完全不变
    for key in ("state", "blob_sha256", "bytes_received", "actual_sha256",
                "declared_size", "declared_sha256"):
        assert before[key] == after[key]

    # 记录不存在 → 404，而不是套用任何旧结果
    with pytest.raises(ReceiptNotFound):
        _verify_published_retransmit(
            settings, database, "GONE-RID",
            total_bytes=len(data), digest=sha,
        )


# ---------------------------------------------------------------------------
# 2. 删除 / 同号重建与慢重传交错
# ---------------------------------------------------------------------------

class _GateStream:
    """首个字节处可挂起的异步字节流：ready 通知已持锁，proceed 放行。"""

    def __init__(self, data: bytes, ready: threading.Event, proceed: threading.Event):
        self._data = data
        self._ready = ready
        self._proceed = proceed
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        self._ready.set()
        # 在线程池线程上阻塞，不卡住事件循环
        await asyncio.to_thread(self._proceed.wait)
        return self._data


def _start_gated_upload(rid: str, data: bytes):
    """在独立线程+独立事件循环中启动一个挂起在首个字节的上传。"""
    ready = threading.Event()
    proceed = threading.Event()
    outcome: dict = {}

    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            row = loop.run_until_complete(
                upload_package(
                    main_mod.settings, main_mod.db, rid,
                    _GateStream(data, ready, proceed), None,
                )
            )
            outcome["row"] = row
        except Exception as exc:  # noqa: BLE001 - 测试要捕获全部结果
            outcome["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=worker)
    t.start()
    assert ready.wait(timeout=10), "上传线程未能在预期时间内持锁并挂起"
    return t, outcome, proceed


def test_delete_committed_first_then_retransmit_returns_404(client, tmp_path, helpers):
    """交错顺序 A（真实 API）：另一实例先删除提交，慢重传不得确认旧记录。"""
    data = make_parquet(tmp_path / "a.parquet")
    rid = "RCV-RACE-DEL1"
    _decl(client, rid, data)
    _publish(client, rid, data)

    assert client.delete(f"/receipts/{rid}").status_code == 204

    # 删除已落库后到达的重传：必须 404，而不是沿用删除前的成功结果
    r = client.post(f"/receipts/{rid}/content", content=data)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "receipt_not_found"
    assert client.get(f"/receipts/{rid}").status_code == 404


def test_inflight_delete_bypassing_lock_kills_stale_confirmation(
    client, tmp_path, helpers
):
    """交错顺序 B：重传已持锁卡在收包，记录在其核验前被删（直接 DB 模拟
    绕过 flock 的他实例落库）→ 锁内重查必须发现记录失效并 404。"""
    data = make_parquet(tmp_path / "a.parquet")
    rid = "RCV-RACE-DEL2"
    _decl(client, rid, data)
    _publish(client, rid, data)

    t, outcome, proceed = _start_gated_upload(rid, data)

    # 重传持锁在途：直接在 SQLite 提交删除（flock 不拦截 DB 写）
    with main_mod.db.transaction() as conn:
        conn.execute("DELETE FROM receipts WHERE receipt_id = ?", (rid,))

    proceed.set()
    t.join(timeout=15)
    assert isinstance(outcome.get("error"), ReceiptNotFound), outcome
    assert "row" not in outcome


def test_api_delete_during_inflight_upload_gets_busy_then_upload_confirms(
    client, tmp_path, helpers
):
    """交错顺序 C：持锁在途期间 API 删除得到 409（既有契约），放行后
    重传对仍有效的记录核验成功——成功与删除永不共存。"""
    data = make_parquet(tmp_path / "a.parquet")
    rid = "RCV-RACE-DEL3"
    _decl(client, rid, data)
    _publish(client, rid, data)

    t, outcome, proceed = _start_gated_upload(rid, data)

    r = client.delete(f"/receipts/{rid}")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "receipt_busy"
    # 记录仍然有效
    assert client.get(f"/receipts/{rid}").json()["state"] == "PUBLISHED"

    proceed.set()
    t.join(timeout=15)
    assert outcome.get("row", {}).get("state") == "PUBLISHED", outcome

    # 上传结束后删除即可生效
    assert client.delete(f"/receipts/{rid}").status_code == 204
    assert client.get(f"/receipts/{rid}").status_code == 404


def test_rebuild_same_id_then_retransmit_goes_through_full_publish(
    client, tmp_path, helpers
):
    """删除后同号重建为在途接收：重传不得套用旧成品，按新声明完整发布。"""
    data = make_parquet(tmp_path / "a.parquet")
    rid = "RCV-RACE-RBLD"
    _decl(client, rid, data)
    _publish(client, rid, data)
    assert client.delete(f"/receipts/{rid}").status_code == 204

    r = _decl(client, rid, data)
    assert r.status_code == 200
    assert r.json()["state"] == "UPLOADING"

    r = client.post(f"/receipts/{rid}/content", content=data)
    assert r.status_code == 200
    assert r.json()["state"] == "PUBLISHED"
    assert client.get(f"/receipts/{rid}/download").content == data


def test_retransmit_while_other_receipts_keep_ingesting(client, tmp_path, helpers):
    """同一接收的重传串行化不影响其他接收入湖。"""
    data = make_parquet(tmp_path / "a.parquet")
    rid = "RCV-RACE-BUSY"
    _decl(client, rid, data)
    _publish(client, rid, data)

    t, outcome, proceed = _start_gated_upload(rid, data)

    other = make_parquet(tmp_path / "b.parquet", salt=4.0)
    orid = "RCV-RACE-OTHER"
    assert _decl(client, orid, other).status_code == 200
    r = client.post(f"/receipts/{orid}/content", content=other)
    assert r.status_code == 200 and r.json()["state"] == "PUBLISHED"
    assert client.get("/receipts").status_code == 200
    assert client.get("/quota").status_code == 200

    proceed.set()
    t.join(timeout=15)
    assert outcome.get("row", {}).get("state") == "PUBLISHED", outcome


def test_repeated_race_iterations_never_confirm_dead_record(client, tmp_path, helpers):
    """压力交错：多轮 在途删除/正常放行 交替，成功应答与记录已删永不共存。"""
    data = make_parquet(tmp_path / "a.parquet")
    for i in range(8):
        rid = f"RCV-RACE-FUZZ{i}"
        _decl(client, rid, data)
        _publish(client, rid, data)

        t, outcome, proceed = _start_gated_upload(rid, data)
        if i % 2 == 0:
            # 在途直接删库 → 重传必须 404
            with main_mod.db.transaction() as conn:
                conn.execute("DELETE FROM receipts WHERE receipt_id = ?", (rid,))
            proceed.set()
            t.join(timeout=15)
            assert isinstance(outcome.get("error"), ReceiptNotFound), (rid, outcome)
            assert client.get(f"/receipts/{rid}").status_code == 404
        else:
            # 正常核验成功后，删除才能生效
            proceed.set()
            t.join(timeout=15)
            assert outcome.get("row", {}).get("state") == "PUBLISHED", (rid, outcome)
            assert client.delete(f"/receipts/{rid}").status_code == 204
            assert client.get(f"/receipts/{rid}").status_code == 404
