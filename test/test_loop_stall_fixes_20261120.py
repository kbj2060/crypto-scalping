"""2026-11-20 loop_stall 고침 확인(네트워크·서버 기동 없음 · 임시 SQLite).
① 느린 읽기(0.5초)에도 bybit_liq_events 는 50ms 안에 돌아온다 ② 나눠 넣은 행(같은 ms 여러 건·창 밖으로 밀림)에서
덧붙이기 == 창 전체 다시 읽기 ③ 읽기 실패 한 번에 가진 값 유지 ④ 큰 응답 압축이 실행기로 가는 설정 ⑤ 음성 대조: 옛 동기 읽기는 ① 을 못 지킨다.
⑥ card30 예측(가짜 느린 모델)이 compute_situation 경로에서 루프를 안 막고 코인끼리 차례로 ⑦ 스레드 예측 == 직접 호출(ctx 값 포함)
⑧ 음성 대조: 미리 채우지 않는 옛 경로는 루프를 붙잡는다.
실행: PYTHONPATH=. python test/test_loop_stall_fixes_20261120.py"""
import asyncio
import sqlite3
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import dashboard.server as srv  # noqa: E402
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import make_mocked_request  # noqa: E402

REAL_READ = srv._read_only_rows


def make_db(d: Path, name: str) -> Path:
    p = d / name
    c = sqlite3.connect(p)   # 수집기(live_bybit_liquidation_collector_20261003.connect)와 같은 표·인덱스
    c.execute("CREATE TABLE bybit_liquidations(ts_ms INTEGER, symbol TEXT, side TEXT, qty REAL, price REAL, recv_ms INTEGER)")
    c.execute("CREATE INDEX bybit_liq_sym_ts ON bybit_liquidations(symbol, ts_ms)")
    c.commit(); c.close()
    return p


def put(p: Path, rows: list) -> None:
    c = sqlite3.connect(p)
    c.executemany("INSERT INTO bybit_liquidations VALUES (?,?,?,?,?,0)", rows)
    c.commit(); c.close()


def old_full(symbol: str, db: Path) -> tuple[list[dict], int | None]:
    """고치기 전 본문 그대로(캐시만 뺌) -- 창 전체를 호출한 스레드에서 다시 읽는다."""
    since = int((srv.time.time() - srv.FOOTPRINT_KEEP_BARS * srv.FOOTPRINT_BAR_SECONDS) * 1000)
    try:
        start = srv._read_only_rows(db, "SELECT min(ts_ms) FROM bybit_liquidations", [])[0][0]
        rows = srv._read_only_rows(db, "SELECT ts_ms, side, qty, price FROM bybit_liquidations WHERE symbol = ? AND ts_ms >= ? "
                                       "ORDER BY ts_ms", [symbol, since])
    except (sqlite3.Error, OSError, IndexError):
        return [], None
    return ([{"ts_ms": int(t), "side": "long" if sd == "Buy" else "short", "qty": float(q), "price": float(px),
              "usd": float(q) * float(px)} for t, sd, q, px in rows], int(start) if start is not None else None)


def settle(symbol: str, db: Path) -> tuple[list[dict], int | None]:
    srv.bybit_liq_events(symbol, db, ttl=0)
    while (symbol, db) in srv._BYBIT_LIQ_BUSY:
        time.sleep(0.005)
    return srv.bybit_liq_events(symbol, db, ttl=1e9)


def slow(*a):
    time.sleep(0.5)
    return REAL_READ(*a)


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    now_ms = int(time.time() * 1000)

    # ① 느린 읽기: 첫 호출(값 없음)도, 묵은 값 갱신도 기다리지 않는다
    db1 = make_db(tmp, "slow.sqlite")
    put(db1, [(now_ms - 5000, "ETHUSDT", "Buy", 1.0, 3000.0)])
    srv._read_only_rows = slow
    try:
        t0 = time.perf_counter(); got = srv.bybit_liq_events("ETHUSDT", db1, ttl=0); dt = time.perf_counter() - t0
        assert got == ([], None) and dt < 0.05, (got, dt)
        ev, start = settle("ETHUSDT", db1)
        assert len(ev) == 1 and start == now_ms - 5000, (ev, start)
        t0 = time.perf_counter(); got = srv.bybit_liq_events("ETHUSDT", db1, ttl=0); dt = time.perf_counter() - t0
        assert got == (ev, start) and dt < 0.05, (got, dt)   # 묵은 값을 바로 준다
        # ⑤ 음성 대조: 옛 방식은 같은 느린 읽기에서 0.5초 넘게 붙잡힌다(① 이 실패할 방식)
        t0 = time.perf_counter(); old_full("ETHUSDT", db1); dt_old = time.perf_counter() - t0
        assert dt_old >= 0.5, dt_old
        while ("ETHUSDT", db1) in srv._BYBIT_LIQ_BUSY:
            time.sleep(0.005)
    finally:
        srv._read_only_rows = REAL_READ
    print(f"① 느린 읽기에서도 즉시 반환 · ⑤ 옛 방식 {dt_old:.2f}초 붙잡힘")

    # ② 덧붙이기 == 창 전체 다시 읽기 (가짜 시계)
    db2 = make_db(tmp, "inc.sqlite")
    clock = [1_800_000_000.0]
    real_time = srv.time
    srv.time = types.SimpleNamespace(time=lambda: clock[0])
    try:
        T = int(clock[0] * 1000)
        W = srv.FOOTPRINT_KEEP_BARS * srv.FOOTPRINT_BAR_SECONDS * 1000
        put(db2, [(T - W - 1000, "ETHUSDT", "Buy", 1, 10),                  # 처음부터 창 밖
                  (T - W + 100_000, "ETHUSDT", "Sell", 2, 11),               # 다음 단계(시계 +300초)에 창 밖으로
                  (T - 100, "ETHUSDT", "Buy", 3, 12), (T - 100, "ETHUSDT", "Sell", 4, 13), (T - 100, "ETHUSDT", "Buy", 5, 14),
                  (T - 50, "BTCUSDT", "Buy", 9, 99)])                        # 다른 종목
        steps = 0

        def check(label: str) -> None:
            nonlocal steps
            assert settle("ETHUSDT", db2) == old_full("ETHUSDT", db2), label
            steps += 1

        check("첫 읽기(전체)")
        st0 = srv._BYBIT_LIQ_START[db2][0]
        clock[0] += 300
        put(db2, [(T - 100, "ETHUSDT", "Sell", 6, 15),                       # 이미 읽은 ms 에 늦게 온 행(꼬리 60초 안)
                  (T + 200_000, "ETHUSDT", "Buy", 7, 16), (T + 200_000, "ETHUSDT", "Buy", 8, 17)])
        check("덧붙이기 · 창 밖 자르기 · 같은 ms")
        assert srv._BYBIT_LIQ_START[db2][0] == st0, "10분 안 -- 전체 다시 읽기가 아니라 덧붙이기여야 한다"
        assert all(e["ts_ms"] >= T - W + 300_000 for e in srv._BYBIT_LIQ_CACHE[("ETHUSDT", db2)][1])
        clock[0] += 5
        put(db2, [(T + 200_000, "ETHUSDT", "Sell", 1, 18), (T + 304_000, "ETHUSDT", "Sell", 2, 19)])
        check("덧붙이기 2")
        assert srv._BYBIT_LIQ_START[db2][0] == st0
        clock[0] += srv.BYBIT_LIQ_FULL_S
        put(db2, [(T + 900_000, "ETHUSDT", "Buy", 3, 20)])
        check("10분 뒤 전체 다시 읽기")
        assert srv._BYBIT_LIQ_START[db2][0] == clock[0]
        n = len(srv._BYBIT_LIQ_CACHE[("ETHUSDT", db2)][1])
    finally:
        srv.time = real_time
    print(f"② 덧붙이기 == 창 전체 다시 읽기 {steps}단계(마지막 {n}건)")

    # ③ 읽기 실패 한 번 -- 가진 값을 지킨다(«모름»으로 안 바뀐다)
    before = srv.bybit_liq_events("ETHUSDT", db1, ttl=1e9)
    assert before[0] and before[1] is not None

    def boom(*a):
        raise sqlite3.OperationalError("database is locked")
    srv._read_only_rows = boom
    try:
        assert settle("ETHUSDT", db1) == before
    finally:
        srv._read_only_rows = REAL_READ
    assert settle("ETHUSDT", db1) == before   # 회복 뒤에도 같은 값
    print("③ 실패 한 번에 가진 값 유지")

    # ④ 큰 응답: 미들웨어를 지나면 zlib_executor_size = 4096(4KB 넘는 본문은 실행기에서 압축)
    app = srv.make_app()
    mw = next(m for m in app.middlewares if m.__name__ == "json_compress_etag")
    big = b"[" + b"1," * 2_000_000 + b"1]"

    async def handler(_req):
        return web.Response(body=big, content_type="application/json", zlib_executor_size=None)   # 서버 aiohttp 3.13 기본값
    raw = asyncio.run(handler(None))
    assert raw._zlib_executor_size is None
    resp = asyncio.run(mw(make_mocked_request("GET", "/api/footprint-week"), handler))
    assert resp._zlib_executor_size == 4096 and len(resp.body) > 4096, resp._zlib_executor_size
    print("④ 큰 응답 압축 = 실행기(zlib_executor_size 4096)")
    card30_checks(app)
    print("통과 8/8")


class SlowModel:
    """predict_proba 가 0.5초 걸리는 가짜 모델(서버 실측 0.4~3.0초 · 실제 sklearn 불필요)."""
    def __init__(self, w: float):
        self.w = w

    def predict_proba(self, row):
        time.sleep(0.5)
        q = 1 / (1 + srv.np.exp(-self.w * float(row[0, 0])))
        return srv.np.array([[1 - q, q]])


def cell(fn, name):
    return fn.__closure__[fn.__code__.co_freevars.index(name)]


def card30_checks(app) -> None:
    """⑥ compute_situation 경로에서 느린 예측도 루프를 안 막는다 ⑦ 스레드 예측 == 직접 호출 ⑧ 음성 대조: 옛 경로(루프 위 예측)는 ⑥ 을 못 지킨다."""
    start = next(f for f in app.on_startup if f.__name__ == "start_micro_ref")
    cs = cell(cell(start, "collect_micro_ref").cell_contents, "compute_situation").cell_contents
    pred, ctx = cell(cs, "_card30_pred").cell_contents, cell(cs, "_flow_read_ctx").cell_contents
    card30, card30_by, cctx = cell(pred, "card30").cell_contents, cell(pred, "card30_by").cell_contents, cell(ctx, "cctx").cell_contents
    card30["m"] = {"dir": {"cols": ["a"], "models": [SlowModel(1.0)], "k": 0.8, "base": 0.5},
                   "reach": {"cols": ["a"], "models": [SlowModel(-2.0)], "k": 1.0, "base": 0.3}}
    real_feat = srv.fr.card30_features
    srv.fr.card30_features = lambda bars: {"a": bars[-1]["close"] / 1000.0}
    now = 1_800_000_000.0
    bar0 = int(now) // 300 * 300

    def bars_upto(last: int) -> list[dict]:
        return [{"time": t, "high": 1.0, "low": 1.0, "close": 1000.0 + t % 997, "volume": 1.0, "taker": 0.5}
                for t in range(last - 299 * 300, last + 1, 300)] + [{"time": last + 300, "close": 9.0, "taker": 0.5}]   # 끝은 형성 중 봉(빠져야 함)

    async def no_net(*a, **k):
        return bars_upto(bar0 - 300) if a and isinstance(a[0], int) else []
    async def swr(*a, **k):
        return {}
    stubs = {"load_5m_day": no_net, "load_market_history": no_net, "swr_cached": swr, "_situation_inputs": lambda *a: {}}
    saved = {k: cell(cs, k).cell_contents for k in stubs}
    try:
        for k, v in stubs.items():
            cell(cs, k).cell_contents = v
        assets = list(cctx)[:3]

        async def worst_gap(coro_fn) -> float:
            gaps, done = [0.0], [False]

            async def ticker():
                t = time.perf_counter()
                while not done[0]:
                    await asyncio.sleep(0.05)
                    gaps[0] = max(gaps[0], time.perf_counter() - t - 0.05); t = time.perf_counter()
            tk = asyncio.create_task(ticker())
            await asyncio.sleep(0.06)
            await coro_fn()
            done[0] = True; await tk
            return gaps[0]

        def clear():
            card30.update(bar=None, p={}); card30_by.clear()

        # ⑦ 직접 호출(옛 경로) == 스레드
        b = [c for c in bars_upto(bar0 - 300) if c["time"] < bar0]
        clear(); direct = {a: dict(pred(b, a)) for a in assets}
        clear(); threaded = {a: dict(asyncio.run(asyncio.to_thread(pred, b, a))) for a in assets}
        assert direct == threaded and direct[assets[0]]["dir"] > 0, (direct, threaded)

        # ⑥ compute_situation 3코인 동시(같은 봉 마감) -- 루프 최대 지연
        clear()
        async def new_path():
            await asyncio.gather(*(cs(now, a) for a in assets))
        t0 = time.perf_counter(); gap_new = asyncio.run(worst_gap(new_path)); wall = time.perf_counter() - t0
        assert wall >= len(assets) * 1.0 - 0.05, wall   # 잠금: 코인마다 1.0초(모델 둘)가 겹치지 않고 차례로
        for a in assets:
            got = cctx[a].situation_state.get("ctx") or {}
            assert (got.get("dir_p"), got.get("reach_p")) == (direct[a]["dir"], direct[a]["reach"]), (a, got, direct[a])
            assert (card30 if a == "eth" else card30_by[a])["bar"] == bar0 - 300   # 갱신 시점 = 마지막 완결 봉
        assert gap_new < 0.2, gap_new

        # ⑧ 음성 대조: 미리 채우지 않으면(옛 compute_situation) _flow_read_ctx 가 루프 위에서 예측해 붙잡힌다
        clear()
        async def old_path():
            for a in assets:
                ctx(now, {}, a)
        gap_old = asyncio.run(worst_gap(old_path))
        assert gap_old >= 0.5, gap_old
    finally:
        for k, v in saved.items():
            cell(cs, k).cell_contents = v
        srv.fr.card30_features = real_feat
        card30.update(m=None, bar=None, p={})
    print(f"⑥ compute_situation {len(assets)}코인({','.join(assets)}) · 느린 예측(0.5초×2)에도 루프 최대 지연 {gap_new * 1000:.0f}ms · 차례로 {wall:.2f}초")
    print(f"⑦ 스레드 예측 == 직접 호출 {len(assets)}코인 (eth dir={direct[assets[0]]['dir']:.4f} reach={direct[assets[0]]['reach']:.4f}) · ctx 의 dir_p·reach_p 도 같음")
    print(f"⑧ 옛 경로(루프 위 예측) 루프 지연 {gap_old:.2f}초")


if __name__ == "__main__":
    main()
