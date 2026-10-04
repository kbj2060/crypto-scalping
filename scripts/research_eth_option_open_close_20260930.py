"""ETH 옵션 체결 -- 신규(open) vs 청산(close) 판별 가능성 (2026-09-30 연구, 커밋·대시보드 수정 없음).

원리: 체결 한 건(수량 a)은 매수자·매도자 두 «쪽»이다. 한 창에서 체결량 V, 미결제 변화 ΔOI 이면
  새로 연 쪽-수량 O = V + ΔOI, 닫은 쪽-수량 C = V − ΔOI   (쪽 단위 합 2V, 체결 수와 무관하게 정확)
모르는 것은 «누가» 열었나(테이커 vs 메이커)다. 테이커가 연 수량 o_t 는
  o_t ∈ [max(0, ΔOI), min(V, V + ΔOI)]   -- 폭 = V − |ΔOI|  ⇒ 확정 비율(거래량 가중) = |ΔOI| / V
테이커 매수 중 신규(TBO) ∈ [max(0, ΔOI − S), min(B, V + ΔOI)]  (B=테이커 매수량, S=테이커 매도량, V=B+S)

모드
  --selftest          판별 규칙 합성 assert
  --live SEC          Deribit 공개 WS 실측(이 PC 호출 허용): trades.option.ETH.100ms + incremental_ticker(전 종목)
                      + ticker.agg2(전 종목, 부하 비교용) + REST get_book_summary 20초 폴링 → OI 갱신 지연·체결 단위 판별
  --server-analyze    서버 option_trades × option_chain_snapshot(미리 pull 한 parquet) 사후 판별
출력 tmp/option_open_close_20260930/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "option_open_close_20260930"
WS = "wss://www.deribit.com/ws/api/v2"
REST = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
EPS = 1e-6


# ───────────────────────── 판별 규칙 ─────────────────────────
def bounds(V, B, S, d):
    """창 하나(또는 배열)의 판별 범위. V=B+S 체결량, d=ΔOI. 반환: dict of arrays."""
    V, B, S, d = (np.asarray(x, float) for x in (V, B, S, d))
    return {
        "opened_side": V + d, "closed_side": V - d,                       # 정확(쪽 단위)
        "taker_open_lo": np.maximum(0, d), "taker_open_hi": np.minimum(V, V + d),
        "tbo_lo": np.maximum(0, d - S), "tbo_hi": np.minimum(B, V + d),   # 테이커 매수 = 신규 롱
        "tso_lo": np.maximum(0, d - B), "tso_hi": np.minimum(S, V + d),   # 테이커 매도 = 신규 숏
        "inconsistent": np.abs(d) > V + EPS,                              # 체결 누락·시각 어긋남
        # «신규 순매수» N = TBO − TSO 의 정확한 범위(TBO+TSO 합 제약까지 반영). 나머지 (B−S)−N 이 «청산 순매수»
        "onet_hi": np.minimum(B, np.minimum(V, V + d)) - np.maximum(0, np.maximum(0, d) - np.minimum(B, np.minimum(V, V + d))),
        "onet_lo": -(np.minimum(S, np.minimum(V, V + d)) - np.maximum(0, np.maximum(0, d) - np.minimum(S, np.minimum(V, V + d)))),
    }


def classify_single(a, d):
    """체결 1건 창: +a 둘 다 신규 / −a 둘 다 청산 / 0 이전(누가 신규인지 모름) / 그 사이 부분 / 초과 불일치."""
    if abs(d) > a + EPS:
        return "inconsistent"
    if abs(d - a) <= EPS:
        return "both_open"
    if abs(d + a) <= EPS:
        return "both_close"
    if abs(d) <= EPS:
        return "transfer"
    return "partial"


def selftest() -> None:
    for d, want in [(5, "both_open"), (-5, "both_close"), (0, "transfer"), (2, "partial"), (7, "inconsistent")]:
        assert classify_single(5, d) == want, (d, classify_single(5, d))
    b = bounds(5, 5, 0, 5)   # 테이커 매수 1건, 둘 다 신규 → 테이커 신규 롱 5 확정
    assert b["taker_open_lo"] == b["taker_open_hi"] == 5 and b["tbo_lo"] == b["tbo_hi"] == 5
    b = bounds(5, 5, 0, -5)  # 둘 다 청산 → 숏 커버 5, 신규 0 확정
    assert b["taker_open_hi"] == 0 and b["tbo_hi"] == 0
    b = bounds(5, 5, 0, 0)   # 이전 → 테이커 신규 0~5 (폭 5 = 전부 모호)
    assert (b["taker_open_lo"], b["taker_open_hi"]) == (0, 5)
    b = bounds(10, 6, 4, 2)  # 다수 체결: O=12, C=8 정확, 테이커 신규 [2,10], TBO [0,6], TSO [0,4]
    assert b["opened_side"] == 12 and b["closed_side"] == 8
    assert (b["taker_open_lo"], b["taker_open_hi"]) == (2, 10)
    assert (b["tbo_lo"], b["tbo_hi"]) == (0, 6) and (b["tso_lo"], b["tso_hi"]) == (0, 4)
    assert (b["onet_lo"], b["onet_hi"]) == (-4, 6)
    b = bounds(10, 6, 4, 9)  # ΔOI 가 크면 좁아진다: TBO ≥ 5, TSO ≥ 3, 신규 순매수 [1,3]
    assert (b["tbo_lo"], b["tso_lo"]) == (5, 3) and b["taker_open_hi"] - b["taker_open_lo"] == 1
    assert (b["onet_lo"], b["onet_hi"]) == (1, 3)
    assert bounds(3, 3, 0, -4)["inconsistent"]
    # 폭 = V − |ΔOI| 항등식
    rng = np.random.default_rng(0)
    V = rng.uniform(1, 50, 200); d = rng.uniform(-1, 1, 200) * V; B = V * rng.uniform(0, 1, 200)
    b = bounds(V, B, V - B, d)
    assert np.allclose(b["taker_open_hi"] - b["taker_open_lo"], V - np.abs(d))
    assert (b["tbo_lo"] <= b["tbo_hi"] + EPS).all() and (b["tso_lo"] <= b["tso_hi"] + EPS).all()
    assert (b["onet_lo"] <= b["onet_hi"] + EPS).all()
    # 무작위 격자 전수: 가능한 (TBO,TSO) 조합의 N 극값 = 공식
    for _ in range(300):
        Bi, Si = rng.integers(0, 6, 2); Vi = Bi + Si
        if Vi == 0:
            continue
        di = int(rng.integers(-Vi, Vi + 1))
        Ns = [x - y for x in range(Bi + 1) for y in range(Si + 1) if max(0, di) <= x + y <= min(Vi, Vi + di)]
        bi = bounds(Vi, Bi, Si, di)
        assert (min(Ns), max(Ns)) == (bi["onet_lo"], bi["onet_hi"]), (Bi, Si, di)
    print("selftest OK")


# ───────────────────────── 1) 실시간 실측 ─────────────────────────
async def _record(sec: int, insts: list[str]) -> dict:
    import websockets, requests
    trades, inc, agg, polls = [], [], [], []
    stat = {"inc": [0, 0], "agg": [0, 0], "trades": [0, 0], "sub_err": [], "reconnect": []}
    t_end = time.time() + sec

    async def conn(channels, sink, key):
        while time.time() < t_end:          # 끊기면 다시 붙는다(09-30 첫 45분 기록이 20분 만에 끊겨 통째로 잃었다)
            try:
                await conn1(channels, sink, key)
            except Exception as e:
                stat["reconnect"].append((key, int(time.time() * 1000), f"{type(e).__name__}"))
                await asyncio.sleep(1)

    async def conn1(channels, sink, key):
        async with websockets.connect(WS, max_size=2**25, ping_interval=20, ping_timeout=20) as ws:
            for i in range(0, len(channels), 100):
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": 100 + i, "method": "public/subscribe",
                                          "params": {"channels": channels[i:i + 100]}}))
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": 9, "method": "public/set_heartbeat", "params": {"interval": 30}}))
            while time.time() < t_end:
                try:
                    raw = await asyncio.wait_for(ws.recv(), 5)
                except asyncio.TimeoutError:
                    continue
                r = int(time.time() * 1000)
                m = json.loads(raw)
                if "error" in m:
                    stat["sub_err"].append(m["error"]); continue
                p = m.get("params") or {}
                if p.get("type") == "test_request":
                    await ws.send(json.dumps({"jsonrpc": "2.0", "id": 8, "method": "public/test", "params": {}})); continue
                ch = p.get("channel")
                if not ch:
                    continue
                k = "trades" if ch.startswith("trades.") else key
                stat[k][0] += 1; stat[k][1] += len(raw)
                if k == "trades":
                    trades.extend([(r, t["timestamp"], t["instrument_name"], t["direction"], t["amount"], t["trade_seq"],
                                bool(t.get("block_trade_id"))) for t in p["data"]])
                else:
                    d = p["data"]
                    if sink is not None:  # agg2 는 부하(메시지·바이트)만 잰다
                        sink.append((r, d["timestamp"], d["instrument_name"], d.get("open_interest"), d.get("type", "")))

    async def poll():
        while time.time() < t_end:
            r0 = int(time.time() * 1000)
            try:
                res = (await asyncio.to_thread(requests.get, REST, params={"currency": "ETH", "kind": "option"}, timeout=20)).json()["result"]
                r1 = int(time.time() * 1000)
                polls.extend((r0, r1, x["instrument_name"], x["open_interest"], x.get("creation_timestamp")) for x in res)
            except Exception as e:
                print("poll fail", e, flush=True)
            await asyncio.sleep(20)

    await asyncio.gather(conn(["trades.option.ETH.100ms"] + [f"incremental_ticker.{i}" for i in insts], inc, "inc"),
                         conn([f"ticker.{i}.agg2" for i in insts], None, "agg"), poll())
    return dict(trades=trades, inc=inc, agg=agg, polls=polls, stat=stat)


def live(sec: int) -> None:
    import requests
    OUT.mkdir(parents=True, exist_ok=True)
    insts = [x["instrument_name"] for x in requests.get(REST, params={"currency": "ETH", "kind": "option"}, timeout=20).json()["result"]]
    print(f"종목 {len(insts)} · {sec}초 기록", flush=True)
    d = asyncio.run(_record(sec, insts))
    pd.DataFrame(d["trades"], columns=["recv", "ts", "inst", "dir", "amount", "seq", "block"]).to_parquet(OUT / "live_trades.parquet")
    pd.DataFrame(d["inc"], columns=["recv", "ts", "inst", "oi", "type"]).to_parquet(OUT / "live_inc.parquet")
    pd.DataFrame(d["polls"], columns=["r0", "r1", "inst", "oi", "cts"]).to_parquet(OUT / "live_polls.parquet")
    json.dump({"sec": sec, "n_inst": len(insts), **{k: v for k, v in d["stat"].items()}},
              open(OUT / "live_stat.json", "w"), ensure_ascii=False, default=str)
    live_analyze()


def _oi_series(tk: pd.DataFrame) -> pd.DataFrame:
    """incremental_ticker: OI 는 바뀔 때만 실린다 → 종목별 앞값 채움. 반환 (inst, ts, recv, oi)."""
    tk = tk.sort_values(["inst", "ts", "recv"]).copy()
    tk["oi"] = tk.groupby("inst")["oi"].ffill()
    return tk.dropna(subset=["oi"])


def live_analyze() -> None:
    st = json.load(open(OUT / "live_stat.json"))
    tr = pd.read_parquet(OUT / "live_trades.parquet").drop_duplicates(["inst", "seq"])
    for k_, t_, _ in st.get("reconnect", []):     # 재접속 틈 주변 체결은 창이 깨졌을 수 있다 -- 뺀다
        tr = tr[~((tr.recv > t_ - 60000) & (tr.recv < t_ + 10000))]
    inc = _oi_series(pd.read_parquet(OUT / "live_inc.parquet"))
    polls = pd.read_parquet(OUT / "live_polls.parquet")
    sec = st["sec"]; L = []
    p = lambda s="": (print(s, flush=True), L.append(s))
    p(f"# 실시간 실측 {sec}초 · ETH 옵션 {st['n_inst']}종목 · 구독 오류 {len(st['sub_err'])}건 {st['sub_err'][:2]} · 재접속 {st.get('reconnect', [])}")
    for k, lab in [("trades", "trades.option.ETH.100ms"), ("inc", "incremental_ticker ×전종목"), ("agg", "ticker.agg2 ×전종목")]:
        n, b = st[k]
        p(f"  {lab:28s} {n/sec:8.1f} msg/s · {b/sec/1024:8.1f} KB/s · 하루 {b/sec*86400/1e9:6.2f} GB")
    p(f"  체결 {len(tr)}건 · {tr.inst.nunique()}종목 · 블록 {int(tr.block.sum())}")
    p(f"  수신 지연(recv−ts) 중앙값: 체결 {np.median(tr.recv - tr.ts):.0f}ms · incremental {np.median(inc.recv - inc.ts):.0f}ms")
    per = inc.groupby("inst").ts.agg(lambda s: np.median(np.diff(np.unique(s))) if s.nunique() > 2 else np.nan)
    p(f"  incremental_ticker 종목당 갱신 간격 중앙값의 중앙값 {per.median():.0f}ms (p90 {per.quantile(.9):.0f}ms)")

    # 종목별 OI 계단(티커 ts, OI) -- 체결된 종목만
    step = {k: (g.drop_duplicates("ts", keep="last").ts.to_numpy(), g.drop_duplicates("ts", keep="last").oi.to_numpy())
            for k, g in inc[inc.inst.isin(tr.inst.unique())].groupby("inst")}
    oi_at = lambda inst, t: step[inst][1][np.searchsorted(step[inst][0], t, "right") - 1]   # 시각 t 까지의 마지막 OI
    # 테이커 주문 = 같은 (종목, ms, 방향) 체결 묶음
    od = tr.groupby(["inst", "ts", "dir"]).amount.sum().reset_index().sort_values(["inst", "ts"])

    # (a) OI 변화 이벤트 ↔ 가장 가까운 주문의 부호 있는 시차(+ = 티커가 체결 뒤). 🔴티커가 체결 ts 보다 먼저 찍히는 경우가 있다
    rows = []
    for inst, (ts, oi) in step.items():
        t = od.ts[od.inst == inst].to_numpy()
        for te in ts[1:][np.diff(oi) != 0]:
            rows.append(te - t[np.argmin(np.abs(t - te))])
    lag = np.array(rows)
    p(f"\n## OI 변화 이벤트 {len(lag)}건 ↔ 가장 가까운 체결(티커 ts − 체결 ts)")
    p(f"  중앙 {np.median(lag):.0f}ms · |시차|≤100ms {np.mean(np.abs(lag) <= 100):.1%} · ≤1초 {np.mean(np.abs(lag) <= 1000):.1%} · "
      f"티커가 먼저(<0) {np.mean(lag < 0):.1%} (최소 {lag.min():.0f}ms) · >1초 {np.sum(np.abs(lag) > 1000)}건")

    # (b) 주문 묶음(같은 종목 주문 간격 ≤ GAP) 단위 판별: OI 전 = 첫 주문 −1초, OI 후 = 마지막 주문 +1초
    GAP, PAD = 2000, 1000
    out = []
    for inst, g in od.groupby("inst"):
        if inst not in step:
            continue
        cid = (g.ts.diff().fillna(GAP + 1) > GAP).cumsum()
        for _, c in g.groupby(cid):
            t0_, t1_ = c.ts.min() - PAD, c.ts.max() + PAD
            if t0_ < step[inst][0][0] or t1_ > step[inst][0][-1]:
                continue
            B = c.amount[c.dir == "buy"].sum(); S = c.amount.sum() - B
            out.append((inst, len(c), B + S, B, S, oi_at(inst, t1_) - oi_at(inst, t0_)))
    w = pd.DataFrame(out, columns=["inst", "n", "V", "B", "S", "d"])
    w.to_parquet(OUT / "live_windows.parquet")
    for key, v in bounds(w.V, w.B, w.S, w.d).items():
        w["inc" if key == "inconsistent" else key] = v
    ok = w[~w.inc]
    dirdet = lambda g: 1 - ((g.tbo_hi - g.tbo_lo).sum() + (g.tso_hi - g.tso_lo).sum()) / g.V.sum()
    one = ok[ok.n == 1]
    p(f"\n## 주문 묶음(간격≤{GAP/1000:.0f}초, 앞뒤 {PAD/1000:.0f}초) {len(w)}개 · 불일치 {w.inc.mean():.1%} · 단일 주문 묶음 {len(one)}개 = 체결량 {one.V.sum()/ok.V.sum():.1%}")
    p(f"  확정(테이커 신규 합계) Σ|ΔOI|/ΣV = {np.abs(ok.d).sum()/ok.V.sum():.1%} · 방향별 확정 {dirdet(ok):.1%} · 단일 주문만 {np.abs(one.d).sum()/one.V.sum():.1%}")
    cls = pd.Series([classify_single(a, d) for a, d in zip(one.V, one.d)], index=one.index)
    p("  단일 주문 분류 (건수 · 거래량):")
    for c in ["both_open", "both_close", "transfer", "partial", "inconsistent"]:
        m = cls == c
        p(f"    {c:13s} {m.mean():6.1%} · {one.V[m].sum()/max(one.V.sum(),1):6.1%}")

    # (b2) 같은 체결·같은 WS OI 로 창만 1분·10분 격자로 바꾸면
    t0 = inc.ts.min()
    for m in (1, 10):
        stp = m * 60000; out = []
        for (inst, k_), c in od.assign(k=(od.ts - t0) // stp + 1).groupby(["inst", "k"]):
            a0, a1 = t0 + (k_ - 1) * stp, t0 + k_ * stp
            if inst not in step or a0 < step[inst][0][0] or a1 > step[inst][0][-1]:
                continue
            B = c.amount[c.dir == "buy"].sum(); S = c.amount.sum() - B
            out.append((B + S, B, S, oi_at(inst, a1) - oi_at(inst, a0)))
        g = pd.DataFrame(out, columns=["V", "B", "S", "d"])
        g = pd.concat([g, pd.DataFrame(bounds(g.V, g.B, g.S, g.d))], axis=1)
        g = g[~g.inconsistent]
        p(f"  {m:>2}분 격자 창 {len(g):>4} · 확정 {np.abs(g.d).sum()/g.V.sum():5.1%} · 방향별 확정 {dirdet(g):5.1%}")

    # (c) REST get_book_summary 의 OI 신선도: 폴 시점 WS OI 와 같은가
    last = inc.sort_values("recv")
    diffs = []
    for (r0, r1), g in polls.groupby(["r0", "r1"]):
        snap = last[last.recv <= r1].groupby("inst").oi.last()
        m = g.set_index("inst").oi.reindex(snap.index).dropna()
        diffs.append((len(m), float((np.abs(m - snap[m.index]) > EPS).mean()) if len(m) else np.nan))
    if diffs:
        p(f"\n## REST get_book_summary OI vs 같은 순간 WS OI: {len(diffs)}회 폴 · 불일치 종목 비율 중앙 {np.nanmedian([d[1] for d in diffs]):.2%}")
    (OUT / "live_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ───────────────────────── 2) 서버 이력 사후 판별 ─────────────────────────
def server_analyze() -> None:
    tr0 = pd.read_parquet(OUT / "srv_trades.parquet")
    ch = pd.read_parquet(OUT / "srv_chain.parquet")
    gaps = pd.read_parquet(OUT / "srv_gaps.parquet") if (OUT / "srv_gaps.parquet").exists() else pd.DataFrame()
    gaps = gaps[gaps.reason.str.startswith("unrecoverable")] if len(gaps) else gaps   # 재접속 틈은 REST 백필로 체결 완비
    L = []
    p = lambda s="": (print(s, flush=True), L.append(s))
    ch["t"] = pd.to_datetime(ch.recorded_at_utc, utc=True).dt.as_unit("ms").astype("int64")   # 🔴[us] 그대로 두면 단위가 틀린다
    oi = ch.pivot_table(index="t", columns="instrument_name", values="open_interest", aggfunc="last")
    col = {c: i for i, c in enumerate(oi.columns)}
    all_snaps = np.sort(ch.t.unique())
    all_snaps = all_snaps[np.searchsorted(all_snaps, tr0.ts_ms.min()) - 1:]          # 체결 수집 시작(09-27) 직전 스냅샷부터

    def build(snaps):
        """(종목 × 인접 스냅샷 창)별 V·B·S·ΔOI 와 판별 범위."""
        A = oi.reindex(snaps).to_numpy()
        tr = tr0[(tr0.ts_ms > snaps[0]) & (tr0.ts_ms <= snaps[-1])].copy()
        tr["k"] = np.searchsorted(snaps, tr.ts_ms.to_numpy(), "left")               # (snaps[k-1], snaps[k]] 창
        tr["B"] = np.where(tr.direction == "buy", tr.amount, 0.0); tr["S"] = tr.amount - tr.B
        g = tr.groupby(["instrument_name", "k"]).agg(n=("amount", "size"), V=("amount", "sum"), B=("B", "sum"), S=("S", "sum"),
                                                      blk=("is_block", "sum")).reset_index()
        g = g[g.instrument_name.isin(col)].copy()
        ci = g.instrument_name.map(col).astype(int).to_numpy(); k = g.k.to_numpy()
        g["ci"] = ci
        g["oi0"], g["oi1"] = A[k - 1, ci], A[k, ci]
        g["t1"] = snaps[k]; g["dt_min"] = (snaps[k] - snaps[k - 1]) / 60000
        bad = np.zeros(len(g), bool)
        for f, t_ in zip(gaps.get("from_ms", []), gaps.get("to_ms", [])):
            bad |= (snaps[k - 1] < t_) & (snaps[k] > f)
        g["new_inst"] = np.isnan(g.oi0)          # 창 시작 때 목록에 없던 종목(신규 상장) → OI 0 에서 시작
        g.loc[g.new_inst, "oi0"] = 0.0
        g = g[~np.isnan(g.oi1) & ~bad].copy()    # 창 끝에 사라진 종목(만기) 제외
        g["d"] = g.oi1 - g.oi0
        for key, v in bounds(g.V, g.B, g.S, g.d).items():
            g["inc" if key == "inconsistent" else key] = v
        return g, A

    agg, A = build(all_snaps)
    # 대조군: 체결 없는 (종목×창)의 ΔOI(= 스냅샷 시각 어긋남·체결 누락 규모)
    Q = np.diff(A, axis=0)
    has = np.zeros_like(Q, bool); has[agg.k.to_numpy() - 1, agg.ci.to_numpy()] = True
    qd = Q[~has & ~np.isnan(Q)]
    p(f"# 서버 사후 판별 · ETH 옵션 체결 {len(tr0):,}건 {pd.to_datetime(tr0.ts_ms.min(), unit='ms')} ~ {pd.to_datetime(tr0.ts_ms.max(), unit='ms')} UTC · 체인 스냅샷 {len(all_snaps)}회")
    p(f"  대조군 · 체결 없는 (종목×창) {len(qd):,}개 중 ΔOI≠0 {np.mean(np.abs(qd) > EPS):.2%} (|ΔOI| 합 {np.abs(qd).sum():,.0f} · 체결 있는 창 ΣV {agg.V.sum():,.0f})")
    agg.to_parquet(OUT / "srv_windows.parquet")
    tot = agg.V.sum()

    def tab(g, lab):
        ok = g[~g.inc]
        exact = ok.V[np.abs(np.abs(ok.d) - ok.V) <= EPS].sum()
        none_ = ok.V[np.abs(ok.d) <= EPS].sum()
        dirdet = 1 - ((ok.tbo_hi - ok.tbo_lo).sum() + (ok.tso_hi - ok.tso_lo).sum()) / max(ok.V.sum(), 1)
        p(f"  {lab:18s} 창 {len(g):>6,} · 체결량 {g.V.sum():>9,.0f} ({g.V.sum()/tot:5.1%}) · 불일치 {g.V[g.inc].sum()/g.V.sum():5.1%} · "
          f"정확 {exact/g.V.sum():5.1%} · ΔOI=0 {none_/g.V.sum():5.1%} · 확정 Σ|ΔOI|/ΣV {np.abs(ok.d).sum()/max(ok.V.sum(),1):5.1%} · "
          f"방향별 확정 {dirdet:5.1%}")

    p("\n## (종목×스냅샷창) 판별 표 (거래량 가중)")
    p("  정확 = |ΔOI|=V 인 창(테이커 몫 신규/청산 전량 확정) · ΔOI=0 = 전부 모호 · 확정 = 테이커 «신규 합계» 범위 폭 0 비율(=Σ|ΔOI|/ΣV)")
    p("  방향별 확정 = 테이커 매수(신규롱 vs 숏커버)·테이커 매도(신규숏 vs 롱청산) 각각의 범위 폭 0 비율")
    tab(agg, "전체")
    for lab, m in [("10분 창(09-28~)", agg.dt_min < 20), ("1시간 창(~09-28)", agg.dt_min >= 20),
                   ("체결 1건 창", agg.n == 1), ("2~5건", agg.n.between(2, 5)), ("6건+", agg.n >= 6), ("블록 포함", agg.blk > 0)]:
        if m.any():
            tab(agg[m], lab)
    one = agg[agg.n == 1]
    cls = pd.Series([classify_single(a, d) for a, d in zip(one.V, one.d)], index=one.index)
    p(f"\n## 체결 1건 창 {len(one):,}개 ({len(one)/len(agg):.1%} 창 · 체결량 {one.V.sum()/tot:.1%}) 분류")
    for c in ["both_open", "both_close", "transfer", "partial", "inconsistent"]:
        m = cls == c
        p(f"  {c:13s} 건수 {m.mean():6.1%} · 거래량 {one.V[m].sum()/max(one.V.sum(),1):6.1%}")

    # 창 길이 민감도: 10분 구간만 떼서 10·30·60분으로 묶으면 확정률이 얼마나 떨어지나
    ten = all_snaps[np.searchsorted(all_snaps, agg.loc[agg.dt_min < 20, "t1"].min()) - 1:]
    p("\n## 창 길이 민감도 (10분 스냅샷 구간만, 같은 체결을 창만 바꿔 재집계)")
    for stride in (1, 3, 6, 18, 36):
        g, _ = build(ten[::stride])
        ok = g[~g.inc]
        dirdet = 1 - ((ok.tbo_hi - ok.tbo_lo).sum() + (ok.tso_hi - ok.tso_lo).sum()) / ok.V.sum()
        p(f"  {10*stride:>4}분 창 · 창 {len(g):>5,} · 체결1건 창 체결량 {g.V[g.n == 1].sum()/g.V.sum():5.1%} · 불일치 {g.V[g.inc].sum()/g.V.sum():5.1%} · "
          f"확정 {np.abs(ok.d).sum()/ok.V.sum():5.1%} · 방향별 확정 {dirdet:5.1%}")

    # 4) 카드 항목 비교: 테이커 순매수 vs «신규 순매수»(범위) + «청산 순매수»
    ok = agg[~agg.inc].copy()
    ok["day"] = pd.to_datetime(ok.t1, unit="ms").dt.date
    ok["hour"] = pd.to_datetime(ok.t1, unit="ms").dt.floor("4h")
    ok["tbc_lo"], ok["tbc_hi"] = ok.B - ok.tbo_hi, ok.B - ok.tbo_lo      # 테이커 매수 = 숏 커버
    ok["tsc_lo"], ok["tsc_hi"] = ok.S - ok.tso_hi, ok.S - ok.tso_lo      # 테이커 매도 = 롱 청산
    sums = dict(V=("V", "sum"), B=("B", "sum"), S=("S", "sum"), dOI=("d", "sum"), opened=("opened_side", "sum"), closed=("closed_side", "sum"),
                tbo_lo=("tbo_lo", "sum"), tbo_hi=("tbo_hi", "sum"), tso_lo=("tso_lo", "sum"), tso_hi=("tso_hi", "sum"),
                tbc_lo=("tbc_lo", "sum"), tbc_hi=("tbc_hi", "sum"), tsc_lo=("tsc_lo", "sum"), tsc_hi=("tsc_hi", "sum"),
                onet_lo=("onet_lo", "sum"), onet_hi=("onet_hi", "sum"))
    d = ok.groupby("day").agg(**sums); d.to_csv(OUT / "srv_daily.csv")
    h = ok.groupby("hour").agg(**sums); h.to_csv(OUT / "srv_4h.csv")

    def line(lab, r):
        net = r.B - r.S
        return (f"  {lab} V {r.V:>7,.0f} · 테이커순매수 {net:>+7,.0f} · ΔOI {r.dOI:>+7,.0f} · 신규순매수 [{r.onet_lo:>+7,.0f},{r.onet_hi:>+7,.0f}] · "
                f"청산순매수 [{net - r.onet_hi:>+7,.0f},{net - r.onet_lo:>+7,.0f}] · 신규롱 [{r.tbo_lo:,.0f},{r.tbo_hi:,.0f}] 신규숏 [{r.tso_lo:,.0f},{r.tso_hi:,.0f}] "
                f"숏커버 [{r.tbc_lo:,.0f},{r.tbc_hi:,.0f}] 롱청산 [{r.tsc_lo:,.0f},{r.tsc_hi:,.0f}]")
    p("\n## 일별(ETH 옵션 계약 수 = ETH, 불일치 창 제외) -- 테이커 순매수를 신규·청산으로 쪼갠 [하한, 상한]")
    for day, r in d.iterrows():
        p(line(str(day), r))
    p("\n## 4시간별(UTC)")
    for hh, r in h.iterrows():
        p(line(hh.strftime("%m-%d %H시"), r))
    ok["cp"] = np.where(ok.instrument_name.str.endswith("-C"), "콜", "풋")
    p("\n## 예시 하루 2026-09-29(UTC) 콜·풋별 -- 카드의 cb/cs/pb/ps 가 신규·청산으로 어떻게 갈리나")
    for cp, r in ok[ok.day.astype(str) == "2026-09-29"].groupby("cp").agg(**sums).iterrows():
        p(line(cp, r))
    sign_flip = ((h.onet_lo > 0) & (h.B - h.S < 0)) | ((h.onet_hi < 0) & (h.B - h.S > 0))
    p(f"  4시간 칸 {len(h)}개 중 «신규 순매수» 부호가 범위 전체로 확정 {((h.onet_lo > 0) | (h.onet_hi < 0)).mean():.0%} · "
      f"그중 테이커 순매수와 부호 반대 {sign_flip.sum()}칸")
    (OUT / "srv_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--live", type=int)
    ap.add_argument("--live-analyze", action="store_true")
    ap.add_argument("--server-analyze", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.live:
        live(a.live)
    if a.live_analyze:
        live_analyze()
    if a.server_analyze:
        server_analyze()
    sys.exit(0)
