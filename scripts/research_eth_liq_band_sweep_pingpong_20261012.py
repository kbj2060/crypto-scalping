"""청산 띠 왕복 스윕 자동매매 백테스트 + «역스윕 뚫림» 손절 비교 (2026-10-12).
사용자 «아래로 청산 히트맵 찍으면서 청산 실시간 데이터가 찍히고 스윕하는 방향으로 포지션 잡고 다시 청산 데이터 쪽으로 가면서
실제 청산 실시간 데이터 찍히고 다시 스윕하는 자동매매 … 백테스트 · 역스윕되면서 뚫렸을 때 손절 분석».

    python3 scripts/research_eth_liq_band_sweep_pingpong_20261012.py DATA_DIR OUT_DIR [--selftest]
    DATA_DIR: eth_tail_risk_1m.parquet(서버 lake tail_risk_1m, t = 분 시작 ms) · dl/ETHUSDT-1m-YYYY-MM-DD.zip(vision 일별)
    1분봉 월별 = /home/kbj20/crypto-scalping/data/binance_vision/klines1m

청산 띠 = 대시보드 풋프린트 밀도 층 그대로: 5분 경계 t 마다 _window_at(닫힌 1시간봉 + 그 시간의 닫힌 5분봉 부분봉, 24줄) →
_spliced_raw(창, 기준가 = t 종가). «보이는 칸» = 쪽마다 직전 144 스냅샷(12h) 최대로 정규화 → 양쪽 합친 양수 90분위 clip →
무게/clip ≥ 0.25 (app.js liqBandEdges 와 같은 임계). 가장자리 = 칸 중심 ∓ 반 칸. 띠 바닥 = 가장자리에서 보이는 칸이 끊길 때까지(한 칸 틈 허용).
🔴근사: 라이브는 144 스냅샷을 **현재가 격자**로 다시 계산하지만 여기선 각 스냅샷을 제 시각 종가 격자로 한 번만 계산한다.

사전등록(결과 전 고정)
- 사건(아래 쪽, 위는 거울): 5분봉 k 의 저가 ≤ 경계 t_{k-1} 지도의 아래 가장자리(찍음) · k 의 실측 롱 청산 5분 합 ≥ 직전 7일 90분위(찍힘) ·
  k 의 1분 5개가 다 있음. 같은 쪽 다음 사건은 1시간 뒤부터(첫 찍음만).
- 진입: t_k 다음 1분봉 종가(1분 지연 -- 사건 라벨 경계 계약: 트리거 봉 k 가 닫힌 뒤, 청산 탐색은 그 다음 분부터).
- 주 규칙 R1 = 반대 방향(아래를 쓸면 롱) · 목표 = t_k 지도의 반대편 가장자리(지정가, 메이커 0bp) · 손절 = 띠 바닥 −0.1%(테이커 4bp) ·
  24시간 시간청산(테이커). 진입 테이커 4bp. 같은 1분에 목표·손절이 다 닿으면 손절 먼저.
- 판정: H2(2026-08-01~10-10, 주) 비용 후 평균 일 블록 부트스트랩 95% CI 하한 > 0 그리고 H1(05-03~07-31) 같은 부호.
- 탐색(판정 아님): 순방향(거울) · 대조 C0 찍음만(청산 무관) · C1 청산만(띠 미도달) · R2 청산 99분위 · R3 R1+되찾기(종가가 가장자리 안쪽) ·
  손절 10종 · 순차 왕복(포지션 하나) 자본곡선.
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · saved_parent_exit_timestamps_used=false · future_rows_used_for_entry=false
"""
import glob, json, os, sys, zipfile
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_liquidation_map_20260824 as LM  # noqa: E402

K1 = "/home/kbj20/crypto-scalping/data/binance_vision/klines1m"
M1, M5, H1 = 60_000, 300_000, 3_600_000
LIT, CLIP_Q, NSNAP = 0.25, 0.90, 144
COST_TAKER = 4.0
HOLD_MIN = 24 * 60
COOLDOWN_BARS = 12
SPLIT_H2 = "2026-08-01"
STOPS = ("band_bottom", "sweep_low", "fix0.5", "fix1", "fix2", "rng3", "close_through", "liq_wave2", "time4h", "none")


# ── 데이터 ─────────────────────────────────────────────────────────────
def load_1m(data_dir, t0, t1):
    parts = [pd.read_parquet(f, columns=["t", "h", "l", "c", "v"]) for f in sorted(glob.glob(f"{K1}/ETHUSDT-1m-2026-0[3-9].parquet"))]
    for z in sorted(glob.glob(f"{data_dir}/dl/ETHUSDT-1m-*.zip")):
        with zipfile.ZipFile(z) as zf:
            d = pd.read_csv(zf.open(zf.namelist()[0]))
            if not str(d.columns[0]).startswith("open"):
                d = pd.read_csv(zf.open(zf.namelist()[0]), header=None)
        d = d.iloc[:, :6]; d.columns = ["t", "o", "h", "l", "c", "v"]
        parts.append(d[["t", "h", "l", "c", "v"]].astype({"t": "int64"}))
    k = pd.concat(parts).drop_duplicates("t").sort_values("t")
    k = k[(k.t >= t0) & (k.t < t1)]
    full = pd.DataFrame({"t": np.arange(k.t.iloc[0], k.t.iloc[-1] + 1, M1)}).merge(k, on="t", how="left")
    full["c"] = full.c.ffill(); full["h"] = full.h.fillna(full.c); full["l"] = full.l.fillna(full.c); full["v"] = full.v.fillna(0.0)
    return full.reset_index(drop=True)


def bars(k, ms):
    g = k.groupby(k.t // ms * ms)
    b = pd.DataFrame({"high": g.h.max(), "low": g.l.min(), "close": g.c.last(), "volume": g.v.sum(), "n": g.size()})
    b = b[b.n == ms // M1].drop(columns="n").reset_index(names="t")
    b["timestamp"] = pd.to_datetime(b.t, unit="ms", utc=True)
    return b


# ── 스냅샷(대시보드 밀도 층의 열) ─────────────────────────────────────────
_G = {}


def _init(h, m):
    _G["h"], _G["m"] = h, m


def _snap(jobs):
    out = []
    for t, cp in jobs:
        raw = LM._spliced_raw(LM._window_at(_G["h"], _G["m"], int(t), 24), float(cp))
        out.append(None if raw is None else (np.array(list(raw[0].keys()), dtype=np.int64), np.array(list(raw[0].values()), dtype=float),
                                             np.array(list(raw[1].keys()), dtype=np.int64), np.array(list(raw[1].values()), dtype=float), raw[2]))
    return out


def snapshots(b1h, b5):
    h, m = LM._bar_arrays(b1h), LM._bar_arrays(b5)
    ts = (b5.t.to_numpy() + M5) // 1000                                     # 경계 = 5분봉 끝(초)
    jobs = list(zip(ts, b5.close.to_numpy()))
    chunks = [jobs[i:i + 500] for i in range(0, len(jobs), 500)]
    with Pool(max(1, os.cpu_count() - 2), initializer=_init, initargs=(h, m)) as p:
        res = p.map(_snap, chunks)
    return [x for c in res for x in c]


def _band(lit, cp, bw, down):
    """보이는 칸 번호(정렬) → (가장자리, 띠 끝). 한 칸 틈까지 같은 띠."""
    if down:
        lit = lit[(lit + 0.5) * bw < cp]
        if not len(lit):
            return np.nan, np.nan
        top = bot = lit[-1]
        for b in lit[::-1][1:]:
            if bot - b <= 2: bot = b
            else: break
        return (top + 0.5) * bw, (bot - 0.5) * bw
    lit = lit[(lit - 0.5) * bw > cp]
    if not len(lit):
        return np.nan, np.nan
    bot = top = lit[0]
    for b in lit[1:]:
        if b - top <= 2: top = b
        else: break
    return (bot - 0.5) * bw, (top + 0.5) * bw


def lit_edges(snaps, closes):
    """경계마다 (아래 가장자리, 아래 띠 바닥, 위 가장자리, 위 띠 꼭대기) -- 직전 144 스냅샷 정규화·90분위 clip."""
    n = len(snaps)
    out = np.full((n, 4), np.nan)
    smax = np.array([s[1].max() if s is not None and len(s[1]) else 0.0 for s in snaps])
    rmax = np.array([s[3].max() if s is not None and len(s[3]) else 0.0 for s in snaps])
    for i in range(NSNAP - 1, n):
        s = snaps[i]
        if s is None:
            continue
        lo = i - NSNAP + 1
        gs, gr = smax[lo:i + 1].max(), rmax[lo:i + 1].max()
        pool = []
        for j in range(lo, i + 1):
            if snaps[j] is not None:
                if gs > 0: pool.append(snaps[j][1] / gs)
                if gr > 0: pool.append(snaps[j][3] / gr)
        pv = np.concatenate(pool) if pool else np.empty(0)
        pv = pv[pv > 0]
        if not len(pv):
            continue
        clip = np.quantile(pv, CLIP_Q, method="lower")
        sk, sw, rk, rw, bw = s
        lit_s = np.sort(sk[sw / gs / clip >= LIT]) if gs > 0 else np.empty(0, np.int64)
        lit_r = np.sort(rk[rw / gr / clip >= LIT]) if gr > 0 else np.empty(0, np.int64)
        out[i, 0], out[i, 1] = _band(lit_s, closes[i], bw, True)
        out[i, 2], out[i, 3] = _band(lit_r, closes[i], bw, False)
    return out


# ── 청산 ───────────────────────────────────────────────────────────────
def simulate(k_t, k_h, k_l, k_c, ei, side, tp, stop_px, close_stop, liq_exit_min, hold=HOLD_MIN):
    """ei = 진입 1분 인덱스(그 분 종가 진입). side +1 롱/−1 숏. tp/stop_px = 가격 또는 nan(1분 고·저로 닿음).
    close_stop = 5분봉 종가가 넘으면 그 종가에 청산(nan 이면 없음). liq_exit_min = 그 분 종가 청산(−1 없음).
    반환 (총수익 bp, 사유, 보유 분)."""
    e = k_c[ei]
    end = min(len(k_c) - 1, ei + hold)
    for j in range(ei + 1, end + 1):
        hit_sl = not np.isnan(stop_px) and (k_l[j] <= stop_px if side > 0 else k_h[j] >= stop_px)
        hit_tp = not np.isnan(tp) and (k_h[j] >= tp if side > 0 else k_l[j] <= tp)
        if hit_sl:
            return side * (stop_px - e) / e * 1e4, "sl", j - ei
        if hit_tp:
            return side * (tp - e) / e * 1e4, "tp", j - ei
        if (k_t[j] + M1) % M5 == 0 and not np.isnan(close_stop) and (k_c[j] < close_stop if side > 0 else k_c[j] > close_stop):
            return side * (k_c[j] - e) / e * 1e4, "close_sl", j - ei
        if j == liq_exit_min:
            return side * (k_c[j] - e) / e * 1e4, "liq_sl", j - ei
    return side * (k_c[end] - e) / e * 1e4, "time", end - ei


def net(gross, reason):
    return gross - COST_TAKER - (0.0 if reason == "tp" else COST_TAKER)


def boot_ci(df, col, nb=2000, seed=7):
    if len(df) < 5:
        return (np.nan, np.nan)
    d = df.groupby("day")[col].agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(d), (nb, len(d)))
    s, c = d["sum"].to_numpy()[idx].sum(1), d["count"].to_numpy()[idx].sum(1)
    return tuple(np.quantile(s / c, [0.025, 0.975]))


def selftest():
    k_t = np.arange(0, 20) * M1; c = np.full(20, 100.0); h = c + 0.1; l = c - 0.1
    h5 = h.copy(); h5[5] = 101.5                                              # 목표 101 이 5분에 닿음
    g, r, _ = simulate(k_t, h5, l, c, 0, +1, 101.0, 99.0, np.nan, -1)
    assert r == "tp" and abs(g - 100.0) < 1e-9, (g, r)
    l2 = l.copy(); l2[3] = 98.5; h2 = h.copy(); h2[3] = 101.5             # 같은 분 둘 다 → 손절 먼저
    g, r, _ = simulate(k_t, h2, l2, c, 0, +1, 101.0, 99.0, np.nan, -1)
    assert r == "sl" and abs(g + 100.0) < 1e-9, (g, r)
    c3 = c.copy(); c3[4] = 98.0                                             # 분 4 = 5분봉 끝(종가 98 < 98.5) → 종가 뚫림
    g, r, _ = simulate(k_t, h, np.minimum(l, c3), c3, 0, +1, np.nan, np.nan, 98.5, -1)
    assert r == "close_sl" and abs(g + 200.0) < 1e-9, (g, r)
    g, r, _ = simulate(k_t, h, l, c, 0, -1, 99.0, 101.0, np.nan, 7)          # 숏 · 2차 파동 청산 분 7
    assert r == "liq_sl" and abs(g) < 1e-9, (g, r)
    g, r, hm = simulate(k_t, h, l, c, 0, +1, np.nan, np.nan, np.nan, -1, hold=10)
    assert r == "time" and hm == 10
    assert net(10.0, "tp") == 6.0 and net(10.0, "sl") == 2.0
    assert _band(np.array([90, 91, 93, 97]), 100.0, 1.0, True) == (97.5, 96.5)     # 97 혼자(93 과 4칸 떨어짐)
    assert _band(np.array([90, 91, 93]), 100.0, 1.0, True) == (93.5, 89.5)         # 93→91(2칸)→90 한 띠
    assert _band(np.array([101, 102, 105]), 100.0, 1.0, False) == (100.5, 102.5)
    print("selftest OK")


# ── 본체 ───────────────────────────────────────────────────────────────
def main(data_dir, out):
    os.makedirs(out, exist_ok=True)
    tr = pd.read_parquet(f"{data_dir}/eth_tail_risk_1m.parquet")
    t0 = int(pd.Timestamp("2026-04-29", tz="UTC").value // 10**6); t1 = int(pd.Timestamp("2026-10-11", tz="UTC").value // 10**6)
    k = load_1m(data_dir, t0, t1)
    b5, b1h = bars(k, M5), bars(k, H1)
    print("1m", len(k), pd.to_datetime(k.t.iloc[[0, -1]], unit="ms").tolist(), "5m", len(b5), flush=True)
    cache = f"{out}/edges.parquet"
    if os.path.exists(cache):
        E = pd.read_parquet(cache)
    else:
        sn = snapshots(b1h, b5)
        E = pd.DataFrame(lit_edges(sn, b5.close.to_numpy()), columns=["dn_edge", "dn_bottom", "up_edge", "up_top"]); E["t"] = b5.t.to_numpy()
        E.to_parquet(cache)
    b5 = b5.merge(E, on="t", how="left")
    tr["b"] = tr.t // M5 * M5
    lq = tr.groupby("b").agg(L=("long_usd_1m", "sum"), S=("short_usd_1m", "sum"), nm=("t", "size"))
    b5 = b5.merge(lq, left_on="t", right_index=True, how="left")
    b5["nm"] = b5.nm.fillna(0)
    ok = (b5.nm == 5).to_numpy()
    for col in ("L", "S"):
        x = b5[col].where(b5.nm == 5)
        b5[f"{col}_q90"] = x.shift(1).rolling(2016, min_periods=1000).quantile(0.90)
        b5[f"{col}_q99"] = x.shift(1).rolling(2016, min_periods=1000).quantile(0.99)
    b5["rng_med"] = (b5.high - b5.low).shift(1).rolling(288, min_periods=200).median()
    print("띠 있는 경계 비율 아래", round(b5.dn_edge.notna().mean(), 3), "위", round(b5.up_edge.notna().mean(), 3),
          "· 아래 가장자리 거리 중앙 bp", round(((b5.close - b5.dn_edge) / b5.close * 1e4).median(), 1),
          "· 위", round(((b5.up_edge - b5.close) / b5.close * 1e4).median(), 1), flush=True)
    k_t, k_h, k_l, k_c = (k[c].to_numpy() for c in ("t", "h", "l", "c"))
    kidx = {int(t): i for i, t in enumerate(k_t)}
    B = {c: b5[c].to_numpy() for c in b5.columns if c != "timestamp"}
    n5 = len(b5)
    rows, last = [], {}
    for i in range(1, n5 - 1):
        if not ok[i]:
            continue
        for sd, edge_c, bot_c, liq_c, opp_c in ((-1, "dn_edge", "dn_bottom", "L", "up_edge"), (+1, "up_edge", "up_top", "S", "dn_edge")):
            edge = B[edge_c][i - 1]
            q90, q99, liq = B[f"{liq_c}_q90"][i], B[f"{liq_c}_q99"][i], B[liq_c][i]
            if np.isnan(edge) or np.isnan(q90):
                continue
            touched = B["low"][i] <= edge if sd < 0 else B["high"][i] >= edge
            big = liq >= q90 and liq > 0
            reclaim = B["close"][i] > edge if sd < 0 else B["close"][i] < edge
            arms = [a for a, c in (("C0", touched), ("R1", touched and big), ("R2", touched and liq >= q99 and liq > 0),
                                   ("R3", touched and big and reclaim), ("C1", (not touched) and big)) if c]
            arms = [a for a in arms if not ((a, sd) in last and i - last[(a, sd)] < COOLDOWN_BARS)]
            if not arms:
                continue
            ei = kidx.get(int(B["t"][i] + M5))                               # 다음 5분봉 첫 1분 종가 진입(1분 지연)
            if ei is None or ei + HOLD_MIN >= len(k_c):
                continue
            for a in arms:
                last[(a, sd)] = i
            day = pd.Timestamp(int(B["t"][i]), unit="ms").strftime("%Y-%m-%d")
            tp = B[opp_c][i]                                                  # t_k 지도의 반대편 가장자리
            sweep_ext = B["low"][i] if sd < 0 else B["high"][i]
            bottom = B[bot_c][i - 1]
            pos = -sd                                                         # 반대(페이드): 아래를 쓸면 롱
            e = k_c[ei]
            rm = B["rng_med"][i]
            stop_lv = {"band_bottom": bottom * (1 - 0.001 * pos), "sweep_low": sweep_ext * (1 - 0.001 * pos),
                       "fix0.5": e * (1 - 0.005 * pos), "fix1": e * (1 - 0.01 * pos), "fix2": e * (1 - 0.02 * pos),
                       "rng3": e - pos * 3 * rm if not np.isnan(rm) else np.nan}
            wave2 = -1                                                        # 2차 파동: 같은 쪽 청산 ≥90분위 + 쓸린 극값 넘는 5분봉 → 그 봉 끝 종가
            for jj in range(i + 1, min(n5, i + 1 + HOLD_MIN // 5)):
                lj = B[liq_c][jj]
                if B["nm"][jj] == 5 and lj >= q90 and lj > 0 and ((B["low"][jj] < sweep_ext) if sd < 0 else (B["high"][jj] > sweep_ext)):
                    wave2 = kidx.get(int(B["t"][jj] + M5 - M1), -1)
                    if wave2 <= ei: wave2 = -1; continue
                    break
            for stop in STOPS:
                sp = stop_lv.get(stop, np.nan)
                hold = 240 if stop == "time4h" else HOLD_MIN
                g, r, hm = simulate(k_t, k_h, k_l, k_c, ei, pos, tp, sp, bottom if stop == "close_through" else np.nan,
                                    wave2 if stop == "liq_wave2" else -1, hold)
                for a in arms:
                    rows.append({"arm": a, "side": "dn" if sd < 0 else "up", "pos": pos, "stop": stop, "day": day, "t": int(B["t"][i]),
                                 "entry": e, "has_tp": not np.isnan(tp), "tp_dist_bp": abs(tp - e) / e * 1e4 if not np.isnan(tp) else np.nan,
                                 "stop_dist_bp": abs(sp - e) / e * 1e4 if not np.isnan(sp) else np.nan,
                                 "liq_x": liq / q90 if q90 > 0 else np.nan,
                                 "gross": g, "reason": r, "hold_min": hm, "net": net(g, r), "net_follow": -g - 2 * COST_TAKER})
    T = pd.DataFrame(rows)
    T.to_parquet(f"{out}/trades.parquet")
    report(T, out)


def report(T, out):
    T = T.copy()
    T["split"] = np.where(T.day < SPLIT_H2, "H1", "H2")
    res = []
    for (arm, stop, split), g in T.groupby(["arm", "stop", "split"]):
        lo, hi = boot_ci(g, "net"); flo, fhi = boot_ci(g, "net_follow")
        res.append({"arm": arm, "stop": stop, "split": split, "n": len(g), "days": g.day.nunique(),
                    "tp_rate": (g.reason == "tp").mean(), "sl_rate": g.reason.isin(["sl", "close_sl", "liq_sl"]).mean(),
                    "gross": g.gross.mean(), "net": g.net.mean(), "net_lo": lo, "net_hi": hi,
                    "follow_net": g.net_follow.mean(), "follow_lo": flo, "follow_hi": fhi,
                    "worst": g.net.min(), "p05": g.net.quantile(0.05), "hold_med": g.hold_min.median(),
                    "tp_dist": g.tp_dist_bp.median(), "stop_dist": g.stop_dist_bp.median()})
    R = pd.DataFrame(res)
    R.to_csv(f"{out}/summary.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    fmt = lambda d: d.round(1).round({"tp_rate": 3, "sl_rate": 3})
    p = R[(R.arm == "R1") & (R.stop == "band_bottom")].set_index("split")
    passed = bool({"H1", "H2"} <= set(p.index) and p.loc["H2", "net_lo"] > 0 and np.sign(p.loc["H1", "net"]) == np.sign(p.loc["H2", "net"]))
    print("\n=== 팔별 · 손절 band_bottom (주 판정 = R1) ===")
    print(fmt(R[R.stop == "band_bottom"]).to_string(index=False))
    print("판정 R1:", "통과" if passed else "불통과")
    print("\n=== R1 손절 비교 ===")
    print(fmt(R[R.arm == "R1"].sort_values(["split", "net"], ascending=[True, False])).to_string(index=False))
    nn = T[T.stop == "none"][["arm", "t", "side", "reason", "gross"]].rename(columns={"reason": "ns_reason", "gross": "ns_gross"})
    J = T[T.stop != "none"].merge(nn, on=["arm", "t", "side"])
    st = J[J.reason.isin(["sl", "close_sl", "liq_sl"])]
    reg = st.groupby(["arm", "stop"]).agg(stopped=("t", "size"), later_tp=("ns_reason", lambda s: (s == "tp").mean()),
                                         saved_bp=("gross", lambda s: 0.0)).reset_index()
    reg["saved_bp"] = st.groupby(["arm", "stop"]).apply(lambda g: (g.gross - g.ns_gross).mean()).to_numpy()
    print("\n=== 손절된 거래: 손절 없었으면 목표 도달 비율 · 손절이 아낀 bp(손절 − 손절 없음, 총수익) ===")
    print(reg.round(3).to_string(index=False))
    seq = T[(T.arm == "R1") & (T.stop == "band_bottom")].sort_values("t")
    busy, eq = -1, []
    for r in seq.itertuples():
        if r.t < busy:
            continue
        busy = r.t + M5 + (r.hold_min + 1) * M1
        eq.append(r.net)
    if eq:
        e = np.cumsum(eq)
        print(f"\n순차 왕복 R1·band_bottom: 거래 {len(eq)} · 합 {e[-1]:.0f}bp · 평균 {e[-1] / len(eq):.1f}bp · 최대낙폭 {(e - np.maximum.accumulate(e)).min():.0f}bp")
    json.dump({"r1_pass": passed, "flags": {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
               "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}}, open(f"{out}/verdict.json", "w"))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    main(sys.argv[1], sys.argv[2])
