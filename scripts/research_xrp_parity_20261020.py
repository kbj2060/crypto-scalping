"""XRP 판 상수 세 가지 -- 권장 배수(vol_mult) · 청산 급증 배지 문턱 · 09시 개장 되돌림 검정 (2026-10-20).

  volmult  ETH 정의 그대로(prereg_eth_pingpong_vol_sizing_fwd_20261007): MED = TRAIN(2022-01~2024-06) 핑퐁 후보 시점 σ24 중앙 ·
           REF = 2025-01~2026-10-05 5분봉 clip(σ24/MED, .25, 3) 중앙. ETH 로 157.394 / 0.76576 재현을 먼저 본다(양성 대조).
           + XRP·ETH σ24 분포 · 규칙 B(commit_rule_l) 결과 분포 · ∓5% 손절 닿음 빈도·1분 갭·L=3 청산 거리 · −0.75% 물타기 닿음.
  liqburst rl_1s_agent.calib_coin 정의: 매초 «직전 60초 그 쪽 청산 USD 합» log1p 의 99.5분위(바이낸스 @forceOrder lake).
           ETH 09-20~26 → 12.732/13.21 · XRP 09-27~10-03 → 11.296/9.866 재현 후 XRP 09-26~10-18 전체·주별.
  openrev  live_eth_direction_forward_recorder_20261010.decide 규칙(|f30| > .5σ30 반대 4h)을 XRP 2022-24/2025/2026 에.
           비용 왕복 8bp(테이커 양쪽) · 4bp(지정가 진입+시장가 청산) · 0bp(지정가 양쪽, 체결 보장 가정 = 상한).
           위약 = (a) 다른 287개 고정 시각 (b) 날마다 무작위 30분 창 500회. 판정 기준(결과 보기 전 고정):
           통과 = 2025~ 8bp 순 건당 평균 > 0 이고 일 재표집 95% CI 하한 > 0 · 2025 와 2026 같은 부호(양) · 비용 전 건당 평균이 무작위 창 위약 95분위 초과.
실행: nice python -I scripts/research_xrp_parity_20261020.py volmult|liqburst|openrev|selftest
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = Path("/home/kbj20/crypto-scalping/tmp/xrp_parity_20261020")
DATA = Path("/home/kbj20/crypto-scalping/data/binance_vision")
SCR = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping--claude-worktrees-soxlusdt-dashboard-add-4fce62/3f3764dd-08ce-440d-9a05-310554eed4a4/scratchpad")
LAKE = Path.home() / "backups/crypto-scalping-server/data/lake/binance/liquidations"
ms = lambda s: int(pd.Timestamp(s, tz="UTC").value // 10**6)  # noqa: E731
rng0 = lambda: np.random.default_rng(20261020)  # noqa: E731


def load_1m(sym: str) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted((DATA / "klines1m").glob(f"{sym}-1m-*.parquet"))]
    parts += [pd.read_parquet(p) for p in sorted((DATA / "klines1m_daily").glob(f"{sym}-1m-*.parquet"))]
    for z in sorted((SCR / "dl_eth1m").glob(f"{sym}-1m-*.zip")):     # ETH 10-01~04(로컬에 없던 날, 새 폴더에 받음)
        with zipfile.ZipFile(z) as f:
            df = pd.read_csv(io.StringIO(f.read(f.namelist()[0]).decode()), header=None)
        df = df.iloc[1:] if not str(df.iat[0, 0]).isdigit() else df
        parts.append(pd.DataFrame({"t": df[0].astype("int64"), "h": df[2].astype(float), "l": df[3].astype(float),
                                   "c": df[4].astype(float), "v": df[5].astype(float), "tb": df[9].astype(float)}))
    k = pd.concat(parts)[["t", "h", "l", "c", "v", "tb"]].drop_duplicates("t", keep="last").sort_values("t")
    k["t"] = k.t.astype("int64")
    full = pd.DataFrame({"t": np.arange(k.t.iloc[0], k.t.iloc[-1] + 1, 60_000)}).merge(k, on="t", how="left")
    miss = int(full.c.isna().sum())
    full["c"] = full.c.ffill(); full["h"] = full.h.fillna(full.c); full["l"] = full.l.fillna(full.c)
    full[["v", "tb"]] = full[["v", "tb"]].fillna(0.0)
    print(f"{sym} 1분봉 {len(full):,} ({pd.Timestamp(full.t.iloc[0], unit='ms')} ~ {pd.Timestamp(full.t.iloc[-1], unit='ms')}) · 빈 분 {miss:,}", flush=True)
    return full


# ── 1. 권장 배수 ─────────────────────────────────────────────────────────────────────────────
def sigma24(k1: pd.DataFrame):
    from research_eth_trend30_60_hold_policy_20261007 import to_5m
    b = to_5m(k1); t5 = b.t.to_numpy()
    r2 = pd.Series((np.diff(np.log(b.c.to_numpy()), prepend=np.nan) * 1e4) ** 2)
    return t5, np.sqrt(r2.rolling(288).sum().to_numpy() / 6), b


def med_ref(sym: str, ref_end: str = "2026-10-06"):
    from prereg_eth_pingpong_vol_sizing_fwd_20261007 import HOLD, candidates
    k = load_1m(sym)
    t5, sig, b = sigma24(k)
    C = candidates(k)
    tdec = C.t.to_numpy("int64") + 60_000
    ei = np.searchsorted(k.t.to_numpy(), tdec)
    bi = np.searchsorted(t5, tdec // 300_000 * 300_000 - 300_000)
    ok = (ei + HOLD < len(k)) & (bi < len(t5)) & np.isfinite(sig[np.minimum(bi, len(t5) - 1)])
    tr = ok & (tdec < ms("2024-07-01"))
    med = float(np.median(sig[bi[tr]]))
    refs = {}
    for nm, lo, hi in (("2025-01~2026-10-05", "2025-01-01", ref_end), ("2025-01~end", "2025-01-01", "2030-01-01"),
                       ("2026-01~end", "2026-01-01", "2030-01-01")):
        m = (t5 >= ms(lo)) & (t5 < ms(hi)) & np.isfinite(sig)
        refs[nm] = float(np.median(np.clip(sig[m] / med, 0.25, 3.0)))
    return dict(med=med, refs=refs, n_cand=int(ok.sum()), n_train_cand=int(tr.sum())), k, t5, sig


def touch_stats(k: pd.DataFrame, t0: int, step_min: int = 60, horizons=(1440, 10080)):
    """2025~ 매시 정각 진입(종가) → 롱 −x / 숏 +x 를 H분 안에 닿는 비율. x = 5%(손절) · 0.75%(물타기) · 31.6%(L=3 청산 근사)."""
    c, h, l = k.c.to_numpy(), k.h.to_numpy(), k.l.to_numpy()
    i0 = np.flatnonzero(k.t.to_numpy() >= t0)[0]
    idx = np.arange(i0, len(c) - max(horizons) - 1, step_min)
    fwd = lambda a, H, f: getattr(pd.Series(a).shift(-1)[::-1].rolling(H, min_periods=H), f)()[::-1].to_numpy()  # noqa: E731 -- 다음 분부터 H 분
    out = {}
    for H in horizons:
        lo = fwd(l, H, "min")[idx]; hi = fwd(h, H, "max")[idx]
        p = c[idx]
        for x in (0.0075, 0.05, 0.10, 0.316):
            out[f"H{H // 60}h_x{x * 100:g}%"] = {"long": float(np.mean(lo <= p * (1 - x))), "short": float(np.mean(hi >= p * (1 + x)))}
    return out, int(len(idx))


def gap_stats(k: pd.DataFrame, t0: int):
    """1분봉 안에서 직전 종가 대비 최대 역행(롱 = (저가/직전 종가 − 1), 숏 = 고가) -- 손절 스탑이 한 봉 안에서 얼마나 미끄러질 수 있나의 상한."""
    m = k.t.to_numpy() >= t0
    pc = k.c.shift(1).to_numpy()
    dn = (k.l.to_numpy() / pc - 1)[m]; up = (k.h.to_numpy() / pc - 1)[m]
    t = k.t.to_numpy()[m]
    q = lambda a, s: {f"q{qq}": float(np.nanquantile(s * a, qq / 100)) * 100 for qq in (99.9, 99.99)}  # noqa: E731
    return {"long_drop_pct": {**q(dn, -1), "max": float(-np.nanmin(dn) * 100), "at": str(pd.Timestamp(t[np.nanargmin(dn)], unit="ms"))},
            "short_rise_pct": {**q(up, 1), "max": float(np.nanmax(up) * 100), "at": str(pd.Timestamp(t[np.nanargmax(up)], unit="ms"))}}


def volmult() -> None:
    from live_manual_peg_entry_20260912 import (COMMIT_CAP_X, COMMIT_LIQ_BUF, COMMIT_MMR, COMMIT_UNIT_X, REC_ADD_DIST,
                                                REC_FIRST_SHARE, RULE_MARGIN_MAX, RULE_STOP_PCT, commit_rule_l)
    res: dict = {"definition": "MED = median σ24 at TRAIN(<2024-07-01) pingpong candidates (ZZ .5%, leg≥40bp, ≤10bp from extreme, first per leg); "
                               "REF = median clip(σ24/MED,.25,3) over 5m bars 2025-01-01..2026-10-05; mult = clip(σ24/MED,.25,3)/REF; "
                               "σ24 = sqrt(Σ288 r_bp²/6)"}
    t0 = ms("2025-01-01")
    S = {}
    for sym in ("ETHUSDT", "XRPUSDT"):
        r, k, t5, sig = med_ref(sym)
        m = (t5 >= t0) & np.isfinite(sig) & (sig > 0)
        s = sig[m]
        ref = r["refs"]["2025-01~2026-10-05"]
        vm = np.clip(s / r["med"], 0.25, 3.0) / ref
        rows = {}
        for side in ("LONG", "SHORT"):
            ls = [commit_rule_l(position_side=side, sigma24_bp=float(a), vol_mult=float(b)) for a, b in zip(s[::12], vm[::12])]  # 매시
            L = np.array([x["l"] for x in ls]); who = pd.Series([x["binding"] for x in ls]).value_counts(normalize=True).round(4).to_dict()
            rows[side] = {"L_q": {q: float(np.quantile(L, q)) for q in (0, .01, .5, 1)}, "binding_share": who}
        tch, n_tch = touch_stats(k, t0)
        S[sym] = s
        res[sym] = {**r, "ref_used": ref, "one_x_sigma_bp": r["med"] * ref,
                    "sigma24_bp_2025~": {q: float(np.quantile(s, q)) for q in (.01, .05, .25, .5, .75, .95, .99)},
                    "sigma_daily_pct_median": float(np.median(s) * np.sqrt(6) / 100),
                    "stop5_in_daily_sigma_median": float(5 / (np.median(s) * np.sqrt(6) / 100)),
                    "vol_mult_2025~": {q: float(np.quantile(vm, q)) for q in (.01, .05, .5, .95, .99)},
                    "vol_mult_floor_ceiling": [0.25 / ref, 3.0 / ref], "unit_x_times_floor": COMMIT_UNIT_X * 0.25 / ref,
                    "rule_b": rows, "touch_from_hourly_entry_2025~": tch, "touch_n": n_tch, "gap_1m_2022~": gap_stats(k, ms("2022-01-01"))}
        print(sym, json.dumps({kk: res[sym][kk] for kk in ("med", "refs", "one_x_sigma_bp", "vol_mult_floor_ceiling", "rule_b")}, default=str), flush=True)
    eth_med, eth_ref = 157.3941696083474, 0.7657638410664205
    res["eth_positive_control"] = {"deployed": [eth_med, eth_ref], "reproduced": [res["ETHUSDT"]["med"], res["ETHUSDT"]["ref_used"]],
                                   "rel_err": [res["ETHUSDT"]["med"] / eth_med - 1, res["ETHUSDT"]["ref_used"] / eth_ref - 1]}
    res["xrp_over_eth_sigma_median"] = float(np.median(S["XRPUSDT"]) / np.median(S["ETHUSDT"]))
    # 규칙 B 상수 점검(코인 공통 산식 · XRP 값)
    mmr_ex = 0.025   # exchangeInfo XRPUSDC/ETHUSDC maintMarginPercent 2.5(공개 · 최저 구간 실제 MMR 은 서명 엔드포인트 leverageBracket 이라 안 봄)
    liq_move = {f"mmr{m}": 1 - (1 - 1 / COMMIT_CAP_X) / (1 - m) for m in (COMMIT_MMR, mmr_ex)}
    res["rule_b_constants"] = {"COMMIT_UNIT_X": COMMIT_UNIT_X, "COMMIT_CAP_X": COMMIT_CAP_X, "RULE_STOP_PCT": RULE_STOP_PCT,
                               "REC_ADD_DIST": REC_ADD_DIST, "REC_FIRST_SHARE": REC_FIRST_SHARE, "COMMIT_MMR": COMMIT_MMR,
                               "COMMIT_LIQ_BUF": COMMIT_LIQ_BUF, "RULE_MARGIN_MAX": RULE_MARGIN_MAX,
                               "loss_at_stop_equity_pct": COMMIT_CAP_X * RULE_STOP_PCT * 100,
                               "cross_liq_move_at_L3_long": liq_move,
                               "isolated_max_lev_for_stop_before_liq": 1 / (RULE_STOP_PCT + COMMIT_LIQ_BUF + mmr_ex),
                               "exchangeInfo_XRPUSDC": {"maintMarginPercent": 2.5, "requiredMarginPercent": 5.0, "tick": 1e-4, "step": 0.1, "min_notional": 5}}
    (OUT / "vol_mult").mkdir(parents=True, exist_ok=True)
    (OUT / "vol_mult/report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res["eth_positive_control"]), res["xrp_over_eth_sigma_median"])


# ── 2. 청산 급증 문턱 ──────────────────────────────────────────────────────────────────────────
def liq_q(coin: str, start: str, end: str) -> dict:
    days = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta("1D"))
    L = pd.concat([pd.read_parquet(LAKE / f"coin={coin}/date={d:%Y-%m-%d}/part.parquet") for d in days
                   if (LAKE / f"coin={coin}/date={d:%Y-%m-%d}/part.parquet").exists()])
    lo, hi = ms(start) // 1000, ms(end) // 1000
    sec = L.ts_ms.to_numpy() // 1000
    out = {"days": len(days), "events": int(len(L))}
    for side in ("long", "short"):
        a = np.zeros(hi - lo)
        m = (L.side.to_numpy() == side) & (sec >= lo) & (sec < hi)
        np.add.at(a, sec[m] - lo, L.usd.to_numpy()[m])
        r = np.log1p(pd.Series(a).rolling(60).sum().to_numpy()[59:])
        q = float(np.quantile(r, 0.995))
        on = r > q
        out[side] = {"q995_log1p": round(q, 3), "usd": round(float(np.expm1(q))), "share_seconds_nonzero60": float((r > 0).mean()),
                     "onsets_per_day": float(np.sum(on[1:] & ~on[:-1]) / len(days)), "usd_total": float(a.sum())}
    return out


def liqburst() -> None:
    res = {"definition": "per-second sum of Binance @forceOrder USD by side over trailing 60s; log1p; 99.5th pct over all seconds of window "
                         "(rl_1s_agent.calib_coin; original also filtered dd_valid seconds -- not available here)",
           "eth_control_0920_0926": liq_q("ETH", "2026-09-20", "2026-09-27"), "eth_deployed": [12.732, 13.21],
           "xrp_control_0927_1003": liq_q("XRP", "2026-09-27", "2026-10-04"), "xrp_coincfg": [11.296, 9.866],
           "xrp_full_0926_1018": liq_q("XRP", "2026-09-26", "2026-10-19"), "eth_full_0926_1018": liq_q("ETH", "2026-09-26", "2026-10-19"),
           "xrp_weekly": {s: liq_q("XRP", s, str((pd.Timestamp(s) + pd.Timedelta("7D")).date())) for s in ("2026-09-26", "2026-10-03", "2026-10-10")},
           "eth_weekly": {s: liq_q("ETH", s, str((pd.Timestamp(s) + pd.Timedelta("7D")).date())) for s in ("2026-09-26", "2026-10-03", "2026-10-10")}}
    (OUT / "liq_burst").mkdir(parents=True, exist_ok=True)
    (OUT / "liq_burst/report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


# ── 3. 09시 개장 되돌림 ────────────────────────────────────────────────────────────────────────
def daily_table(d5: pd.DataFrame, off: np.ndarray | int) -> pd.DataFrame:
    """5분 종가 격자(빈 봉 NaN) → 날마다 기준 시각 T = 그날 00:00 + off(5분 단위, 날마다 다를 수 있음).
    p0 = T−5분 봉 종가 · p30 = T+25분 봉 종가 · σ30 = T−5분 봉까지 289 종가의 로그수익 std(ddof=1)×√6 · 청산 = T+4h25분 봉 종가."""
    t = d5.t.to_numpy(); c = d5.c.to_numpy()
    lc = np.log(c)
    r = np.diff(lc, prepend=np.nan)
    sd = pd.Series(r).rolling(288).std(ddof=1).to_numpy() * np.sqrt(6) * 1e4
    day0 = np.arange(t[0] // 86_400_000 * 86_400_000 + 86_400_000, t[-1] - 86_400_000, 86_400_000)
    T = day0 + np.asarray(off) * 300_000
    pos = lambda x: np.searchsorted(t, x)  # noqa: E731
    i0, i30, ix = pos(T - 300_000), pos(T + 1_500_000), pos(T + 4 * 3_600_000 + 1_500_000)
    ok = (ix < len(t))
    i0, i30, ix, T, day0 = i0[ok], i30[ok], ix[ok], T[ok], day0[ok]
    ok = (t[i0] == T - 300_000) & (t[i30] == T + 1_500_000) & (t[ix] == T + 4 * 3_600_000 + 1_500_000)
    f30 = (lc[i30] - lc[i0]) * 1e4; s30 = sd[i0]; fwd = (lc[ix] - lc[i30]) * 1e4
    D = pd.DataFrame({"day": day0, "T": T, "f30": f30, "s30": s30, "fwd": fwd})[ok & np.isfinite(f30) & np.isfinite(s30) & np.isfinite(fwd)]
    D["pos"] = np.where(np.abs(D.f30) > 0.5 * D.s30, -np.sign(D.f30), 0.0)
    D["yr"] = pd.to_datetime(D.day, unit="ms").dt.year
    return D.reset_index(drop=True)


def grid5(d: pd.DataFrame) -> pd.DataFrame:
    t = (pd.to_datetime(d.timestamp).astype("int64") // 10**6).to_numpy()
    g = pd.DataFrame({"t": np.arange(t[0], t[-1] + 1, 300_000)})
    return g.merge(pd.DataFrame({"t": t, "c": d.close.to_numpy(float)}), on="t", how="left")


def summarize(D: pd.DataFrame, rng, n_boot=2000) -> dict:
    out = {}
    for nm, m in (("2022-24", D.yr <= 2024), ("2025", D.yr == 2025), ("2026", D.yr == 2026), ("2025~", D.yr >= 2025), ("all", D.yr > 0)):
        x = D[m & (D.pos != 0)]
        g = x.pos * x.fwd
        r = {"n_days": int(m.sum()), "n_trades": int(len(x)), "hit": float((g > 0).mean()) if len(x) else None, "gross_bp": float(g.mean()) if len(x) else None}
        for cost in (8, 4, 0):
            v = (g - cost).to_numpy()
            if len(v) > 5:
                bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
                daily = pd.Series(v, x.day).reindex(D.day[m], fill_value=0.0)
                r[f"net{cost}"] = {"mean_bp": float(v.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                   "per_day_bp": float(daily.mean()), "sharpe_ann": float(daily.mean() / daily.std() * np.sqrt(365))}
        out[nm] = r
    return out


def openrev(sym_files=None) -> None:
    from live_eth_direction_forward_recorder_20261010 import decide
    rng = rng0()
    src = sym_files or {"XRPUSDT": OUT / "vol4h/XRPUSDT_5m.parquet",
                        "ETHUSDT": Path.home() / "backups/crypto-scalping-server/data/live/vol_panel/ETHUSDT_5m.parquet"}
    res: dict = {"rule": "f30 = ln(c[00:25]/c[23:55])·1e4, σ30 = std(288 5m log rets to 23:55, ddof1)·√6·1e4, |f30|>.5σ30 → −sign(f30), "
                         "enter c[00:25], exit c[04:25] (UTC)", "costs_rt_bp": {"taker_both": 8, "maker_in_taker_out": 4, "maker_both(ideal fill)": 0},
                 "pass_rule": "2025~ net8 mean>0 & CI low>0 & 2025,2026 net8 both >0 & gross mean > random-window placebo q95"}
    for sym, p in src.items():
        d = pd.read_parquet(p)
        d = d[pd.to_datetime(d.timestamp) >= "2021-12-01"]
        g5 = grid5(d)
        D = daily_table(g5, 0)
        # 파리티: 기록기 decide() 와 날 40개 대조
        rows = g5.dropna().to_numpy()
        for j in rng.choice(len(D), 40, replace=False):
            day0 = int(D.day.iloc[j]); i = np.searchsorted(rows[:, 0], day0 + 1_500_000)
            m5 = [[int(a), 0, 0, 0, float(b)] for a, b in rows[max(0, i - 400):i + 1]]
            dec = decide([[0, 0, 0, 0, 1.0]] * 91, m5, day0 + 1_860_000, sym, 0)
            assert dec["open_pos"] == int(D.pos.iloc[j]) and abs(dec["f30_bp"] - round(D.f30.iloc[j], 2)) < 0.011, (sym, j, dec, D.iloc[j])
        r = {"parity_decide_40days": "ok", "summary": summarize(D, rng)}
        # 위약 (a) 고정 시각 288개
        fx = {}
        for off in range(288):
            Do = D if off == 0 else daily_table(g5, off)
            for nm, msk in (("2022-24", Do.yr <= 2024), ("2025~", Do.yr >= 2025)):
                x = Do[msk & (Do.pos != 0)]
                fx.setdefault(nm, []).append(float((x.pos * x.fwd).mean()))
        r["fixed_offset_placebo"] = {nm: {"actual_gross": v[0], "rank_of_0000_desc": int(1 + np.sum(np.array(v[1:]) > v[0])), "of": 288,
                                          "placebo_q": {q: float(np.quantile(v[1:], q)) for q in (.05, .5, .95)}} for nm, v in fx.items()}
        # 위약 (b) 날마다 무작위 창(00:00 제외) 500회
        pb = {"2022-24": [], "2025": [], "2026": [], "2025~": []}
        nday = int(g5.t.iloc[-1] - g5.t.iloc[0]) // 86_400_000 + 1
        for _ in range(500):
            Do = daily_table_var(g5, rng.integers(1, 288, nday))
            for nm, msk in (("2022-24", Do.yr <= 2024), ("2025", Do.yr == 2025), ("2026", Do.yr == 2026), ("2025~", Do.yr >= 2025)):
                x = Do[msk & (Do.pos != 0)]
                pb[nm].append(float((x.pos * x.fwd).mean()))
        r["random_window_placebo"] = {nm: {"actual_gross": r["summary"][nm]["gross_bp"], "pctile": float(np.mean(np.array(v) < r["summary"][nm]["gross_bp"]) * 100),
                                           "q": {q: float(np.quantile(v, q)) for q in (.05, .5, .95)}} for nm, v in pb.items()}
        s = r["summary"]
        r["verdict"] = {"net8_2025~_ci_low>0": s["2025~"]["net8"]["ci95"][0] > 0, "net8_mean_2025>0": s["2025"]["net8"]["mean_bp"] > 0,
                        "net8_mean_2026>0": s["2026"]["net8"]["mean_bp"] > 0,
                        "gross_above_placebo_q95": r["random_window_placebo"]["2025~"]["pctile"] > 95}
        r["verdict"]["PASS"] = all(r["verdict"].values())
        res[sym] = r
        print(sym, json.dumps(r, default=str, ensure_ascii=False)[:3000], flush=True)
    (OUT / "open_rev").mkdir(parents=True, exist_ok=True)
    (OUT / "open_rev/report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


def daily_table_var(g5: pd.DataFrame, offs: np.ndarray) -> pd.DataFrame:
    """daily_table 의 날마다 다른 기준 시각판 -- offs 길이는 그 날 수 이상(앞에서부터 쓴다)."""
    t = g5.t.to_numpy()
    n = len(np.arange(t[0] // 86_400_000 * 86_400_000 + 86_400_000, t[-1] - 86_400_000, 86_400_000))
    return daily_table(g5, offs[:n])


def selftest() -> None:
    # 개장 되돌림 표: 23:55 → 00:25 +40bp 오르고 σ 작으면 숏, 4h 뒤 −10bp 이면 +10bp
    t = np.arange(ms("2026-01-01"), ms("2026-01-05"), 300_000)
    c = np.full(len(t), 100.0) * (1 + 1e-4 * np.sin(np.arange(len(t))))
    j = np.searchsorted(t, ms("2026-01-03"))
    c[j:j + 6] = c[j - 1] * np.exp(np.linspace(0.0007, 0.004, 6)); c[j + 6:] = c[j + 5] * np.exp(-0.001)
    D = daily_table(pd.DataFrame({"t": t, "c": c}), 0)
    row = D[D.day == ms("2026-01-03")].iloc[0]
    assert row.pos == -1 and abs(row.f30 - 1e4 * np.log(c[j + 5] / c[j - 1])) < 1e-6 and abs(row.fwd + 10) < 1e-6, row
    # 청산 60초 합: 같은 초 두 건 합 + 61초 뒤 빠짐
    a = np.zeros(200); np.add.at(a, np.array([10, 10, 100]), np.array([1.0, 2.0, 5.0]))
    r = pd.Series(a).rolling(60).sum().to_numpy()
    assert r[69] == 3.0 and r[70] == 0.0 and r[120] == 5.0
    print("selftest OK")


if __name__ == "__main__":
    {"volmult": volmult, "liqburst": liqburst, "openrev": openrev, "selftest": selftest}[sys.argv[1]]()
