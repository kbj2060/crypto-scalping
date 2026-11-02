#!/usr/bin/env python3
"""세션 VWAP 거리(H1) · 가격↔델타 다이버전스(H2) · 세션 첫/마지막 30분 모멘텀(H3) — 4코인 5분봉 (2026-11-02).

사전등록: docs/prereg/session_vwap_divergence_intraday_momo_20261102.md (계산 전 커밋 170aff6).
원천: data/live/dirshort_panel/{ETH,BTC,SOL,XRP}USDT.parquet (5분, timestamp = 봉 시작). 열: high·low·close·volume·taker_buy_base.
결정 = 봉 t 마감 값만 · 체결 = c_t(봉 t+1 시가) · 청산 = c_{t+6} · 거래 원장·저장 예측을 입력으로 쓰지 않는다.

    python scripts/research_session_vwap_div_momo_20261102.py --selftest   # 합성 데이터 · 파케 불필요
    python scripts/research_session_vwap_div_momo_20261102.py              # (서버) → tmp/session_vwap_div_momo_20261102/report.json

🔴이 파일을 고친 뒤 실데이터를 돌리면 사전등록의 «계산 전 커밋»이 깨진다 — 사양을 바꾸려면 새 사전등록.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from market_ctx import market_session, session_vwap  # noqa: E402

PANEL = ROOT / "data/live/dirshort_panel"
OUT = ROOT / "tmp/session_vwap_div_momo_20261102"
COINS = ["ETH", "BTC", "SOL", "XRP"]
TRAIN = ("2024-01-01", "2025-12-31")
VAL = ("2026-01-01", "2026-05-31")
TEST = ("2026-06-01", "2100-01-01")           # 시험 끝 = 패널 끝(보고서에 실제 끝을 적는다)
H = 6                                          # 보유 6봉 = 30분
W = 6                                          # 다이버전스 창
MINBARS = 12                                   # 세션 시작 후 최소 봉 수(H1)
Z_THR = 2.0
RANGE_FRAC = 0.25
SD_WIN = 288
COST_STRICT = 0.0010                           # 왕복 10bp (테이커 5bp × 2)
COST_CAND = 0.0005                             # 왕복 5bp (진입 메이커 0 · 청산 테이커 5 — 체결 모델 없음, 낙관)
BOOT = 2000
PLACEBO = 200
SEED = 20261102
ALPHA = 0.05
FLAGS = dict(fresh_forward_bar_by_bar=True, trade_ledgers_used_as_input=False,
             saved_parent_exit_timestamps_used=False, future_rows_used_for_entry=False)


# ── 데이터 ───────────────────────────────────────────────────────────────
def load(sym: str) -> pd.DataFrame:
    cols = ["timestamp", "high", "low", "close", "volume", "taker_buy_base"]
    d = pd.read_parquet(PANEL / f"{sym}USDT.parquet")
    miss = [c for c in cols if c not in d.columns]
    assert not miss, f"{sym} 열 없음 {miss}"
    d = d[cols].sort_values("timestamp").reset_index(drop=True)
    assert d.timestamp.dt.tz is None, f"{sym} timestamp 가 tz-aware"
    assert (d.timestamp.diff().dropna() == pd.Timedelta("5min")).all(), f"{sym} 5분 구멍"
    return d


def prep(d: pd.DataFrame) -> dict:
    c, h, lo, v, tb = (d[k].to_numpy(float) for k in ("close", "high", "low", "volume", "taker_buy_base"))
    ts = pd.DatetimeIndex(d.timestamp)
    tsec = ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy(dtype="int64")
    sst = np.array([market_session(int(t))[0] for t in tsec], dtype="int64")
    vw, vsd = session_vwap(tsec.tolist(), h.tolist(), lo.tolist(), c.tolist(), v.tolist(), key=lambda t: market_session(t)[0])
    n = len(c)
    new = np.r_[True, sst[1:] != sst[:-1]]
    starts = np.flatnonzero(new)
    grp = np.cumsum(new) - 1
    sess_idx = np.arange(n) - starts[grp] + 1
    valid_start = tsec[starts[grp]] == sst
    r6 = np.full(n, np.nan)
    r6[:-H] = c[H:] / c[:-H] - 1
    return dict(c=c, h=h, l=lo, v=v, tb=tb, ts=ts, tsec=tsec, sst=sst, n=n, new=new, starts=starts,
                vw=np.array([np.nan if x is None else x for x in vw]), vsd=np.array([np.nan if x is None else x for x in vsd]),
                sess_idx=sess_idx, valid_start=valid_start, r6=r6, day=tsec // 86400)


# ── 신호 (봉마다 −1/0/+1 = 봉 t 마감 결정의 방향) ─────────────────────────────
def sig_h1(I: dict, thr: float = Z_THR) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (I["c"] - I["vw"]) / I["vsd"]
    ok = np.isfinite(z) & (I["vsd"] > 0) & (I["sess_idx"] >= MINBARS) & I["valid_start"]
    s = np.zeros(I["n"])
    m = ok & (np.abs(z) >= thr)
    s[m] = -np.sign(z[m])          # VWAP 쪽으로 되돌림
    return s


def feats_h2(I: dict) -> dict:
    c, h, lo, v, tb = I["c"], I["h"], I["l"], I["v"], I["tb"]
    d = 2 * tb - v
    cs = pd.Series(c)
    prev = cs.shift(W).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        dp = c / prev - 1
        V6 = pd.Series(v).rolling(W).sum().to_numpy()
        D6 = pd.Series(d).rolling(W).sum().to_numpy()
        iota = np.where(V6 > 0, D6 / V6, np.nan)
        sd = pd.Series(iota).rolling(SD_WIN, min_periods=SD_WIN).std().to_numpy()
        rng = (pd.Series(h).rolling(W).max() - pd.Series(lo).rolling(W).min()).to_numpy() / prev
    return dict(dp=dp, iota=iota, sd=sd, rng=rng)


def sig_h2(I: dict) -> tuple[np.ndarray, np.ndarray]:
    f = feats_h2(I)
    dp, iota, sd, rng = f["dp"], f["iota"], f["sd"], f["rng"]
    base = (np.isfinite(dp) & np.isfinite(iota) & np.isfinite(sd) & np.isfinite(rng)
            & (np.abs(iota) >= sd) & (np.abs(dp) >= RANGE_FRAC * rng) & (dp != 0) & (iota != 0))
    same = np.sign(dp) == np.sign(iota)
    div = np.where(base & ~same, np.sign(dp), 0.0)      # 다이버전스 → 가격 방향 지속
    norm = np.where(base & same, np.sign(dp), 0.0)      # 대조군(정상 정렬) → 같은 방향 지속
    return div, norm


def sig_h3(I: dict, session_filter: str | None = None, first_bars: int = 6, min_len: int = 24) -> np.ndarray:
    """세션 첫 30분 방향 → 마지막 6봉 진입(봉 e−6 마감 = 청산 e). 신호는 e−6 위치에 둔다."""
    c, tsec, sst, starts, n = I["c"], I["tsec"], I["sst"], I["starts"], I["n"]
    ends = np.r_[starts[1:] - 1, n - 1]
    s3 = np.zeros(n)
    for k, (s, e) in enumerate(zip(starts, ends)):
        if k == len(starts) - 1:
            continue                              # 마지막 세션은 끝이 불완전
        if tsec[s] != sst[s] or s < 1 or e - s + 1 < min_len:
            continue
        if session_filter and market_session(int(tsec[s]))[1] != session_filter:
            continue
        F = c[s + first_bars - 1] / c[s - 1] - 1
        if F == 0:
            continue
        s3[e - H] = np.sign(F)
    return s3


# ── 사건 선택·통계 ──────────────────────────────────────────────────────
def pick(sig: np.ndarray, r6: np.ndarray) -> np.ndarray:
    """시간순 탐욕적 겹침 제거: 한 번 발동하면 보유(H봉) 동안 다음 발동을 버린다."""
    idx = np.flatnonzero((sig != 0) & np.isfinite(r6))
    keep, last = [], -10**9
    for i in idx:
        if i - last >= H:
            keep.append(i)
            last = i
    return np.array(keep, dtype=int)


def split_mask(ts: pd.DatetimeIndex, rng: tuple[str, str]) -> np.ndarray:
    lo, hi = pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta("1D")
    return np.asarray((ts >= lo) & (ts < hi))


def collect(sigs: dict, data: dict, rng: tuple[str, str], shift_days: int = 0) -> dict:
    """코인 풀링 사건. 값 = (방향 × r6) 비용 전, 방향, r6, UTC 날짜, 코인."""
    y, side, r6s, day, coin = [], [], [], [], []
    for ci, k in enumerate(COINS):
        I = data[k]
        s = np.roll(sigs[k], shift_days * 288) if shift_days else sigs[k]
        ix = pick(s, I["r6"])
        ix = ix[split_mask(I["ts"][ix], rng)]
        y.append(s[ix] * I["r6"][ix]); side.append(s[ix]); r6s.append(I["r6"][ix]); day.append(I["day"][ix]); coin.append(np.full(len(ix), ci))
    cat = lambda a: np.concatenate(a) if a else np.array([])
    return dict(y=cat(y), side=cat(side), r6=cat(r6s), day=cat(day), coin=cat(coin))


def uncond_mean(data: dict, rng: tuple[str, str]) -> float:
    xs = [data[k]["r6"][split_mask(data[k]["ts"], rng) & np.isfinite(data[k]["r6"])] for k in COINS]
    x = np.concatenate(xs)
    return float(x.mean()) if len(x) else float("nan")


def day_boot(y: np.ndarray, day: np.ndarray, rng: np.random.Generator, B: int) -> dict:
    if len(y) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), p=float("nan"))
    u, inv = np.unique(day, return_inverse=True)
    S = np.bincount(inv, weights=y, minlength=len(u)); N = np.bincount(inv, minlength=len(u)).astype(float)
    ix = rng.integers(0, len(u), (B, len(u)))
    m = S[ix].sum(1) / N[ix].sum(1)
    return dict(mean=float(y.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)),
                p=float((np.sum(m <= 0) + 1) / (B + 1)))


def ex_top3(y: np.ndarray, day: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    u, inv = np.unique(day, return_inverse=True)
    S = np.bincount(inv, weights=y, minlength=len(u))
    drop = set(np.argsort(S)[-3:]) if len(u) > 3 else set()
    keep = ~np.isin(inv, list(drop))
    return float(y[keep].mean()) if keep.any() else float("nan")


def holm(ps: dict, alpha: float = ALPHA) -> dict:
    order = sorted(ps, key=lambda k: ps[k])
    out, m, ok = {}, len(order), True
    for i, k in enumerate(order):
        ok = ok and ps[k] <= alpha / (m - i)
        out[k] = bool(ok)
    return out


def evaluate(name: str, sigs: dict, data: dict, rng: np.random.Generator, cost_s: float = COST_STRICT,
             cost_c: float = COST_CAND, B: int = BOOT, P: int = PLACEBO) -> dict:
    ev = {sp: collect(sigs, data, r) for sp, r in (("train", TRAIN), ("val", VAL), ("test", TEST))}
    te = ev["test"]
    net_s = te["y"] - cost_s
    bs = day_boot(net_s, te["day"], rng, B)
    # 원형 밀기 위약 (코인 간 같은 밀기, 288봉 정수배 1~30일)
    pm = []
    for _ in range(P):
        k = int(rng.integers(1, 31))
        e = collect(sigs, data, TEST, shift_days=k)
        pm.append(float((e["y"] - cost_s).mean()) if len(e["y"]) else float("nan"))
    pm = np.array([x for x in pm if np.isfinite(x)])
    obs = bs["mean"]
    pl_pct = float((pm < obs).mean() * 100) if len(pm) else float("nan")
    pl95 = float(np.percentile(pm, 95)) if len(pm) else float("nan")
    # 다리별 대조 (비용 전, 같은 구간 무조건 평균 대비)
    um = uncond_mean(data, TEST)
    L, S = te["side"] > 0, te["side"] < 0
    long_ex = float(te["r6"][L].mean() - um) if L.any() else float("nan")
    short_ex = float(-te["r6"][S].mean() + um) if S.any() else float("nan")
    # 코인별
    per_coin = {}
    for ci, k in enumerate(COINS):
        m = te["coin"] == ci
        per_coin[k] = dict(n=int(m.sum()), gross_bp=float(te["y"][m].mean() * 1e4) if m.any() else None,
                           cand_bp=float((te["y"][m] - cost_c).mean() * 1e4) if m.any() else None)
    coins_pos_gross = sum(1 for v in per_coin.values() if v["gross_bp"] is not None and v["gross_bp"] > 0)
    coins_pos_cand = sum(1 for v in per_coin.values() if v["cand_bp"] is not None and v["cand_bp"] > 0)
    val_cand = float((ev["val"]["y"] - cost_c).mean()) if len(ev["val"]["y"]) else float("nan")
    test_cand = float((te["y"] - cost_c).mean()) if len(te["y"]) else float("nan")
    wins = int((net_s > 0).sum())
    out = dict(
        name=name,
        n=dict(train=int(len(ev["train"]["y"])), val=int(len(ev["val"]["y"])), test=int(len(te["y"]))),
        test=dict(gross_bp=float(te["y"].mean() * 1e4) if len(te["y"]) else None, strict_bp=obs * 1e4,
                  strict_ci_bp=[bs["lo"] * 1e4, bs["hi"] * 1e4], p_one_sided=bs["p"],
                  strict_ex_top3days_bp=ex_top3(net_s, te["day"]) * 1e4, cand_bp=test_cand * 1e4,
                  wins_strict=f"{wins}/{len(net_s)}",
                  cost_sensitivity_bp={str(rt): float((te["y"] - rt / 1e4).mean() * 1e4) for rt in (3, 5, 7, 10, 15)} if len(te["y"]) else {}),
        val=dict(gross_bp=float(ev["val"]["y"].mean() * 1e4) if len(ev["val"]["y"]) else None, cand_bp=val_cand * 1e4),
        train=dict(gross_bp=float(ev["train"]["y"].mean() * 1e4) if len(ev["train"]["y"]) else None),
        placebo=dict(n=int(len(pm)), strict_p95_bp=pl95 * 1e4, observed_percentile=pl_pct),
        legs=dict(long_excess_bp=long_ex * 1e4, short_excess_bp=short_ex * 1e4, n_long=int(L.sum()), n_short=int(S.sum())),
        per_coin_test=per_coin, coins_pos_gross=coins_pos_gross, coins_pos_cand=coins_pos_cand,
    )
    c1 = bool(np.isfinite(bs["lo"]) and bs["lo"] > 0)
    c2 = bool(np.isfinite(pl95) and obs > pl95)
    c3 = bool(np.isfinite(long_ex) and np.isfinite(short_ex) and long_ex > 0 and short_ex > 0)
    c4 = bool(np.isfinite(val_cand) and val_cand > 0 and coins_pos_gross >= 3)
    out["criteria"] = dict(c1_ci_lower_gt0=c1, c2_placebo_gt95=c2, c3_both_legs_beat_passive=c3, c4_val_cand_pos_and_3of4_coins=c4)
    out["candidate_pass"] = bool(np.isfinite(val_cand) and val_cand > 0 and np.isfinite(test_cand) and test_cand > 0
                                 and coins_pos_cand >= 2 and np.isfinite(pl_pct) and pl_pct >= 50)
    return out


# ── 메인 ─────────────────────────────────────────────────────────────────
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data = {k: prep(load(k)) for k in COINS}
    panel_end = {k: str(data[k]["ts"][-1]) for k in COINS}
    sigs = {"H1": {k: sig_h1(data[k]) for k in COINS},
            "H2": {k: sig_h2(data[k])[0] for k in COINS},
            "H3": {k: sig_h3(data[k]) for k in COINS}}
    res = {h: evaluate(h, sigs[h], data, rng) for h in sigs}
    hp = holm({h: res[h]["test"]["p_one_sided"] for h in res})
    for h in res:
        c = res[h]["criteria"]
        res[h]["holm_reject"] = hp[h]
        res[h]["pass_strict"] = bool(all(c.values()) and hp[h])
    aux = {}
    # 보조: H1 문턱 1.5·2.5 · H2 정상 정렬 대조 · H3 세션별 — 판정에 안 씀
    for thr in (1.5, 2.5):
        aux[f"H1_thr{thr}"] = evaluate(f"H1_thr{thr}", {k: sig_h1(data[k], thr) for k in COINS}, data, rng, B=500, P=50)
    aux["H2_normal_alignment_control"] = evaluate("H2_normal", {k: sig_h2(data[k])[1] for k in COINS}, data, rng, B=500, P=50)
    for s_name in ("아시아", "유럽", "미국"):
        aux[f"H3_{s_name}"] = evaluate(f"H3_{s_name}", {k: sig_h3(data[k], session_filter=s_name) for k in COINS}, data, rng, B=500, P=50)
    rep = dict(prereg="docs/prereg/session_vwap_divergence_intraday_momo_20261102.md", prereg_commit="170aff6",
               flags=FLAGS, splits=dict(train=TRAIN, val=VAL, test=[TEST[0], max(panel_end.values())]),
               panel_end=panel_end, n_bars={k: data[k]["n"] for k in COINS},
               cost_rt_bp=dict(strict=COST_STRICT * 1e4, candidate=COST_CAND * 1e4),
               primary=res, aux=aux,
               aux_not_run=["지평 3·12봉", "H1 문턱·H2 창 변주", "강세/약세 다이버전스 분리", "월별 부호 표", "OI 조건부 분할"],
               trials=dict(primary=3, aux=len(aux), grid_selection=0))
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    for h, r in res.items():
        t = r["test"]
        print(f"{h}: n={r['n']} test 비용전 {t['gross_bp']:+.2f}bp · 엄격 {t['strict_bp']:+.2f}bp CI[{t['strict_ci_bp'][0]:+.1f},{t['strict_ci_bp'][1]:+.1f}] "
              f"· 위약분위 {r['placebo']['observed_percentile']:.0f} · 기준 {r['criteria']} · holm {r['holm_reject']} → 통과 {r['pass_strict']} · 후보 {r['candidate_pass']}")
    print("report →", OUT / "report.json")


# ── 자체 점검 (합성 데이터) ─────────────────────────────────────────────
def _synth(n_days: int = 300, seed: int = 1, plant_h3: float = 0.0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    n = n_days * 288
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    lr = r.normal(0, 0.001, n)
    base = pd.DataFrame({"timestamp": ts})
    if plant_h3:
        tsec = ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy(dtype="int64")
        sst = np.array([market_session(int(t))[0] for t in tsec])
        new = np.r_[True, sst[1:] != sst[:-1]]
        st = np.flatnonzero(new); en = np.r_[st[1:] - 1, n - 1]
        for k, (s, e) in enumerate(zip(st, en)):
            if k == len(st) - 1 or tsec[s] != sst[s] or s < 1 or e - s + 1 < 24:
                continue
            F = lr[s:s + 6].sum()
            lr[e - 5:e + 1] += plant_h3 * F / 6
    c = 2500 * np.exp(np.cumsum(lr))
    hi = c * (1 + np.abs(r.normal(0, 0.0005, n))); lo = c * (1 - np.abs(r.normal(0, 0.0005, n)))
    v = np.exp(r.normal(3, 0.5, n))
    base["close"], base["high"], base["low"], base["volume"] = c, hi, lo, v
    base["taker_buy_base"] = v * r.uniform(0.4, 0.6, n)
    return base


def selftest() -> None:
    d = _synth()
    I = prep(d)
    # 1) 인과성: 앞 60% 만 가지고 다시 계산해도 그 구간의 신호·피쳐가 같다
    m = int(I["n"] * 0.6)
    J = prep(d.iloc[:m].reset_index(drop=True))
    a1, b1 = sig_h1(I)[:m], sig_h1(J)
    assert np.array_equal(a1, b1), "H1 인과성 위반"
    fa, fb = feats_h2(I), feats_h2(J)
    for k in fa:
        assert np.allclose(fa[k][:m], fb[k], equal_nan=True), f"H2 피쳐 {k} 인과성 위반"
    assert np.array_equal(sig_h2(I)[0][:m], sig_h2(J)[0]), "H2 신호 인과성 위반"
    a3, b3 = sig_h3(I)[: m - 300], sig_h3(J)[: m - 300]
    assert np.array_equal(a3, b3), "H3 인과성 위반"
    # 2) 겹침 제거: 간격 ≥ H
    ix = pick(sig_h2(I)[0], I["r6"])
    assert len(ix) == 0 or np.all(np.diff(ix) >= H), "겹침 제거 실패"
    # 3) Holm
    assert holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}, "Holm 전부 기각"
    assert holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}, "Holm 순차 중단"
    assert holm({"a": 0.5, "b": 0.6, "c": 0.7}) == {"a": False, "b": False, "c": False}
    # 인과성 비교가 빈 배열끼리의 비교가 아니다
    assert (a1 != 0).sum() > 100 and (sig_h2(I)[0][:m] != 0).sum() > 100 and (a3 != 0).sum() > 100, "신호가 비어 있어 인과성 점검이 무의미"
    # 4) 심은 효과를 찾고(H3, 비용 0), 귀무에서는 0 을 포함한다
    global COINS
    old = COINS
    COINS = ["ETH"]
    try:
        r = np.random.default_rng(7)
        pl = {"ETH": prep(_synth(plant_h3=0.3, seed=2))}
        full = ("2024-01-01", "2100-01-01")
        sg = {"ETH": sig_h3(pl["ETH"])}
        e = collect(sg, pl, full)
        bs = day_boot(e["y"], e["day"], r, 500)
        assert bs["lo"] > 0, f"심은 H3 효과를 못 찾음 {bs}"
        nl = {"ETH": prep(_synth(plant_h3=0.0, seed=3))}
        e0 = collect({"ETH": sig_h3(nl["ETH"])}, nl, full)
        b0 = day_boot(e0["y"], e0["day"], r, 500)
        assert b0["lo"] < 0 < b0["hi"], f"귀무인데 0 을 안 포함 {b0}"
    finally:
        COINS = old
    print("selftest OK (인과성 · 겹침 제거 · Holm · 심은 효과 · 귀무)")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
