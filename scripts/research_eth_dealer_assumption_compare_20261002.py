#!/usr/bin/env python3
"""딜러 포지션 가정 비교 -- 수요압력·만기 소멸·물리 제약 (2026-10-02, 사용자 «체결 기반 말고 다른 가정과 비교»).

가정(딜러 순포지션 w, 고객 순수요 = −w):
  T        체결 기반(현재 화면): w = −Σ(테이커 부호×수량), 종목 첫 체결부터.
  T_block  블록만(요청자 = 고객).   T_screen  화면(비블록)만.
  T_small  비블록 & 체결 수량 ≤ 25 ETH 만(저장소 09-30 정확도 연구 SMALL 규약 그대로 -- 이번 결과 보기 전 값).
  C        관행: 콜 +OI · 풋 −OI.     R  뒤집은 관행: 콜 −OI · 풋 +OI.   (C·R 은 정확히 거울 → 검정 1·2 의 효과는 부호만 반대)
체결: 재구성 스크립트(research_eth_dealer_gex_reconstruct_2026_20261001)의 load_trades 그대로(seq 1 부터 완결 판정)
  + 블록 표시(block_trade_id). 분석 끝 END = 2026-09-30 17:00 UTC(seq 가 있는 체결 끝 17:16 직전).
  eth_opt_trades_2026.parquet 와 trade_id 대조(양성 대조): 2026 캐시 행 100% 포함, 빠진 건 09-30 17:17 이후뿐.
가능 구간: T 계열 2026-01-01~09-30 · C·R 은 체인 스냅샷 08-15~09-30(미결제 필요). 검정 1·3 은 스냅샷 구간만(전 가정 공통).
시점 경계: 스냅샷 시각 s 의 포지션 = ts < s − 120초 체결만(GUARD -- 체인 수집 소요 시간 동안의 체결 배제).

검정 1 (Gârleanu·Pedersen·Poteshman 2009 수요압력): 이론 = 고객 순수요가 큰 행사가일수록 IV 잔차 +.
  스냅샷 = 시각별 첫 체인. 행사가 단위(콜·풋 합산 -- 패리티로 같은 IV): iv = 콜·풋 mark_iv 평균, 수요 = Σ(고객 순 계약)×베가(1vol pt, USD).
  만기 필터: 잔존 ≥ 1일 · |ln(K/F)|/(iv·√T) ≤ 2 · 행사가 ≥ 8. 잔차 = iv − 3차 다항식(x = ln(K/F)/√T) 적합.
  주(횡단면): (스냅샷, 만기) 그룹마다 Spearman(수요베가, 잔차) → 그룹 크기 가중 평균(일별 합 → 일 블록 부트스트랩).
  보조(시계열): 연속 시간 스냅샷(간격 ≤ 1.5h)에서 Δ잔차 vs Δ수요×베가, 같은 방식. 이론 부호 둘 다 +.
검정 2 (만기 소멸 자연 실험): 매일 08:00 UTC 만기.
  y = ln RV[08:05,12:00) − ln RV[04:00,07:30) (바이낸스 ETHUSDT 선물 1분 로그수익, 정산 평균 07:30~08:05 제외, 빈 분 있으면 버림).
  x = 08:00 만기 종목의 딜러 GEX$(γ·F²·1%·w, 재구성 exposures 의 all 범위)를 표본 안 백분위 순위(0~1)로.
    T 계열: 07:00 시점, IV = 재구성 선택안 B(직전 체결 iv × DVOL 비), S = Deribit 무기한 5분 종가.
    C·R: [07:00,07:30) 첫 체인 스냅샷의 OI·mark_iv·underlying_price.
  통제 = ln DVOL(07:00 에 닫힌 1h 봉) + ln RV(직전 24h, 1분) + 요일 더미 6 + 상수. OLS, 일(=행) 부트스트랩.
  이론 β < 0(딜러 롱감마 소멸 → 변동성 증가). 주 = 각 가정 전체 가능 구간. 보조 = T 계열을 C·R 과 같은 날로 자른 것.
검정 3 (물리 제약): 스냅샷마다 종목별 |딜러 순| vs OI. 모집단 = 살아있는 체인 종목 중 OI>0 또는 T 계열 |순|>0.
  주 = 위반율(|w| > 1.01·OI + 1 ETH 인 종목-스냅샷 비율), 보조 = Σ|w|/ΣOI(설명 비율) · Σmax(|w|−OI,0)/Σ|w|(불가능 몫)
  · 무작위 부호 기준선(같은 체결 수량, 부호 50/50, seed 7, 1회) · T 대비 차이(일 블록 부트스트랩). C·R 은 정의상 0 → 판정 제외.
다중 비교: 주 칸 = 가정 6 × 검정 3(C·R 검정 3 제외 16칸). Bonferroni(16) 표시 = |효과/SE| > 2.95 (판정은 95% CI 기준 그대로).
사후 점검은 키에 _posthoc 를 붙이고 판정에 안 쓴다.
출력 tmp/dealer_assumption_compare_20261002/ : results.json · summary.md

  python scripts/research_eth_dealer_assumption_compare_20261002.py
  python scripts/research_eth_dealer_assumption_compare_20261002.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_eth_dealer_gex_reconstruct_2026_20261001 as R  # noqa: E402  load_trades · positions_at · exposures · Bars

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/dealer_assumption_compare_20261002"
VAL = ROOT / "tmp/opt_validate_20261002"
H_MS, D_MS, M_MS = 3_600_000, 86_400_000, 60_000
END = pd.Timestamp("2026-09-30 17:00", tz="UTC").value // 10**6
GUARD = 120_000
SMALL = 25.0
ASM = ("T", "T_block", "T_screen", "T_small", "C", "R")
TFAM = ASM[:4]
NB = 2000

# 결과 보기 전 고정(2026-10-02). 결과를 보고 바꾸지 않는다.
CRITERIA = {
    "test1_demand_iv": {"main": "횡단면 Spearman(고객 수요×베가, IV 잔차) 그룹가중 평균", "theory_sign": +1, "meaningful_effect": 0.05,
                        "aux": ["시계열 Δ수요 vs Δ잔차"]},
    "test2_expiry": {"main": "β(x=만기 딜러 GEX 백분위) on y=ln RV 후/전, 통제 DVOL·RV24·요일", "theory_sign": -1, "meaningful_effect": 0.10,
                     "aux": ["T 계열을 C·R 날짜로 자른 것"]},
    "test3_constraint": {"main": "위반율 |w|>1.01·OI+1", "consistent_if_ci_hi_le": 0.02, "inconsistent_if_ci_lo_gt": 0.02,
                         "aux": ["Σ|w|/ΣOI", "불가능 몫", "무작위 부호 기준선", "T 대비 차이"], "excluded": ["C", "R"]},
    "rule": "부호 검정: 이론 부호 & 일 블록 부트스트랩 95% CI 0 배제 = 지지 · 반대 부호 & CI 0 배제 = 반대 · "
            "CI 0 포함이고 CI ⊂ (−E,+E) = 근거 없음(E 이상 배제) · 그 외 = 검정력 부족(n_req = 효과 E 를 CI 로 가를 일수 = n·(1.96·SE/E)²)",
    "multiple_comparisons": "주 칸 16개, Bonferroni |z|>2.95 별도 표시",
}


# ───────────────────────── 가정 ─────────────────────────
def subset_masks(tr: pd.DataFrame) -> dict:
    blk = tr["is_block"].to_numpy()
    return {"T": np.ones(len(tr), bool), "T_block": blk, "T_screen": ~blk,
            "T_small": ~blk & (tr["amount"].to_numpy() <= SMALL)}


def dealer_w(asm: str, sg, oi, tpos: dict):
    """딜러 순포지션. T 계열 = −Σ테이커(미리 계산된 tpos), C = 콜 +OI·풋 −OI, R = 그 반대."""
    if asm == "C":
        return sg * oi
    if asm == "R":
        return -sg * oi
    return tpos[asm]


def positions(tr: pd.DataFrame, t_eval: np.ndarray, masks: dict, q_override: np.ndarray | None = None) -> pd.DataFrame:
    """T 계열 딜러 포지션 (k, inst) 행 -- ts < t 체결만(R.positions_at). 열 pos_<가정> · cov(seq 완결)."""
    q = tr["q"].to_numpy() if q_override is None else q_override
    out = None
    for a, m in masks.items():
        P = R.positions_at(tr.assign(q=np.where(m, q, 0.0)), t_eval)
        if out is None:
            out = P.rename(columns={"pos": f"pos_{a}"})
        else:
            out[f"pos_{a}"] = P["pos"].to_numpy()       # 같은 tr 정렬·같은 t_eval → 행 순서 동일
    return out


# ───────────────────────── 통계 ─────────────────────────
def boot_ratio(num: np.ndarray, den: np.ndarray, seed: int = 11) -> tuple[float, float, float]:
    """일별 (분자, 분모) → Σ분자/Σ분모 와 일 부트스트랩 95% CI."""
    ok = den > 0; num, den = num[ok], den[ok]
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(num), (NB, len(num)))
    b = num[ix].sum(1) / den[ix].sum(1)
    return float(num.sum() / den.sum()), *map(float, np.percentile(b, [2.5, 97.5]))


def judge(est, lo, hi, n, sign, E) -> dict:
    se = (hi - lo) / 3.92
    if (lo > 0 or hi < 0):
        v = "지지" if np.sign(est) == sign else "반대"
    elif -E < lo and hi < E:
        v = "근거 없음"
    else:
        v = "검정력 부족"
    r = {"est": round(est, 4), "ci": [round(lo, 4), round(hi, 4)], "n_days": int(n), "verdict": v,
         "bonferroni16": bool(abs(est) / se > 2.95) if se > 0 else False}
    if v == "검정력 부족":
        r["n_req_days_for_E"] = int(np.ceil(n * (1.96 * se / E) ** 2))
    return r


def group_spearman(df: pd.DataFrame, gcols: list, a: str, b: str) -> pd.DataFrame:
    """그룹마다 Spearman(a,b) 과 크기. 상수 열이면 NaN."""
    g = df.groupby(gcols, sort=False)
    d = pd.DataFrame({"g": g.ngroup().to_numpy(), "ra": g[a].rank().to_numpy(), "rb": g[b].rank().to_numpy(), "day": df["day"].to_numpy()})
    gm = d.groupby("g")[["ra", "rb"]].transform("mean")
    xa, xb = d["ra"] - gm["ra"], d["rb"] - gm["rb"]
    s = pd.DataFrame({"g": d["g"], "ab": xa * xb, "aa": xa * xa, "bb": xb * xb, "n": 1, "day": d["day"]}) \
        .groupby("g").agg(ab=("ab", "sum"), aa=("aa", "sum"), bb=("bb", "sum"), n=("n", "sum"), day=("day", "first"))
    with np.errstate(invalid="ignore", divide="ignore"):
        s["rho"] = s["ab"] / np.sqrt(s["aa"] * s["bb"])
    return s[np.isfinite(s["rho"])]


def daily_rho(s: pd.DataFrame, sign=+1, E=0.05) -> dict:
    dd = s.assign(w=s["rho"] * s["n"]).groupby("day")[["w", "n"]].sum()
    est, lo, hi = boot_ratio(dd["w"].to_numpy(), dd["n"].to_numpy(float))
    return {**judge(est, lo, hi, len(dd), sign, E), "n_groups": len(s)}


def ols_boot(y, X, seed=13) -> tuple[float, float, float, float]:
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    b0 = np.linalg.lstsq(X[:, 1:], y, rcond=None)[0]
    r2_0 = 1 - ((y - X[:, 1:] @ b0) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(seed); bs = []
    for _ in range(NB):
        ix = rng.integers(0, len(y), len(y)); bs.append(np.linalg.lstsq(X[ix], y[ix], rcond=None)[0][0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(b[0]), float(lo), float(hi), float(r2 - r2_0)


# ───────────────────────── 스마일 ─────────────────────────
def smile_resid(st: pd.DataFrame) -> pd.DataFrame:
    """행사가 표(k, exp_ms, K, iv, F, T) → 필터 통과 행 + resid(iv − 3차 적합) + vega(1vol pt, USD/ETH)."""
    z = np.log(st["K"] / st["F"]) / (st["iv"] / 100 * np.sqrt(st["T"]))
    st = st[(st["T"] >= 1 / 365) & (st["iv"] > 0) & (z.abs() <= 2)].copy()
    st["x"] = np.log(st["K"] / st["F"]) / np.sqrt(st["T"])
    st = st[st.groupby(["k", "exp_ms"])["K"].transform("size") >= 8]
    res = np.empty(len(st)); xs, ivs = st["x"].to_numpy(), st["iv"].to_numpy()
    for idx in st.groupby(["k", "exp_ms"], sort=False).indices.values():
        res[idx] = ivs[idx] - np.polyval(np.polyfit(xs[idx], ivs[idx], 3), xs[idx])
    st["resid"] = res
    s = st["iv"] / 100
    d1 = (np.log(st["F"] / st["K"]) + 0.5 * s * s * st["T"]) / (s * np.sqrt(st["T"]))
    st["vega"] = st["F"] * norm.pdf(d1) * np.sqrt(st["T"]) / 100
    return st


def rv_win(lr: pd.Series, a_ms: np.ndarray, b_ms: np.ndarray) -> np.ndarray:
    """1분 로그수익(인덱스 = 봉 open ms) 중 open ∈ [a, b) 의 RV. 빈 분이 있으면 NaN."""
    t = lr.index.to_numpy(); c2 = np.concatenate([[0.0], np.cumsum(lr.to_numpy() ** 2)])
    i, j = np.searchsorted(t, a_ms, "left"), np.searchsorted(t, b_ms, "left")
    full = (j - i) == (b_ms - a_ms) // M_MS
    return np.where(full, np.sqrt(c2[j] - c2[i]), np.nan)


# ───────────────────────── 메인 ─────────────────────────
def load():
    tr = R.load_trades(END)
    blk = set()
    for p in (R.CACHE / "opt_trades_hist.parquet", R.OUT / "tail_trades.parquet", R.OUT / "prelisted_trades.parquet"):
        b = pd.read_parquet(p, columns=["trade_id", "block_trade_id"]); blk |= set(b.loc[b["block_trade_id"].notna(), "trade_id"])
    tr["is_block"] = tr["trade_id"].isin(blk)
    tr = tr[tr["timestamp"] < END].reset_index(drop=True)
    new = pd.read_parquet(VAL / "eth_opt_trades_2026.parquet", columns=["ts_ms", "trade_id", "block_trade_id"])
    new = new[new["ts_ms"] < END]
    ctrl = {"new_in_ours": round(float(new["trade_id"].isin(set(tr["trade_id"])).mean()), 6), "n_new": len(new),
            "block_flag_agree": round(float((new["block_trade_id"].notna().to_numpy() ==
                                             new["trade_id"].isin(blk).to_numpy()).mean()), 6)}
    return tr, ctrl


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, ctrl = load()
    masks = subset_masks(tr)
    vol = {a: round(float(tr.loc[m & (tr["exp_ms"] > R.T0).to_numpy(), "amount"].sum() / tr.loc[tr["exp_ms"] > R.T0, "amount"].sum()), 4)
           for a, m in masks.items()}
    res = {"criteria": CRITERIA, "control": ctrl, "volume_share_2026_expiries": vol, "end_utc": str(pd.Timestamp(END, unit="ms"))}
    print("체결", len(tr), ctrl, vol, flush=True)

    # ── 스냅샷 공통(검정 1·3)
    ch = pd.read_parquet(VAL / "eth_chain.parquet")
    ch["ts"] = ch["recorded_at_utc"].astype("int64") // 1000
    ch = ch[ch["ts"] < END]
    st_ = np.sort(ch["ts"].unique()); s_eval = pd.Series(st_).groupby(st_ // H_MS).min().to_numpy()
    sn = ch[ch["ts"].isin(s_eval)].copy()
    sn["exp_ms"] = sn["expiration_ts"].astype("int64") // 1000
    sn = sn[sn["exp_ms"] > sn["ts"]]
    sn["k"] = np.searchsorted(s_eval, sn["ts"].to_numpy())
    sn["sg"] = np.where(sn["option_type"] == "call", 1.0, -1.0)
    P = positions(tr, s_eval - GUARD, masks).rename(columns={"inst": "instrument_name"})
    sn = sn.merge(P[["k", "instrument_name", "cov"] + [f"pos_{a}" for a in TFAM]], on=["k", "instrument_name"], how="left")
    sn["cov"] = sn["cov"].astype("boolean").fillna(True).astype(bool)      # 체결 0 인 종목 = 포지션 0 확정
    for a in TFAM:
        sn[f"pos_{a}"] = sn[f"pos_{a}"].fillna(0.0)
    tpos = {a: sn[f"pos_{a}"].to_numpy() for a in TFAM}
    for a in ASM:
        sn[f"w_{a}"] = dealer_w(a, sn["sg"].to_numpy(), sn["open_interest"].to_numpy(), tpos)
    sn["day"] = sn["ts"] // D_MS
    res["snap"] = {"n_snapshots": len(s_eval), "days": int(sn["day"].nunique()), "first": str(pd.Timestamp(s_eval[0], unit="ms")),
                   "cov_share_rows": round(float(sn["cov"].mean()), 6)}
    print("스냅샷", res["snap"], flush=True)

    # ── 검정 1
    sn["iv_ok"] = sn["mark_iv"].where(sn["mark_iv"] > 0)
    for a in ASM:
        sn[f"c_{a}"] = -sn[f"w_{a}"]                                            # 고객 순수요 = −딜러, 아래서 콜·풋 합산
    agg = {"iv": ("iv_ok", "mean"), "F": ("underlying_price", "mean"), "day": ("day", "first")}
    agg.update({f"D_{a}": (f"c_{a}", "sum") for a in ASM})
    stk = sn.groupby(["k", "exp_ms", "strike"], sort=False).agg(**agg).reset_index().rename(columns={"strike": "K"})
    stk["T"] = (stk["exp_ms"] - s_eval[stk["k"]]) / (365 * D_MS)
    sm = smile_resid(stk.dropna(subset=["iv"]))
    t1 = {"n_strike_rows": len(sm), "resid_sd": round(float(sm["resid"].std()), 3)}
    for a in ASM:
        sm[f"X_{a}"] = sm[f"D_{a}"] * sm["vega"]
        t1[a] = {"cross_section": daily_rho(group_spearman(sm, ["k", "exp_ms"], f"X_{a}", "resid"))}
    prev = sm[["k", "exp_ms", "K", "resid"] + [f"D_{a}" for a in ASM]].copy(); prev["k"] += 1
    ts_ = sm.merge(prev, on=["k", "exp_ms", "K"], suffixes=("", "_p"))
    ts_ = ts_[(s_eval[ts_["k"]] - s_eval[ts_["k"] - 1]) <= 1.5 * H_MS]
    ts_["dres"] = ts_["resid"] - ts_["resid_p"]
    for a in ASM:
        ts_[f"dX_{a}"] = (ts_[f"D_{a}"] - ts_[f"D_{a}_p"]) * ts_["vega"]
        t1[a]["time_series_aux"] = daily_rho(group_spearman(ts_, ["k", "exp_ms"], f"dX_{a}", "dres"))
        t1[a]["time_series_aux"]["share_rows_dD_nonzero"] = round(float((ts_[f"dX_{a}"] != 0).mean()), 4)
    res["test1"] = t1
    print("검정1", json.dumps({a: t1[a]["cross_section"] for a in ASM}, ensure_ascii=False), flush=True)

    # ── 검정 3
    U = sn[(sn["open_interest"] > 0) | (np.abs(sn[[f"w_{a}" for a in TFAM]]).sum(1) > 0)].copy()
    rng = np.random.default_rng(7)
    qn = rng.choice([-1.0, 1.0], len(tr)) * tr["amount"].to_numpy()
    Pn = positions(tr, s_eval - GUARD, masks, q_override=qn).rename(columns={"inst": "instrument_name"})
    U = U.merge(Pn[["k", "instrument_name"] + [f"pos_{a}" for a in TFAM]].rename(columns={f"pos_{a}": f"null_{a}" for a in TFAM}),
                on=["k", "instrument_name"], how="left").fillna({f"null_{a}": 0.0 for a in TFAM})
    oi = U["open_interest"].to_numpy(); day = U["day"].to_numpy()
    lim = 1.01 * oi + 1.0
    t3 = {"n_rows": len(U), "n_days": int(U["day"].nunique())}
    viol_T = None
    for a in ASM:
        w = np.abs(U[f"w_{a}"].to_numpy()); v = (w > lim).astype(float)
        g = pd.DataFrame({"day": day, "v": v, "n": 1.0, "w": w, "oi": oi, "ex": np.maximum(w - oi, 0)}).groupby("day").sum()
        r = {}
        if a in TFAM:
            est, lo, hi = boot_ratio(g["v"].to_numpy(), g["n"].to_numpy())
            thr = CRITERIA["test3_constraint"]["consistent_if_ci_hi_le"]
            r = {"viol_rate": round(est, 4), "ci": [round(lo, 4), round(hi, 4)],
                 "verdict": "일관" if hi <= thr else ("불일치" if lo > thr else "경계")}
            wn = np.abs(U[f"null_{a}"].to_numpy())
            r["null_random_sign_viol_rate"] = round(float((wn > lim).mean()), 4)
            r["null_random_sign_explain"] = round(float(wn.sum() / oi.sum()), 4)
            if a == "T":
                viol_T = v
            else:
                gd = pd.DataFrame({"day": day, "d": v - viol_T, "n": 1.0}).groupby("day").sum()
                e2, l2, h2 = boot_ratio(gd["d"].to_numpy(), gd["n"].to_numpy())
                r["diff_vs_T"] = [round(e2, 4), round(l2, 4), round(h2, 4)]
        else:
            r = {"viol_rate": round(float(v.mean()), 4), "verdict": "정의상 0 -- 판정 제외"}
        est, lo, hi = boot_ratio(g["w"].to_numpy(), g["oi"].to_numpy())
        r["explain_sum_abs_over_oi"] = [round(est, 4), round(lo, 4), round(hi, 4)]
        r["impossible_share"] = round(float(g["ex"].sum() / max(g["w"].sum(), 1e-9)), 4)
        t3[a] = r
    res["test3"] = t3
    print("검정3", json.dumps(t3, ensure_ascii=False), flush=True)

    # ── 검정 2
    m1 = pd.read_parquet(VAL / "ethusdt_1m.parquet")
    m1 = m1.drop_duplicates("open_time").sort_values("open_time")
    lr = pd.Series(np.log(m1["close"].to_numpy()), index=m1["open_time"].to_numpy()).diff()
    lr = lr[np.r_[False, np.diff(lr.index.to_numpy()) == M_MS]]                 # 앞 분이 있는 수익만
    bars, dvol = R.fetch_series(END)
    B = R.Bars(bars, dvol)
    days = np.arange(R.T0, END - 12 * H_MS, D_MS)                                # 12:00 < END 인 날
    t7 = days + 7 * H_MS

    def frame(t_x: np.ndarray, d0: np.ndarray) -> pd.DataFrame:
        f = pd.DataFrame({"day": d0 // D_MS})
        f["y"] = np.log(rv_win(lr, d0 + 8 * H_MS + 5 * M_MS, d0 + 12 * H_MS)) - np.log(rv_win(lr, d0 + 4 * H_MS, d0 + 7 * H_MS + 30 * M_MS))
        tm = t_x - t_x % M_MS
        f["rv24"] = np.log(rv_win(lr, tm - 24 * H_MS, tm))
        f["dvol"] = np.log(dvol.set_index("ts")["c"].reindex((t_x // H_MS) * H_MS - H_MS).to_numpy())
        f["dow"] = ((d0 // D_MS) + 3) % 7                                         # 1970-01-01 = 목(3)
        return f

    def reg(f: pd.DataFrame, x: str, sign=-1, E=0.10) -> dict:
        d = f[["y", x, "rv24", "dvol", "dow"]].replace([np.inf, -np.inf], np.nan).dropna()
        X = np.column_stack([d[x].rank(pct=True), d[["rv24", "dvol"]], pd.get_dummies(d["dow"]).to_numpy(float)[:, 1:], np.ones(len(d))])
        b, lo, hi, dr2 = ols_boot(d["y"].to_numpy(), X)
        return {**judge(b, lo, hi, len(d), sign, E), "dR2": round(dr2, 4), "share_gex_pos": round(float((d[x] > 0).mean()), 3)}

    # T 계열: 07:00, 만기 08:00 종목만
    S7 = B.at(t7)["S"].to_numpy()
    PT = positions(tr, t7, masks)
    PT = PT[PT["exp_ms"].to_numpy() == t7[PT["k"].to_numpy()] + H_MS].copy()
    dv = B.dvol
    PT["iv_B"] = PT["iv_last"] * dv.reindex((t7[PT["k"]] // H_MS) * H_MS - H_MS).to_numpy() / \
        dv.reindex((np.nan_to_num(PT["t_last"].to_numpy()) // H_MS).astype("int64") * H_MS - H_MS).to_numpy()
    FT = frame(t7, days)
    for a in TFAM:
        FT[f"g_{a}"] = R.exposures(PT.assign(w=PT[f"pos_{a}"]), t7, S7, "iv_B")["gex_all"].to_numpy()
    # C·R: [07:00, 07:30) 첫 스냅샷
    cs = pd.Series(st_); cs = cs[(cs % D_MS >= 7 * H_MS) & (cs % D_MS < 7 * H_MS + 30 * M_MS)]
    t_cr = cs.groupby(cs // D_MS).min().to_numpy(); d_cr = (t_cr // D_MS) * D_MS
    cc = ch[ch["ts"].isin(t_cr)].copy()
    cc["exp_ms"] = cc["expiration_ts"].astype("int64") // 1000
    cc = cc[cc["exp_ms"] == (cc["ts"] // D_MS) * D_MS + 8 * H_MS]
    cc["k"] = np.searchsorted(t_cr, cc["ts"].to_numpy()); cc["K"] = cc["strike"]; cc["sg"] = np.where(cc["option_type"] == "call", 1.0, -1.0)
    Scr = B.at(t_cr)["S"].to_numpy()
    FC = frame(t_cr, d_cr)
    for a in ("C", "R"):
        FC[f"g_{a}"] = R.exposures(cc.assign(w=dealer_w(a, cc["sg"].to_numpy(), cc["open_interest"].to_numpy(), {})),
                                   t_cr, Scr, "mark_iv", fwd_col="underlying_price")["gex_all"].to_numpy()
    t2 = {"T_family_days": int(FT["y"].notna().sum()), "CR_days": int(FC["y"].notna().sum())}
    for a in TFAM:
        t2[a] = {"main": reg(FT, f"g_{a}"), "aux_same_days_as_CR": reg(FT[FT["day"].isin(FC["day"])], f"g_{a}")}
    for a in ("C", "R"):
        t2[a] = {"main": reg(FC, f"g_{a}")}
    J = FT.merge(FC[["day", "g_C"]], on="day")
    t2["spearman_T_vs_C_same_days"] = round(float(J["g_T"].corr(J["g_C"], method="spearman")), 3)
    t2["spearman_T_vs_screen_vs_block_full"] = {"T~screen": round(float(FT["g_T"].corr(FT["g_T_screen"], method="spearman")), 3),
                                                "T~block": round(float(FT["g_T"].corr(FT["g_T_block"], method="spearman")), 3)}
    res["test2"] = t2
    print("검정2", json.dumps(t2, ensure_ascii=False), flush=True)

    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    write_summary(res)
    print("저장", OUT)


def write_summary(res: dict):
    def c(r):
        if "ci" not in r:
            return "-"
        extra = f" · n_req {r['n_req_days_for_E']}일" if "n_req_days_for_E" in r else ""
        return f"{r['verdict']} {r.get('est', r.get('viol_rate'))} [{r['ci'][0]}, {r['ci'][1]}] ({r.get('n_days', '')}일){extra}"
    L = ["| 가정 | 검정1 수요→IV 잔차 ρ (주: 횡단면) | 검정1 보조: 시계열 ρ | 검정2 만기 소멸 β (주) | 검정2 보조: C·R 날짜 | 검정3 위반율 | 검정3 Σ|w|/ΣOI |",
         "|---|---|---|---|---|---|---|"]
    for a in ASM:
        t1, t2, t3 = res["test1"][a], res["test2"][a], res["test3"][a]
        v3 = (f"{t3['verdict']} {t3['viol_rate']} [{t3['ci'][0]}, {t3['ci'][1]}] (무작위 {t3['null_random_sign_viol_rate']})"
              if "ci" in t3 else f"{t3['verdict']}")
        L.append(f"| {a} | {c(t1['cross_section'])} | {c(t1['time_series_aux'])} | {c(t2['main'])} | "
                 f"{c(t2['aux_same_days_as_CR']) if 'aux_same_days_as_CR' in t2 else '-'} | {v3} | {t3['explain_sum_abs_over_oi'][0]} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def selftest():
    h = H_MS
    # 1) 가정별 부호: 콜 X 에 블록 테이커 매수 10, 화면 소형 매도 3, 화면 대형(30) 매수 30
    tr = pd.DataFrame({"instrument_name": ["X"] * 3, "timestamp": [h, h + 1, h + 2], "trade_seq": [1, 2, 3],
                       "q": [10.0, -3.0, 30.0], "amount": [10.0, 3.0, 30.0], "is_block": [True, False, False], "iv": [50.0] * 3,
                       "K": [100.0] * 3, "sg": [1.0] * 3, "exp_ms": [100 * h] * 3})
    P = positions(tr, np.array([2 * h]), subset_masks(tr)).iloc[0]
    assert (P["pos_T"], P["pos_T_block"], P["pos_T_screen"], P["pos_T_small"]) == (-37.0, -10.0, -27.0, 3.0)
    tp = {"T": np.array([-37.0])}
    assert dealer_w("C", np.array([1.0, -1.0]), np.array([20.0, 5.0]), tp).tolist() == [20.0, -5.0]
    assert dealer_w("R", np.array([1.0, -1.0]), np.array([20.0, 5.0]), tp).tolist() == [-20.0, 5.0]
    assert dealer_w("T", None, None, tp)[0] == -37.0
    # 2) 시점 경계: 평가 시각 − GUARD 와 같은 시각의 체결은 안 보고, 1ms 앞은 본다
    s = 2 * h + GUARD
    tr2 = tr.assign(timestamp=[h, 2 * h - 1, 2 * h])
    P2 = positions(tr2, np.array([s - GUARD]), subset_masks(tr2)).iloc[0]
    assert P2["pos_T"] == -7.0                                  # 10 − 3, 정각(2h) 30 은 제외
    # 3) RV 창: 07:30~08:05 에만 움직이면 전·후 RV 둘 다 0 근처, 빈 분이 있으면 NaN
    # 봉 open t 의 수익 = 종가(t) − 종가(t−1) = 실제 시각 [t, t+1분). 움직임은 open ∈ [07:30, 08:04) 봉에서만.
    t = np.arange(0, D_MS, M_MS); px = np.full(len(t), 100.0)
    jump = (t >= 7 * h + 30 * M_MS) & (t < 8 * h + 4 * M_MS); px[jump] = 101.0 + (np.arange(jump.sum()) % 2)
    lr = pd.Series(np.log(px), index=t).diff().dropna()
    a = rv_win(lr, np.array([4 * h, 8 * h + 5 * M_MS, 7 * h + 30 * M_MS]), np.array([7 * h + 30 * M_MS, 12 * h, 8 * h + 5 * M_MS]))
    assert a[0] == 0 and a[1] == 0 and a[2] > 0                # 전·후 창은 제외 구간의 움직임을 안 본다
    assert np.isnan(rv_win(lr.drop(lr.index[300]), np.array([4 * h]), np.array([6 * h]))[0])
    # 4) 스마일 잔차: 매끈한 스마일 + 한 행사가 +5pt 혹 → 그 행사가 잔차가 최대이고 양수
    K = np.arange(80.0, 125.0, 2.5); F, T = 100.0, 30 / 365
    iv = 60 + 40 * np.log(K / F) ** 2; iv[9] += 5
    st = pd.DataFrame({"k": 0, "exp_ms": 1, "K": K, "F": F, "T": T, "iv": iv})
    r = smile_resid(st)
    assert r["resid"].idxmax() == 9 and r.loc[9, "resid"] > 2
    # 5) 판정: 이론 부호 CI 0 배제 = 지지, 반대 = 반대, 좁게 0 포함 = 근거 없음, 넓게 = 검정력 부족
    assert judge(0.1, 0.02, 0.2, 40, +1, 0.05)["verdict"] == "지지"
    assert judge(0.1, 0.02, 0.2, 40, -1, 0.05)["verdict"] == "반대"
    assert judge(0.0, -0.03, 0.03, 40, +1, 0.05)["verdict"] == "근거 없음"
    assert judge(0.0, -0.2, 0.2, 40, +1, 0.05)["verdict"] == "검정력 부족"
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else main()
