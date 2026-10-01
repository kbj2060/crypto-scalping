#!/usr/bin/env python3
"""딜러 가정 × 검정 매트릭스 (2026-10-02, 사용자 «체결 기반으로 쌓은 데이터라 그럴 수 있다 — 가정을 바꾸고 다시 검증»).

가정(딜러 순포지션 w):
  T        체결 기반(화면 현행): w = −Σ테이커, 종목 첫 체결부터.     T_block  블록 체결만(요청자 = 고객).
  C        관행: 콜 +OI · 풋 −OI.                                      R        뒤집은 관행: 콜 −OI · 풋 +OI.
  Tflip    «테이커 = 딜러» = T 의 부호 반전. 이 매트릭스의 검정은 전부 w(또는 그 선형 변환)에 선형이거나 순위라
           β → −β, CI 는 거울 → **따로 계산하지 않고** T 결과를 뒤집어 판정만 다시 낸다. 물리 제약은 |w| 라 T 와 같다.
  C·R 은 미결제가 필요해 체인 스냅샷 구간(2026-08-15~09-30, 47일)만. 모든 검정에서 이 47일 공통 구간 칸을 둔다.

재사용(다시 안 돌림, results.json 을 읽기만): 수요압력·만기 소멸·물리 제약(compare 20261002) · 기존 T/C 변동성(H1·H3)
  · 재헤지 원 판정(두 반기) · 헤지 흔적 T 2026(분 단위) · 흐름 지표(flow_result).
새로 계산:
  ① 변동성 T_block — 재구성 스크립트와 같은 사양(week · k=4h · 통제 RV 1h/24h/7d·DVOL·시 더미 · 백분위 순위).
     2026 전체 = H1 사양(iv 대안 B, 반기 01-01~05-31 / 06-01~09-30) · 47일 = H3 사양(스냅샷 mark_iv·underlying_price, 중앙값 분할).
     같은 코드로 T·C 를 다시 내어 H1·H3 반기 값과 소수 4자리 일치를 확인(양성 대조) + 창 전체 β(주 판정용) 추가.
  ② 재헤지 T_block — 재헤지 스크립트의 분 패널·회귀 그대로(z·Δp15 → Ys_f15). T·C 도 같은 코드로 원 반기 값 재현 + 창 전체 β.
  ③ 헤지 흔적(시간 단위 새 설계) — T·T_block·C·R, 47일:
     시각 s_k = 매 정시 첫 체인 스냅샷(recorded_at = 응답 받은 뒤 시각 → 그 OI 는 s_k 에 이미 알려져 있다).
     ΔD_k = Σ_i (w_i(s_k) − w_i(s_{k−1})) × δ_i(s_{k−1})  [ETH] — 직전 스냅샷 델타로 고정해 가격 변화 효과를 뺀 «수량 기여분».
       C·R: Δw = ±ΔOI · T 계열: Δw = −(구간 안 테이커 순). 새로 상장된 종목(직전 스냅샷에 없음)은 w_prev=0, δ = s_k 값.
       s_k 에 만기 지난 종목은 빠진다(만기 정산은 헤지 흐름 정의에서 제외). 스냅샷 간격 > 1.5h 인 구간은 버린다.
     라벨 Y = s_k 뒤 첫 완결 1분봉(open ≥ s_k)부터 60분 선물 테이커 순 ETH(바이낸스 2·tb−v + Deribit 무기한 블록 다리 제외).
     통제 = 같은 구간 Y(open ∈ [s_{k−1}, s_k) 이고 닫힌 분만 — s_k 를 품은 분은 어느 쪽에도 안 씀) · 그 전 구간 Y
       · 구간 수익률·그 전 구간 수익률(s_k 전에 닫힌 분 종가) · UTC 시 더미. OLS, 일 블록 부트스트랩(HF.ols_boot 400회).
     이론: 딜러 델타가 줄면(ΔD<0) 선물을 산다 → β < 0. 한 거래소가 다음 1시간에 헤지를 전부 받으면 β ≈ −1.
판정 기준(CRITERIA)은 새 칸 결과를 보기 전에 고정. 바이낸스 REST 호출 없음(data.binance.vision 일 zip 만).
출력 tmp/dealer_assumption_matrix_20261002/ : results.json · summary.md · tblock_gex_hourly.parquet

  python scripts/research_eth_dealer_assumption_matrix_20261002.py
  python scripts/research_eth_dealer_assumption_matrix_20261002.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_dealer_gex_reconstruct_2026_20261001 as R      # noqa: E402  load_trades · positions_at · exposures · Bars · beta_test · halves
import research_eth_dealer_assumption_compare_20261002 as A        # noqa: E402  positions(마스크별) · judge · END
import research_eth_gamma_rehedge_footprint_2026_20261001 as G     # noqa: E402  minutes · attach · prep · fit · halves · conv_gex
import research_eth_dealer_hedge_footprint_2026_20261001 as HF     # noqa: E402  binance_minutes · ols_boot · fetch_binance_day

OUT = ROOT / "tmp/dealer_assumption_matrix_20261002"
VAL = ROOT / "tmp/opt_validate_20261002"
H_MS, M_MS, D_MS = 3_600_000, 60_000, 86_400_000
ms = lambda s: pd.Timestamp(s, tz="UTC").value // 10**6
END = A.END                                   # 2026-09-30 17:00 — seq 있는 체결 끝 직전(compare 와 같음)
VOL47_END = ms("2026-09-30 14:30")            # H3·사전등록 §6 과 같은 끝(재현 대조)
Y26_END = ms("2026-10-01")

# 결과 보기 전 고정(2026-10-02). 새 칸(변동성 T_block · 재헤지 T_block · 헤지 흔적 시간 단위 전부)과 집계 규칙.
CRITERIA = {
    "rule": A.CRITERIA["rule"],
    "primary_rule_for_matrix": "창 전체 β 의 일 블록 부트스트랩 95% CI(위 rule). 변동성·재헤지는 원 스크립트 판정(두 반기 모두 CI 0 배제)도 병기",
    "primary_cells": {
        "vol": {"def": "week GEX 백분위 → 다음 4h log RV, 통제 RV1h·24h·7d·DVOL·시 더미, 창 전체", "theory_sign": -1, "E": 0.10},
        "rehedge": {"def": "week · 바이낸스+Deribit · h=15 · k=15 의 z·Δp 계수, 창 전체", "theory_sign": -1,
                    "E": "0.05 × sd(Ys_f15) / sd(dp15) [ETH per 1σGEX·bp]"},
        "hedge": {"def": "시간 ΔD(Δw × 직전 스냅샷 δ) → 다음 1h 바이낸스+Deribit 테이커 순 ETH, 통제", "theory_sign": -1, "E": 0.10},
        "demand": "compare test1 cross_section(재사용, 이론 +, E 0.05)",
        "expiry": "compare test2(재사용, 이론 −, E 0.10) — 47일: T 계열 aux_same_days_as_CR · C/R main / 2026: T 계열 main",
        "constraint": "compare test3 위반율(재사용) — 일관=지지 · 불일치=반대 · 경계=불가 · C/R 정의상 0 → 집계 제외",
    },
    "aux_not_judged": ["헤지 흔적: 바이낸스만 · Deribit 만 · 같은 구간(동시, 인과 모호) · 두 반쪽(09-07 분할)",
                       "변동성·재헤지: 원 반기 판정", "흐름 지표(vg24·블록 순베가·순델타) — 표만"],
    "aggregation": "가정마다 주 칸의 지지/반대/불가 수. 47일 공통 구간(6칸)과 2026 전체(T 계열 4칸: 변동성·재헤지·헤지 흔적·만기)를 따로 센다. "
                   "C↔R · T↔Tflip 은 정확한 거울이라 독립 증거가 아니다 — 한쪽 지지 = 다른 쪽 반대.",
    "Tflip": "T 결과 부호 반전(β→−β, CI 거울, 판정 지지↔반대). 물리 제약은 T 와 동일. 따로 계산 안 함",
}


# ───────────────────────── 데이터 ─────────────────────────
def load_trades() -> pd.DataFrame:
    """재구성 스크립트 체결(캐시만 읽는다) + 블록 표시."""
    tr = R.load_trades(int(time.time() * 1000))
    blk = set()
    for p in (R.CACHE / "opt_trades_hist.parquet", R.OUT / "tail_trades.parquet", R.OUT / "prelisted_trades.parquet"):
        b = pd.read_parquet(p, columns=["trade_id", "block_trade_id"]); blk |= set(b.loc[b["block_trade_id"].notna(), "trade_id"])
    tr["is_block"] = tr["trade_id"].isin(blk)
    return tr


def masks_of(tr):
    return {"T": np.ones(len(tr), bool), "T_block": tr["is_block"].to_numpy()}


def hourly_first(ch: pd.DataFrame) -> np.ndarray:
    st = np.sort(ch["ts"].unique())
    return pd.Series(st).groupby(st // H_MS).min().to_numpy()


def chain(end_ms: int) -> pd.DataFrame:
    ch = pd.read_parquet(VAL / "eth_chain.parquet", columns=["recorded_at_utc", "instrument_name", "option_type", "strike", "expiration_ts",
                                                             "open_interest", "mark_iv", "underlying_price"])
    assert str(ch["recorded_at_utc"].dtype).startswith("datetime64[us")      # µs → ms 변환이 맞는지
    ch["ts"] = ch["recorded_at_utc"].astype("int64") // 1000
    ch["exp_ms"] = ch["expiration_ts"].astype("int64") // 1000
    return ch[(ch["ts"] < end_ms) & (ch["exp_ms"] > ch["ts"])]


# ───────────────────────── ① 변동성 ─────────────────────────
def vol_tests(tr, B) -> tuple[dict, pd.DataFrame]:
    out = {}
    # 2026 전체(H1 사양)
    t_eval = np.arange(R.T0, Y26_END, H_MS)
    L = B.at(t_eval); S = L["S"].to_numpy()
    P = A.positions(tr, t_eval, masks_of(tr))
    dv = B.dvol
    P["iv_B"] = P["iv_last"] * dv.reindex((t_eval[P["k"]] // H_MS) * H_MS - H_MS).to_numpy() / \
        dv.reindex((P["t_last"].to_numpy() // H_MS).astype("int64") * H_MS - H_MS).to_numpy()
    Hh = pd.concat([pd.DataFrame({"ts": t_eval}), L], axis=1)
    tb = pd.DataFrame({"ts": t_eval})
    for a in ("T", "T_block"):
        E = R.exposures(P.assign(w=np.where(P["cov"], P[f"pos_{a}"], 0.0)), t_eval, S, "iv_B")
        Hh[f"{a}_week"] = E["gex_week"].to_numpy()
        if a == "T_block":
            tb[[f"gex_{sc}" for sc in R.SCOPES]] = E[[f"gex_{sc}" for sc in R.SCOPES]].to_numpy()
    tb.to_parquet(OUT / "tblock_gex_hourly.parquet")
    for a in ("T", "T_block"):
        r = R.halves(Hh, f"{a}_week", "y4")
        f = R.beta_test(Hh, f"{a}_week", "y4", False)
        out[f"{a}|2026"] = {"halves": {**r, **R.verdict(r, -1)}, "full": f,
                            "judge": A.judge(f["beta"], f["ci"][0], f["ci"][1], f["days"], -1, 0.10)}
    out["spearman_T_vs_Tblock_week_2026"] = round(float(Hh["T_week"].corr(Hh["T_block_week"], method="spearman")), 3)
    print("vol 2026", json.dumps({k: v for k, v in out.items() if "2026" in k}, ensure_ascii=False), flush=True)

    # 47일(H3 사양)
    ch = chain(VOL47_END); s_eval = hourly_first(ch)
    Ls = B.at(s_eval); Ss = Ls["S"].to_numpy()
    sn = ch[ch["ts"].isin(s_eval)].copy(); sn["k"] = np.searchsorted(s_eval, sn["ts"].to_numpy())
    Ps = A.positions(tr, s_eval, masks_of(tr)).merge(
        sn[["k", "instrument_name", "mark_iv", "underlying_price"]].rename(columns={"instrument_name": "inst"}), on=["k", "inst"], how="inner")
    H3 = pd.concat([pd.DataFrame({"ts": s_eval}), Ls], axis=1)
    for a in ("T", "T_block"):
        H3[f"{a}_week"] = R.exposures(Ps.assign(w=np.where(Ps["cov"], Ps[f"pos_{a}"], 0.0)), s_eval, Ss, "mark_iv",
                                      fwd_col="underlying_price")["gex_week"].to_numpy()
    sn["K"] = sn["strike"]; sn["sg"] = np.where(sn["option_type"] == "call", 1.0, -1.0)
    sn = sn.rename(columns={"instrument_name": "inst"})
    for a, sgn in (("C", 1.0), ("R", -1.0)):
        H3[f"{a}_week"] = R.exposures(sn.assign(w=sgn * sn["sg"] * sn["open_interest"]), s_eval, Ss, "mark_iv",
                                      fwd_col="underlying_price")["gex_week"].to_numpy()
    mid = int(np.median(s_eval))
    for a in ("T", "T_block", "C", "R"):
        r = R.halves(H3, f"{a}_week", "y4", split=mid)
        f = R.beta_test(H3, f"{a}_week", "y4", False)
        out[f"{a}|47d"] = {"halves": {**r, **R.verdict(r, -1)}, "full": f,
                           "judge": A.judge(f["beta"], f["ci"][0], f["ci"][1], f["days"], -1, 0.10)}
    out["n_hours_47d"] = len(s_eval)
    out["spearman_47d"] = {f"{a}~{b}": round(float(H3[f"{a}_week"].corr(H3[f"{b}_week"], method="spearman")), 3)
                           for a, b in (("T", "T_block"), ("T", "C"), ("T_block", "C"))}
    print("vol 47d", json.dumps({k: v for k, v in out.items() if "47d" in k}, ensure_ascii=False), flush=True)
    return out, tb


# ───────────────────────── ② 재헤지 ─────────────────────────
def rehedge_tests(tb: pd.DataFrame) -> dict:
    df = G.minutes()
    Hd = pd.read_parquet(R.OUT / "hourly_dealer_gex.parquet"); C = G.conv_gex()
    df["T_g"] = G.attach(df, Hd, ["gex_week"]).to_numpy()[:, 0]
    df["T_block_g"] = G.attach(df, tb, ["gex_week"]).to_numpy()[:, 0]
    df["C_g"] = G.attach(df, C, ["gex_week"]).to_numpy()[:, 0]
    df["R_g"] = -df["C_g"]
    df = G.prep(df)
    E = 0.05 * float(df["Ys_f15"].std()) / float(df["dp15"].std())
    wins = {"2026": (G.D0, G.SPLIT, G.D1), "47d": (G.SW0, G.SW_SPLIT, G.D1)}
    out = {"E": round(E, 3)}
    ts = df.index.to_numpy() * M_MS
    for a in ("T", "T_block", "C", "R"):
        for wn, (lo, sp, hi) in wins.items():
            if a in ("C", "R") and wn == "2026":
                continue
            h = G.halves(df, lo, sp, hi, f"{a}_g", 15, "Ys_f15", "Ys")
            f = G.fit(df[(ts >= lo) & (ts < hi)], f"{a}_g", 15, "Ys_f15", "Ys")
            out[f"{a}|{wn}"] = {"halves": h, "full": f, "judge": A.judge(f[0], f[1], f[2], f[3], -1, E)}
            print("rehedge", a, wn, out[f"{a}|{wn}"], flush=True)
    return out


# ───────────────────────── ③ 헤지 흔적(시간 단위) ─────────────────────────
def delta_change(sn: pd.DataFrame, asm: list) -> pd.DataFrame:
    """sn 행 = (k, instrument_name) 스냅샷 k 에 살아 있는 종목, 열 delta · w_<가정>.
    ΔD_k = Σ (w_k − w_{k−1}) × δ_{k−1}(직전 스냅샷에 없으면 δ_k). 반환 index = k, 열 dD_<가정>."""
    prev = sn[["k", "instrument_name", "delta"] + [f"w_{a}" for a in asm]].copy(); prev["k"] += 1
    m = sn.merge(prev, on=["k", "instrument_name"], how="left", suffixes=("", "_p"))
    dref = m["delta_p"].fillna(m["delta"]).to_numpy()
    d = pd.DataFrame({"k": m["k"]})
    for a in asm:
        d[f"dD_{a}"] = (m[f"w_{a}"] - m[f"w_{a}_p"].fillna(0.0)).to_numpy() * dref
    return d.groupby("k").sum()


def windows(s_prev: np.ndarray, s_cur: np.ndarray) -> dict:
    """분 번호(open//60s) 구간. 라벨 = [m, m+60) (open ≥ s_cur 인 첫 분부터), 같은 구간 = [m_prev, m−1) (닫힌 분만),
    가격 = 분 m−2 의 종가(닫힘 (m−1)·60s ≤ s_cur). s_cur 를 품은 분 m−1 은 어디에도 안 쓴다."""
    m, mp = -(-s_cur // M_MS), -(-s_prev // M_MS)
    return {"lab": (m, m + 60), "same": (mp, m - 1), "px": m - 2, "px_prev": mp - 2}


def wsum(y: pd.Series, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """y(분 번호 연속 인덱스)의 [a, b) 합. 빈 분이 있거나 범위 밖이면 NaN."""
    base = int(y.index[0]); v = y.to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))]); nn = np.concatenate([[0], np.cumsum(np.isnan(v))])
    i, j = a - base, b - base
    ok = (i >= 0) & (j <= len(v)) & (i < j)
    i, j = np.clip(i, 0, len(v)), np.clip(j, 0, len(v))
    return np.where(ok & (nn[j] == nn[i]), cs[j] - cs[i], np.nan)


def futures_minutes() -> pd.DataFrame:
    """분 패널(08-14~09-30): Yb 바이낸스 · c 종가 · Yd Deribit 무기한(블록 다리 제외) · Ys 합. 09-30 은 vision 일 zip."""
    (OUT / "binance").mkdir(parents=True, exist_ok=True)
    with mock.patch.object(HF, "OUT", OUT):
        HF.fetch_binance_day(pd.Timestamp("2026-09-30", tz="UTC"))
    k = pd.concat([pd.read_parquet(f) for f in sorted((OUT / "binance").glob("*.parquet"))]).drop_duplicates("t")
    extra = pd.DataFrame({"Yb": (2 * k["tb"] - k["v"]).to_numpy(), "c": k["c"].to_numpy()}, index=k["t"].to_numpy() // M_MS)
    Bm = pd.concat([HF.binance_minutes(), extra]); Bm = Bm[~Bm.index.duplicated()]
    idx = np.arange(ms("2026-08-14") // M_MS, ms("2026-10-01") // M_MS)
    df = pd.DataFrame(index=idx).join(Bm)
    fs = sorted((HF.OUT / "perp").glob("*.parquet"))
    df["Yd"] = pd.concat([pd.read_parquet(f) for f in fs])["y_nb"].reindex(idx)
    got = pd.to_datetime(idx * 60, unit="s").strftime("%Y-%m-%d").isin([f.stem for f in fs])
    df.loc[got, "Yd"] = df.loc[got, "Yd"].fillna(0.0)
    df["Ys"] = df["Yb"] + df["Yd"]
    return df


def hedge_tests(tr: pd.DataFrame) -> dict:
    asm = ["T", "T_block", "C", "R"]
    ch = chain(END); s_eval = hourly_first(ch)
    sn = ch[ch["ts"].isin(s_eval)].copy(); sn["k"] = np.searchsorted(s_eval, sn["ts"].to_numpy())
    sn["sg"] = np.where(sn["option_type"] == "call", 1.0, -1.0)
    nd1, _ = R.bs(sn["underlying_price"].to_numpy(), sn["strike"].to_numpy(), sn["mark_iv"].to_numpy(),
                  (sn["exp_ms"].to_numpy() - sn["ts"].to_numpy()) / (365 * D_MS))
    sn["delta"] = np.where(sn["sg"] > 0, nd1, nd1 - 1.0)                  # 수집기 _bs_delta 와 같은 식(r=0, 선도가)
    sn.loc[~(sn["mark_iv"] > 0), "delta"] = 0.0
    P = A.positions(tr, s_eval, masks_of(tr)).rename(columns={"inst": "instrument_name"})
    sn = sn.merge(P[["k", "instrument_name", "cov", "pos_T", "pos_T_block"]], on=["k", "instrument_name"], how="left")
    sn["cov"] = sn["cov"].astype("boolean").fillna(True).astype(bool)      # 체결 0 종목 = 포지션 0 확정
    for a in ("T", "T_block"):
        sn[f"w_{a}"] = np.where(sn["cov"], sn[f"pos_{a}"].fillna(0.0), 0.0)
    sn["w_C"] = sn["sg"] * sn["open_interest"]; sn["w_R"] = -sn["w_C"]
    D = delta_change(sn, asm).reindex(range(len(s_eval)))
    gap = np.r_[np.inf, np.diff(s_eval)]
    F = pd.DataFrame({"s": s_eval, "gap_ok": gap <= 1.5 * H_MS}).join(D)
    F.loc[~F["gap_ok"], [f"dD_{a}" for a in asm]] = np.nan
    fm = futures_minutes()
    W = windows(np.r_[s_eval[0], s_eval[:-1]], s_eval)
    lc = np.log(fm["c"])
    for y in ("Ys", "Yb", "Yd"):
        F[f"{y}_next"] = wsum(fm[y], *W["lab"])
        F[f"{y}_same"] = np.where(F["gap_ok"], wsum(fm[y], *W["same"]), np.nan)
        F[f"{y}_prev"] = F[f"{y}_same"].shift(1)
    F["r_same"] = np.where(F["gap_ok"], lc.reindex(W["px"]).to_numpy() - lc.reindex(W["px_prev"]).to_numpy(), np.nan)
    F["r_prev"] = F["r_same"].shift(1)
    F["day"] = F["s"] // D_MS; F["hod"] = (F["s"] // H_MS) % 24
    H = pd.get_dummies(F["hod"], drop_first=True).to_numpy(float)

    def reg(sub, a, y, same=False):
        tgt = F[f"{y}_same"] if same else F[f"{y}_next"]
        cols = [np.ones(len(F)), F[f"dD_{a}"], F["r_same"], F["r_prev"], F[f"{y}_prev"]] + ([] if same else [F[f"{y}_same"]])
        Z = np.column_stack([np.asarray(c, float) for c in cols] + [H])
        ok = sub & np.isfinite(Z).all(1) & np.isfinite(tgt.to_numpy())
        return HF.ols_boot(Z[ok], tgt.to_numpy()[ok], F["day"].to_numpy()[ok])

    allw = np.ones(len(F), bool); split = ms("2026-09-07")
    out = {"n_snap_hours": len(s_eval), "n_pairs_ok": int(F["gap_ok"].sum()),
           "first": str(pd.Timestamp(int(s_eval[0]), unit="ms")), "last": str(pd.Timestamp(int(s_eval[-1]), unit="ms")),
           "sd_dD": {a: round(float(F[f"dD_{a}"].std()), 1) for a in asm},
           "sd_Ys_next": round(float(F["Ys_next"].std()), 1),
           "spearman_dD": {f"{a}~{b}": round(float(F[f"dD_{a}"].corr(F[f"dD_{b}"], method="spearman")), 3)
                           for a, b in (("T", "T_block"), ("T", "C"), ("T_block", "C"))}}
    for a in asm:
        p = reg(allw, a, "Ys")
        out[a] = {"primary_sum_next1h": p, "judge": A.judge(p[0], p[1], p[2], p[3], -1, 0.10),
                  "aux_binance": reg(allw, a, "Yb"), "aux_deribit": reg(allw, a, "Yd"),
                  "aux_same_interval_sum": reg(allw, a, "Ys", same=True),
                  "aux_half_a": reg(F["s"].to_numpy() < split, a, "Ys"), "aux_half_b": reg(F["s"].to_numpy() >= split, a, "Ys")}
        print("hedge", a, json.dumps(out[a], ensure_ascii=False), flush=True)
    return out


# ───────────────────────── 매트릭스 ─────────────────────────
FLIP = {"지지": "반대", "반대": "지지"}


def cell(j: dict, note: str = "") -> dict:
    return {"verdict": j["verdict"], "est": j["est"], "ci": j["ci"], "days": j["n_days"], "note": note,
            **({"n_req": j["n_req_days_for_E"]} if "n_req_days_for_E" in j else {})}


def flip(c: dict) -> dict:
    if "est" not in c:
        return dict(c)
    return {**c, "verdict": FLIP.get(c["verdict"], c["verdict"]), "est": -c["est"], "ci": [-c["ci"][1], -c["ci"][0]],
            "note": "T 부호 반전(계산 안 함)"}


def matrix(vol, reh, hed) -> dict:
    cmp_ = json.loads((ROOT / "tmp/dealer_assumption_compare_20261002/results.json").read_text())
    hf = json.loads((HF.OUT / "results.json").read_text())
    t1, t2, t3 = cmp_["test1"], cmp_["test2"], cmp_["test3"]
    M = {}
    for a in ("T", "T_block", "C", "R"):
        c47 = {"vol": cell(vol[f"{a}|47d"]["judge"]), "rehedge": cell(reh[f"{a}|47d"]["judge"]), "hedge": cell(hed[a]["judge"]),
               "demand": cell(t1[a]["cross_section"])}
        c47["expiry"] = cell(t2[a]["aux_same_days_as_CR"] if a in A.TFAM else t2[a]["main"])
        r3 = t3[a]
        c47["constraint"] = ({"verdict": {"일관": "지지", "불일치": "반대"}.get(r3["verdict"], "불가"), "est": r3["viol_rate"], "ci": r3["ci"],
                              "days": t3["n_days"], "note": f"위반율 · 원 판정 {r3['verdict']} · 무작위 부호 {r3['null_random_sign_viol_rate']}"}
                             if "ci" in r3 else {"verdict": "제외", "note": "정의상 |w| = OI → 위반 0 (판정 제외)"})
        M[f"{a}|47d"] = c47
    for a in ("T", "T_block"):
        hk = "binance|X_main|all" if a == "T" else "binance|X_blk|all"
        b = hf["H1"][hk]["k5"]
        M[f"{a}|2026"] = {
            "vol": cell(vol[f"{a}|2026"]["judge"]), "rehedge": cell(reh[f"{a}|2026"]["judge"]),
            "hedge": cell(A.judge(b[0], b[1], b[2], b[3], +1, 0.10),
                          "분 단위 원 설계(고객 델타 X → 다음 5분 바이낸스, 이론 +) · " + ("X_main" if a == "T" else "선물 다리 없는 블록 X_blk")),
            "expiry": cell(t2[a]["main"]),
            "demand": {"verdict": "미계산", "note": "IV 잔차·수요는 체인 스냅샷(08-15~)이 필요"},
            "constraint": {"verdict": "미계산", "note": "미결제(OI)가 필요 — 체인 스냅샷 08-15~ 뿐"}}
    for w in ("47d", "2026"):
        M[f"Tflip|{w}"] = {k: (flip(v) if k != "constraint" else {**v, "note": "|w| 동일 → T 와 같음"}) for k, v in M[f"T|{w}"].items()}
    for a in ("C", "R"):
        M[f"{a}|2026"] = {k: {"verdict": "미계산", "note": "미결제 필요 — 체인 스냅샷 2026-08-15~ 뿐"} for k in M["T|2026"]}
    agg = {}
    for key, c in M.items():
        a, w = key.split("|")
        cols = ("vol", "rehedge", "hedge", "demand", "expiry", "constraint") if w == "47d" else ("vol", "rehedge", "hedge", "expiry")
        vs = [c[x]["verdict"] for x in cols if c[x]["verdict"] not in ("제외", "미계산")]
        agg[key] = {"지지": vs.count("지지"), "반대": vs.count("반대"), "불가": len(vs) - vs.count("지지") - vs.count("반대"), "n_cells": len(vs)}
    return {"cells": M, "aggregate": agg}


def write_summary(res: dict):
    M, agg = res["matrix"]["cells"], res["matrix"]["aggregate"]
    def f(c):
        if "est" not in c:
            return f"{c['verdict']} ({c.get('note', '')})"
        nr = f" · 필요 {c['n_req']}일" if "n_req" in c else ""
        return f"**{c['verdict']}** {c['est']:+.4g} [{c['ci'][0]:+.4g}, {c['ci'][1]:+.4g}] {c['days']}일{nr}"
    cols = [("vol", "변동성"), ("rehedge", "재헤지"), ("hedge", "헤지 흔적"), ("demand", "수요압력"), ("expiry", "만기 소멸"), ("constraint", "물리 제약")]
    L = []
    for w, title in (("47d", "47일 공통 구간(2026-08-15~09-30)"), ("2026", "2026 전체(T 계열)")):
        L += [f"## {title}", "", "| 가정 | " + " | ".join(n for _, n in cols) + " | 지지/반대/불가 |", "|---" * (len(cols) + 2) + "|"]
        for a in ("T", "T_block", "Tflip", "C", "R"):
            c = M[f"{a}|{w}"]; g = agg[f"{a}|{w}"]
            L.append(f"| {a} | " + " | ".join(f(c[k]) for k, _ in cols) + f" | {g['지지']}/{g['반대']}/{g['불가']} |")
        L.append("")
    (OUT / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_trades()
    B = R.Bars(pd.read_parquet(R.OUT / "perp_5m.parquet"), pd.read_parquet(R.OUT / "dvol_1h.parquet"))
    print(f"체결 {len(tr):,} · 블록 {int(tr['is_block'].sum()):,}", flush=True)
    res = {"criteria": CRITERIA}
    res["vol"], tb = vol_tests(tr, B)
    res["rehedge"] = rehedge_tests(tb)
    res["hedge"] = hedge_tests(tr[tr["timestamp"] < END].reset_index(drop=True))
    res["matrix"] = matrix(res["vol"], res["rehedge"], res["hedge"])
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    write_summary(res)
    print("저장", OUT)


def selftest():
    # 1) ΔOI 델타 기여: 콜 X OI 10→15(δ 0.6→0.9: 가격이 움직임), 풋 Y OI 5→5(δ −0.4→−0.2), 새 콜 Z OI 4(δ 0.5)
    sn = pd.DataFrame({"k": [0, 0, 1, 1, 1], "instrument_name": ["X", "Y", "X", "Y", "Z"], "sg": [1.0, -1.0, 1.0, -1.0, 1.0],
                       "oi": [10.0, 5.0, 15.0, 5.0, 4.0], "delta": [0.6, -0.4, 0.9, -0.2, 0.5]})
    sn["w_C"] = sn["sg"] * sn["oi"]; sn["w_R"] = -sn["w_C"]
    d = delta_change(sn, ["C", "R"])
    assert abs(d.loc[1, "dD_C"] - (5 * 0.6 + 4 * 0.5)) < 1e-12      # 직전 δ 로만(가격 변화 0.6→0.9 효과 없음) · Y 는 ΔOI 0 → 0
    assert d.loc[1, "dD_R"] == -d.loc[1, "dD_C"]
    sn2 = sn.assign(oi=[10.0, 5.0, 10.0, 7.0, 0.0]); sn2["w_C"] = sn2["sg"] * sn2["oi"]
    assert abs(delta_change(sn2, ["C"]).loc[1, "dD_C"] - (-2) * (-0.4)) < 1e-12   # 관행 딜러 풋 −OI: 풋 OI +2 → 딜러 풋 숏 +2 → 델타 +0.8
    # 2) 시점 경계: s = 06:03:40 → 라벨 첫 분 06:04, 같은 구간은 06:02 까지(06:03 은 s 를 품어 제외), 가격 = 06:02 종가
    s = np.array([6 * H_MS + 3 * M_MS + 40_000]); sp = np.array([5 * H_MS + 3 * M_MS + 40_000])
    W = windows(sp, s)
    assert W["lab"][0][0] == 364 and W["same"] == (np.array([304]), np.array([363])) and W["px"][0] == 362
    y = pd.Series(np.arange(500.0), index=np.arange(500))
    assert wsum(y, *W["lab"])[0] == sum(range(364, 424)) and wsum(y, *W["same"])[0] == sum(range(304, 363))
    assert np.isnan(wsum(y.where(y != 400), *W["lab"])[0])          # 라벨 창 안 빈 분 → NaN
    # 3) Tflip = T 부호 반전
    c = flip({"verdict": "반대", "est": -0.2, "ci": [-0.3, -0.1], "days": 47, "note": ""})
    assert c["verdict"] == "지지" and c["ci"] == [0.1, 0.3]
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else main()
