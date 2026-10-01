#!/usr/bin/env python3
"""실현 감마 국면(RG) vs 체결 기반 딜러 GEX(T) 정면 비교 (2026-10-02, 사용자 «실현 감마 국면 가정이 딜러 체결 기반 가정보다 더 나은지»).

질문: «지금 시장이 움직임을 누르는 장(롱감마)인지 키우는 장(숏감마)인지»를 어느 쪽이 더 정확히 알려 주나. 수익률 아님.
정의는 재사용만 한다:
  RG = research_eth_realized_gamma_regime_20261002.panel (바이낸스 5분 · 가격점 ≤ t · 결과 가격점 ≥ t+5분)
  T  = tmp/dealer_gex_reconstruct_2026_20261001/hourly_dealer_gex.parquet (ts = t, 체결 ts < t) · 보조 T_block(같은 시각 규약)
  C  = tmp/gamma_rehedge_footprint_2026_20261001/conv_gex_hourly.parquet (ts = 스냅샷 응답 시각 → t 이하 최신 스냅샷만)
방향 맞추기: 롱감마 점수 = −RG · +GEX → 같은 표본 안 백분위 순위(0~1).
기준은 아래 CRITERIA(결과 보기 전 고정). 바이낸스 REST 호출 없음.

  python scripts/research_eth_realized_gamma_vs_dealer_gex_20261002.py
  python scripts/research_eth_realized_gamma_vs_dealer_gex_20261002.py --selftest
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_realized_gamma_regime_20261002 as G   # noqa: E402  load_5m · panel · dummies · ms · A.judge

OUT = ROOT / "tmp/realized_gamma_vs_dealer_gex_20261002"
ms, D_MS = G.ms, G.D_MS
T0, T_END, C0 = ms("2026-01-01"), ms("2026-09-30 17:00"), ms("2026-08-15")   # T_END = hourly_dealer_gex 마지막 ts
NB, SEED = 2000, 20261002

# 결과 보기 전 고정(2026-10-02).
CRITERIA = {
    "sample": "2026-01-01 00:00 ~ 2026-09-30 17:00 UTC 매 정시(1h 격자). 비교마다 참가 지표 전부 + 목표가 유효한 공통 행만. "
              "겹치지 않는 창 = 4h 격자(UTC 0·4·8·…시). 47일 보조 = 2026-08-15 ~ (C 있는 구간).",
    "score": "롱감마 점수: RG → −RG, GEX → +GEX, 그다음 그 비교의 공통 표본 안 백분위 순위(0~1). 계수 단위 = 점수 최저→최고.",
    "primary_pair": "RG1_4h vs T_week. RG2_4h vs T_week 병기. RG 12h · T_all · T_front 는 보조 표.",
    "Y1": {"def": "y = ln RV(다음 4h, 5분 제곱합, 가격점 [t+5m, t+4h5m]) ~ UTC 시 더미 24 + ln RV 1h·24h·7d + ln DVOL(t 에 닫힌 1h 봉) + 점수. "
                  "점수 계수 = HAR 잔차에 대한 효과(FWL 과 동치).", "theory_sign": -1, "E": 0.05, "role": "주 판정"},
    "Y2": {"def": "y = r_next_z(다음 1h, t+5m 부터) ~ 1 + r_past_z + 점수 + r_past_z×점수. 상호작용 계수(롱감마일수록 되돌림 → 음수).",
           "theory_sign": -1, "E": 0.05, "role": "보조"},
    "Y3": {"def": "y = 다음 4h 5분 수익률 1차 자기상관 ~ 1 + 점수.", "theory_sign": -1, "E": 0.02,
           "role": "보조 — RG 는 같은 측정치의 과거 값이라 지속성만으로 유리(구조적 편향)"},
    "verdict": {
        "1_single": "단독 모형 점수 계수: 이론 부호 & 일 블록 부트스트랩 95% CI 0 배제 = «유효»(A.judge: 지지/반대/근거 없음/검정력 부족)",
        "2_both": "둘을 같이 넣은 모형에서 각자의 계수(같은 판정 규칙)",
        "3_dR2": "ΔR²(X) = R²(기준 + X) − R²(기준). 차이 ΔR²(RG) − ΔR²(T) 의 같은 부트스트랩 95% CI: 0 배제면 큰 쪽이 «더 낫다», 포함이면 «구분 불가»",
        "4_primary": "주 판정 = Y1 · 1h 격자 · 일 블록. Y2·Y3·4h 격자·주 블록·47일은 보조",
    },
    "bootstrap": f"UTC 일 블록 {NB}회(같은 추출을 모든 모형에 공유 → 차이의 CI 가 짝지어짐). 주 블록(일//7) 은 주 쌍 보조.",
    "controls": "대조군: (a) 점수 셔플(행 순열) (b) 지표 1일(24행) 지연 — 둘 다 계수·ΔR² 가 0 근처여야.",
    "boundary": "RG: 가격점 ≤ t. T·T_block: 체결 ts < t(행 ts = t). C: 스냅샷 응답 시각 ≤ t 인 최신 값(merge_asof backward). "
                "DVOL: [t−1h, t) 봉 종가. 목표: 가격점 ≥ t+5분(한 봉 공백).",
}


# ───────────────────────── 통계 ─────────────────────────
def fit_many(Xs: dict, y: np.ndarray, blk: np.ndarray, nb=NB, seed=SEED) -> tuple[dict, int]:
    """여러 설계를 같은 행·같은 블록 부트스트랩 추출로 OLS. {이름: (계수, 부트 계수 nb×k, R², 부트 R² nb)}.
    블록별 충분통계(X'X, X'y, y'y, Σy, n)를 추출 횟수로 가중합 → SSR = y'y − b'X'y, SST = y'y − (Σy)²/n."""
    m = np.isfinite(y) & np.all([np.isfinite(X).all(1) for X in Xs.values()], 0)
    o = np.argsort(blk[m], kind="stable"); y = y[m][o]; b = blk[m][o]
    st = np.flatnonzero(np.r_[True, b[1:] != b[:-1]]); D = len(st)
    rng = np.random.default_rng(seed)
    C = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(nb)]).astype(float)
    yy, sy, n = np.add.reduceat(y * y, st), np.add.reduceat(y, st), np.add.reduceat(np.ones_like(y), st)
    yyb, syb, nbk = C @ yy, C @ sy, C @ n
    out = {}
    for nm, X in Xs.items():
        X = X[m][o]; k = X.shape[1]
        XX = np.add.reduceat(np.einsum("ni,nj->nij", X, X).reshape(len(X), -1), st).reshape(D, k, k)
        Xy = np.add.reduceat(X * y[:, None], st)
        bf = np.linalg.lstsq(XX.sum(0), Xy.sum(0), rcond=None)[0]
        r2 = 1 - (yy.sum() - bf @ Xy.sum(0)) / (yy.sum() - sy.sum() ** 2 / n.sum())
        Xyb = C @ Xy
        bb = np.linalg.solve(np.einsum("bd,dij->bij", C, XX) + 1e-10 * np.eye(k), Xyb[..., None])[..., 0]
        r2b = 1 - (yyb - (bb * Xyb).sum(1)) / (yyb - syb ** 2 / nbk)
        out[nm] = (bf, bb, float(r2), r2b)
    return out, D


def ci(est, bs):
    lo, hi = np.percentile(bs, [2.5, 97.5]); return [round(float(est), 5), round(float(lo), 5), round(float(hi), 5)]


def rank01(x: pd.Series) -> np.ndarray:
    return x.rank(pct=True).to_numpy()


def design(Y: str, s: pd.DataFrame, scores: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """기준 설계 + 점수들. 반환 = (X, 각 점수의 «관심 계수» 열 번호)."""
    if Y == "Y1":
        base = np.c_[G.dummies(s["hour"].to_numpy(), 24), np.log(s[["rv1h", "rv24h", "rv7d"]].to_numpy()), s["ldvol"].to_numpy()]
        add = scores
    elif Y == "Y2":
        rp = s["rp_z"].to_numpy(); base = np.c_[np.ones(len(s)), rp]
        add = [c for p in scores for c in (p, rp * p)]
    else:
        base = np.ones((len(s), 1)); add = scores
    X = np.c_[base, *add] if add else base
    k0 = base.shape[1]
    idx = [k0 + i for i in range(len(scores))] if Y != "Y2" else [k0 + 2 * i + 1 for i in range(len(scores))]
    return X, idx


YCOL = {"Y1": "ly4", "Y2": "rn_z", "Y3": "f_rg1_4h"}
NEED = {"Y1": ["rv1h", "rv24h", "rv7d", "ldvol"], "Y2": ["rp_z"], "Y3": []}     # 기준 설계 열 — 공통 표본에 포함


def compare(df: pd.DataFrame, names: dict, Y: str, blk_col="day", pairs=None, shuffle=False) -> dict:
    """names = {표시명: (열, 부호)}. 공통 표본 → 점수 순위 → 기준·단독·쌍 모형을 한 부트스트랩으로."""
    cols = [c for c, _ in names.values()]
    s = df[np.isfinite(df[cols + [YCOL[Y]] + NEED[Y]]).all(1)].reset_index(drop=True)
    sc = {nm: rank01(sg * s[c]) for nm, (c, sg) in names.items()}
    if shuffle:
        rng = np.random.default_rng(SEED)
        sc = {nm: rng.permutation(v) for nm, v in sc.items()}
    pairs = list(itertools.combinations(names, 2)) if pairs is None else pairs
    Xs, idx = {"base": design(Y, s, [])[0]}, {}
    for nm in names:
        Xs[nm], idx[nm] = design(Y, s, [sc[nm]])
    for a, b in pairs:
        Xs[f"{a}+{b}"], idx[f"{a}+{b}"] = design(Y, s, [sc[a], sc[b]])
    fits, D = fit_many(Xs, s[YCOL[Y]].to_numpy(), s[blk_col].to_numpy())
    sign, E = CRITERIA[Y]["theory_sign"], CRITERIA[Y]["E"]
    jd = lambda f, j: G.A.judge(float(f[0][j]), *map(float, np.percentile(f[1][:, j], [2.5, 97.5])), D, sign, E)   # noqa: E731
    base = fits["base"]
    r = {"n": len(s), "n_blocks": D, "r2_base": round(base[2], 5), "single": {}, "both": {}, "dR2_diff": {},
         "score_spearman": {f"{a}~{b}": round(float(pd.Series(sc[a]).corr(pd.Series(sc[b]), method="spearman")), 3) for a, b in pairs}}
    for nm in names:
        f = fits[nm]
        r["single"][nm] = {**jd(f, idx[nm][0]), "dR2": ci(f[2] - base[2], f[3] - base[3])}
    for a, b in pairs:
        f = fits[f"{a}+{b}"]
        r["both"][f"{a}+{b}"] = {a: jd(f, idx[f"{a}+{b}"][0]), b: jd(f, idx[f"{a}+{b}"][1]), "dR2_joint": ci(f[2] - base[2], f[3] - base[3]),
                                 f"unique_{a}": ci(f[2] - fits[b][2], f[3] - fits[b][3]), f"unique_{b}": ci(f[2] - fits[a][2], f[3] - fits[a][3])}
        d = ci(fits[a][2] - fits[b][2], fits[a][3] - fits[b][3])
        r["dR2_diff"][f"{a}-{b}"] = {"est_ci": d, "verdict": (f"{a} 더 낫다" if d[1] > 0 else f"{b} 더 낫다" if d[2] < 0 else "구분 불가")}
    return r


def null_check(df: pd.DataFrame, names: dict, Y: str, n=500) -> dict:
    """사후 진단(판정 아님): 단독 계수의 귀무분포 — 행 순열 vs 원형 이동(±7일 밖, 점수 자기상관 보존). p = |귀무| ≥ |관측| 비율."""
    cols = [c for c, _ in names.values()]
    s = df[np.isfinite(df[cols + [YCOL[Y]] + NEED[Y]]).all(1)].reset_index(drop=True); y = s[YCOL[Y]].to_numpy(); N = len(s)
    rng = np.random.default_rng(SEED); out = {}
    coef = lambda p: (lambda X, j: np.linalg.lstsq(X, y, rcond=None)[0][j[0]])(*design(Y, s, [p]))   # noqa: E731
    for nm, (c, sg) in names.items():
        sc = rank01(sg * s[c]); obs = coef(sc)
        perm = np.array([coef(rng.permutation(sc)) for _ in range(n)])
        circ = np.array([coef(np.roll(sc, rng.integers(168, N - 168))) for _ in range(n)])
        out[nm] = {"obs": round(float(obs), 4), "acf_24h": round(float(pd.Series(sc).autocorr(24)), 3),
                   "perm_p": round(float((np.abs(perm) >= abs(obs)).mean()), 3), "perm_95": [round(float(v), 4) for v in np.percentile(perm, [2.5, 97.5])],
                   "circ_p": round(float((np.abs(circ) >= abs(obs)).mean()), 3), "circ_95": [round(float(v), 4) for v in np.percentile(circ, [2.5, 97.5])]}
    return out


# ───────────────────────── 데이터 ─────────────────────────
def build() -> pd.DataFrame:
    d = G.panel(np.log(G.load_5m()), 12)
    d = d[(d["t"] >= T0) & (d["t"] <= T_END)].reset_index(drop=True)
    d["ly4"] = np.log(d["rv_next4h"])
    H = pd.read_parquet(ROOT / "tmp/dealer_gex_reconstruct_2026_20261001/hourly_dealer_gex.parquet",
                        columns=["ts", "gex_week", "gex_all", "gex_front", "dvol"]).rename(columns={"ts": "t", "dvol": "ldvol"})
    B = pd.read_parquet(ROOT / "tmp/dealer_assumption_matrix_20261002/tblock_gex_hourly.parquet", columns=["ts", "gex_week"]) \
        .rename(columns={"ts": "t", "gex_week": "tblock_week"})
    C = pd.read_parquet(ROOT / "tmp/gamma_rehedge_footprint_2026_20261001/conv_gex_hourly.parquet", columns=["ts", "gex_week"]) \
        .rename(columns={"ts": "t_c", "gex_week": "c_week"}).sort_values("t_c")
    d = d.merge(H, on="t", how="left").merge(B, on="t", how="left")
    d = pd.merge_asof(d.sort_values("t"), C, left_on="t", right_on="t_c", direction="backward")
    d.loc[d["t"] < C0, "c_week"] = np.nan
    d.loc[(d["t"] - d["t_c"]) > 2 * 3_600_000, "c_week"] = np.nan          # 2h 넘게 묵은 스냅샷은 안 씀
    assert (d["t_c"].dropna() <= d.loc[d["t_c"].notna(), "t"]).all()
    for c in ("rg1_4h", "rg2_4h", "rg1_12h", "rg2_12h", "gex_week", "gex_all", "gex_front"):
        d[f"{c}_lag1d"] = d[c].shift(24)                                   # 1h 격자 연속(아래 assert)
    assert (np.diff(d["t"].to_numpy()) == 3_600_000).all()
    return d


IND = {"RG1_4h": ("rg1_4h", -1), "RG2_4h": ("rg2_4h", -1), "RG1_12h": ("rg1_12h", -1), "RG2_12h": ("rg2_12h", -1),
       "T_week": ("gex_week", +1), "T_all": ("gex_all", +1), "T_front": ("gex_front", +1),
       "T_block": ("tblock_week", +1), "C": ("c_week", +1)}
pick = lambda *k: {n: IND[n] for n in k}                                    # noqa: E731


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = build(); d4 = d[d["hour"] % 4 == 0].reset_index(drop=True); d["week"] = d["day"] // 7
    d47 = d[d["t"] >= C0].reset_index(drop=True)
    res = {"criteria": CRITERIA, "n_hours": len(d), "c_hours": int(d["c_week"].notna().sum())}
    Ys = ("Y1", "Y2", "Y3")
    run = lambda key, f: (res.setdefault(key, {}).update(f()), print(key, flush=True))   # noqa: E731
    # 주 쌍 + 병기
    for rg in ("RG1_4h", "RG2_4h"):
        run(f"{rg}_vs_T_week", lambda: {Y: {"1h": compare(d, pick(rg, "T_week"), Y), "4h_nonoverlap": compare(d4, pick(rg, "T_week"), Y),
                                            "week_block": compare(d, pick(rg, "T_week"), Y, blk_col="week")} for Y in Ys})
    # 대조군(주 쌍)
    lagd = {"RG1_4h": ("rg1_4h_lag1d", -1), "T_week": ("gex_week_lag1d", +1)}
    run("controls_primary", lambda: {Y: {"shuffled": compare(d, pick("RG1_4h", "T_week"), Y, shuffle=True),
                                         "lag1d": compare(d, lagd, Y)} for Y in Ys})
    run("null_check_posthoc", lambda: {Y: null_check(d, pick("RG1_4h", "RG2_4h", "T_week", "T_block"), Y) for Y in Ys})
    # 보조 격자: RG {1_4h,2_4h,1_12h,2_12h} × T {week,all,front,T_block}
    run("grid", lambda: {f"{rg}_vs_{t}": {Y: compare(d, pick(rg, t), Y) for Y in Ys}
                         for rg in ("RG1_4h", "RG2_4h", "RG1_12h", "RG2_12h") for t in ("T_week", "T_all", "T_front", "T_block")})
    # 47일 공통: RG vs T vs C (+T_block)
    run("d47", lambda: {Y: compare(d47, pick("RG1_4h", "T_week", "C", "T_block"), Y,
                                   pairs=[("RG1_4h", "T_week"), ("RG1_4h", "C"), ("T_week", "C"), ("RG1_4h", "T_block")]) for Y in Ys})
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    write_summary(res)


def write_summary(res):
    f = lambda j: f"**{j['verdict']}** {j['est']:+.4f} [{j['ci'][0]:+.4f}, {j['ci'][1]:+.4f}]"   # noqa: E731
    g = lambda c: f"{c[0]:+.5f} [{c[1]:+.5f}, {c[2]:+.5f}]"                                     # noqa: E731

    def table(title, blob, a, b):
        L = ["", f"## {title}", "", f"| Y | 표본 | 단독 {a} | 단독 {b} | 같이: {a} | 같이: {b} | ΔR² {a} | ΔR² {b} | ΔR²({a})−ΔR²({b}) | 판정 | ρ(점수) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
        for Y, v in blob.items():
            for sp, r in (v.items() if "single" not in v else [("—", v)]):
                bo = r["both"][f"{a}+{b}"]; dd = r["dR2_diff"][f"{a}-{b}"]
                L.append(f"| {Y} | {sp} n={r['n']}·{r['n_blocks']}블록 | {f(r['single'][a])} | {f(r['single'][b])} | {f(bo[a])} | {f(bo[b])} | "
                         f"{g(r['single'][a]['dR2'])} | {g(r['single'][b]['dR2'])} | {g(dd['est_ci'])} | **{dd['verdict']}** | {r['score_spearman'][f'{a}~{b}']} |")
        return L

    L = ["# 실현 감마 국면(RG) vs 체결 기반 딜러 GEX(T) (2026-10-02)", "",
         f"표본 {res['n_hours']}시간(2026-01-01~09-30 17:00) · C 있는 시간 {res['c_hours']}. 점수 = 롱감마 방향 백분위(−RG, +GEX). "
         "이론 부호: 세 목표 모두 음(−). Y1 E=0.05 · Y2 E=0.05 · Y3 E=0.02.", ""]
    for rg in ("RG1_4h", "RG2_4h"):
        L += table(f"주 쌍 — {rg} vs T_week (1h 격자·일 블록 = 주 판정 / 4h 비겹침 / 주 블록)", res[f"{rg}_vs_T_week"], rg, "T_week")
    L += table("대조군 — 셔플", {Y: v["shuffled"] for Y, v in res["controls_primary"].items()}, "RG1_4h", "T_week")
    L += table("대조군 — 1일 지연", {Y: v["lag1d"] for Y, v in res["controls_primary"].items()}, "RG1_4h", "T_week")
    L += ["", "## 사후 진단(판정 아님) — 단독 계수 귀무분포: 행 순열 vs 원형 이동(±7일 밖, 자기상관 보존) 각 500회", "",
          "| Y | 지표 | 관측 | 점수 24h 자기상관 | 순열 p · 95% | 원형 이동 p · 95% |", "|---|---|---|---|---|---|"]
    for Y, v in res["null_check_posthoc"].items():
        for nm, x in v.items():
            L.append(f"| {Y} | {nm} | {x['obs']:+.4f} | {x['acf_24h']} | {x['perm_p']} · {x['perm_95']} | {x['circ_p']} · {x['circ_95']} |")
    for k, v in res["grid"].items():
        a, b = k.split("_vs_")
        L += table(f"보조 격자 — {k}", v, a, b)
    L += ["", "## 47일 공통(2026-08-15~) — RG1_4h · T_week · C · T_block", "", "| Y | n | 단독 RG1_4h | 단독 T_week | 단독 C | 단독 T_block | ΔR² 차 RG−T | RG−C | T−C | RG−T_block |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for Y, r in res["d47"].items():
        L.append(f"| {Y} | {r['n']}·{r['n_blocks']}일 | " + " | ".join(f(r["single"][n]) for n in ("RG1_4h", "T_week", "C", "T_block")) + " | "
                 + " | ".join(f"{g(r['dR2_diff'][p]['est_ci'])} {r['dR2_diff'][p]['verdict']}" for p in ("RG1_4h-T_week", "RG1_4h-C", "T_week-C", "RG1_4h-T_block")) + " |")
    L += ["", "## 주의", "",
          "- Y3 는 RG 에 구조적으로 유리하다(같은 측정치의 과거 값). 판정은 Y1.",
          "- GEX 는 느리게 움직여 실효 표본이 시간 수보다 작다 — 주 블록 칸이 그 점검.",
          "- 백분위 순위는 표본 전체 분포로 매긴다(단조 변환이라 부호·순위 정보는 인과적, 크기 척도만 표본 의존).",
          "- RG 는 가격만 써서 2021~2025 로도 검증 가능하다(tmp/realized_gamma_regime_20261002) — 옵션은 2026 만이라 이 비교는 2026 만."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


def selftest():
    # ① 충분통계 R²·계수 = 직접 OLS, 그리고 같은 추출 공유(모형 간 차의 부트 분포가 결정적)
    rng = np.random.default_rng(1); n = 600
    X = np.c_[np.ones(n), rng.normal(size=(n, 2))]; y = X @ [0.1, 0.5, -0.3] + rng.normal(size=n); blk = np.arange(n) // 10
    fits, D = fit_many({"a": X, "b": X[:, :2]}, y, blk, nb=50)
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert D == 60 and np.allclose(fits["a"][0], b) and abs(fits["a"][2] - r2) < 1e-10
    assert (fits["a"][3] >= fits["b"][3] - 1e-12).all()                   # 같은 추출 → 내포 모형 R² 는 매 추출 ≤
    # ② 순위는 부호 반영: −RG 점수는 RG 와 역순
    assert (np.argsort(rank01(pd.Series([3.0, 1.0, 2.0]) * -1)) == [0, 2, 1]).all()
    G.selftest()                                                            # RG 시점 경계(재사용 정의)
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else main()
