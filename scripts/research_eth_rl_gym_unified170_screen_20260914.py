"""**170컬럼 선별** — 09-12 절제에 쓴 `rl_training_*_unified.csv` 를 gym 라벨로 다시 잰다 (2026-09-14).

사용자 *"150개 피쳐들이 있었는데 그 중 골라서 피쳐 추가해보는건 어때?"*

## 🔴이 자료의 한계를 먼저 고정한다 (실행 전)
- 커버리지 **2024-01-01 ~ 2026-02-28**. 내 창 기준: TRAIN 전부 · VAL 전부 · **OOS 2/3** · **TEST 0%**.
  ⇒ §12 의 3창 규칙(OOS·TEST 둘 다)을 **쓸 수 없다**. 이 실행은 «선별」이고 판정이 아니다.
- CSV 생성 2026-05-08 = M7 제거 커밋(4c46d20, 2026-08-10) **이전의 화석**. `m7_*`(47) · `ai_*`(16) ·
  `pred_*`/`conf_*`/`patchtst_*`/`tide_*`/`timesnet_*`/`dlinear_*` 는 **지금 코드로 재생산 불가**다.
  ⇒ 이들은 `parity_risk=True` 로 표시하고, 통과해도 **라이브에 못 쓴다**(전 구간 재생성 불가).
- 09-12 절제는 죽은 열을 기록했다(`vol_rank`·`sl_offset_norm` 고유값 1 · `m7_vae_error` OOS 2값 ·
  `garch_vol_z` OOS 상수 · 창별 가용성 다른 30열). **세 창 모두에서 변동하는 열만** 후보로 둔다.

## 선별 규칙 (실행 전 고정 — 3창을 못 쓰므로 §12 보다 **엄격하게**)
군이 «전 구간 재생성 후보」가 되려면 둘 다:
  (a) 그 군에 |TRAIN IC| ≥ 0.05 이고 **VAL·OOS 둘 다 부호 유지**인 열이 있다.
  (b) 43열 기준 대비 쌍둥이 표본외 IC 증분이 **VAL·OOS 둘 다** ≥ +0.01.
⚠️둘 다 요구하는 이유: 평가창이 둘뿐이고 그중 하나가 잘려 있어 우연 통과 확률이 §12 보다 높다.
⚠️이 실행은 **170열 × 다중검정**이다. 통과 군의 다음 단계는 «믿는다」가 아니라 «원자료에서 전 구간
재생성해 TEST 까지 포함해 다시 잰다」이다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

CSVS = [ROOT / f"data/rl_training_{y}_unified.csv" for y in (2024, 2025, 2026)]
# 재생산 불가(화석) 접두어 — 통과해도 라이브 불가라 따로 표시한다
PARITY_RISK = ("m7_", "ai_", "pred_", "conf_", "patchtst_", "tide_", "timesnet_", "dlinear_")
# 내 43D 상태에 이미 있거나 klines 원본이라 새 정보가 아닌 열
DROP = {"timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "close_btc", "volume_btc", "quote_volume_btc",
        "hour_sin", "hour_cos", "minute_sin", "minute_cos"}


def family_of(c: str) -> str:
    if c.startswith("m7_"):
        return "m7(화석)"
    if c.startswith("ai_"):
        return "ai앵커(화석)"
    if c.startswith(("pred_", "conf_", "patchtst_", "tide_", "timesnet_", "dlinear_")):
        return "신경망출력(화석)"
    if c.startswith(("sig_",)):
        return "sig규칙"
    if c.startswith(("cvp_", "fvg")) or c in ("turtle_signal", "breakout_strength", "volume_profile_signal"):
        return "구조"
    if c.startswith(("funding", "ou_", "mta_")) or c in ("last_funding_rate", "long_squeeze_risk"):
        return "펀딩"
    if c.startswith(("sum_", "count_")) or c in ("oi_change_rate",):
        return "OI·포지셔닝"
    if c in ("whale_retail_ratio", "smart_money_flow", "net_taker_ratio", "taker_acceleration",
             "trade_intensity", "big_trade_ratio", "ofi_acceleration", "ofti", "whale_conviction",
             "amihud_illiquidity_z", "svps", "kel"):
        return "주문흐름·고래"
    if c.startswith(("regime_", "chop_")) or c in ("mtf_trend_1h", "mtf_trend_4h", "hurst_48", "kalman_velocity"):
        return "레짐·추세"
    if c.startswith(("jump_", "evt_")) or c in ("realized_skewness", "squeeze_power"):
        return "점프·꼬리"
    if c in ("log_return", "rsi", "macd_hist", "hma_slope", "dual_momentum", "mean_reversion_z",
             "btc_corr_60", "eth_btc_ratio_change"):
        return "모멘텀·교차"
    return "변동성·기타"


def load_unified() -> pd.DataFrame:
    fr = [pd.read_csv(p, parse_dates=["timestamp"]) for p in CSVS]
    u = pd.concat(fr, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    return u.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-parity-risk", action="store_true", help="화석(재생산 불가) 열을 아예 뺀다")
    a = ap.parse_args()
    d, sm, win, S43, cols43, _ = P.prepare(G.DEFAULT_FAMILIES)
    u = load_unified()
    print(f"unified {len(u):,}행 {u.timestamp.min()} ~ {u.timestamp.max()}", flush=True)

    # 봉 t 의 행을 봉 t 에 붙인다(같은 timestamp = 같은 5분봉 시작). 라벨은 봉 t 이후만 본다.
    num = [c for c in u.columns if c not in DROP and pd.api.types.is_numeric_dtype(u[c])]
    if a.no_parity_risk:
        num = [c for c in num if not c.startswith(PARITY_RISK)]
    m = d[["timestamp"]].merge(u[["timestamp"] + num], on="timestamp", how="left")
    assert len(m) == len(d) and (m.timestamp.to_numpy() == d.timestamp.to_numpy()).all()
    X = m.drop(columns=["timestamp"])

    z_ = np.load(P.OUT / "direction_labels.npz", allow_pickle=True)
    lab = {k: (z_[f"{k}_idx"], z_[f"{k}_y"]) for k in z_["names"]}
    EV = ("VAL", "OOS")                       # 🔴TEST 는 자료가 없다
    # 창별 커버리지(그 창 라벨 중 unified 가 값을 가진 비율)를 먼저 찍는다 — 판정 해석의 전제다
    cover_w = {}
    for w in ("TRAIN",) + EV + ("TEST",):
        idx = lab[w][0]
        cover_w[w] = float(np.isfinite(X.iloc[idx, 0].to_numpy(float)).mean()) if len(idx) else 0.0
    print("창별 unified 커버리지: " + " · ".join(f"{w} {100*v:.0f}%" for w, v in cover_w.items()), flush=True)
    assert cover_w["TEST"] < 0.05, "TEST 가 덮인다면 이 스크립트의 전제(선별 전용)를 다시 쓴다"

    # 🔴세 창 모두에서 살아 있고 변동하는 열만 (09-12 죽은 열 교훈)
    alive = []
    for c in num:
        v = X[c].to_numpy(float)
        if all(pd.Series(v[lab[w][0]]).nunique(dropna=True) > 1 for w in ("TRAIN",) + EV):
            alive.append(c)
    print(f"후보 {len(num)}열 → 세 창 모두 변동 {len(alive)}열", flush=True)

    normx = G.fit_normalizer(X.assign(timestamp=m.timestamp), alive, *win["TRAIN"])
    SX = G.build_state(X, normx)
    pos = {c: i for i, c in enumerate(alive)}

    rows = []
    for c in alive:
        j = pos[c]; r = {"feature": c, "family": family_of(c), "parity_risk": c.startswith(PARITY_RISK)}
        for w in ("TRAIN",) + EV:
            idx, y = lab[w]; x = SX[idx, j]
            r[w] = float(spearmanr(x, y).statistic) if np.std(x) > 0 else np.nan
        r["sign_keep"] = int(sum(np.sign(r[w]) == np.sign(r["TRAIN"]) for w in EV
                                 if np.isfinite(r[w]) and np.isfinite(r["TRAIN"])))
        rows.append(r)
    df = pd.DataFrame(rows)

    from sklearn.ensemble import HistGradientBoostingRegressor
    ti, ty = lab["TRAIN"]
    def twin_ic(Xtr, Xev: dict) -> dict:
        mm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=200, l2_regularization=1.0,
                                           random_state=P.SEEDS[0])
        mm.fit(Xtr, ty)
        return {w: float(spearmanr(mm.predict(Xv), lab[w][1]).statistic) for w, Xv in Xev.items()}
    base = twin_ic(S43[ti], {w: S43[lab[w][0]] for w in EV})
    print(f"기준 43열 쌍둥이 표본외 IC: " + " · ".join(f"{w} {base[w]:+.4f}" for w in EV), flush=True)

    fams = sorted(df.family.unique())
    verdict = {}
    print("\n군별 (열 · 화석? · max|TRAIN IC| · 부호유지 2/2 열수 · 쌍둥이 증분 VAL/OOS)")
    for f in fams:
        sub = df[df.family == f]
        js = [pos[c] for c in sub.feature]
        inc = twin_ic(np.hstack([S43[ti], SX[ti][:, js]]),
                      {w: np.hstack([S43[lab[w][0]], SX[lab[w][0]][:, js]]) for w in EV})
        inc = {w: inc[w] - base[w] for w in EV}
        rule_a = bool(((sub["TRAIN"].abs() >= 0.05) & (sub.sign_keep == 2)).any())
        rule_b = bool(all(inc[w] >= 0.01 for w in EV))
        verdict[f] = {"n": len(sub), "parity_risk": bool(sub.parity_risk.any()),
                      "max_abs_train_ic": float(sub["TRAIN"].abs().max()),
                      "best": str(sub.loc[sub["TRAIN"].abs().idxmax()].feature),
                      "sign_keep2": int((sub.sign_keep == 2).sum()), "twin_inc": inc,
                      "rule_a": rule_a, "rule_b": rule_b, "select": rule_a and rule_b}
        v = verdict[f]
        print(f"  {f:<16}{v['n']:>3}열 {'화석' if v['parity_risk'] else '   '} "
              f"max|IC| {v['max_abs_train_ic']:.4f}({v['best']}) · 2/2 {v['sign_keep2']:>2} · "
              f"증분 {inc['VAL']:+.4f}/{inc['OOS']:+.4f} · {'⭐선택' if v['select'] else '탈락'}", flush=True)
    allj = list(range(SX.shape[1]))
    inc_all = twin_ic(np.hstack([S43[ti], SX[ti]]), {w: np.hstack([S43[lab[w][0]], SX[lab[w][0]]]) for w in EV})
    inc_all = {w: inc_all[w] - base[w] for w in EV}
    print(f"  {'전체':<16}{len(alive):>3}열      증분 {inc_all['VAL']:+.4f}/{inc_all['OOS']:+.4f}")

    print("\n|TRAIN IC| 상위 15")
    top = df.sort_values("TRAIN", key=lambda s: s.abs(), ascending=False).head(15)
    for _, r in top.iterrows():
        print(f"  {r.feature:<26}{r.family:<16}{r.TRAIN:>+8.4f}{r.VAL:>+8.4f}{r.OOS:>+8.4f}  {r.sign_keep}/2"
              f"  {'화석' if r.parity_risk else ''}")
    sel = [f for f, v in verdict.items() if v["select"]]
    live_ok = [f for f in sel if not verdict[f]["parity_risk"]]
    print(f"\n선택 {sel if sel else '없음'} · 그중 재생성 가능 {live_ok if live_ok else '없음'}")
    rep = {"coverage_by_window": cover_w, "n_candidate": len(num), "n_alive": len(alive),
           "base_twin_ic": base, "families": verdict, "all_inc": inc_all,
           "selected": sel, "selected_reproducible": live_ok,
           "per_feature": df.to_dict("records"),
           "limitation": "unified CSV 는 2026-02-28 까지 — TEST 미포함, OOS 부분. 선별 전용."}
    p = P.OUT / "report_unified170_screen.json"
    json.dump(rep, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"저장 {p}")
    # 자체점검: 라벨 부호 대칭 · 커버리지 범위
    idx, y = lab["VAL"]
    a0 = spearmanr(SX[idx, 0], y).statistic; a1 = spearmanr(SX[idx, 0], -y).statistic
    assert abs(a0 + a1) < 1e-12, "IC 계산이 라벨 부호에 비대칭"
    assert 0.0 <= cover_w["VAL"] <= 1.0 and cover_w["VAL"] > 0.9, f"VAL 커버리지 이상 {cover_w['VAL']}"
    print("확인: IC 부호 대칭 · VAL 커버리지 정상")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
