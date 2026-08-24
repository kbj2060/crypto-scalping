#!/usr/bin/env python3
"""정식 스태킹 메타모델(로지스틱회귀) -- zigzag/cusum을 독립 방향 피처, h48qual의 품질점수를
별도 축으로 취급해 LEARN한 결합 규칙. `docs/experiments/
ilias_eth_label_fusion_combined_model_research_20260821.md` §5 추천사항(옵션1, h48qual을
3번째 투표로 세지 않는 버전)의 실제 구현+테스트 -- 사용자 승인 후 진행.

방법론(walk-forward causal, OOS 미접촉):
  - 피처: zz_signal=dir_p_long-dir_p_short, cs_signal=동일(cusum), h4_quality=h48qual의
    quality_for_action(품질축, 방향투표 아님).
  - 타겟: 순방향 48bar(4h, h48_conservative와 동일 horizon 관례) 수익률 부호.
  - 학습: 2024만. 평가: 2025만(학습에 전혀 안 쓰인 held-out) -- 여전히 기존에 승인된 TRAIN+VAL
    샌드박스 내부이고, OOS(2026)는 전혀 접촉하지 않는다.
  - 공정비교를 위해 solo/vote/consensus도 이 스크립트 안에서 동일 2025-only 윈도우로 재계산한다
    (기존 `eth_ilias_label_fusion_train_period_feasibility_20260821.py`의 2024-2025 풀링 결과와는
    학습에 쓰인 데이터가 다르므로 직접 비교 불가 -- 별도 표로 취급).
  - `eth_ilias_label_fusion_train_period_feasibility_20260821.py`(검증됨)의 `_load_label_decisions`
    패턴/`_metrics_with_ledger` 로직을 그대로 재사용한다(재구현 아님).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import eth_ilias_label_fusion_train_period_feasibility_20260821 as fsn  # noqa: E402

omega = fsn.omega
parent = fsn.parent
_metrics_with_ledger = fsn.ledger_mod._metrics_with_ledger
SEEDS = fsn.SEEDS
FEE, SLIP = omega._load_fee_slip()
COST_MULT = 3.0

PREFIX = "omega1_regime3_expertdq_oof_"
FORWARD_BARS = 48
OUT_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/stacking_meta_model_2025holdout"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_label_preds(label: str, seed: int) -> pd.DataFrame:
    out_dir = fsn.OUT_ROOT / f"{fsn.MODEL_ID}_label5way_{label}_154feat_ilias_anchored_seed{seed}_20260821"
    report = json.loads((out_dir / "report.json").read_text())
    best = report["ranking_by_validation_pnl"][0]
    q_tag = f"q{int(round(float(best['quality_threshold']) * 100.0)):03d}"
    preds = pd.concat(
        [pd.read_csv(out_dir / f"train_predictions_{q_tag}.csv", parse_dates=["timestamp"]),
         pd.read_csv(out_dir / f"validation_predictions_{q_tag}.csv", parse_dates=["timestamp"])],
        ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)
    return preds


def _combine_from_side(dec_template: pd.DataFrame, side_combined: np.ndarray) -> pd.DataFrame:
    active = side_combined != 0
    return pd.DataFrame({
        "timestamp": dec_template["timestamp"].to_numpy(),
        "action": np.where(side_combined == 1, 1, np.where(side_combined == -1, 2, 0)),
        "side": side_combined,
        "notional_exposure": np.where(active, dec_template["notional_exposure"].to_numpy(), 0.0),
        "leverage": np.where(active, dec_template["leverage"].to_numpy(), 1.0),
        "take_profit": np.where(active, dec_template["take_profit"].to_numpy(), 0.0),
        "stop_loss": np.where(active, dec_template["stop_loss"].to_numpy(), 0.0),
        "max_hold_bars": dec_template["max_hold_bars"].to_numpy(),
        "cooldown_bars": dec_template["cooldown_bars"].to_numpy(),
    })


def main() -> None:
    full_raw = pd.concat(
        [pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{y}.csv",
                      usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
         for y in (2024, 2025)], ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)
    full_raw["fwd_ret"] = full_raw["close"].shift(-FORWARD_BARS) / full_raw["close"] - 1.0
    full_raw["y_up"] = (full_raw["fwd_ret"] > 0).astype(int)

    raw_2025 = full_raw[full_raw["timestamp"].dt.year == 2025]
    bench = (float(raw_2025["close"].iloc[-1]) / float(raw_2025["close"].iloc[0]) - 1.0) * 100.0
    print(f"2025-only always-long benchmark: {bench:+.2f}% (no leverage), {bench * 2:+.2f}% (2x)", flush=True)

    summary_rows: list[dict] = []
    coef_rows: list[dict] = []

    for seed in SEEDS:
        preds = {lb: _load_label_preds(lb, seed) for lb in ("zigzag", "cusum", "h48qual")}
        n0 = len(preds["zigzag"])
        for lb in preds:
            assert len(preds[lb]) == n0, (lb, len(preds[lb]), n0)
            assert (preds[lb]["timestamp"].to_numpy() == preds["zigzag"]["timestamp"].to_numpy()).all(), lb

        decs = {lb: parent._to_decisions(preds[lb], oof=True) for lb in preds}
        for lb in decs:
            decs[lb].insert(0, "timestamp", preds[lb]["timestamp"].to_numpy())

        feat = pd.DataFrame({
            "timestamp": preds["zigzag"]["timestamp"].to_numpy(),
            "zz_signal": (preds["zigzag"][f"{PREFIX}dir_p_long"] - preds["zigzag"][f"{PREFIX}dir_p_short"]).to_numpy(),
            "cs_signal": (preds["cusum"][f"{PREFIX}dir_p_long"] - preds["cusum"][f"{PREFIX}dir_p_short"]).to_numpy(),
            "h4_quality": preds["h48qual"][f"{PREFIX}quality_for_action"].to_numpy(),
        })
        feat = feat.merge(full_raw[["timestamp", "y_up", "fwd_ret"]], on="timestamp", how="inner")
        feat = feat.dropna(subset=["fwd_ret"]).sort_values("timestamp").reset_index(drop=True)

        year = feat["timestamp"].dt.year
        fit_mask = (year == 2024).to_numpy()
        eval_mask = (year == 2025).to_numpy()
        X = feat[["zz_signal", "cs_signal", "h4_quality"]].to_numpy(dtype=np.float64)
        y = feat["y_up"].to_numpy(dtype=np.int64)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[fit_mask], y[fit_mask])
        p_up = clf.predict_proba(X[eval_mask])[:, 1]
        stack_side = np.where(p_up > 0.55, 1, np.where(p_up < 0.45, -1, 0)).astype(np.int64)
        coef_rows.append({
            "seed": seed, "coef_zigzag": clf.coef_[0][0], "coef_cusum": clf.coef_[0][1],
            "coef_h48qual_quality": clf.coef_[0][2], "intercept": clf.intercept_[0],
            "fit_n": int(fit_mask.sum()), "eval_n": int(eval_mask.sum()),
        })

        eval_ts = feat.loc[eval_mask, ["timestamp"]].reset_index(drop=True)
        eval_frame = full_raw.merge(eval_ts, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
        assert len(eval_frame) == len(eval_ts), (len(eval_frame), len(eval_ts))

        dec_by_label = {}
        for lb in decs:
            d = decs[lb].merge(eval_ts, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
            assert len(d) == len(eval_ts), (lb, len(d), len(eval_ts))
            dec_by_label[lb] = d

        side_z = dec_by_label["zigzag"]["side"].to_numpy()
        side_c = dec_by_label["cusum"]["side"].to_numpy()
        side_h = dec_by_label["h48qual"]["side"].to_numpy()

        long_votes = (side_z == 1).astype(int) + (side_c == 1).astype(int) + (side_h == 1).astype(int)
        short_votes = (side_z == -1).astype(int) + (side_c == -1).astype(int) + (side_h == -1).astype(int)
        vote_side = np.where(long_votes >= 2, 1, np.where(short_votes >= 2, -1, 0))

        zc_long = (side_z == 1) & (side_c == 1)
        zc_short = (side_z == -1) & (side_c == -1)
        consensus_side = np.where(zc_long & (side_h == 1), 1, np.where(zc_short & (side_h == -1), -1, 0))

        tags = {
            "solo_zigzag": side_z, "solo_h48qual": side_h, "solo_cusum": side_c,
            "vote": vote_side, "consensus": consensus_side, "stack": stack_side,
        }
        for tag, side_arr in tags.items():
            dec_combined = _combine_from_side(dec_by_label["zigzag"], side_arr)
            agg, ledger = _metrics_with_ledger(eval_frame, dec_combined, fee=FEE, slip=SLIP, cost_mult=COST_MULT)
            ledger.to_csv(OUT_DIR / f"{tag}_seed{seed}_2025holdout_trade_ledger.csv", index=False)
            lf = agg["long_entries"] / max(agg["long_entries"] + agg["short_entries"], 1)
            summary_rows.append({
                "tag": tag, "seed": seed, "trades": agg["trades"], "pnl": agg["pnl"], "mdd": agg["mdd"],
                "long_entries": agg["long_entries"], "short_entries": agg["short_entries"], "long_frac": lf,
            })
            print(f"[{tag:12s} seed={seed}] pnl={agg['pnl']:7.2f}% trades={agg['trades']:3d} long_frac={lf:.3f}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(OUT_DIR / "stack_coefficients.csv", index=False)

    print(f"\n=== 2025-holdout aggregate (N=6 seeds), always-long bench = {bench:+.2f}% ===")
    for tag in ["solo_zigzag", "solo_h48qual", "solo_cusum", "vote", "consensus", "stack"]:
        sub = summary_df[summary_df["tag"] == tag]
        corr = sub["long_frac"].corr(sub["pnl"])
        print(f"{tag:12s} mean={sub['pnl'].mean():7.2f} std={sub['pnl'].std():6.2f} "
              f"mean_long_frac={sub['long_frac'].mean():.3f} corr={corr:6.3f} "
              f"trades=[{sub['trades'].min()},{sub['trades'].max()}]")

    print("\nstack coefficients per seed (양수=LONG쪽 가중, 음수=SHORT쪽 가중):")
    print(coef_df.to_string(index=False))
    print(f"\nsaved {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
