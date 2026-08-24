#!/usr/bin/env python3
"""일리아스 라벨 퓨전(zigzag/h48qual/cusum) 저비용 feasibility 체크 -- 신규 학습 없음, TRAIN구간
(2025 전체)만 사용, OOS는 전혀 건드리지 않음.
`docs/experiments/ilias_eth_label_fusion_combined_model_research_20260821.md`의 실증 근거.

이미 저장된 18개 run(3라벨 x N=6 시드)의 train_predictions_qXXX.csv + validation_predictions_qXXX.csv
(2025 전체, oof)를 그대로 재사용한다 -- 재추론/재학습 없음.
`eth_ilias_anchored_train_period_ledger_20260821.py`의 검증된 _metrics_with_ledger 로직을 그대로
import(재구현 아님).

같은 seed 번호끼리(zigzag/h48qual/cusum 각각 독립 학습된 모델)를 하나의 "결합 트라이얼"로 짝지어
N=6 트라이얼을 구성한다 -- seed 번호 자체에 교차-라벨 의미는 없으나(각 라벨의 6개 무작위 시드 중
인덱스 매칭일 뿐), 6개의 독립적 결합 결과를 얻기 위한 절차적 장치로 쓴다.

두 가지 결합 규칙:
  (A) vote  -- 다수결 투표: 3개 라벨의 side(-1/0/+1) 중 과반(>=2)이 같은 방향이면 그 방향으로
      진입, 아니면 CASH. "신호 평균/배깅" 계열 앙상블에 해당.
  (B) consensus -- 합의필터: zigzag와 cusum(구조적으로 독립인 두 방향소스)이 같은 방향에 동의하고,
      h48qual도 같은 방향으로 active(quality 게이트 통과)일 때만 진입 -- zigzag/h48qual/cusum의
      실제 비대칭 구조(zigzag=방향, h48qual=zigzag방향+품질게이트, cusum=독립방향)를 반영한
      meta-labeling에 가까운 결합. "노이즈 트레이드 제거" 계열에 해당.

TP/SL/notional/leverage는 BASE_TEMPLATE 고정 상수(이 3-way 스크리닝 자체가 리스크 사이드카 없이
고정 템플릿을 씀, 라벨 무관)이므로 결합 시 그대로 재사용하되, active bar마다 세 라벨의 값이 실제로
동일한지 검증(assert)한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eth_ilias_anchored_train_period_ledger_20260821 as ledger_mod  # noqa: E402

omega = ledger_mod.omega
parent = ledger_mod.parent

SEEDS = ledger_mod.SEEDS
LABELS = ("zigzag", "h48qual", "cusum")
OUT_ROOT = ledger_mod.OUT_ROOT
MODEL_ID = ledger_mod.MODEL_ID
OUT_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/train_period_fusion_feasibility"


def _load_label_decisions(label: str, seed: int) -> pd.DataFrame:
    out_dir = OUT_ROOT / f"{MODEL_ID}_label5way_{label}_154feat_ilias_anchored_seed{seed}_20260821"
    report = json.loads((out_dir / "report.json").read_text())
    best = report["ranking_by_validation_pnl"][0]
    q_tag = f"q{int(round(float(best['quality_threshold']) * 100.0)):03d}"
    train_csv = out_dir / f"train_predictions_{q_tag}.csv"
    val_csv = out_dir / f"validation_predictions_{q_tag}.csv"
    preds = pd.concat(
        [pd.read_csv(train_csv, parse_dates=["timestamp"]), pd.read_csv(val_csv, parse_dates=["timestamp"])],
        ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)
    dec = parent._to_decisions(preds, oof=True)
    dec.insert(0, "timestamp", preds["timestamp"].to_numpy())
    return dec


def _combine(dec_z: pd.DataFrame, dec_h: pd.DataFrame, dec_c: pd.DataFrame, *, rule: str) -> pd.DataFrame:
    n = len(dec_z)
    assert len(dec_h) == n and len(dec_c) == n, (len(dec_z), len(dec_h), len(dec_c))
    assert (dec_z["timestamp"].to_numpy() == dec_h["timestamp"].to_numpy()).all()
    assert (dec_z["timestamp"].to_numpy() == dec_c["timestamp"].to_numpy()).all()

    side_z = dec_z["side"].to_numpy()
    side_h = dec_h["side"].to_numpy()
    side_c = dec_c["side"].to_numpy()

    if rule == "vote":
        long_votes = (side_z == 1).astype(int) + (side_h == 1).astype(int) + (side_c == 1).astype(int)
        short_votes = (side_z == -1).astype(int) + (side_h == -1).astype(int) + (side_c == -1).astype(int)
        side_combined = np.where(long_votes >= 2, 1, np.where(short_votes >= 2, -1, 0))
    elif rule == "consensus":
        z_c_agree_long = (side_z == 1) & (side_c == 1)
        z_c_agree_short = (side_z == -1) & (side_c == -1)
        side_combined = np.where(
            z_c_agree_long & (side_h == 1), 1, np.where(z_c_agree_short & (side_h == -1), -1, 0)
        )
    else:
        raise ValueError(rule)

    active = side_combined != 0
    # 세 라벨이 "동시에 개별적으로도" active인 bar에서만 등가 검증 -- 다수결/합의 조합상 한쪽이
    # CASH(비활성, TP/SL/notional=0)인 채로 combined-active에 포함되는 경우가 정상적으로 있으므로,
    # 그 경우까지 강제로 동일해야 한다고 요구하면 잘못된 실패가 난다(실측으로 확인: router_expert가
    # bar마다 세 라벨 전부 100% 일치 -> TP/SL/notional/leverage는 개별active bar에서는 항상 동일).
    all_individually_active = (side_z != 0) & (side_h != 0) & (side_c != 0)
    for col in ("take_profit", "stop_loss", "notional_exposure", "leverage"):
        vz = dec_z[col].to_numpy()[all_individually_active]
        vh = dec_h[col].to_numpy()[all_individually_active]
        vc = dec_c[col].to_numpy()[all_individually_active]
        if len(vz) and not (np.allclose(vz, vh) and np.allclose(vz, vc)):
            raise RuntimeError(f"{col} mismatch across labels on jointly-active bars -- fixed-template assumption broken")

    out = pd.DataFrame(
        {
            "timestamp": dec_z["timestamp"].to_numpy(),
            "action": np.where(side_combined == 1, 1, np.where(side_combined == -1, 2, 0)),
            "side": side_combined,
            "notional_exposure": np.where(active, dec_z["notional_exposure"].to_numpy(), 0.0),
            "leverage": np.where(active, dec_z["leverage"].to_numpy(), 1.0),
            "take_profit": np.where(active, dec_z["take_profit"].to_numpy(), 0.0),
            "stop_loss": np.where(active, dec_z["stop_loss"].to_numpy(), 0.0),
            "max_hold_bars": dec_z["max_hold_bars"].to_numpy(),
            "cooldown_bars": dec_z["cooldown_bars"].to_numpy(),
        }
    )
    return out


def main() -> None:
    # ⚠️ 실행 중 발견: 사전계산된 tmp/ilias_labellogic_recheck_20260821/train_period_trade_ledgers/
    # summary.csv는 2024 데이터가 학습에 반영되기 *이전*(train_predictions 78,605행=2025 1~9월 뿐)
    # 시점 산출물이었고, 이후 "2024 실제학습 반영" 단계가 같은 출력 디렉토리를 덮어써
    # train_predictions_qXXX.csv가 지금은 210,481행(2024-01~2025-12, 2024+2025 전체)이다 -- 즉 그
    # summary.csv는 지금 디스크에 있는 예측 파일들과 더 이상 재현되지 않는 stale 산출물이다(직접
    # 재실행해 확인: `data/splits/year_oos/training_features_2025.csv` 단독으로 merge하면 raw join
    # 이 210,481행 중 105,101행만 남기고 나머지를 버려 원본 스크립트의 자체 정합성 검사
    # `len(frame)==len(preds)`에 걸린다). 그래서 이 스크립트는 2025 단독이 아니라 예측 파일이 실제로
    # 커버하는 2024-01~2025-12 전체를 raw 기준으로 삼고, 개별 라벨 벤치마크도 stale summary.csv를
    # 재사용하지 않고 이 스크립트 안에서 동일 코드로 새로 계산한다(사과 대 사과 비교 보장).
    full_raw = pd.concat(
        [
            pd.read_csv(
                ROOT / f"data/splits/year_oos/training_features_{year}.csv",
                usecols=["timestamp", "open", "high", "low", "close"],
                parse_dates=["timestamp"],
            )
            for year in (2024, 2025)
        ],
        ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)
    px0 = float(full_raw["close"].iloc[0])
    px1 = float(full_raw["close"].iloc[-1])
    always_long = (px1 / px0 - 1.0) * 100.0
    print(f"2024-2025 always-long(raw close, no fee/leverage): {always_long:.2f}%  [{px0:.2f} -> {px1:.2f}]", flush=True)
    print(f"2024-2025 always-long(raw close x2 leverage, no fee): {always_long * 2:.2f}%", flush=True)

    def _run_one(dec_frame: pd.DataFrame, tag: str, seed: int) -> dict:
        frame = (
            full_raw.merge(dec_frame[["timestamp"]], on="timestamp", how="inner")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if len(frame) != len(dec_frame):
            raise RuntimeError(f"{tag} seed={seed}: raw join lost rows ({len(frame)} vs {len(dec_frame)})")
        dec_aligned = dec_frame.set_index("timestamp").reindex(frame["timestamp"]).reset_index()
        agg, ledger = ledger_mod._metrics_with_ledger(
            frame, dec_aligned, fee=ledger_mod.FEE, slip=ledger_mod.SLIP, cost_mult=ledger_mod.COST_MULT
        )
        n_entries = max(agg["long_entries"] + agg["short_entries"], 1)
        long_frac = agg["long_entries"] / n_entries
        ledger.to_csv(OUT_DIR / f"{tag}_seed{seed}_2024_2025_trade_ledger.csv", index=False)
        return {
            "tag": tag,
            "seed": seed,
            "trades": agg["trades"],
            "wr": agg["wr"],
            "pnl": agg["pnl"],
            "mdd": agg["mdd"],
            "long_entries": agg["long_entries"],
            "short_entries": agg["short_entries"],
            "long_frac": long_frac,
        }

    rows = []
    for seed in SEEDS:
        dec_by_label = {label: _load_label_decisions(label, seed) for label in LABELS}
        common_ts = pd.DatetimeIndex(
            sorted(
                set(dec_by_label["zigzag"]["timestamp"])
                & set(dec_by_label["h48qual"]["timestamp"])
                & set(dec_by_label["cusum"]["timestamp"])
            ),
            name="timestamp",
        )
        coverage = len(common_ts) / max(len(dec_by_label["zigzag"]), 1)
        aligned = {
            label: dec_by_label[label].set_index("timestamp").reindex(common_ts).reset_index()
            for label in LABELS
        }

        # 개별 라벨(결합 없음) -- stale summary.csv를 재사용하지 않고 이 스크립트로 새로 계산.
        for label in LABELS:
            r = _run_one(aligned[label], f"solo_{label}", seed)
            r["coverage_frac"] = coverage
            rows.append(r)
            print(
                f"[solo_{label} seed={seed}] trades={r['trades']} pnl={r['pnl']:.2f} mdd={r['mdd']:.2f} "
                f"long_frac={r['long_frac']:.3f} (L={r['long_entries']}/S={r['short_entries']})",
                flush=True,
            )

        for rule in ("vote", "consensus"):
            dec_combo = _combine(aligned["zigzag"], aligned["h48qual"], aligned["cusum"], rule=rule)
            r = _run_one(dec_combo, rule, seed)
            r["coverage_frac"] = coverage
            rows.append(r)
            print(
                f"[{rule} seed={seed}] coverage={coverage:.3f} trades={r['trades']} pnl={r['pnl']:.2f} "
                f"mdd={r['mdd']:.2f} long_frac={r['long_frac']:.3f} (L={r['long_entries']}/S={r['short_entries']})",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nsaved {OUT_DIR / 'summary.csv'}", flush=True)
    for tag in [f"solo_{l}" for l in LABELS] + ["vote", "consensus"]:
        sub = df[df["tag"] == tag]
        corr = sub["long_frac"].corr(sub["pnl"]) if len(sub) > 2 else float("nan")
        print(
            f"\n[{tag}] N={len(sub)} long_frac<->pnl corr={corr:.3f} pnl_mean={sub['pnl'].mean():.2f} "
            f"pnl_std={sub['pnl'].std():.2f} pos_sign={int((sub['pnl'] > 0).sum())}/{len(sub)} "
            f"long_frac_range=[{sub['long_frac'].min():.3f},{sub['long_frac'].max():.3f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
