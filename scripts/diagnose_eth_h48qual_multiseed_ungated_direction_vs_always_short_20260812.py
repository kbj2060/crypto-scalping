"""팀장(Model Architect) 리서치 문서(eth_h48qual_quality_head_replacement_research_20260812.md)의
후보 9 핵심 질문을 이 세션 표준 다중시드 방법론으로 정식 검증: quality_head 게이트 없이
direction_head 원본 픽(dir_action)만으로 거래하면 always-short를 이기는가?

오늘 이미 라이브 번들 1회 실행으로 확인했지만(VAL 거의 동률, OOS -18.3pp 격패) 그건 단일
실행이었다. 이 스크립트는 scripts/diagnose_eth_h48qual_short_only_vs_always_short_20260811.py의
h48orig(5시드)+h384(15시드) 재현판 저장 예측을 재사용해(재학습 없음) N>=5 다양시드로 정식
재현한다."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

VARIANTS = {
    "h384_v2": {
        "seeds": [260620, 481003, 26611, 903174, 155827, 44452, 51724, 179660, 240382, 375044,
                  378518, 692713, 711841, 750878, 821662],
        "tag": "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2_e40_r30000_s",
    },
    "h48orig": {
        "seeds": [260620, 481003, 26611, 903174, 155827],
        "tag": "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h48orig_20260811_r30000_s",
    },
}
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")


def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


train_all = _read(TRAIN_CSV)
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = _read(EVAL_CSV)

fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def forced_side_dec(dec, side_value):
    out = dec.copy()
    active = omega._active(dec)
    out.loc[active, "side"] = side_value
    out.loc[active, "action"] = omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT
    return out


def ungated_decisions(src: pd.DataFrame, prefix: str, oof: bool) -> pd.DataFrame:
    src2 = src.copy()
    src2[f"{prefix}_final_action"] = src2[f"{prefix}_dir_action"]
    return omega._to_fixed_decisions(src2, oof=oof)


rows = []
for variant, cfg in VARIANTS.items():
    for seed in cfg["seeds"]:
        d = RUN_ROOT / f"{cfg['tag']}{seed}"
        for split_name, frame, oof, fname, prefix in [
            ("VAL", val_raw, True, "validation_predictions_q050.csv", "omega1_regime3_expertdq_oof"),
            ("OOS", oos_raw, False, "oos_predictions_q050.csv", "omega1_regime3_expertdq"),
        ]:
            src_path = d / fname
            if not src_path.exists():
                print(f"[스킵] {variant} seed={seed} {split_name}: {src_path} 없음")
                continue
            src = pd.read_csv(src_path, parse_dates=["timestamp"])
            f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            assert len(f) == len(src_aligned), f"{variant} {seed} {split_name} length mismatch"

            dec_gated = omega._to_fixed_decisions(src_aligned, oof=oof)
            dec_ungated = ungated_decisions(src_aligned, prefix, oof)
            dec_short = forced_side_dec(dec_ungated, -1)
            dec_long = forced_side_dec(dec_ungated, 1)

            m_gated = omega._metrics(f, dec_gated, fee=fee, slip=slip, cost_mult=cost_mult)
            m_ungated = omega._metrics(f, dec_ungated, fee=fee, slip=slip, cost_mult=cost_mult)
            m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cost_mult)
            m_long = omega._metrics(f, dec_long, fee=fee, slip=slip, cost_mult=cost_mult)

            rows.append({
                "variant": variant, "seed": seed, "split": split_name,
                "gated_pnl": m_gated["pnl"], "gated_trades": m_gated["trades"],
                "ungated_pnl": m_ungated["pnl"], "ungated_trades": m_ungated["trades"], "ungated_wr": m_ungated["wr"],
                "always_short_pnl": m_short["pnl"], "always_short_trades": m_short["trades"],
                "always_long_pnl": m_long["pnl"],
                "ungated_beats_always_short": m_ungated["pnl"] > m_short["pnl"],
            })

df = pd.DataFrame(rows)
out_path = ROOT / "tmp/eth_h48qual_odyssey_regression_analysis_20260811/multiseed_ungated_vs_always_short.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

pd.set_option("display.width", 220)
for variant in VARIANTS:
    sub_v = df[df["variant"] == variant]
    if sub_v.empty:
        continue
    print(f"\n===================== {variant} (n={len(cfg['seeds'])}시드) =====================")
    for split in ["VAL", "OOS"]:
        sub = sub_v[sub_v["split"] == split]
        if sub.empty:
            continue
        n = len(sub)
        beat = int(sub["ungated_beats_always_short"].sum())
        # one-sided Wilcoxon signed-rank: ungated_pnl - always_short_pnl > 0 ?
        diff = (sub["ungated_pnl"] - sub["always_short_pnl"]).to_numpy()
        if n >= 5 and np.any(diff != 0):
            wstat, wp = stats.wilcoxon(diff, alternative="greater")
        else:
            wstat, wp = np.nan, np.nan
        print(f"\n--- {split} ({n}시드) ---")
        print(f"  gated(현재 라이브 방식):   pnl={sub['gated_pnl'].mean():+7.2f}±{sub['gated_pnl'].std():5.2f}  trades={sub['gated_trades'].mean():5.1f}")
        print(f"  ungated(direction_head 원본): pnl={sub['ungated_pnl'].mean():+7.2f}±{sub['ungated_pnl'].std():5.2f}  trades={sub['ungated_trades'].mean():5.1f}  wr={sub['ungated_wr'].mean()*100:5.1f}%")
        print(f"  always_short:              pnl={sub['always_short_pnl'].mean():+7.2f}±{sub['always_short_pnl'].std():5.2f}")
        print(f"  always_long:               pnl={sub['always_long_pnl'].mean():+7.2f}±{sub['always_long_pnl'].std():5.2f}")
        print(f"  >> ungated가 always_short 이긴 시드: {beat}/{n}   |   Wilcoxon one-sided p(ungated>always_short)={wp:.4f}")

print("\n=== 저장 ===")
print(out_path)
