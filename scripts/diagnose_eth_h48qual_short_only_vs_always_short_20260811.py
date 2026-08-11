"""사용자 지적: 하락장이라 always-short이 방향 스킬 없이도 이길 수 있다 -- "always-short이 모델을
이긴다"는 게 quality_head의 숏 선별 자체가 무가치하다는 뜻이 아니라, 모델이 고른 소수의 롱
트레이드(승률 6.7~31.7%, breakeven=40% 훨씬 밑)가 전체를 깎아먹고 있을 뿐일 수 있다. 실제로
`verify_eth_h48qual_always_short_baseline_*`의 always_short는 모델과 완전히 같은 active bar
집합(quality gate 통과 후 final_action!=CASH)에서 방향만 숏으로 강제한 것 -- 즉 모델이 롱으로
골랐던 bar까지 전부 숏으로 바꿔치기한 값이라, 롱이 나쁘면 그걸 숏으로 바꾸기만 해도 이겨버리는
구조다. 이 스크립트는 "모델이 실제로 숏을 고른 bar만" 골라서(model_short_only) 같은 active set
전체를 강제숏한 always_short와 pnl/거래수/승률을 직접 비교한다 -- 이게 "quality_head의 숏 선별이
그냥 강제숏보다 나은가"에 대한 진짜 답이다."""
from pathlib import Path
import sys
import numpy as np, pandas as pd

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


def side_only_dec(dec, side_value):
    """dec_model 그대로 두되, side!=side_value인 active bar는 notional_exposure=0으로 꺼서
    비활성화 -- 모델이 실제로 그 방향을 고른 bar만 남긴다(강제 변환 없음)."""
    out = dec.copy()
    active = omega._active(dec)
    keep = active & (pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64) == side_value)
    drop = active & ~keep
    out.loc[drop, "notional_exposure"] = 0.0
    return out


rows = []
for variant, cfg in VARIANTS.items():
    for seed in cfg["seeds"]:
        d = RUN_ROOT / f"{cfg['tag']}{seed}"
        for split_name, frame, oof, fname in [
            ("VAL", val_raw, True, "validation_predictions_q050.csv"),
            ("OOS", oos_raw, False, "oos_predictions_q050.csv"),
        ]:
            src_path = d / fname
            if not src_path.exists():
                print(f"[스킵] {variant} seed={seed} {split_name}: {src_path} 없음")
                continue
            src = pd.read_csv(src_path, parse_dates=["timestamp"])
            f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            assert len(f) == len(src_aligned), f"{variant} {seed} {split_name} length mismatch"

            dec_model = omega._to_fixed_decisions(src_aligned, oof=oof)
            dec_short_forced = forced_side_dec(dec_model, -1)   # always_short: 같은 active set, 방향만 숏 강제
            dec_short_only = side_only_dec(dec_model, -1)       # model_short_only: 모델이 실제로 숏 고른 bar만
            dec_long_only = side_only_dec(dec_model, 1)         # model_long_only: 모델이 실제로 롱 고른 bar만

            m_model = omega._metrics(f, dec_model, fee=fee, slip=slip, cost_mult=cost_mult)
            m_always_short = omega._metrics(f, dec_short_forced, fee=fee, slip=slip, cost_mult=cost_mult)
            m_short_only = omega._metrics(f, dec_short_only, fee=fee, slip=slip, cost_mult=cost_mult)
            m_long_only = omega._metrics(f, dec_long_only, fee=fee, slip=slip, cost_mult=cost_mult)

            rows.append({
                "variant": variant, "seed": seed, "split": split_name,
                "model_pnl": m_model["pnl"], "model_trades": m_model["trades"], "model_wr": m_model["wr"],
                "always_short_pnl": m_always_short["pnl"], "always_short_trades": m_always_short["trades"], "always_short_wr": m_always_short["wr"],
                "short_only_pnl": m_short_only["pnl"], "short_only_trades": m_short_only["trades"], "short_only_wr": m_short_only["wr"],
                "long_only_pnl": m_long_only["pnl"], "long_only_trades": m_long_only["trades"], "long_only_wr": m_long_only["wr"],
            })

df = pd.DataFrame(rows)
out_path = ROOT / "tmp/eth_h48qual_odyssey_regression_analysis_20260811/short_only_vs_always_short.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

pd.set_option("display.width", 220)
for variant in VARIANTS:
    sub_v = df[df["variant"] == variant]
    if sub_v.empty:
        continue
    print(f"\n===================== {variant} =====================")
    for split in ["VAL", "OOS"]:
        sub = sub_v[sub_v["split"] == split]
        if sub.empty:
            continue
        n = len(sub)
        print(f"\n--- {split} ({n}시드) ---")
        print(f"  model(전체, 롱+숏 실제선택):   pnl={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}  trades={sub['model_trades'].mean():5.1f}  wr={sub['model_wr'].mean()*100:5.1f}%")
        print(f"  always_short(같은 active set, 방향 강제): pnl={sub['always_short_pnl'].mean():+7.2f}±{sub['always_short_pnl'].std():5.2f}  trades={sub['always_short_trades'].mean():5.1f}  wr={sub['always_short_wr'].mean()*100:5.1f}%")
        print(f"  model_short_only(실제 숏 선택만):  pnl={sub['short_only_pnl'].mean():+7.2f}±{sub['short_only_pnl'].std():5.2f}  trades={sub['short_only_trades'].mean():5.1f}  wr={sub['short_only_wr'].mean()*100:5.1f}%")
        print(f"  model_long_only(실제 롱 선택만):   pnl={sub['long_only_pnl'].mean():+7.2f}±{sub['long_only_pnl'].std():5.2f}  trades={sub['long_only_trades'].mean():5.1f}  wr={sub['long_only_wr'].mean()*100:5.1f}%")
        beat_as = int((sub['short_only_pnl'] > sub['always_short_pnl']).sum())
        wr_beat_as = int((sub['short_only_wr'] > sub['always_short_wr']).sum())
        print(f"  >> short_only가 always_short(pnl) 이긴 시드: {beat_as}/{n}   |   short_only가 always_short(승률) 이긴 시드: {wr_beat_as}/{n}")

print("\n=== 저장 ===")
print(out_path)
