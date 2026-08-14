"""'최종 보스'(train_eval_eth_h48qual_final_boss_20260812.py) N개 시드 결과에 대해 거래
시뮬레이션(omega._metrics) 실행. 승격 기준은 절대 pnl(사용자 지시: always_short 대조 무시)
이지만, 맥락 참고용으로 always_short/always_long도 같이 계산해서 리포트에 남긴다."""
from pathlib import Path
import sys
import json
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

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


def build_dec(action):
    action = action.astype(np.int64)
    active = action != omega.ACTION_CASH
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    return pd.DataFrame({
        "action": action, "side": side,
        "notional_exposure": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "leverage": np.where(active, float(omega.BASE_TEMPLATE["leverage"]), 1.0),
        "position_fraction": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "take_profit": np.where(active, float(omega.BASE_TEMPLATE["take_profit"]), 0.0),
        "stop_loss": np.where(active, float(omega.BASE_TEMPLATE["stop_loss"]), 0.0),
        "max_hold_bars": np.where(active, int(omega.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
        "cooldown_bars": np.where(active, int(omega.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
    })


TP_PRICE_MOVE = float(omega.BASE_TEMPLATE["take_profit"]) / float(omega.BASE_TEMPLATE["notional"])
SL_PRICE_MOVE = float(omega.BASE_TEMPLATE["stop_loss"]) / float(omega.BASE_TEMPLATE["notional"])


def build_dec_dynamic(action, margin_fraction, leverage):
    """라이브 사이드카와 같은 원리 -- 행마다 다른 margin_fraction/leverage(사전에
    train_eval_eth_h48qual_final_boss_20260812.py의 percentile-rank 매핑으로 계산됨,
    같은 캡: leverage<=5.0/notional<=1.8)를 그대로 사용.

    레포의 Futures Risk Sizing Contract(CLAUDE.md): omega._metrics는 take_profit/stop_loss를
    notional-스케일 계정pnl 기준(unreal=raw_price_move*notional과 직접 비교)으로 취급한다.
    fixed 사이징판(notional=0.45 고정)은 BASE_TEMPLATE의 take_profit=0.026/stop_loss=0.014를
    그대로 썼는데, 이건 notional=0.45에서 암묵적으로 raw 가격변동 5.78%/3.11%를 뜻한다.
    notional이 행마다 바뀌는 동적 사이징에서 이 값을 고정으로 두면 notional이 클수록 훨씬
    작은 가격변동에도 TP/SL이 발동해버려(레버리지를 또 곱하는 이중계산과 같은 효과) barrier
    자체가 왜곡된다 -- 최초 구현에서 실제로 발생(거래수 59->245건, MDD -13%->-42%로 급증,
    "사이징만 바꾸는" 실험의 전제가 깨짐). take_profit/stop_loss를 매 행 notional에 다시
    곱해 raw 가격변동 목표(5.78%/3.11%)를 사이즈와 무관하게 고정시켜 수정."""
    action = action.astype(np.int64)
    active = action != omega.ACTION_CASH
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    notional = np.where(active, margin_fraction * leverage, 0.0)
    return pd.DataFrame({
        "action": action, "side": side,
        "notional_exposure": notional,
        "leverage": np.where(active, leverage, 1.0),
        "position_fraction": np.where(active, margin_fraction, 0.0),
        "take_profit": np.where(active, TP_PRICE_MOVE * notional, 0.0),
        "stop_loss": np.where(active, SL_PRICE_MOVE * notional, 0.0),
        "max_hold_bars": np.where(active, int(omega.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
        "cooldown_bars": np.where(active, int(omega.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
    })


def forced_side(dec, side_value):
    out = dec.copy()
    active = omega._active(dec)
    out.loc[active, "side"] = side_value
    out.loc[active, "action"] = omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT
    return out


def run(out_tag, seeds, sizing="fixed", base_dir="tmp/eth_h48qual_final_boss_20260812"):
    d = ROOT / base_dir / out_tag
    rows = []
    for seed in seeds:
        for split_name, frame, fname in [("VAL", val_raw, f"val_decisions_s{seed}.csv"), ("OOS", oos_raw, f"oos_decisions_s{seed}.csv")]:
            src = pd.read_csv(d / fname, parse_dates=["timestamp"])
            f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            assert len(f) == len(src_aligned)

            if sizing == "dynamic":
                dec_model = build_dec_dynamic(src_aligned["final_action"].to_numpy(),
                                               src_aligned["margin_fraction"].to_numpy(),
                                               src_aligned["leverage"].to_numpy())
            else:
                dec_model = build_dec(src_aligned["final_action"].to_numpy())
            dec_short = forced_side(dec_model, -1)
            dec_long = forced_side(dec_model, 1)

            for cm, tag in [(1.0, "cost1"), (2.0, "cost2"), (3.0, "cost3")]:
                m_model = omega._metrics(f, dec_model, fee=fee, slip=slip, cost_mult=cm)
                m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cm)
                m_long = omega._metrics(f, dec_long, fee=fee, slip=slip, cost_mult=cm)
                rows.append({
                    "seed": seed, "split": split_name, "cost": tag,
                    "model_pnl": m_model["pnl"], "model_mdd": m_model["mdd"], "model_trades": m_model["trades"], "model_wr": m_model["wr"],
                    "model_long": m_model["long_entries"], "model_short": m_model["short_entries"],
                    "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
                })
    df = pd.DataFrame(rows)
    out_path = d / f"pnl_comparison_{sizing}.csv"
    df.to_csv(out_path, index=False)

    pd.set_option("display.width", 220)
    for split in ["VAL", "OOS"]:
        print(f"\n===================== {split} =====================")
        for cost in ["cost1", "cost2", "cost3"]:
            sub = df[(df.split == split) & (df.cost == cost)]
            if sub.empty:
                continue
            n = len(sub)
            beat_short = int((sub["model_pnl"] > sub["always_short_pnl"]).sum())
            beat_long = int((sub["model_pnl"] > sub["always_long_pnl"]).sum())
            print(f"[{split}/{cost}] model={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}  "
                  f"mdd={sub['model_mdd'].mean():+6.2f}  "
                  f"trades={sub['model_trades'].mean():.1f}(L={sub['model_long'].mean():.1f}/S={sub['model_short'].mean():.1f})  "
                  f"wr={sub['model_wr'].mean()*100:.1f}%  "
                  f"always_short={sub['always_short_pnl'].mean():+7.2f}  always_long={sub['always_long_pnl'].mean():+7.2f}  "
                  f"승(short)={beat_short}/{n}  승(long)={beat_long}/{n}")
    print(f"\n=== 저장 === {out_path}")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--sizing", choices=["fixed", "dynamic"], default="fixed")
    ap.add_argument("--base-dir", default="tmp/eth_h48qual_final_boss_20260812")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    run(args.out_tag, seeds, sizing=args.sizing, base_dir=args.base_dir)
