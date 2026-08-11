"""3-1 진단(연구문서 항목 3-1, 계약 L4 사이징 후보): TabM k=8 앙상블 멤버간 불일치(direction_head)가
실현 결과와 순위상관이 있는가. quality_for_action 순위상관 진단과 동일 방법론(dir_action 기준
pre-gate 시뮬레이션, spearmanr(신호, trade_return), 시드별 rho가 1차 근거)이지만 신호가 다르다 --
quality_head의 자기 확률이 아니라 direction_head의 k=8 멤버가 서로 얼마나 동의하는지(Depeweg et al.
2018 상호정보량 분해의 epistemic 성분).

.mean(dim=1) 풀링 전 per-member 출력은 저장된 예측 CSV에 없어서(평균만 저장됨) 새로 학습한 v2b
번들(scripts/ops/run_h48qual_ensemble_disagreement_5seed_20260811.sh, 서버에서 실행, epochs=40/
patience/rows=30000, v2와 동일 시드 5개)로 직접 inference를 재실행한다."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811 as h384script  # noqa: E402

omega = h384script.omega
parent_script = h384script.parent_script  # train_eval_omega4_3head_parent72_loose_entry_quality_20260620
tabm = parent_script.parent               # train_eval_omega1_2_tabm_3head_20260603 (ThreeHeadTabM, CFG, _standardize_apply)
hard = parent_script.hard                 # train_omega1_regime3_expert_direction_head_volpca_20260602 (routing)

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

SEEDS = [260620, 481003, 26611, 903174, 155827]
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
TAG = "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2b_e40_r30000_s"
SPLIT_TS = pd.Timestamp("2025-10-01")
DEVICE = torch.device("cpu")  # inference-only, no GPU needed for a handful of forward passes

fee, slip = omega._load_fee_slip()
cost_mult = 3.0


@torch.no_grad()
def predict_members(payload: dict, frame: pd.DataFrame, base_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """payload: 번들의 models[expert] 딕셔너리 하나. 반환: (N,k,3) direction/quality softmax,
    .mean(dim=1) 풀링 전. _base_input()으로 POS_COLS(추론 시 0)를 base_cols 뒤에 붙여야 스케일러
    컬럼 계약(base_cols + POS_COLS)과 맞는다 -- parent._predict_payload 호출부(line 1078-1079)와 동일 패턴."""
    x = tabm._base_input(frame, base_cols)
    model = tabm.ThreeHeadTabM(int(payload["n_features"]), cfg=tabm.CFG).to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = tabm._standardize_apply(x, payload["scaler"])
    dir_chunks, qual_chunks = [], []
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start:start + 8192]).to(DEVICE)
        out = model(xb)
        dir_chunks.append(torch.softmax(out["direction"], dim=-1).cpu().numpy())
        qual_chunks.append(torch.softmax(out["quality"], dim=-1).cpu().numpy())
    return np.concatenate(dir_chunks, axis=0), np.concatenate(qual_chunks, axis=0)


def route_combine(per_expert: dict, route: np.ndarray) -> np.ndarray:
    n, k, c = per_expert[hard.EXPERT_NAMES[0]].shape
    out = np.zeros((n, k, c), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        out[mask] = per_expert[expert][mask]
    return out


def mi_decomposition(probs_nkc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depeweg et al. 2018: total = H[mean dist], aleatoric = mean(H[member dist]), epistemic = total - aleatoric."""
    mean_dist = probs_nkc.mean(axis=1)
    total = -(mean_dist * np.log(np.clip(mean_dist, 1e-12, 1.0))).sum(axis=1)
    member_h = -(probs_nkc * np.log(np.clip(probs_nkc, 1e-12, 1.0))).sum(axis=2)
    aleatoric = member_h.mean(axis=1)
    return total, aleatoric, total - aleatoric


def pre_gate_decisions(dir_action: np.ndarray, epistemic: np.ndarray) -> pd.DataFrame:
    active = dir_action != omega.ACTION_CASH
    side = np.where(dir_action == omega.ACTION_LONG, 1, np.where(dir_action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    return pd.DataFrame({
        "action": dir_action, "side": side,
        "notional_exposure": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "take_profit": np.where(active, float(omega.BASE_TEMPLATE["take_profit"]), 0.0),
        "stop_loss": np.where(active, float(omega.BASE_TEMPLATE["stop_loss"]), 0.0),
        "max_hold_bars": np.where(active, int(omega.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
        "cooldown_bars": np.where(active, int(omega.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
        "epistemic": epistemic,
    })


def trades_with_signal(frame: pd.DataFrame, dec: pd.DataFrame):
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = (dec["action"].to_numpy() != omega.ACTION_CASH) & (dec["side"].to_numpy() != 0) & (dec["notional_exposure"].to_numpy() > 0)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = 1.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    entry_signal = 0.0
    notional = take_profit = stop_loss = 0.0
    max_hold = cooldown = next_cooldown = 0
    records = []
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            hold = int(i) - int(entry_idx)
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "max_hold"
            if reason:
                filled, exit_px, exit_fee, _r = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                records.append({"entry_idx": entry_idx, "epistemic": entry_signal, "trade_return": (cash - entry_equity) / entry_equity})
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _r = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = px
        entry_equity = cash
        entry_idx = int(i)
        entry_signal = float(row["epistemic"])
        notional = float(omega.BASE_TEMPLATE["notional"])
        take_profit = float(omega.BASE_TEMPLATE["take_profit"])
        stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])
        max_hold = int(omega.BASE_TEMPLATE["max_hold"])
        next_cooldown = int(omega.BASE_TEMPLATE["cooldown"])
        cash -= cash * entry_fee * notional
    if pos != 0:
        fill_i = len(frame) - 1
        exit_px = omega._fill_price(arrays, fill_i, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        records.append({"entry_idx": entry_idx, "epistemic": entry_signal, "trade_return": (cash - entry_equity) / entry_equity})
    return records


def main():
    train, eval_df, _audit = omega._load_omega_frames()
    val_frame = train[train["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    oos_frame = eval_df.reset_index(drop=True)
    final12 = h384script.FINAL12

    per_seed_rho = {"VAL": [], "OOS": []}
    pooled = {"VAL": [], "OOS": []}
    sanity_rows = []

    for seed in SEEDS:
        d = RUN_ROOT / f"{TAG}{seed}"
        bundle_path = d / "true_3head_tabm_bundle.pt"
        if not bundle_path.exists():
            print(f"[스킵] seed={seed}: {bundle_path} 없음 (아직 서버 학습 안 끝났거나 pull 전)")
            continue
        payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
        assert payload["base_cols"] == final12, f"base_cols mismatch for seed {seed}"

        for split_name, frame in [("VAL", val_frame), ("OOS", oos_frame)]:
            route = hard._route_id(frame)
            per_expert_dir, per_expert_qual = {}, {}
            for expert in hard.EXPERT_NAMES:
                d_probs, q_probs = predict_members(payload["models"][expert], frame, final12)
                per_expert_dir[expert] = d_probs
                per_expert_qual[expert] = q_probs
            dir_probs = route_combine(per_expert_dir, route)  # (N,k,3)

            mean_dist = dir_probs.mean(axis=1)
            dir_action = mean_dist.argmax(axis=1).astype(np.int64)
            _total, _aleatoric, epistemic = mi_decomposition(dir_probs)

            # sanity check seed=260620 VAL/OOS against the existing v2 (a-run) saved predictions
            if seed == 260620:
                ref_dir = "predictions_q050.csv"
                ref_path = (RUN_ROOT / f"omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2_e40_r30000_s260620" /
                            ("validation_" + ref_dir if split_name == "VAL" else "oos_" + ref_dir))
                if ref_path.exists():
                    ref = pd.read_csv(ref_path, parse_dates=["timestamp"])
                    ref_a = ref.merge(frame[["timestamp"]], on="timestamp", how="inner")
                    prefix = "omega1_regime3_expertdq_oof" if split_name == "VAL" else "omega1_regime3_expertdq"
                    ref_dir_action = pd.to_numeric(ref_a[f"{prefix}_dir_action"], errors="raise").to_numpy()
                    match = float((ref_dir_action == dir_action[: len(ref_dir_action)]).mean()) if len(ref_dir_action) == len(dir_action) else None
                    sanity_rows.append({"split": split_name, "dir_action_match_vs_existing_csv": match})

            dec = pre_gate_decisions(dir_action, epistemic)
            recs = trades_with_signal(frame, dec)
            if len(recs) >= 10:
                e = [r["epistemic"] for r in recs]
                r = [r["trade_return"] for r in recs]
                rho, p = spearmanr(e, r)
                per_seed_rho[split_name].append(rho)
                print(f"  seed={seed:>7} {split_name}  n={len(recs):>4}  rho(epistemic,return)={rho:+.4f}  p={p:.4f}")
            else:
                print(f"  seed={seed:>7} {split_name}  n={len(recs):>4} (10건 미만, 상관 생략)")
            pooled[split_name].extend(recs)

    if sanity_rows:
        print("\n=== 정합성 체크 (seed=260620, 기존 v2 저장 예측과 dir_action 일치율) ===")
        for row in sanity_rows:
            print(f"  {row}")

    print("\n=== 요약 ===")
    for split_name in ["VAL", "OOS"]:
        arr = np.array(per_seed_rho[split_name])
        if len(arr) == 0:
            print(f"{split_name}: 유효 시드 없음")
            continue
        print(f"{split_name}: 시드 {len(arr)}개 -- 평균 rho={arr.mean():+.4f}  중앙값={np.median(arr):+.4f}  "
              f"양수 시드={int((arr > 0).sum())}/{len(arr)}  음수(useful, 불일치 클수록 결과 나쁨) 시드={int((arr < 0).sum())}/{len(arr)}")
        pr = pooled[split_name]
        if len(pr) >= 10:
            prho, pp = spearmanr([x["epistemic"] for x in pr], [x["trade_return"] for x in pr])
            print(f"        풀링 rho(n={len(pr)}) = {prho:+.4f}  p={pp:.4f}  (참고용)")


if __name__ == "__main__":
    main()
