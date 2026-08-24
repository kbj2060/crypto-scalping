"""102-feature vs 78-feature (dedup) zig075 3-Head TabM 모델의 순수 추론 속도 벤치마크.

사용자 질문: "78-feature 버전이 더 빠른가?" -- PnL/skill은 이미 동일(둘 다 always_short에
완패, `docs/experiments/eth_omega461_dedup78feature_nseed_skill_retest_20260815.md`)한 것으로
확정됐으므로 이번엔 순수 성능(속도) 비교만 한다. 재학습 없음 -- 기존에 저장된 5-seed 번들
(`docs/experiments/eth_omega461_zig075_direction_head_skill_formal_nseed_20260815.md`의
pinned102 zig075 5-seed, `eth_omega461_dedup78feature_nseed_skill_retest_20260815.md`의
pinned78 zig075 5-seed)의 `.pt` 체크포인트만 로드.

모델 로딩/추론 코드는 라이브 경로(`trading_bot_modules/omega4_6_1_live.py:_Component._build_model`,
`scripts/train_eval_omega1_2_tabm_3head_20260603.py:ThreeHeadTabM/_predict_payload`)와 동일한
클래스/state_dict 로딩 방식을 그대로 재사용(재구현 없음). 라이브 기본 device="cpu"
(`Omega461LiveAdapter.__init__(..., device: str = "cpu")`)이고 이 dev 머신도
`torch.cuda.is_available()=False`(GPU 없음)라 CPU가 곧 라이브가 실제 쓰는 경로와 일치.

입력 텐서는 랜덤 float32(정확한 shape만 중요 -- forward pass의 연산량/latency는 입력 값이 아니라
텐서 shape에 의해 결정되므로 랜덤 데이터로 충분).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402

torch.set_grad_enabled(False)

BUNDLES = {
    "102feature": {
        946043153: ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_zig075_formal5seed_20260815_seed946043153/true_3head_tabm_bundle.pt",
        542143953: ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_zig075_formal5seed_20260815_seed542143953/true_3head_tabm_bundle.pt",
    },
    "78feature": {
        946043153: ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned78_zig075_dedup_seed946043153/true_3head_tabm_bundle.pt",
        542143953: ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned78_zig075_dedup_seed542143953/true_3head_tabm_bundle.pt",
    },
}
EXPERT = "bull"  # 3개 expert(bull/bear/chop) 전부 동일 아키텍처/n_features -- 하나만 시간측정 대표로 사용
DEVICE = torch.device("cpu")
N_SINGLE_WARMUP = 50
N_SINGLE_TIMED = 300
BATCH_N = 10_000
N_BATCH_REPS = 10


def build_model(payload):
    cfg = parent.ThreeHeadConfig(**dict(payload["config"]))
    model = parent.ThreeHeadTabM(int(payload["n_features"]), cfg=cfg).to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def param_count(state_dict) -> int:
    return int(sum(v.numel() for v in state_dict.values()))


def time_single_row(model, n_features: int):
    x = torch.randn(1, n_features, dtype=torch.float32)
    for _ in range(N_SINGLE_WARMUP):
        model(x)
    times_ms = []
    for _ in range(N_SINGLE_TIMED):
        t0 = time.perf_counter()
        model(x)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "p95_ms": times_ms[int(0.95 * len(times_ms)) - 1],
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
    }


def time_batch(model, n_features: int, batch_n: int):
    x = torch.randn(batch_n, n_features, dtype=torch.float32)
    for _ in range(3):
        model(x)
    times_ms = []
    for _ in range(N_BATCH_REPS):
        t0 = time.perf_counter()
        model(x)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = statistics.mean(times_ms)
    return {"mean_ms": mean_ms, "rows_per_sec": batch_n / (mean_ms / 1000.0)}


results = {}
for tag, seeds in BUNDLES.items():
    results[tag] = {}
    for seed, path in seeds.items():
        assert path.exists(), f"missing bundle: {path}"
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        n_base_cols = len(bundle["base_cols"])
        payload = bundle["models"][EXPERT]
        n_features = int(payload["n_features"])
        model = build_model(payload)
        n_params_single_expert = param_count(payload["state_dict"])
        n_params_all_3_experts = sum(param_count(bundle["models"][e]["state_dict"]) for e in bundle["models"])

        single = time_single_row(model, n_features)
        batch = time_batch(model, n_features, BATCH_N)

        results[tag][seed] = {
            "bundle_path": str(path),
            "n_base_cols": n_base_cols,
            "n_features_total": n_features,
            "n_params_single_expert_model": n_params_single_expert,
            "n_params_all_3_experts": n_params_all_3_experts,
            "single_row_latency_ms": single,
            "batch_10000": batch,
        }
        print(f"[{tag}] seed={seed} n_base_cols={n_base_cols} n_features={n_features} "
              f"params(1expert)={n_params_single_expert} params(3experts)={n_params_all_3_experts}")
        print(f"    single-row: mean={single['mean_ms']:.4f}ms median={single['median_ms']:.4f}ms "
              f"p95={single['p95_ms']:.4f}ms min={single['min_ms']:.4f}ms max={single['max_ms']:.4f}ms")
        print(f"    batch({BATCH_N}): mean={batch['mean_ms']:.2f}ms  throughput={batch['rows_per_sec']:.0f} rows/sec")

out_path = ROOT / "tmp/eth_omega461_102vs78feature_speed_benchmark_20260815/results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))
print("\n저장:", out_path)

# 요약: 102 vs 78 delta (seed 평균)
for metric_path, label in [
    (lambda r: r["single_row_latency_ms"]["mean_ms"], "single-row mean ms"),
    (lambda r: r["single_row_latency_ms"]["median_ms"], "single-row median ms"),
    (lambda r: r["single_row_latency_ms"]["p95_ms"], "single-row p95 ms"),
    (lambda r: r["batch_10000"]["rows_per_sec"], "batch throughput rows/sec"),
    (lambda r: r["n_params_single_expert_model"], "params (1 expert model)"),
]:
    v102 = statistics.mean(metric_path(r) for r in results["102feature"].values())
    v78 = statistics.mean(metric_path(r) for r in results["78feature"].values())
    delta_pct = (v78 - v102) / v102 * 100.0
    print(f"{label}: 102feat={v102:.4f}  78feat={v78:.4f}  delta={delta_pct:+.2f}%")
