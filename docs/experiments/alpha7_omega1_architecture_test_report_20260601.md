# Alpha7/Omega1 Architecture Test Report - 2026-06-01

## Executive Summary

오늘 테스트한 후보 중 active/live 후보를 교체할 만한 모델은 없었다.

현재 active 기준선은 계속 `Alpha7 Regime3 current-context MoE`
`bull0.85_bear1.15_chop1.25`이다.

- Active validation Cost1/2/3: `+350.75% / +361.91% / +270.24%`
- Active validation Cost3: MDD `-37.74%`, trades `167`, WR `14.97%`
- Active 2026 OOS Cost1/2/3: `+117.46% / +113.87% / +103.72%`
- Active 2026 OOS Cost3: MDD `-27.81%`, trades `133`, WR `15.04%`

오늘의 공통 결론:

1. 현재 active는 고승률 모델이 아니라 낮은 WR을 큰 payoff로 보상하는 구조다.
2. WR을 올리는 gate/TabM 계열은 승률은 개선하지만 큰 수익 타점을 같이 잘라내거나 drawdown이 악화된다.
3. Soft MoE는 OOS Cost3만 보면 가능성이 있어 보이지만 validation이 약해 승격하면 안 된다.
4. full TabM은 standalone parent로 의미 있는 연구 후보지만 active HGB MoE를 대체할 수준은 아니다.

## Promotion Decision

| Candidate | Promote | Reason |
|---|---:|---|
| Active `bull0.85_bear1.15_chop1.25` | Keep | Validation/OOS 모두 가장 실전적 |
| Soft MoE | No | OOS Cost3는 높지만 validation이 약함 |
| Two-stage entry gate | No | WR 개선, PnL 급감 |
| Shared MLP backbone | No | OOS Cost3 음수 |
| FT-Transformer | No | Validation Cost3 음수 |
| TabM-CryptoMamba sidecar | No | 기존 CryptoMamba direction sidecar보다 2026 OOS 약함 |
| Full TabM parent | No | WR은 높지만 OOS PnL/MDD가 active보다 약함 |

## Results

### 1. Active Baseline

Artifact:
`tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601`

Current active:
`bull0.85_bear1.15_chop1.25`

| Split | Cost1 | Cost2 | Cost3 | Cost3 MDD | Cost3 Trades | Cost3 WR |
|---|---:|---:|---:|---:|---:|---:|
| Validation | `+350.75%` | `+361.91%` | `+270.24%` | `-37.74%` | `167` | `14.97%` |
| 2026 OOS | `+117.46%` | `+113.87%` | `+103.72%` | `-27.81%` | `133` | `15.04%` |

Decision: retained.

### 2. Soft MoE

Script:
`scripts/eval_alpha7_regime3_current_moe_soft_blend_20260601.py`

Report:
`tmp/causal_regen_20260516/alpha7_regime3_current_moe_soft_blend_20260601/report.json`

Design:
existing bull/bear/chop expert outputs are blended at decision level using current-Regime3 probabilities. No expert retraining.

Selected:
`p1.0_conf0.65_side0.15`

| Split | Cost1 | Cost2 | Cost3 | Cost3 MDD | Cost3 Trades | Cost3 WR |
|---|---:|---:|---:|---:|---:|---:|
| Validation | `+49.81%` | `+26.17%` | `+8.83%` | `-43.78%` | `263` | `12.55%` |
| 2026 OOS | `+147.73%` | `+103.49%` | `+109.82%` | `-34.56%` | `188` | `14.36%` |

Interpretation:
OOS Cost3만 보면 active보다 조금 높지만, validation Cost3가 매우 약하고 MDD도 악화된다. OOS를 보고 승격하면 selection leak이 되므로 research-only로 둔다.

Decision: no promotion.

### 3. Two-stage Entry Gate

Script:
`scripts/eval_alpha7_regime3_current_moe_two_stage_entry_gate_20260601.py`

Report:
`tmp/causal_regen_20260516/alpha7_regime3_current_moe_two_stage_entry_gate_20260601/report.json`

Design:
Stage 1 HGB binary gate learns whether the active MoE entry was profitable on pre-validation executed entries. Stage 2 remains the existing bull/bear/chop planner.

Selected:
`gate0.35`

| Split | Cost1 | Cost2 | Cost3 | Cost3 MDD | Cost3 Trades | Cost3 WR |
|---|---:|---:|---:|---:|---:|---:|
| Validation | `+68.81%` | `+63.83%` | `+48.37%` | `-28.37%` | `71` | `12.68%` |
| 2026 OOS | `+19.70%` | `+17.49%` | `+15.79%` | `-20.92%` | `50` | `20.00%` |

Interpretation:
WR is improved, but payoff capture is destroyed. This confirms the current strategy is not a high-WR policy; it is payoff-skewed.

Decision: no promotion.

### 4. Shared Backbone / FT-Transformer Contract Test

Script:
`scripts/eval_alpha7_shared_backbone_ft_contract_test_20260601.py`

Report:
`tmp/causal_regen_20260516/alpha7_shared_backbone_ft_contract_test_20260601/report.json`

Design:
small standalone PyTorch lifecycle parents using existing Alpha7 feature contract, lifecycle label builder, and `_combo_metrics`. OOS was evaluated once after validation selection.

| Candidate | Val Cost3 | Val MDD | Val Trades | Val WR | OOS Cost3 | OOS MDD | OOS Trades | OOS WR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Shared MLP | `+23.94%` | `-27.22%` | `36` | `25.00%` | `-15.84%` | `-32.69%` | `63` | `17.46%` |
| FT-Transformer | `-14.04%` | `-31.93%` | `59` | `22.03%` | `+12.11%` | `-32.98%` | `63` | `25.40%` |

Interpretation:
The deep tabular replacements did not generalize. They improve hit rate in places but do not reproduce the active HGB MoE payoff profile.

Decision: no promotion.

### 5. TabM-CryptoMamba Direction Sidecar

Script:
`scripts/build_omega1_dir3_tabm_cryptomamba_direction_20260601.py`

Reports:

- `tmp/causal_regen_20260516/omega1_dir3_tabm_cryptomamba_20260601/dir3_tabm_cryptomamba_audit.json`
- `tmp/causal_regen_20260516/omega1_dir3_tabm_cryptomamba_128_20260601/dir3_tabm_cryptomamba_audit.json`
- `tmp/causal_regen_20260516/omega1_dir3_tabm_cryptomamba_e3_20260601/dir3_tabm_cryptomamba_audit.json`

Design:
only the CryptoMamba input projection is replaced with a TabM/BatchEnsemble frontend. Outputs are kept separate as `dir3_tabm_cmamba_*`; active Omega1 feature contract was not changed.

Baseline existing CryptoMamba sidecar:
`tmp/causal_regen_20260516/omega1_dir3_cryptomamba_20260531/dir3_cryptomamba_audit.json`

| Candidate | Features | Val bacc | Val AUC | 2025 bacc | 2025 AUC | 2026 bacc | 2026 AUC | 2026 proxy WR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline CryptoMamba | `128` | `0.5661` | `0.7513` | `0.5484` | `0.7457` | `0.5671` | `0.7486` | `0.6267` |
| TabM e5 f200 | `154` | `0.5695` | `0.7546` | `0.5509` | `0.7391` | `0.5640` | `0.7458` | `0.6187` |
| TabM e5 f128 | `128` | `0.5608` | `0.7494` | `0.5415` | `0.7358` | `0.5626` | `0.7461` | `0.6080` |
| TabM e3 f200 | `154` | `0.5686` | `0.7446` | `0.5404` | `0.7285` | `0.5536` | `0.7287` | `0.6125` |

Interpretation:
TabM slightly improves internal validation in one case, but fixed 2026 OOS is weaker than the existing CryptoMamba direction sidecar.

Decision: no promotion; do not add `dir3_tabm_cmamba_*` to active Omega1 feature contract.

### 6. Full TabM Parent

Script:
`scripts/eval_alpha7_full_tabm_parent_contract_test_20260601.py`

Report:
`tmp/causal_regen_20260516/alpha7_full_tabm_parent_contract_test_20260601/report.json`

Design:
full BatchEnsemble/TabM tabular lifecycle parent. It directly outputs action, quality, notional, leverage, TP, SL, max_hold, cooldown.

Important finding:
standard loss collapsed into near-all-cash decisions, so the final run used a trade-biased loss while preserving the same feature/backtest contract.

Selected:
`full_tabm_parent_c0.50_q0.010_s1.00_cap3.00_u0.070`

| Split | Cost1 | Cost2 | Cost3 | Cost3 MDD | Cost3 Trades | Cost3 WR |
|---|---:|---:|---:|---:|---:|---:|
| Validation | `+25.41%` | `+25.75%` | `+16.53%` | `-31.85%` | `116` | `25.00%` |
| 2026 OOS | `+46.14%` | `+32.10%` | `+26.66%` | `-43.63%` | `98` | `27.55%` |

Interpretation:
Full TabM is meaningfully higher WR than active, but total OOS PnL and MDD are much worse. It is a high-WR research branch, not a replacement.

Decision: no promotion.

## Technical Notes

### Feature / Data Contract

- No active/live feature fallback or silent alias was added.
- TabM-CryptoMamba outputs use a new prefix: `dir3_tabm_cmamba_*`.
- These outputs are not added to Omega1 active contract.
- Full TabM parent uses the existing Alpha7 clean parent feature list.

### Normalization

Deep models did not receive raw unscaled values.

- CryptoMamba / TabM-CryptoMamba: median fill + `StandardScaler`, fit on train split only.
- Shared MLP / FT / full TabM parent: quantile-normal transform, fit on train label set only.
- No 2026 normalization fit was used.

### Why WR Improved But PnL Did Not

The active candidate has low WR because it earns through payoff skew:

- many small losses,
- fewer large wins,
- high notional/payoff capture when it is right.

Post-hoc gates and high-WR neural parents tend to remove the rare high-payoff trades together with bad trades. That improves hit rate but lowers total OOS PnL.

## Next Recommendations

1. Do not replace active MoE with today’s candidates.
2. Keep Soft MoE as research-only; revisit only with a stricter validation-month stability selector.
3. Keep full TabM as a high-WR branch; do not use it for active execution yet.
4. If the goal is higher WR, retrain objectives must optimize expected value and payoff distribution together, not just entry filtering.
5. If TabM is revisited, the next useful test is not another standalone parent; it should be a residual/risk overlay on active MoE outputs.

