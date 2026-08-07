# Alpha1.4 Phase2 DL/RL Upgrade Suite - 2026-05-14

## 기준선

Alpha1.4 soft execution proxy는 기존 Alpha1/V31 decision stack을 동결하고 execution cost proxy만 추가한 기준 모델이다.

| Model | Verdict | Cost1 PnL | Cost1 MDD | Cost2 PnL | Cost3 PnL | Note |
|---|---:|---:|---:|---:|---:|---|
| Alpha1 baseline taker | baseline | +361.19% | -31.74% | +88.74% | +0.58% | no soft execution |
| Alpha1.4 soft execution proxy | promote | +385.98% | -31.68% | +94.35% | +10.35% | selected live candidate |

## Phase2 실험 결과

| Idea / Experiment | Layer Changed | Retrained Dependent Layer | Verdict | Cost1 PnL | Cost1 MDD | Cost2 PnL | Cost3 PnL |
|---|---|---|---:|---:|---:|---:|---:|
| L2 execution replay | Execution router | no model retrain, route replay only | shadow_collect_l2 | +642.43% | -30.54% | +434.61% | +402.96% |
| RL exit + sizing | Deep alpha exit/sizing | exit policy + notional allocator retrained | iterate | +361.19% | -31.74% | +88.74% | +0.58% |
| DSAC execution overlay | Deep alpha execution | DSAC checkpoint reused, route thresholds selected | iterate | +361.19% | -31.74% | +88.74% | +0.58% |
| Cost3 CVaR notional guard | Notional risk guard | runtime guard selected on 2025Q4 | iterate | +385.98% | -31.68% | +94.35% | +10.35% |
| Residual veto/size RL | Candidate-level RL critic | skip/size critics retrained | iterate | +361.19% | -31.74% | +88.74% | +0.58% |
| DT/Liquid parent | Parent replacement | sequence parent retrained | iterate | +361.19% | -31.74% | +88.74% | +0.58% |
| Teacher-constrained deep parent | Parent constrained overlay | sequence teacher model retrained | promote | +398.59% | -31.74% | +88.55% | +0.86% |
| Fresh DSAC parent-lite | Parent replacement | DSAC parent-lite selected | iterate | +361.19% | -31.74% | +88.74% | +0.58% |
| Selective hazard pruning | Parent notional hazard guard | hazard model retrained | iterate | +273.45% | -31.72% | +79.69% | -2.09% |

## 레드팀 판정

- Blocking issue는 없음. 모든 실험은 train/selection/OOS 분리를 유지했다.
- 반복 경고: 일부 feature가 train/eval에서 zero-fill된다. 특히 `garch_vol_z`, `liquidity_vacuum`, `execution_quality`, `jump_z`, `funding_pressure`, `patchtst_pred`, `patchtst_confidence` 계열은 추후 feature completeness 개선 필요.
- L2 execution replay는 수치가 가장 좋지만 `shadow_collect_l2` 판정이다. historical L2 snapshot/queue fill 모델이 충분하지 않아 live 승격 후보가 아니라 shadow 검증 후보로 둔다.
- Teacher-constrained deep parent는 Cost1 PnL을 Alpha1.4보다 높였지만 Cost2/Cost3 내구성이 Alpha1.4 soft execution proxy보다 약하다.

## 결론

현재 메인 유지 후보는 여전히 Alpha1.4 soft execution proxy다.

다음 후보는 두 갈래다.

1. Alpha1.4 + teacher-constrained deep parent + soft execution proxy를 결합해 Cost1 상승분과 Alpha1.4 cost relief를 같이 가져오는 조합 실험.
2. L2 replay를 live shadow orderbook snapshot으로 재검증해 maker fill assumption을 실제 데이터로 검증.

Parent를 DSAC/DT/Liquid로 완전 교체하는 방향은 이번 재검증에서도 OOS가 무너졌다. 당분간은 완전 교체보다 frozen parent 주변의 constrained overlay 방식이 더 안전하다.
