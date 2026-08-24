# ETH zig075 direction_head 스킬 Stage 1 예비 체크 (2026-08-15)

## 배경

Odyssey1(`docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`)이
N≥5 진짜 다양시드 × 7개 이상 독립 모델/라벨 조합으로 **h48qual의 `direction_head`는 스킬이
없다**(ungated `direction_head` argmax가 VAL/OOS 매번 always-short에 완패)를 formal하게
확정했다. zig075는 h48qual과 **같은 아키텍처(3-Head TabM, k=8, hidden=192, 115피쳐)와 같은
라벨(`zigzag_action`)**을 공유하지만(`quality_mode=same_as_direction`,
`docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`), 이 formal
ungated-vs-always-short 테스트를 한 번도 받은 적이 없었다. 기존 zig075 체크들
(`scripts/diagnose_eth_zig075_short_only_vs_always_short_20260812.py`,
`scripts/diagnose_eth_zig075_final15_multiseed_always_short_20260812.py` 등, 결과는
`docs/experiments/eth_zig075_final15_multiseed_pnl_validation_20260812.md`)은 전부
**quality_threshold=0.75로 이미 게이트된 active set 위에서** model(gated) vs always_short를
비교한 것으로, gate 자체를 무시한 순수 direction_head 스킬 테스트가 아니었다. 사용자가 이 갭을
싸게(재학습 없이) 먼저 닫으라고 지시했다 — 이 문서가 그 Stage 1 결과다.

## 방법

`scripts/diagnose_eth_h48qual_ungated_direction_vs_always_short_20260812.py`(Odyssey1의 formal
ungated 테스트 스크립트)를 **로직 변경 없이** zig075 데이터 경로로만 바꿔 재사용
(`scripts/diagnose_eth_zig075_ungated_direction_vs_always_short_20260815.py`, git diff는
번들/CSV 경로·prefix뿐).

- **배포 번들 확인**: `trading_bot_modules/runtime_config.py`의
  `FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH` →
  `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt`.
  `trading_bot_modules/omega4_6_1_live.py:290`에서 이 컴포넌트가 `quality_threshold=0.75`로
  인스턴스화됨(ETH 라이브는 h48qual→zig075 우선순위 confluence, `PRIORITY = ("h48qual", "zig075")`).
- **기존 예측 재사용**: 위 번들 디렉토리에 이미 `train/validation/oos_predictions_q075.csv`가
  존재(2026-06-30 생성, Omega Artifact Integrity 정책이 요구하는 정확한 threshold 일치) — 신규
  추론 없음, 재학습 없음.
- **ungated 시뮬레이션**: `_to_fixed_decisions`가 읽는 `{prefix}_final_action`(quality gate
  통과분)을 `{prefix}_dir_action`(direction_head 원본 argmax, 게이트 없음)으로 치환한 뒤 동일
  TP/SL/비용 컨벤션(`train_eval_omega1_2_tabm_diffusion_risk_20260603._metrics`, cost_mult=3.0,
  `max_hold=0`/`cooldown=0`)으로 백테스트. `always_short`/`always_long`은 같은 ungated active
  set에서 방향만 강제.
- **구간**: 기존 zig075/h48qual 라이브 번들 진단 스크립트 전체와 동일한 라이브 VAL 시작점인
  VAL=2025-10-01~12-31(라이브 TRAIN CSV의 VAL 파티션), OOS=2026-01-01~03-31
  (`trade_candidates_2026_alpha6_current_tail111_exact.csv`). 저장소 표준 스플릿(VAL 09-01
  시작)과 VAL 시작일이 다른데, 이는 이 프로젝트의 zig075/h48qual 라이브 번들 진단 스크립트들이
  전부 공유하는 기존 관행(TRAIN CSV 자체의 VAL 파티션 경계)을 그대로 따른 것 — 비교 가능성을
  위해 새 경계를 만들지 않았다.
- **단일 인스턴스(N=1)**: 이미 배포된 시드 하나의 저장 예측만 사용. Seed-Diversity Ensemble
  Promotion Gate 정책상 이 결과 단독으로는 확정 결론(스킬 있음/없음)을 낼 수 없다 — 아래
  결과는 전부 **preliminary, single-seed**로 취급.

## 결과

| split | ungated pnl | trades | wr | always_short pnl | trades | wr | always_long pnl | trades | wr | ungated이 always_short 이김? | ungated이 max(둘) 이김? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| VAL | **+20.30** | 33 | 48.5% | +10.38 | 36 | 41.7% | -13.78 | 41 | 29.3% | **YES** | **YES** |
| OOS | -2.34 | 27 | 33.3% | **+17.11** | 25 | 52.0% | -16.16 | 23 | 17.4% | NO | NO |

(참고: 게이트 있는 라이브 정책(`quality_threshold=0.75`) 재현값 VAL +11.03/OOS +14.71 —
`eth_zig075_final15_multiseed_pnl_validation_20260812.md`의 260620 시드 q050 계열 수치와
같은 방향, 파이프라인 정합성 확인용.)

**VAL에서는 ungated direction_head가 always_short과 always_long 둘 다 이기지만, OOS에서는
always_short에 진다(always_long은 이김).** 2칸 중 1칸(VAL)만 승 — h48qual의 formal 테스트가
보인 완패 패턴(N=40칸 중 2칸, 전부 Wilcoxon p≈1.0, h384 VAL 한정)과는 다르다: 여기서는
승리가 h48qual보다 하나 더 있고(자체 리스크셋이 다르므로 직접 비교 불가하지만), 무엇보다
**게이트 정책값과 부호가 다른 스플릿(VAL)에서 ungated가 게이트값보다도 크게 이긴다**
(+20.30 vs +11.03) — h48qual에서는 이런 패턴이 관측된 적이 없다.

## 해석

이 단일 시드 결과는 h48qual의 완패 패턴을 그대로 재현하지 않는다 — VAL에서 명확한 승리가
있다는 점은 h48qual과 다른 신호다. 그러나:

1. N=1이므로 [[tabm_hp_low_signal_pattern]]/Seed-Diversity Gate 정책상 노이즈와 신호를 구분할
   수 없다. h48qual도 individual 시드 단위로는 부호가 자주 뒤집혔다(예: `final15` 903174 시드가
   구버전과 거의 같은 confidence gap을 보인 것처럼) — 이 zig075 VAL 승리가 시드 노이즈인지
   진짜 신호인지는 이 결과 하나로 판단 불가.
2. OOS에서는 always_short에 진다 — "최근 구간에서 스킬이 있다"는 결론도 아직 이르다.
3. 그럼에도 불구하고, h48qual formal 테스트(40칸 중 2칸, 전부 유의하지 않음)와 달리 **여기선
   2칸 중 1칸이 명확한 큰 폭 승리**라, "zig075도 h48qual과 똑같이 완전히 죽었다"고 사전에
   가정할 근거는 약하다 — 이 gap을 formal하게 닫을 가치가 h48qual 케이스보다 조금 더 있어
   보인다.

## 산출물

- 스크립트: `scripts/diagnose_eth_zig075_ungated_direction_vs_always_short_20260815.py`
- 결과 CSV: `tmp/eth_zig075_ungated_direction_vs_always_short_20260815/ungated_vs_always_short.csv`

## 다음 단계 (Stage 2 여부는 사람 판단 필요)

**Stage 2(N≥5 진짜 다양시드 formal 재학습, Seed-Diversity Ensemble Promotion Gate 충족)는 이
문서에서 실행하지 않는다** — 이 태스크 스코프 밖(GPU/서버 비용이 드는 결정, 사람이 먼저
판단해야 함). 이 Stage 1 결과가 주는 신호: h48qual과 달리 완전한 죽은 신호로 보이지 않으므로
(VAL 승리 존재), Stage 2를 고려할 여지는 있다 — 다만 N=1로는 "볼 가치가 있다" 이상을 주장할
수 없다. Stage 2를 진행한다면 h48qual formal 테스트와 동일한 방법론
(`scripts/diagnose_eth_h48qual_multiseed_ungated_direction_vs_always_short_20260812.py` 패턴,
N≥5 랜덤 시드, VAL+OOS 전부, Wilcoxon)을 그대로 이식하면 된다.
