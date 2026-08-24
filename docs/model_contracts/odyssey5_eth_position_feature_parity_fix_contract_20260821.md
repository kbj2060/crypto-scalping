# Odyssey5 — ETH position-feature train/inference parity 버그수정 계약 문서 (2026-08-21)

slug: `odyssey5_eth_position_feature_parity_fix`

**계보**: 2026-08-18 "일리아스 1"로 명명됐던 오디세이4 버그수정 재학습판을 사용자 지시로
2026-08-21 이 문서로 이관·재명명했다. 이 축은 애초부터 일리아스 고유의 연구목표(사람 방향입력+
능동적 리스크관리)와 무관한 **오디세이4 자체의 버그수정 계승판**이라, 오디세이4/오디세이5
연속선상에 두는 게 taxonomy상 정확하다는 판단. 내용 자체(버그수정 3건, REJECTED_SIGN_MISMATCH
판정, N=5/N=6 시드검증)는 무변경 그대로 옮겼다 — 재검증하지 않았다.

## 상태

| 항목 | 상태 |
|---|---|
| position-feature parity 버그수정 재학습(h48qual/zig075) | `완료, Baseline(오디세이4 G0) 대비 REJECTED_SIGN_MISMATCH` |
| 단일시드 6창 정식비교 | `완료` — 아래 "G0 참조값 대비 결과" 참고 |
| N=5(zig075)+N=5(h48qual) dual 시드재현성 검증 | `완료 — CONFIRMED REJECTED_SIGN_MISMATCH(6/6 시드 동일패턴, 시드노이즈 아님)` |
| zig075 단독 always-benchmark N=5 재현성 | `완료 — 단일시드 "진짜스킬" 결론이 시드노이즈였음 확인, 착수 안 함` |

## 범위

오디세이4 G0(h48qual/zig075)이 갖고 있던 3건의 학습-추론 피쳐 불일치 버그를 수정하고 재학습한
모델. 아키텍처/피쳐공간(102 base + 13 pos)은 오디세이4와 완전히 동일 — 이 문서가 다루는 건
순수 버그수정 효과 하나뿐이다.

**수정한 버그 3건** (`.claude/CLAUDE.md` "Position-Feature Train/Inference Parity Contract"
정책 신설의 근거, 상세 `docs/experiments/eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_20260818.md`):
1. `pos_unrealized`/`pos_mfe`/`pos_mae`: 학습 시 notional로 스케일압축됐으나 라이브는 비압축
   원시 가격변동률을 씀.
2. `pos_notional`/`pos_leverage`/`pos_exposure`: 학습 시 BASE_TEMPLATE 고정상수라 무분산이었으나
   라이브는 실제 risk sidecar 예측값이 매 순간 다름.
3. `pos_tp`/`pos_sl`: 학습 시 BASE_TEMPLATE 고정값이었으나 라이브는 ATR-adaptive.

**추가 변경**: canonical 데이터 재학습(base_cols을 원본과 정확히 동일한 102개로 고정), 전용
risk sidecar 신규 학습, VAL-best quality_threshold 재선택(h48qual 0.50→0.40, zig075 0.75→0.80).

## 모델 자산

| 컴포넌트 | 위치 | 비고 |
|---|---|---|
| h48qual TabM 번들 | `tmp/causal_regen_20260516/odyssey5_eth_h48qual_bundle_20260821/true_3head_tabm_bundle.pt` | 2026-08-21 `omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_pinned102_20260818/`에서 이동. **구 경로는 심볼릭링크로 보존**(19개 의존 스크립트 무수정 동작) |
| zig075 TabM 번들 | `tmp/causal_regen_20260516/odyssey5_eth_zig075_bundle_20260821/true_3head_tabm_bundle.pt` | 2026-08-21 `..._current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818/`에서 이동. 구 경로 심볼릭링크 보존 |
| h48qual risk sidecar | `tmp/causal_regen_20260516/odyssey5_eth_h48qual_risk_sidecar_20260821/risk_sidecar.pkl` | 2026-08-21 `omega4_2_trade_risk_sidecar_20260622_h48qual_pinned102_q040_20260818/`에서 이동. 구 경로 심볼릭링크 보존 |
| zig075 risk sidecar | `tmp/causal_regen_20260516/odyssey5_eth_zig075_risk_sidecar_20260821/risk_sidecar.pkl` | 2026-08-21 `omega4_2_trade_risk_sidecar_20260622_zig075_pinned102_q080_20260818/`에서 이동. 구 경로 심볼릭링크 보존 |
| N=5(zig075) 신규시드 번들 | `scripts/train_eth_ilias1_zig075_pinned102_seed_variant_20260818.py`로 재학습, 개별 산출경로는 그 스크립트 출력 참고 | **이동 안 함** — 검증용 파생물, 파일명에 "ilias1" 접두어가 남아있으나(2026-08-18 명명 당시 생성) 스크립트/데이터 자체는 무수정, 이름만 역사적 잔재 |
| N=5(h48qual) 신규시드 번들 | `scripts/train_eth_ilias1_h48qual_pinned102_seed_variant_20260818.py` | 위와 동일 사유로 이동 안 함 |

**주의**: 위 4개 핵심 자산은 물리적으로 이동했으나, `ilias1_*`/`zigzagfix_06_*` 등 옛 이름을 쓰는
19개 스크립트(`grep -rl` 확인됨: `eval_eth_ilias1_*`, `train_eth_ilias1_*`,
`train_eval_omega4_2_risk_sidecar_eth_pinned102_*` 등)는 심볼릭링크 덕분에 전부 무수정으로
계속 동작한다. N=5 시드 검증 산출물은 핵심 4개 자산이 아니라 다수의 파생 파일이라 이번
이관에서는 이동하지 않았다 — 필요시 후속 정리.

## G0 참조값 (오디세이4 Baseline v1, 재계산 불필요)

| 창 | PnL | MDD | 거래수 |
|---|---:|---:|---:|
| VAL(판정) | +77.31% | −21.76% | 26 |
| OOS-Q1(판정) | +67.25% | −15.48% | 19 |
| OOS-Q2(판정) | −12.69% | −20.76% | 10 |

## G0 참조값 대비 결과 (단일시드 260620, 6창 정식비교)

| 창 | tier | Baseline(원본) | Odyssey5(버그수정) | delta |
|---|---|---:|---:|---:|
| 2025-Q1(참고) | context | +28.54%/−20.62%/19건 | +0.88%/−39.94%/27건 | −27.65pp/−19.32pp |
| 2025-Q2(참고) | context | +39.99%/−10.82%/15건 | −9.94%/−26.83%/24건 | −49.93pp/−16.01pp |
| 2025-Q3(참고) | context | −9.73%/−44.37%/19건 | −10.72%/−38.50%/24건 | −1.00pp/+5.87pp |
| VAL(판정) | val | +54.88%/−31.11%/22건 | +114.03%/−25.88%/27건 | +59.15pp/+5.23pp |
| **OOS-Q1(판정)** | oos_confirm | +28.17%/−15.48%/19건 | +22.75%/−19.99%/31건 | **−5.42pp/−4.50pp** |
| **OOS-Q2(판정)** | oos_confirm | +9.85%/−15.00%/10건 | +36.21%/−9.17%/14건 | **+26.36pp/+5.83pp** |

**판정: REJECTED_SIGN_MISMATCH**(사전등록기준 — OOS-Q1이 PnL·MDD 둘 다 non-worse 기준 미달,
OOS-Q2는 통과, single-touch 양쪽동시통과 미충족). Baseline(오디세이4 G0)을 대체하지 않는다.

⚠️ VAL 수치는 이 번들 자신의 VAL-PnL 기준으로 threshold를 재선택했기 때문에 극적으로
개선된 것(+59.15pp) — 일반화 근거 아님(VAL 선택편향과 같은 모양일 위험).

## N=5 시드 재현성 검증 — REJECTED_SIGN_MISMATCH는 시드노이즈 아님, CONFIRMED

quality_threshold(h48qual=0.40, zig075=0.80)와 risk sidecar는 고정, 인코더 시드만 단독 변수로
격리(원본 260620 + 신규 5개, 총 6시드 dual 페어링).

| 시드 | OOS-Q1 판정 | OOS-Q1 PnL delta | OOS-Q2 판정 | OOS-Q2 PnL delta | 최종판정 |
|---|---|---:|---|---:|---|
| 260620(원본) | fail | −44.50pp | PASS | +48.90pp | REJECTED_SIGN_MISMATCH |
| 121026 | fail | −11.94pp | PASS | +85.22pp | REJECTED_SIGN_MISMATCH |
| 337153 | fail | −62.82pp | PASS | +54.08pp | REJECTED_SIGN_MISMATCH |
| 390529 | fail | −62.33pp | PASS | +17.19pp | REJECTED_SIGN_MISMATCH |
| 640787 | fail | −89.68pp | PASS | +23.37pp | REJECTED_SIGN_MISMATCH |
| 794920 | fail | −58.13pp | PASS | +73.26pp | REJECTED_SIGN_MISMATCH |

**결과: 6/6 시드 전부 REJECTED_SIGN_MISMATCH, 예외 없음.** OOS-Q1은 6개 시드 전부 fail(전부
음수), OOS-Q2는 6개 시드 전부 PASS(전부 양수) — 실패/통과 창의 조합이 시드에 걸쳐 완전히
동일하다. **구조적으로 안정된(seed-robust) 결과, 단일시드 노이즈가 아니다.**

⚠️ 범위: threshold/sidecar를 원본 고정값으로 유지한 상태의 시드 재현성만 격리했다. threshold
재선택이나 시드별 전용 sidecar까지 포함한 완전 재현은 범위 밖.

## zig075 단독 always-benchmark N=5 재현성 (부수 조사)

Odyssey5 재학습 이후 진행된 부수 조사에서 zig075 단독 성과(VAL +192.21% 등)가 눈에 띄게 강해
"zig075 단독 위에 리스크관리 스택을 새로 쌓자"는 제안이 나왔으나, always-long/always-short
벤치마크 대조([[h48qual_standalone_replay_invalid]] 방법론)로 먼저 검증:

- 단일시드(원본 260620): real이 always_long/always_short 둘 다 압도하는 창은 2025-Q2·VAL
  2곳뿐, 판정 2창(OOS-Q1/OOS-Q2)은 스킬 증거 없음.
- N=5 신규 랜덤시드 확장(6시드×6창×3variant): beats_both 8/36(22.2%) — 3택1 무작위 기준선
  (~33.3%)보다도 낮음. 원본시드의 "VAL/2025-Q2에서만 진짜스킬" 패턴 자체가 재현 안 됨(seed337153은
  VAL 패배·OOS-Q1,Q2 승리로 원본과 정반대).

**판정**: N=5 재현성 테스트로 "zig075 단독이 진짜 방향스킬을 가진다"는 단일시드 결론이
시드노이즈였음을 확인. "zig075 단독 위에 리스크관리 스택을 새로 쌓자"는 제안은 이 근거로는
착수하지 않는다.

## 미해결 이슈

- **CLAUDE.md Omega Artifact Integrity Gate 관련**: 공식 `promotion_pass=true`는 아직 아님 —
  parent/sidecar 스크립트 둘 다 `dataset_lineage` 필드를 원천적으로 안 씀(그 감사 스크립트
  자체가 "이 게이트 신설 이전 모든 report.json은 의도적으로 전부 fail"이라고 명시). 스크립트
  자체 수정이 필요한 별도 과제.
- N=5 시드 검증 산출물(다수 파일)은 이번 이관에서 물리적으로 이동하지 않았다 — 필요시 후속
  정리 대상.
- REJECTED_SIGN_MISMATCH 판정 자체를 뒤집을 만한 새 접근(예: OOS-Q1이 왜 구조적으로 fail하는지
  원인분석)은 미착수.

## 승격 게이트

이 축은 REJECTED_SIGN_MISMATCH로 확정됐으므로 현재 승격 후보가 아니다. 재시도한다면
`.claude/CLAUDE.md` Omega Artifact Integrity Gate + Fresh-Forward Rule + Seed-Diversity
Ensemble Promotion Gate를 전부 통과해야 하며, 무엇보다 OOS-Q1이 구조적으로 실패하는 원인을
먼저 규명해야 한다.
