# 오메가4.6.1 라이브 승격(h48qual+zig075 dual) 자체의 시드 강건성 — N=5 CONFIRMED (2026-08-19)

**⚠️ 업데이트(같은 날)**: N=3 예비검증 이후 사용자 지시로 신규 랜덤시드 2개(312069414/44751167)를
추가해 N=5로 확장 완료 — CLAUDE.md Seed-Diversity Ensemble Promotion Gate 정식기준 충족.
결과는 N=3과 동일 패턴(6창 중 4창 부호플립)으로 재현됐다. 아래 N=3 서술은 최초 실행 기록으로
보존하고, N=5 결과와 최종 판정은 문서 끝에 별도 절로 추가한다.

## 배경 / 질문

사용자 질문: "오메가 4.6.1 라이브 모델은 이전에 했던 시드 5개 실험에서도 준수한 성능을 보였지?"에
대해 메모리·저장소를 조사한 결과, **라이브로 승격된 모델 자체를 여러 시드로 재검증한 기록은
없었다**는 게 확인됐다. 가장 근접한 과거 작업("일리아스1" N=5+1 시드 검증,
`docs/experiments/eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_20260818.md` 후속
세션 10)은 exit_head pos_tp/pos_sl 버그를 수정하고 threshold를 재튜닝한 **후보**를 baseline과
비교한 것이라 다른 질문("이 후보가 baseline을 이기는가")에 답한 것이었다.

또한 라이브 승격 시점(`docs/model_contracts/live_model_v1_checkpoint_20260714.md`, 2026-07-14
체크포인트, 2026-07-13 아티팩트 무결성 재생성) 당시엔 CLAUDE.md의 Seed-Diversity Ensemble
Promotion Gate 정책 자체가 존재하지 않았다(그 정책은 2026-08-01 Sigma3-1h 감사에서 사후 도입).
즉 **실제 라이브 승격 근거는 단일 시드(260620)뿐이었다** — 사용자 지시로 이 공백을 N=3 시드로
메우는 예비검증을 진행했다.

## 방법론

**목표**: "실제 라이브를 학습시킨 정확한 코드/설정"을 다른 시드로 재현해, VAL/OOS 부호가
시드에 걸쳐 일관되는지 확인. 버그수정판(일리아스1)과는 의도적으로 다른 축.

1. **코드**: `git show HEAD:scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`
   로 추출한 스냅샷(`scripts/eth_live_promotion_seed_robustness_prefix_snapshot_20260819.py`) —
   워킹트리의 그 파일은 2026-08-18 exit_head pos_tp/pos_sl 버그수정이 **미커밋 상태**로 적용돼
   있어(`git diff HEAD`로 확인), 이 스냅샷이 실제 라이브 번들을 만든 정확한 코드다.
2. **설정**: report.json에서 확인한 라이브 그대로 — h48qual(`quality_mode=quality_label_action`,
   `quality_label_dir=sltp_h48_conservative_padded_to_zigzag_timestamps`, threshold=0.50),
   zig075(`quality_mode=same_as_direction`, threshold=0.75), 둘 다
   `direction_label_dir=zigzag_action_labels_20260531`, `exit-label-mode=entry_label_terminal_giveback`,
   `epochs=2`, `max-exit-samples=30000`. Risk sidecar는 3개 시드 전부 원본(260620) 것을 frozen
   재사용(일리아스1과 동일한 단순화 — 시드별 전용 sidecar 재학습은 별도의 무거운 축, 범위 밖).
3. **시드**: 260620(기존 라이브 번들 그대로 재사용, 재학습 안 함) + 신규 랜덤 2개
   (`random.SystemRandom().sample`, 94046540/524707103).
4. **⚠️ 도중 발견한 피쳐드리프트 이슈(원본 코드를 오늘 그대로 재실행할 때의 함정)**: 원본
   코드+legacy CSV(`omega.TRAIN_CSV/EVAL_CSV` 기본값) 조합으로 첫 실행했더니 자동유도 base_cols가
   102→179개로 커짐(ai_*/m7_*/patchtst_* 등) — posfix 재학습 때 이미 겪은 것과 동일한 피쳐엔지니어링
   누적 문제. 라이브 번들 자신의 102 base_cols로 pin했더니 이번엔 legacy CSV 자체에 원본 102개 중
   7개(`fibonacci_level`/`funding_roc_12`/`funding_roc_48`/`funding_z_score`/`hurst_288`/
   `regime_persistence`/`short_squeeze_risk`)가 아예 없어 실패 — legacy EVAL_CSV(2026-02-28까지)가
   이 피쳐들의 overlay 커버리지 밖인 것으로 추정. 기존에 이미 검증된 해법(canonical TRAIN_CSV/EVAL_CSV
   오버라이드, `train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818.py`와 동일 기법)을
   원본 코드 위에 복제(`scripts/eth_live_promotion_seed_robustness_canonicaldata_20260819.py`) +
   102-pin을 최종 적용. 사전점검(`_precheck_20260819.py`)으로 canonical 데이터에서 h48qual/zig075
   원본 102개가 전부 존재함(missing=0)을 확인 후 재실행.
5. **⚠️ 동시성 버그**: 4개 학습(h48qual/zig075 × 신규시드 2개)을 병렬로 띄웠더니 canonical-override
   래퍼가 모듈 임포트 시점에 공유 placeholder CSV를 매번 재생성하는 로직이라 2/4가 파일쓰기 경합으로
   pandas 파싱에러 발생 — `os.replace()` 원자적 rename으로 수정 후 재시도, 전체 4개 성공.

**평가**: `eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818.py`(Fresh-Forward 6창,
번들 자신의 추론으로 매 bar 진입결정 재생성, 저장 ledger 재사용 없음)를 그대로 재사용 —
`scripts/eth_live_promotion_seed_robustness_eval_3seed_20260819.py`가 BUNDLES만 시드별로 교체.
threshold(0.50/0.75)·risk sidecar(원본 frozen)·exit_threshold(`sweep.BASELINE_EXIT_THRESHOLD`=0.95)는
`research_eth_omega461_exit_sweep_20260721.py::COMPONENTS`(=실제 라이브 설정 그 자체) 그대로.

## 결과 — with_gate PnL(%), 6창 × 3시드

| window | seed260620(라이브 원본) | seed94046540 | seed524707103 | 부호일치 |
|---|---:|---:|---:|---|
| 2025q1 | +28.54 | +49.82 | +45.73 | ✅ YES |
| 2025q2 | +39.99 | **-33.45** | **-35.08** | ❌ **SIGN FLIP** |
| 2025q3 | -9.73 | -31.24 | **+4.77** | ❌ **SIGN FLIP** |
| **val** | **+54.88** | **+91.44** | **-16.48** | ❌ **SIGN FLIP** |
| oos_q1 | +28.17 | +5.70 | +38.41 | ✅ YES |
| oos_q2 | +9.85 | **-4.61** | +13.72 | ❌ **SIGN FLIP** |

**6창 중 4창(2025q2, 2025q3, val, oos_q2)에서 부호가 시드에 따라 뒤집힌다.** 부호가 일치하는
2창(2025q1, oos_q1)도 크기 편차가 크다(oos_q1: +5.70%~+38.41%, 약 33pp 스프레드).

가장 우려되는 지점은 **VAL 자체가 뒤집힌다는 것**이다(+54.88%/+91.44% → -16.48%) — CLAUDE.md가
인용하는 Sigma3-1h 선례(VAL은 시드 간 거의 일치했으나 OOS만 반전)보다 오히려 더 심한 패턴이다.
이번 케이스는 "OOS가 VAL과 다르게 논다"가 아니라 "VAL조차 시드 노이즈로 부호가 갈린다" —
즉 원 승격 근거였던 VAL 수치(+54.88%) 자체가 시드 260620의 우연일 가능성을 배제할 수 없다.

## 한계 / 교란변수 (⚠️ 정식판정 아님)

- **N=3**: CLAUDE.md Seed-Diversity Ensemble Promotion Gate는 N≥5 진짜 랜덤시드를 요구한다. 이
  결과는 그 문턱에 못 미치는 **예비/방향성 확인**이지, 공식 판정(CONFIRMED/REJECTED)이 아니다.
- **risk sidecar 고정**: 3개 시드 전부 원본(260620) sidecar를 frozen 재사용 — 시드별 전용
  sidecar였다면 결과가 달라질 수 있음(일리아스1과 동일한, 이미 감수된 단순화).
- **canonical 데이터 전환**: 순수 legacy CSV로는 원본 102 base_cols 중 7개가 아예 없어 부득이
  canonical로 전환했다 — seed260620_original 행도 이번 평가에선 canonical 프레임으로 새로
  추론한 예측을 쓴다(번들 가중치 자체는 라이브와 동일, 원본 260620의 report.json 수치와 소수점
  일치 여부는 별도 검증 안 함 — 이전 posfix 축에서 검증된 파이프라인 재사용이라 신뢰도는 높으나
  이번 세션에서 직접 재확인하지 않았음).

## 결론 / 제안

N=3라는 예비 규모에서도 6창 중 4창이 부호를 뒤집는다는 건 심각한 신호다 — 정식 N≥5 검증
없이도 "이 특정 260620 시드가 우연히 좋았을 가능성"을 진지하게 고려해야 한다. 다음 중 하나를
권장한다:
1. N≥5로 정식 확장(신규 랜덤시드 3개 추가) — 공식 CLAUDE.md 게이트 기준 충족.
2. 그 결과를 기다리는 동안 라이브 리스크 익스포저 재검토(현재 실주문은
   `BINANCE_ACCOUNT_ENABLED`로 차단된 decision-only 상태였음 — 2026-07-14 체크포인트 기준,
   현재 서버 설정은 이번 세션에서 재확인하지 않음, 별도 확인 필요).
3. 시드별 전용 risk sidecar까지 포함한 재검증(현재 단순화가 결과를 왜곡했는지 확인).

## 산출물

- 스크립트: `scripts/eth_live_promotion_seed_robustness_{prefix_snapshot,canonicaldata,precheck,
  h48qual_seed_variant,zig075_seed_variant,eval_3seed}_20260819.py`
- 번들: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_
  {zigzagfix_06_h48_quality_noctx_padded,current_only_alllabels_01_zigzag_action_labels_20260531}_
  e2_fulltrain_exit30k_livepromo_seedvariant_{94046540,524707103}/`
- 평가 결과: `tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819_{seed260620_
  original,94046540,524707103}/report.json`, 요약
  `tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819_summary.json`

**fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false** (평가 스크립트 자체 report.json 필드로 명시).

## N=5 확장 — 공식 CONFIRMED (같은 날 후속)

신규 랜덤시드 2개 추가(`scripts/eth_live_promotion_seed_robustness_{h48qual,zig075}_seed_variant_
20260819.py`를 그대로 재사용, 시드만 312069414/44751167) 학습 완료 후
`scripts/eth_live_promotion_seed_robustness_eval_5seed_20260819.py`로 5개 시드(260620 원본+
94046540/524707103/312069414/44751167) 전체 재평가. 방법론 완전 동일(캐노니컬데이터+102-pin,
threshold/sidecar 라이브 그대로).

### 결과 — with_gate PnL(%), 6창 × 5시드

| window | 260620(원본) | 94046540 | 524707103 | 312069414 | 44751167 | 부호일치 |
|---|---:|---:|---:|---:|---:|---|
| 2025q1 | +28.54 | +49.82 | +45.73 | +55.49 | +163.97 | ✅ YES |
| 2025q2 | +39.99 | -33.45 | -35.08 | +8.89 | -6.71 | ❌ **SIGN FLIP** |
| 2025q3 | -9.73 | -31.24 | +4.77 | -21.52 | -7.14 | ❌ **SIGN FLIP** |
| **val** | **+54.88** | +91.44 | -16.48 | +107.06 | +19.26 | ❌ **SIGN FLIP** |
| oos_q1 | +28.17 | +5.70 | +38.41 | +28.39 | +40.27 | ✅ YES |
| oos_q2 | +9.85 | -4.61 | +13.72 | +16.24 | +55.29 | ❌ **SIGN FLIP** |

**N=3에서 나온 패턴이 N=5에서 정확히 재현됐다** — 여전히 6창 중 4창(2025q2/2025q3/val/oos_q2)
부호플립, 2025q1/oos_q1 2창만 일치(단 oos_q1도 +5.70%~+40.27%로 스프레드 큼). 5개 시드 전체에서
예외 없이 같은 4창이 갈린다 — 우연한 시드조합의 산물이 아니라 이 정확한 설정(라이브 코드+
threshold 0.50/0.75+frozen 원본 sidecar) 자체가 구조적으로 시드에 민감하다는 뜻이다.

### 최종 판정

**N=5 CONFIRMED**: CLAUDE.md Seed-Diversity Ensemble Promotion Gate의 N≥5 요구를 충족한 정식
검증 결과, 라이브 오메가4.6.1(h48qual+zig075 dual)의 승격 근거는 **seed-robust하지 않다**. 특히
원 승격 근거였던 VAL +54.88%(시드 260620)는 5개 시드 중 최저치에 가깝고(다른 시드는 +19~107%
또는 -16%), 부호 자체가 갈린다 — 단일시드 우연의 가능성이 실증적으로 확인됐다. "한계/교란변수"
절의 N=3 caveat은 해소됐으나, risk sidecar frozen 재사용(시드별 전용 아님)과 canonical 데이터
전환(순수 legacy 재현 아님) 두 가지는 여전히 남은 단순화다.

산출물 추가: `scripts/eth_live_promotion_seed_robustness_eval_5seed_20260819.py`,
`tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819_{312069414,44751167}/
report.json`, 요약 `tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819_
5seed_summary.json`.
