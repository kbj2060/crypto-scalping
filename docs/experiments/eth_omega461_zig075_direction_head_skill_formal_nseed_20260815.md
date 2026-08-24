# ETH zig075 direction_head 스킬 Stage 2: N=5 진짜 다양시드 formal 재학습 (2026-08-15)

## 배경

`docs/experiments/eth_omega461_zig075_direction_head_skill_stage1_prelim_20260815.md`(Stage 1)가
이미 배포된 zig075 번들(단일 시드, N=1) 하나의 저장 예측만으로 ungated `direction_head` argmax
vs always_short/always_long을 비교했다: VAL에서는 ungated이 둘 다 이겼지만(+20.30 vs
always_short +10.38), OOS에서는 always_short에 졌다(-2.34 vs +17.11). N=1이라 CLAUDE.md
**Seed-Diversity Ensemble Promotion Gate**(N≥5 진짜 다양시드, OOS 부호 일치 확인, 시드 리스트
기록 의무) 상 노이즈와 신호를 구분할 수 없어 사용자가 이 Stage 2(N≥5 진짜 다양시드 formal
재학습)를 승인했다. 이 문서가 그 결과다.

## 레시피 식별

1. `trading_bot_modules/runtime_config.py`의 `FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH` →
   `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt`
   (`omega4_6_1_live.py:290`에서 `quality_threshold=0.75`로 인스턴스화, ETH 라이브
   `PRIORITY=("h48qual","zig075")` confluence의 두 번째 컴포넌트). 이 매핑은
   `docs/model_contracts/omega4_6_plus_t12_nohold_risk1_20260630_contract.md`의 컴포넌트 표와도
   일치.
2. 번들의 `report.json`을 읽어 `model_id=omega4_3head_parent72_loose_entry_quality_20260620`,
   `label_contract.direction_label_dir=.../omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531`,
   `quality_mode=same_as_direction`, `exit_label.mode=entry_label_terminal_giveback`(터미널 윈도우
   3, adverse -0.01, giveback_min 0.65, min_mfe_for_giveback 0.006 — 전부 스크립트 기본값),
   `epochs_ran=2`(bull/bear/chop 각), `input_contract.total_features=115`
   (`base_feature_count=102` + `position_feature_count=13`)를 확인.
3. 학습기 본체는 `scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`
   (기본 시드 260620, `--epochs`/`--direction-label-dir`/`--quality-mode`/`--exit-label-mode`/
   `--max-exit-samples`/`--max-train-rows`/`--quality-thresholds`를 CLI로 받음). `report.json`에
   시드 필드가 직접 없지만(체크포인트 payload에도 seed 키 없음), model_id에 시드 접미사가 없는
   점과 이 스크립트의 CLI 기본값(260620)이 저장소 전반의 명명 관례와 일치해 배포 번들은 기본
   시드로 학습된 것으로 판단.
4. **함정 발견 (본 세션)**: 위 학습기는 `omega._numeric_feature_cols(train_all, eval_df)`로
   feature 목록을 **런타임에 자동 탐지**한다. 06-29 당시엔 102개였지만, 그 사이 여러 리서치
   세션이 후보 CSV에 새 feature(cmamba/AI-패턴 컬럼 등)를 추가해서, 오늘 같은 인자로 그대로
   재실행하면 172개 base feature(115→185 total)를 잡는다 — `scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py`의
   docstring이 2026-07-27에 이미 이 정확한 버그를 문서화해뒀다(그때는 h48qual/exit-head 재학습
   맥락). 이 스크립트는 배포된 zig075 번들의 `base_cols`(102개, live 순서 그대로)에
   `--pin-component zig075`로 고정하고, 2025 학습 프레임에서 사라진 7개 컬럼(`fibonacci_level`,
   `funding_roc_12/48`, `funding_z_score`, `short_squeeze_risk`, `hurst_288`,
   `regime_persistence`)을 `data/splits/year_oos/training_features_2025.csv`에서 timestamp 조인으로
   복구한다. 아키텍처/피쳐/라벨/하이퍼파라미터/epoch를 전부 고정하고 시드만 바꾸려면 이 wrapper가
   필수라고 판단해 채택. (첫 시도 946043153을 pin 없이 먼저 돌렸다가 172-feature 오염을 발견하고
   즉시 폐기·재실행 — 로그는
   `tmp/eth_zig075_direction_head_formal_nseed_20260815/DISCARDED_unpinned172feat_parent_seed946043153.log`에
   증거로 남겨뒀다.)

**최종 재학습 커맨드** (5개 시드 각각에 대해 `--seed`만 교체):
```
python scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py \
  --pin-component zig075 \
  --epochs 2 --quality-mode same_as_direction \
  --direction-label-dir tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531 \
  --quality-thresholds 0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 \
  --max-exit-samples 30000 --max-train-rows 0 \
  --exit-label-mode entry_label_terminal_giveback \
  --out-suffix pinned102_zig075_formal5seed_20260815_seed<SEED> \
  --device cpu --seed <SEED>
```
드라이버: `scripts/run_zig075_direction_head_formal_5seed_dev_20260815.sh`. 재학습 후 각 번들에서
`input_contract.total_features=115`/`base_feature_count=102`를 재확인해 피쳐 오염이 없음을
검증했다.

## 서버 용량 체크 및 dev/server 분배 결정

작업 시작 전 `scripts/ops/handoff.sh status server`와 직접 ssh로 서버 상태를 확인:
`free -h` 가용 23GB/31GB, `nproc`=12, `uptime` load average 0.08/0.12/0.15 — 선례
(23GB free, 12 core, load~0.3)와 동등하거나 더 여유. `trading_bot.py`(197분 누적 CPU, 살아있음),
BTC multislot shadow loop ×2, `live_eth_regime_aware_exit_guard_shadow_20260814.py`,
`live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py` 전부 정상 구동 중 확인.

**단일 시드 실측 타이밍으로 견적 재조정**: 같은 zig075 parent 레시피(epochs=2, JM 레짐3
버전이지만 학습 비용은 동일 아키텍처)를 5-시드로 이미 돌린 선례
(`tmp/jm_full_retrain_seed_robustness_20260813/zig075_parent_seed*.log`, 2026-08-13)를 찾았고,
그 로그의 시작/종료 타임스탬프를 보면 시드당 **약 3~4.5분**(CPU, dev와 동일 스펙)이었다. 이는
지시문이 예시로 든 exit-head-only 재학습(55~96분/fold, 인코더 동결)과는 비교가 안 되게 가볍다
— 이번 재학습은 인코더+3-헤드 전체를 처음부터 학습하지만 epoch 수가 2로 매우 작고
`max_train_rows=0`(무제한이지만 실제로도 183,936행 규모)이라 fold 학습보다 훨씬 짧다. 실측으로
현재 재학습(pinned102 wrapper, feature 복구 조인 포함)도 시드당 **3.2분**으로 확인됐다(아래 표).

이 견적에 따라 총 5-시드 wall-clock이 ~16분으로 예상돼 **서버 분배 없이 dev 단독 순차 실행**을
선택했다 — dev는 12코어/13GB 여유/load 0.3로 유휴 상태였고, 라이브 트레이딩 서버에 손대지 않는
쪽이 더 안전하며, 분배해도 wall-clock 이득이 사실상 없었다(작업 자체가 이미 충분히 짧음). 서버는
1건도 큐에 넣지 않았다 — "한 번에 최대 1건" 정책과도 충돌 없음(0건이므로).

**실측 총 소요**: 5-시드 순차 실행 13:04:43~13:20:45(KST), 총 **16분 2초**, 시드당 평균 3분 12초.

## 시드 5개 생성

`random.SystemRandom().sample(range(1, 1_000_000_000), 5)` (OS 엔트로피, 고정 증분 아님) —
`eth_omega461_live_jm_full_retrain_seed_robustness_20260813.md`가 확립한 동일 방식. 생성된 5개:

**946043153, 932925759, 74851798, 975176982, 542143953**

(참고용 6번째 지점: 배포 번들 자체의 시드로 추정되는 260620 — Stage 1 결과가 이 시드에 해당,
Seed-Diversity Gate의 N=5 카운트에는 포함하지 않음.)

## 평가 방법

`scripts/diagnose_eth_zig075_ungated_direction_vs_always_short_20260815.py`에 `--bundle-dir`/
`--out-csv` 인자를 추가해 일반화(로직 변경 없음 — 기존 하드코딩 `BUNDLE_DIR`를 argparse 기본값으로
옮긴 것뿐). 인자 없이 실행하면 Stage 1과 정확히 같은 결과(VAL +20.30/-2.34 OOS)를 재현하는 것으로
회귀 확인 완료. `quality_threshold`는 완전히 무시(`direction_head` 원본 argmax `dir_action`만
사용), `always_short`/`always_long`은 같은 ungated active set에서 방향만 강제. TP/SL/비용
컨벤션은 Stage 1과 동일(`cost_mult=3.0`, `max_hold=0`/`cooldown=0`). 구간은 저장소의 zig075/h48qual
라이브 번들 진단 스크립트 전체가 공유하는 기존 관행대로 VAL=2025-10-01~12-31,
OOS=2026-01-01~03-31(`trade_candidates_2026_alpha6_current_tail111_exact.csv`).

## 결과 (5 시드 × VAL/OOS, cherry-picking 없음)

| seed | split | ungated pnl | trades | wr | always_short pnl | trades | wr | always_long pnl | trades | wr | ungated이 always_short 이김? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 946043153 | VAL | -10.86 | 45 | 28.9% | +13.25 | 37 | 43.2% | -15.61 | 42 | 28.6% | **NO** |
| 946043153 | OOS | +9.22 | 25 | 48.0% | +22.71 | 22 | 59.1% | -19.64 | 29 | 20.7% | **NO** |
| 932925759 | VAL | -13.43 | 45 | 28.9% | +11.63 | 39 | 43.6% | -12.16 | 46 | 30.4% | **NO** |
| 932925759 | OOS | -3.65 | 25 | 32.0% | +21.56 | 25 | 56.0% | -15.99 | 29 | 24.1% | **NO** |
| 74851798 | VAL | +6.44 | 42 | 40.5% | +9.48 | 40 | 42.5% | -16.20 | 45 | 28.9% | **NO** |
| 74851798 | OOS | +11.45 | 27 | 48.1% | +18.42 | 27 | 51.9% | -18.94 | 31 | 22.6% | **NO** |
| 975176982 | VAL | -10.19 | 36 | 27.8% | +12.50 | 35 | 42.9% | -16.34 | 42 | 26.2% | **NO** |
| 975176982 | OOS | +7.92 | 24 | 45.8% | +20.17 | 23 | 56.5% | -17.22 | 30 | 23.3% | **NO** |
| 542143953 | VAL | -11.69 | 43 | 30.2% | +13.20 | 35 | 42.9% | -16.69 | 43 | 27.9% | **NO** |
| 542143953 | OOS | -0.72 | 26 | 34.6% | +17.11 | 25 | 52.0% | -13.11 | 28 | 25.0% | **NO** |

(참고, Seed-Diversity Gate N=5 카운트 외: 배포 번들 시드=Stage 1 결과 — VAL ungated +20.30 이김,
OOS -2.34 짐.)

**10칸 중 10칸(5시드 × VAL/OOS 전부) ungated direction_head가 always_short에 완패.** Stage 1의
단일 시드(260620)만 유일하게 VAL에서 이겼던 것과 달리, 새로 뽑은 5개 진짜 다양시드는 VAL/OOS
가리지 않고 전부 always_short에 진다 — VAL에서도 3/5는 큰 폭 손실(-10~-13), 나머지 2/5도
always_short보다 작다.

## 해석 및 formal 검증

Seed-Diversity Ensemble Promotion Gate: "N≥5개의 진짜 다양한 시드... OOS 부호 일치를 보여야
한다." 5개 신규 시드 중 **OOS에서 always_short를 이긴 경우는 0/5**, VAL까지 포함해도
**10칸 중 0칸**이 ungated 승리다. 이는 h48qual의 formal 테스트가 보였던 완패 패턴(40칸 중 2칸,
전부 유의하지 않음)과 사실상 동일한 결론이며, 오히려 이번 zig075 결과는 h48qual보다도 더
일관되게(2칸조차 없이) 진다.

Stage 1의 단일 배포-시드 VAL 승리(+20.30)는 이번 5-시드 재확인으로 **시드 노이즈였음이
formal하게 확인됐다** — 배포 시드 하나가 우연히 VAL 구간에서 좋았을 뿐, 재학습된 5개 독립 시드
중 어느 것도 그 패턴을 재현하지 못했다.

## 최종 판정: **REJECTED**

**zig075의 `direction_head`는 entry-side 스킬이 confirmed되지 않는다.** h48qual과 동일한 결론 —
N≥5 진짜 다양시드 중 강한 다수(이 경우 전부)가 OOS에서 always_short에 지므로, Seed-Diversity
Ensemble Promotion Gate 기준 CONFIRMED 판정은 성립하지 않는다. zig075도 h48qual과 같은
"direction_head에 방향 스킬 없음" 벽에 부딪힌 것으로 판단한다.

## Fresh-Forward 체크리스트

`fresh_forward_bar_by_bar=true`(고정 VAL/OOS 구간을 5분봉 단위 causal 예측 → TP/SL/time-exit
시뮬레이션, `omega._to_fixed_decisions`/`omega._metrics`가 매 bar 인과적으로 확정),
`trade_ledgers_used_as_input=false`(저장 원장을 입력으로 쓰지 않음 — 이번에 생성된 예측 CSV는
전부 새 재학습의 자체 출력), `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`. 라이브 파일(`trading_bot_modules/omega4_6_1_live.py`,
`trading_bot.py`, `runtime_config.py`, `.env`)은 전혀 건드리지 않았다 — `git status`로 확인
완료, 이 세션이 수정한 것은 연구용 스크립트 3개(`scripts/diagnose_eth_zig075_ungated_direction_vs_always_short_20260815.py`
일반화, `scripts/run_zig075_direction_head_formal_5seed_dev_20260815.sh` 신규)와 이 문서뿐이다.
서버에는 어떤 파일도 push/수정하지 않았다(상태 조회만 ssh로 수행).

## 산출물

- 재학습 드라이버: `scripts/run_zig075_direction_head_formal_5seed_dev_20260815.sh`
- 재학습 베이스 스크립트(변경 없음, 재사용): `scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py`
  (`--pin-component zig075`), 내부적으로 `scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py` 호출
- 평가 스크립트(일반화): `scripts/diagnose_eth_zig075_ungated_direction_vs_always_short_20260815.py`
- 5개 번들: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_zig075_formal5seed_20260815_seed{946043153,932925759,74851798,975176982,542143953}/`
- 학습 로그: `tmp/eth_zig075_direction_head_formal_nseed_20260815/pinned102_parent_seed<SEED>.log`,
  드라이버 로그 `tmp/eth_zig075_direction_head_formal_nseed_20260815/driver.log`
- 시드별 진단 결과 CSV: `tmp/eth_zig075_direction_head_formal_nseed_20260815/diag_out/ungated_vs_always_short_seed<SEED>.csv`
- 폐기된 첫 시도(feature 오염) 증거: `tmp/eth_zig075_direction_head_formal_nseed_20260815/DISCARDED_unpinned172feat_parent_seed946043153.log`

## 다음 단계

이 문서는 verdict만 확정한다 — `docs/model_contracts/`의 zig075/Odyssey 관련 계약 문서 갱신은
사용자가 직접 처리(작업 지시 원문에 따라 이 세션은 contract 문서를 스스로 편집하지 않음).
