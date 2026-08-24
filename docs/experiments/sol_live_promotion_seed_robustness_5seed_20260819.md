# SOL 라이브 승격(zig075 v2, adaptive_squeeze) 자체의 시드 강건성 — N=5 검증 (2026-08-19)

## 배경 / 질문

같은 세션에서 ETH dual(h48qual+zig075)의 시드 강건성 예비검증(N=3, `docs/experiments/
eth_live_promotion_seed_robustness_3seed_20260819.md`)을 완료한 뒤, 동일 질문을 SOL 라이브
모델(zig075 단일 컴포넌트, adaptive_squeeze v2, `docs/model_contracts/
sol_adaptive_squeeze_v2_20260720.md`)에 적용했다. SOL 라이브 승격 근거도 ETH와 마찬가지로
**단일 시드**뿐이었고(2026-07-20 학습, 시드값 자체는 report.json에 기록되지 않음 — 이 스크립트들의
report 구조 자체가 seed를 기록하지 않는 공통 한계), CLAUDE.md의 Seed-Diversity Ensemble
Promotion Gate(N≥5 진짜 랜덤시드)를 충족한 적이 없었다.

**목표**: "실제 라이브를 학습시킨 정확한 코드"로 신규 랜덤시드 4개를 재학습(기존 라이브 번들 =
1번째 시드로 재사용, 합쳐서 N=5)한 뒤 Fresh-Forward 평가로 VAL/OOS 부호가 시드에 걸쳐
일관되는지 확인 — ETH와 달리 N=5를 예비검증 없이 바로 정식 규모로 수행했다(CLAUDE.md 게이트
문턱을 첫 시도에서 충족).

## 방법론

### 코드 스냅샷

`git show HEAD:scripts/train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py`로
추출한 스냅샷(`scripts/sol_live_promotion_seed_robustness_prefix_snapshot_20260819.py`) — 워킹
트리의 그 파일은 2026-08-18 exit_head pos_tp/pos_sl/risk_margin/risk_leverage 버그수정이
**미커밋 상태**로 적용돼 있어(`git diff HEAD`로 확인, 89줄 추가/14줄 삭제), 이 스냅샷이 실제
라이브 번들(`tmp/causal_regen_20260516/
sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720/
true_3head_tabm_bundle.pt`)을 만들어낸 정확한 코드다. 워킹트리 dirty 파일은 전혀 건드리지 않았다.

### 설정 (report.json 대조)

`quality_mode=same_as_direction`, `quality_thresholds=0.40..0.75`(8개, q070 포함),
`exit_label.mode=entry_label_terminal_giveback`(SOL HEAD 스냅샷의 유일한 choice이자 기본값),
`epochs=4`, `max_exit_samples=12000`(report.json `exit_label.diag.rows=12000`과 실측 대조
확인 — ETH 세션의 epochs=2/max_exit_samples=30000 오버라이드와 다름, SOL 원본은 두 값 모두
argparse 기본값을 그대로 썼다), `direction_label_dir`도 기본값(`tmp/causal_regen_20260516/
sol_zigzag_action_labels_20260707`). 데이터는 `data/splits/year_oos_adaptive_squeeze_sol_20260720/
sol_features_{2025,2026}.csv`(adaptive_squeeze 재구축, 기존 committed 래퍼
`scripts/train_eval_omega4_3head_parent72_loose_entry_quality_sol_adaptive_squeeze_20260720.py`와
동일 오버라이드). Risk sidecar는 5개 시드 전부 원본(`sol_omega4_2_trade_risk_sidecar_20260707_
adaptive_squeeze_q070_20260720`) 것을 frozen 재사용 — ETH와 동일한 단순화(범위 밖, 명시적
caveat). 라이브 설정(`trading_bot_modules/runtime_config.py`
`OMEGA4_6_1_SHADOW_ASSET_CONFIG["sol"]`)에서 확인: `quality_threshold=0.70`,
`scale_map={zig075_L:1.0, zig075_S:1.75}`, `duration_threshold=0.0055208323`,
`exit_threshold=0.95`(risk sidecar 자신의 report.json `contract.exit_threshold`).

### 시드

`seed1_live_original`(기존 라이브 번들 그대로 재사용, 재학습 안 함) + 신규 랜덤 4개
(`random.SystemRandom().sample`, 848498120/732130789/193749676/534479280).

## SOL이 ETH와 달랐던 지점 (사전 지시대로 직접 실측)

1. **피쳐드리프트: 없었다.** ETH는 원본 코드를 오늘 그대로 재실행하면 자동유도 base_cols가
   102→179개로 부풀고, legacy CSV엔 원본 102개 중 7개가 아예 없어 canonical 데이터 전환까지
   필요했다. SOL은 정반대였다 — precheck(`scripts/sol_live_promotion_seed_robustness_
   precheck_20260819.py`) 실측 결과 오늘(2026-08-19) 시점 adaptive_squeeze 데이터에서
   자동유도 피쳐 개수가 **정확히 147개**(라이브 번들 자신의 base_cols 개수와 동일), missing=0.
   1개월 간 피쳐 엔지니어링 누적이 SOL 쪽엔 반영되지 않은 것으로 보인다.
2. **pin 메커니즘이 이미 내장돼 있었다.** SOL의 HEAD 스냅샷 학습스크립트는
   `--base-feature-contract-bundle` CLI 옵션을 자체적으로 갖고 있어(지정된 번들의 base_cols로
   자동유도 결과를 pin, 부족하면 RuntimeError), ETH처럼 `omega._numeric_feature_cols`를 직접
   monkey-patch할 필요가 없었다. 위 1번과 맞물려 이번엔 이 pin이 사실상 무해한
   확인사살이었지만(missing=0), 메커니즘 자체는 그대로 활용했다.
3. **⚠️ 예상 못한 데이터 결측을 새로 발견했다: 공유 regime3 HMM 오버레이 파일 자체가 없었다.**
   `data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707/`에 2025년
   오버레이 CSV가 아예 없고, 2026년 것도 `.bak_pre_extend_20260721` 백업만 남아있었다(dev/
   server 둘 다 동일하게 실측 확인). 이를 만든 스크립트(`scripts/
   extend_regime3_wide24_sol_btc_20260721.py`)를 찾아, SOL 분기만 별도 scratch 경로에
   재현했다(`scripts/sol_live_promotion_seed_robustness_canonicaldata_20260819.py`) — 공유
   정식 경로나 BTC 데이터는 건드리지 않았다. HMM이 소비하는 `WIDE24_EXTRA_COLS`는 funding/
   squeeze 피쳐와 무관한 별도 피쳐군이라(`scripts/
   experiment_regime3_current_hmm_wide24_20260529.py` 확인), legacy 소스로 재계산해도
   adaptive_squeeze 수정과 무관하게 라이브가 실제 학습에 쓴 값과 같다.
4. **동시성**: 애초 "SOL은 새 파일을 안 만들어 ETH의 동시쓰기 경합이 구조적으로 없다"고
   예상했으나, 3번 발견 때문에 결국 새 파일(regime3 오버레이 2개)을 만들게 됐고 ETH와 동일한
   `os.replace()` 원자적 rename이 다시 필요해졌다. 4개 시드 학습을 병렬로 띄웠고 전부 성공 —
   `os.replace()`가 사전에 이 경합을 막았다(idempotent 재사용 체크: 이미 존재하면 재계산 생략).
5. **Duration gate 컨벤션 불일치를 발견**: `docs/model_contracts/
   sol_adaptive_squeeze_v2_20260720.md`는 v1/v2 비교를 "gate off"로 서술하지만,
   `trading_bot_modules/runtime_config.py`의 오늘 시점 기본값
   (`FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_DURATION_GATE_OFF=False`)은 실제 duration_threshold
   (0.0055208323)를 적용한다(`trading_bot.py:11327-11331`). 그 문서의 "gate off"는 그 문서
   자신의 비교 방법론 서술이지 오늘의 라이브 기본값이 아닌 것으로 보인다 — 이 불일치를
   해소하지 않고 이번 평가는 `no_gate`/`with_gate` 둘 다 계산해 명시적으로 남겼다.
6. **재사용 가능한 SOL 전용 replay 엔진을 발견**: `scripts/
   eval_sol_dual_structure_router_20260729.py`(SOL zig075/h24wide dual 라우터 후보 평가에
   실제 쓰였던 스크립트)가 SOL 전용 `_prepare_frames`/`prepare_component`/SPLIT_TS(2025-09-01)/
   VAL_END(2026-01-01)/OOS_END(2026-04-01) 상수를 이미 CLAUDE.md 기본값과 일치하게 갖고
   있었다. 단, 그 스크립트 자신의 `dual_replay()`는 **scale_map(롱숏 비대칭 레버리지)을 전혀
   적용하지 않는다**(자체 `replay_variant()`가 `greedy.SCALE_MAP`을 전부 1.0으로 중립화하고
   별도 risk_scale/regime_margin_scale 그리드서치를 쓰는 구조탐색 축 설계) — 실제 라이브
   SOL 설정(`scale_map={zig075_L:1.0, zig075_S:1.75}`)과 다르다. 그래서 실제 bar-by-bar
   walk는 `dual_replay` 대신 `replay_omega4_6_1_greedy_router_20260706.py::greedy_replay`를
   썼다(SCALE_MAP을 실제로 적용하는 쪽) — `prepare_component`만 그 라우터 스크립트에서 그대로
   재사용(SOL 전용 sidecar 모듈을 이미 올바르게 참조하고 있어 재구현 불필요).

## 결과 — 5시드 × 3창(VAL, OOS-Q1, OOS-Q2)

CLAUDE.md 필수 2창(VAL 2025-09-01~12-31, OOS 2026-01-01~03-31)에 보너스 1창(OOS-Q2
2026-04-01~06-30, adaptive_squeeze 데이터가 2026-07-21까지 커버해 가능)을 추가했다. ETH의
6창(2025q1/q2/q3 포함)은 만들지 않았다 — `parent.SPLIT_TS=2025-10-01` 기준 2025 Q1-Q3는
train_raw(인샘플)에 들어가 있어 ETH의 그 3창도 사실은 "OOS형" 진단이 아니고, SOL 전용
WINDOW_DEFS 인프라도 없어 새로 만드는 대신 이미 검증된 2창 관례 + 데이터가 허용하는 보너스
1창으로 범위를 좁혔다.

### no_gate (duration gate 미적용) PnL(%)

| window | seed1(라이브 원본) | seed848498120 | seed732130789 | seed193749676 | seed534479280 | 부호일치 |
|---|---:|---:|---:|---:|---:|---|
| val | -27.41 | +1.38 | -8.05 | +21.28 | **+49.36** | ❌ **SIGN FLIP** |
| oos_q1 | **+46.92** | +22.51 | -29.23 | -9.77 | -26.77 | ❌ **SIGN FLIP** |
| oos_q2 | -23.86 | -20.08 | +24.64 | +12.05 | -11.35 | ❌ **SIGN FLIP** |

### with_gate (duration_threshold=0.0055208323 적용) PnL(%)

| window | seed1(라이브 원본) | seed848498120 | seed732130789 | seed193749676 | seed534479280 | 부호일치 |
|---|---:|---:|---:|---:|---:|---|
| val | -16.16 | +20.56 | -16.24 | +27.27 | **+41.54** | ❌ **SIGN FLIP** |
| oos_q1 | +29.62 | **+44.50** | -12.97 | +3.78 | -22.24 | ❌ **SIGN FLIP** |
| oos_q2 | -25.85 | -23.82 | +22.48 | -13.48 | -15.18 | ❌ **SIGN FLIP** |

**3창 전부(no_gate/with_gate 둘 다) 부호가 시드에 따라 뒤집힌다.** ETH의 N=3 예비검증(6창 중
4창 플립)보다 더 심하다 — SOL은 **테스트한 모든 창에서 부호가 시드 노이즈에 좌우된다**, N=5로
CLAUDE.md 정식 게이트 문턱을 충족한 결과다(예비검증이 아니라 최종판정).

크기 편차도 크다: no_gate 기준 val은 -27.41%~+49.36%(76.77pp 스프레드), oos_q1은
-29.23%~+46.92%(76.15pp), oos_q2는 -23.86%~+24.64%(48.50pp). 5시드 표준편차는 창별로
17.5~29.6pp — 어떤 실제 신호보다 시드 분산 자체가 압도적으로 크다.

### 신규 4시드만 따로 봐도(원본 제외) 부호가 안 갈린다

"원본(seed1)이 우연히 운 좋았던 이상치일 뿐, 신규시드끼리는 일관될 것"이라는 가설도 확인해
기각했다 — 신규 4시드만의 부호 분포(no_gate): val=[+,-,+,+], oos_q1=[+,-,-,-],
oos_q2=[-,+,+,-]. 3-1, 1-3, 2-2로 매번 갈린다. 즉 원본 시드를 빼도 신규시드끼리 조차 어느
창에서도 만장일치가 없다 — 문제는 "원본이 특이했다"가 아니라 **이 학습 레시피/데이터 크기
자체가 시드 노이즈에 압도된다**는 것이다.

## 한계 / 교란변수

- **risk sidecar 고정**: 5개 시드 전부 원본 sidecar를 frozen 재사용 — 시드별 전용 sidecar였다면
  margin/leverage 값 자체가 달라져 결과가 바뀔 수 있음(마진값은 실제 sidecar 예측치이지 상수는
  아니지만, sidecar 자체가 원본 번들 예측 분포에 맞춰 학습됐다는 의미의 단순화).
- **Duration gate 컨벤션 불일치 미해소**: `sol_adaptive_squeeze_v2_20260720.md`(gate off
  서술) vs `runtime_config.py` 오늘 기본값(gate on, threshold=0.0055208323) 중 어느 쪽이
  실제 배포 시점의 진짜 라이브 동작인지 이번 세션에서 재확인하지 않았다 — 그래서 두 값 다
  계산했고, 결론(3/3 부호 플립)은 **어느 쪽을 봐도 동일**하다(표 참고).
  `FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE=False`가 기본값이라 이 불일치가
  실제 주문에는 영향을 준 적이 없다는 점도 확인했다(`runtime_config.py` 직접 대조).
- **절대 수치는 승격 당시 문서와 비교 불가**: 이 평가는 VAL 시작일(2025-09-01, 라이브 번들
  자신의 train/val split 2025-10-01보다 이름) / ATR-adaptive+sidecar 기반 greedy_replay를
  쓴다 — `sol_adaptive_squeeze_v2_20260720.md`가 인용한 원래 승격시점 수치(VAL +16.75%/OOS
  +57.94%, gate off)는 다른 스크립트/다른 윈도우 경계로 만들어져 직접 비교 대상이 아니다.
  이번 평가는 **5시드 전부에 동일하게 적용된 내부일관 방법론**으로 상대비교(부호일치 여부)를
  묻는 것이지, 절대 PnL을 원래 승격 수치와 맞춰보는 것이 목적이 아니다.
  `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`,
  `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false` (평가
  스크립트 자체 report.json 필드로 명시).
- **N=5는 5개뿐**: CLAUDE.md 게이트 문턱은 충족하지만, 더 큰 N에서 우연히 부호가 맞아떨어질
  가능성 자체를 배제하진 못한다(다만 그럴 경우 그 신규 다수결도 원 승격 근거였던 seed1 자체와
  일치하리라는 보장은 없다는 점은 이미 위 결과가 시사한다).

## 결론 / 제안

N=5 정식 규모에서 **테스트한 3개 창 전부, no_gate/with_gate 두 컨벤션 전부**가 시드에 따라
부호를 뒤집는다 — ETH(N=3, 6창 중 4창 플립)보다도 명확한 결과다. 원 라이브 승격의 유일한
근거였던 단일 시드(2026-07-20 학습)가 우연이었을 가능성을 사실상 배제할 수 없는 수준을 넘어,
**신규 4시드끼리도 서로 일치하지 않는다** — 이 특정 학습 레시피(zig075 v2, 8주치
zigzag_action + terminal_giveback exit 라벨, 147피쳐, 시드당 트레이드 22~53개, 승률
23.5~47.8%)가 구조적으로 시드 노이즈에 압도된다는 뜻이다.

권고:
1. **현재 라이브 SOL zig075 v2 포지션의 리스크 익스포저 재검토** — 현재 실주문은
   `FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE=False`로 차단된 decision-only
   상태(오늘 `runtime_config.py`로 재확인)지만, 이 플래그가 바뀌는 시점 전에 이 발견이
   검토돼야 한다.
2. **트레이드 수를 늘리는 방향(더 긴 학습기간/더 낮은 quality threshold)이나 앙상블
   접근(N개 시드 평균)이 이 노이즈를 줄이는지 별도 검증** — 단, CLAUDE.md의
   Seed-Diversity Ensemble Promotion Gate가 요구하는 "N≥5 진짜 다양한 시드"로 그 앙상블
   자체도 재검증해야 한다(단순 배깅이 VAL에서만 노이즈를 지우고 OOS에서 재현 안 되는 패턴은
   Sigma3-1h 선례로 이미 경고됨).
3. risk sidecar까지 시드별로 재학습한 재검증(현재 frozen 단순화가 결과를 왜곡했는지 확인) —
   ETH 쪽과 동일하게 아직 미해결.

## 산출물

- 스크립트: `scripts/sol_live_promotion_seed_robustness_{prefix_snapshot,canonicaldata,
  precheck,zig075_seed_variant,eval_5seed}_20260819.py`
- 재생성된 regime3 오버레이(scratch, 공유 경로 아님):
  `tmp/causal_regen_20260516/sol_live_promotion_seed_robustness_20260819/
  regime3_overlay_rebuild/sol_features_{2025,2026}_regime3_current_sensitive_hmm_wide24.csv`
- 신규 시드 번들: `tmp/causal_regen_20260516/
  sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_seedvariant_
  {848498120,732130789,193749676,534479280}/true_3head_tabm_bundle.pt`
- 평가 결과: `tmp/causal_regen_20260516/sol_live_promotion_seed_robustness_20260819/
  summary_report.json`, 시드×창별 원장 `portfolio_ledger_<seed>_<window>.csv`

**fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false** (평가 스크립트
자체 report.json 필드로 명시).
