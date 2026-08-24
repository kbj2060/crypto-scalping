# BTC h48qual+swingtransition 라이브 승격 자체의 시드 강건성 — N=5 정식검증 (2026-08-19)

## 배경 / 목표

같은 날 먼저 진행한 ETH dual(h48qual+zig075) 시드 강건성 검증(`docs/experiments/
eth_live_promotion_seed_robustness_3seed_20260819.md`, 이후 N=5로 확장돼 최종 CONFIRMED 판정—
6창 중 4창 부호플립, VAL 자체도 플립)에서 확립한 기법을 BTC 라이브 parent 모델(h48qual+
swingtransition, 실주문은 `BINANCE_ACCOUNT_ENABLED=False`로 차단돼 있지만 결정은 실시간 계산 중)
에 그대로 적용한다. BTC 승격 시점(2026-08-06 학습, 2026-08-07 라이브 배포로 추정)도 CLAUDE.md의
Seed-Diversity Ensemble Promotion Gate 정책(2026-08-01 도입) 도입 전후 관계가 불명확하지만, 어느
쪽이든 **N≥5 시드 검증 자체가 지금까지 한 번도 없었다**는 사실은 확실하다. 이 세션에서 신규 랜덤시드
4개를 학습해 기존 라이브 번들(시드 1개)과 합쳐 N=5로 정식 검증했다.

## 방법론

**목표**: "실제 라이브를 학습시킨 정확한 코드/설정"을 다른 시드로 재현해, VAL/OOS 부호가 시드에
걸쳐 일관되는지 확인 — ETH와 동일한 질문, 동일한 기법 계보.

1. **코드**: `git show HEAD:scripts/train_eval_omega4_3head_parent72_loose_entry_quality_btc_
   swingtransition_20260806.py`로 추출한 스냅샷(`scripts/btc_live_promotion_seed_robustness_
   prefix_snapshot_20260819.py`) — 워킹트리의 그 파일은 2026-08-18 exit_head pos_tp/pos_sl/
   pos_notional 버그수정이 **미커밋 상태**로 적용돼 있어(`git diff HEAD`로 확인, ETH 원본과 동일한
   패턴의 수정), 이 스냅샷이 실제 라이브 번들을 만든 정확한 코드다. 서버(`llewyn@...`)의 워킹트리도
   동일 파일이 동일 diff로 dirty함을 직접 확인(diffstat 라인 수까지 일치) — dev/server 어느 쪽
   기준으로도 이 스냅샷이 맞다.
2. **설정**: report.json 및 argparse 기본값에서 확인한 라이브 그대로 —
   `direction-label-dir=tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708`,
   `quality-mode=quality_label_action`, `quality-label-dir=tmp/causal_regen_20260516/
   btc_h48_conservative_padded_to_zigzag_timestamps_20260708`, `quality-threshold=0.55`(q055).
   epochs=4/quality-thresholds="0.40,...,0.60"(5개)/max-exit-samples=12000/max-train-rows=30000/
   cost-mult=3.0/exit-label 관련 전부는 스크립트 자체 argparse 기본값 그대로 사용 — report.json의
   간접증거(`exit_label.diag.rows=12000`, `summaries.*.epochs_ran=4`, 조기중단 흔적 없음,
   `risk_sidecar_precomputed_prediction_tag_values=[q040,q045,q050,q055,q060]`가 기본
   threshold grid와 정확히 일치)로 재확인했다. report.json에 원본 argv 자체가 저장돼 있지 않아
   100% 확정은 아니라는 caveat은 ETH와 동일하게 남는다. Risk sidecar는 5개 시드 전부 원본
   (`btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260806_swingtransition/
   risk_sidecar.pkl`) frozen 재사용(ETH/일리아스1과 동일한 명시적 단순화).
3. **시드**: 260620(기존 라이브 번들 그대로 재사용, 재학습 안 함 — report.json에 seed 필드가
   없어 argparse 기본값 260620으로 가정, 이를 벗어날 근거를 찾지 못함) + 신규 랜덤 4개
   (`random.SystemRandom().sample(range(10_000_000,999_999_999), 4)`, 등간격 아님):
   750703416 / 160125165 / 626578270 / 179796523.
4. **평가**: BTC는 dual-component 라우터(h48qual+zig075)가 아니라 단일 컴포넌트(h48qual,
   swingtransition은 학습피쳐 추가일 뿐 별도 컴포넌트 아님)이므로, ETH의 `greedy_replay` 대신
   BTC 자신의 검증된 단일-컴포넌트 리플레이 엔진 `train_eval_omega4_2_risk_sidecar_btc_20260708.
   py::_replay_with_risk`를 그대로 재사용했다 — 이건 실제 배포된 "HEADLINE" BTC 평가 스크립트
   (`apply_final_scale_map_btc_freshforward_ext_swingtransition_20260806.py`,
   `audit_live_models_fresh_forward_20260808.py`가 "BTC h48qual+swingtransition (promoted
   live)"의 HEADLINE으로 명시)가 실제로 쓰는 바로 그 함수다. BTC 전용 6창 multiwindow gate
   인프라는 없었으므로, ETH의 `eth_omega461_multiwindow_confirmation_gate_20260814.py::
   WINDOW_DEFS`와 동일한 날짜 경계(2025q1/q2/q3/val=2025-10-01~12-31/oos_q1=2026-01-01~03-31/
   oos_q2=2026-04-01~06-30)를 BTC 데이터에 그대로 적용해 직접 구성했다(`scripts/btc_live_
   promotion_seed_robustness_eval_5seed_20260819.py`). 진입 예측은 각 시드 학습 스크립트가
   자기 자신의 bundle로 이미 만들어둔 `{train,validation,oos}_predictions_q055.csv`(causal,
   저장 ledger 아님)를 그대로 슬라이싱해 사용 — `apply_final_scale_map_btc...py`의
   `_scaled_margin_leverage`/`_compound_metrics`/scale(0.5/2.5)/exit_threshold(0.95)/
   cost_mult(3.0)를 그대로 import해 재사용, 재구현하지 않았다.

### ETH 대비 BTC에서 실제로 달랐던 점 (실측 기반)

- **피쳐드리프트/pin: 불필요로 확정**. ETH는 legacy CSV의 7주치 피쳐엔지니어링 누적 드리프트 때문에
  canonicaldata 오버라이드 + 102-pin이 필요했다. BTC omega 모듈(`train_eval_omega1_2_tabm_
  diffusion_risk_btc_swingtransition_20260806.py`)의 TRAIN_CSV/EVAL_CSV 기본값(`data/splits/
  year_oos/btc_features_{2025,2026}_swingtransition.csv`)은 라이브 번들과 같은 날(2026-08-06)
  만들어진 정적 파일이라 드리프트 여지가 원천적으로 적었고, 사전점검(`_precheck_20260819.py`)에서
  **자동유도 152개 피쳐가 라이브 번들의 152 base_cols와 순서까지 완전히 일치**함을 실측 확인했다
  (`missing_from_auto_derived=0`, `auto_derived_extra_vs_live=0`, `ordered_list_identical=True`).
  102-pin 같은 명시적 pin 로직 자체가 불필요했다.
- **⚠️ ETH에 없던 신규 문제: 오버레이 데이터 결측**. BTC omega 모듈이 요구하는 `REGIME3_CURRENT_
  2025/2026` 오버레이(6개 wide24 HMM regime 컬럼)가 공유 정식 경로(`data/ensemble/supervised/
  btc_regime3_current_hmm_sensitive_wide24_20260708/`)에서 **2025년 파일은 아예 없고 2026년
  파일도 `.bak_pre_extend_20260721` 백업만 남아있었다**(dev/server 둘 다 동일, 원인 불명 — 정식
  파일이 언제 왜 사라졌는지 이번 세션에서 밝히지 못함). 같은 날 SOL 쪽 세션이 동일 문제를 먼저
  발견해(`scripts/sol_live_promotion_seed_robustness_canonicaldata_20260819.py`) 해결한 전례를
  그대로 BTC 분기로 재현했다 — 원본 빌더(`scripts/extend_regime3_wide24_sol_btc_20260721.py`,
  SOL+BTC 공유)의 `ASSETS["btc"]`(joblib=`regime3_current_sensitive_hmm_wide24_2024.joblib`,
  소스=legacy `data/splits/year_oos/btc_features_{2025,2026}.csv`)를 별도 scratch 경로에서
  그대로 호출(`_transform`/`_read` 재사용, 재구현 아님, `scripts/btc_live_promotion_seed_
  robustness_canonicaldata_20260819.py`). **추가 교차검증**: 서버 전체를 검색해 우연히 발견한
  독립 산출물(`tmp/causal_regen_20260516/btc_regime3_current_hmm_tuning_20260720/sensitive/
  btc_features_2025_regime3_current_sensitive_hmm_wide24.csv`, 7/20 튜닝실험 부산물로 추정)과
  재생성 결과를 6개 컬럼 전부 대조한 결과 최대오차 3.2e-9(부동소수점 잡음 수준) — 재생성 접근이
  올바름을 독립적으로 재확인했다.
- **동시성 방어**: 오버레이 재생성이 새 파일을 만들므로 ETH/SOL과 동일하게 `os.replace()` 원자적
  rename을 적용(4개 신규시드 학습을 병렬로 띄웠지만, 이미 사전점검 단계에서 오버레이 파일이 먼저
  만들어져 있어 실제로는 경합이 발생하지 않았다).
- **학습 소요시간**: ETH는 시드당 2.5~4분이었으나 BTC는 시드당 약 23.5~23.9분(4개 병렬 실행,
  1411.7~1434.5초) — max-train-rows/max-exit-samples 등 설정값은 비슷한데도 훨씬 오래 걸렸다.
  정확한 원인은 이번 세션에서 규명하지 않았다(BTC 데이터 로딩 자체가 무거운지, 서버 12코어를 4개
  프로세스가 나눠써서 그런지 등 미확인) — 향후 유사 작업 시간산정에 참고할 것.

**Fresh-Forward 준수**: `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`,
`saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false` (평가 스크립트
자체 report.json 필드로 명시).

## 결과 — compound PnL(%) / MDD(%) / trades, 6창 × 5시드

| window | 260620(원본) | 750703416 | 160125165 | 626578270 | 179796523 | 부호일치 |
|---|---:|---:|---:|---:|---:|---|
| 2025q1 | +55.67 | +30.18 | +21.03 | +23.30 | +33.42 | ✅ YES |
| 2025q2 | -13.39 | -31.30 | -27.02 | **+7.91** | -19.74 | ❌ **SIGN FLIP** |
| 2025q3 | +0.02 | -10.12 | -11.24 | -16.57 | -1.87 | ❌ **SIGN FLIP**(경계값) |
| **val** | **+18.03** | +7.80 | +15.86 | +26.00 | +12.73 | ✅ **YES** |
| oos_q1 | +10.95 | **-22.11** | +17.42 | +17.28 | +31.45 | ❌ **SIGN FLIP** |
| oos_q2 | -1.20 | -20.14 | -2.90 | -5.04 | -7.84 | ✅ YES |

MDD(%, with_gate 없이 compound):

| window | 260620 | 750703416 | 160125165 | 626578270 | 179796523 |
|---|---:|---:|---:|---:|---:|
| 2025q1 | -3.49 | -14.67 | -31.51 | -7.18 | -16.75 |
| 2025q2 | -20.67 | -34.57 | -28.24 | -6.90 | -30.24 |
| 2025q3 | -12.04 | -12.79 | -12.76 | -21.53 | -10.86 |
| val | -8.17 | -16.70 | -11.45 | -5.84 | -13.57 |
| oos_q1 | -10.34 | -37.41 | -13.22 | -15.75 | -18.21 |
| oos_q2 | -8.75 | -20.70 | -16.15 | -13.81 | -12.55 |

sign_flip_windows = `['2025q2', '2025q3', 'oos_q1']` — **6창 중 3창 부호플립** (2025q3는
+0.02%라는 경계값이 만든 형식적 플립이라 사실상 2.5창에 가깝다).

## 해석 — ETH N=5 CONFIRMED 결과와의 비교

ETH dual(h48qual+zig075)의 최종 N=5 결과는 6창 중 **4창** 부호플립(2025q2/2025q3/**val**/
oos_q2)이었고, 가장 심각했던 지점은 원 승격근거였던 **VAL 자체가 뒤집힌 것**(+54.88% → 최저
-16.48%)이었다. BTC는 이 결과와 비교했을 때 **부분적으로 더 강건하다**:

- **VAL은 5개 시드 전부 양(+7.80%~+26.00%)** — ETH처럼 VAL 자체가 뒤집히는 최악의 패턴은
  BTC에서 재현되지 않았다. 스프레드는 있지만(약 18pp) 부호는 일관된다.
- oos_q2도 5개 시드 전부 음(-1.20%~-20.14%)으로 일관된다.
- 그러나 **oos_q1은 부호가 갈린다** — 4개 시드(+10.95~+31.45%)는 양, 시드 750703416만
  -22.11%로 음. CLAUDE.md가 요구하는 "OOS 부호일치"를 문자 그대로 적용하면, oos_q1/oos_q2
  두 OOS 창 중 하나가 갈리므로 **완전한 OOS 부호일치는 성립하지 않는다**.
- 2025q2/2025q3(컨텍스트 창, ETH 자신의 gate 프레임워크에서도 "진단 전용, pass/fail 안 함"으로
  분류됨)도 부호가 갈리지만, 이 두 창은 원래 판정에 넣지 않는 창이다.

**시드 750703416이 유독 약해 보인다** — 2025q2 최저, val 최저, oos_q1 유일한 음수, oos_q2도
최저(-20.14%). 5개 창(2025q1 제외 전부)에서 최저치이거나 유일한 부호이탈이다. 이게 "우연히 약한
시드 하나" 때문인지 "설정 자체가 시드에 민감해서 생기는 자연스러운 분산"인지는 N=5로는 통계적으로
확정할 수 없다 — 바로 이 구분 불가능성 자체가 CLAUDE.md Seed-Diversity Gate가 존재하는 이유다
(정책 배경의 Sigma3-1h 5-seed 사례와 동일 구조: 소수 시드로는 "신호"와 "시드분산 노이즈"를 구분
못한다).

## 한계 / 교란변수

- **risk sidecar 고정**: 5개 시드 전부 원본(260620) sidecar를 frozen 재사용 — 시드별 전용
  sidecar였다면 결과가 달라질 수 있음(ETH/일리아스1과 동일한, 이미 감수된 단순화).
- **오버레이 데이터 재생성**: `REGIME3_CURRENT_2025/2026`을 legacy 소스+원본 joblib으로 직접
  재생성했다 — 정식 경로 파일이 결측이라 부득이했고, 독립 산출물과의 교차검증(최대오차 3.2e-9)으로
  강한 확신을 얻었지만, "swingtransition 피쳐 추가가 wide24 HMM 입력열에 영향 없다"는 핵심 논리는
  SOL 세션의 선례를 신뢰해 그대로 적용한 것이지 이 세션에서 처음부터 재도출하지 않았다.
- **N=5 경계**: CLAUDE.md 문턱을 충족하지만 여전히 소표본이다 — 750703416이 정말 이상치인지는
  추가 시드로만 확인 가능하다.
- **거래빈도가 낮은 창**: 2025q3/oos_q2 등 trades=4~17 수준의 창은 한두 건의 트레이드 결과가
  전체 부호를 좌우할 수 있어, "부호"라는 이진 지표 자체의 통계적 무게가 창마다 다르다(표에 trades
  칼럼을 남겨 이 점을 투명하게 남김).

## 결론 / 제안

**N=5 결과: 부분적으로만 강건함 (완전한 CONFIRMED도, ETH만큼의 명백한 REJECTED도 아님)**.
VAL과 oos_q2는 5개 시드 전부 부호가 일치해 ETH보다 뚜렷이 나은 패턴을 보이지만, oos_q1(문자
그대로의 OOS 확인 창 중 하나)에서 시드 하나가 부호를 이탈해 "OOS 부호일치" 요건을 완전히 충족하지
못한다. 원 승격 근거였던 VAL(+18.03%, 시드 260620)은 5개 시드 중앙값 부근(+7.80~+26.00% 범위,
260620은 세 번째로 높음)이라 ETH처럼 "우연히 좋았던 시드"로 보이지는 않는다.

권장:
1. oos_q1 플립이 750703416 하나 때문인지, 진짜 구조적 민감성인지 확인하려면 N≥8로 확장(신규
   랜덤시드 3개 추가)해 750703416이 이상치인지 재확인하는 것이 다음 단계로 자연스럽다.
2. 라이브 리스크 익스포저는 현재 `BINANCE_ACCOUNT_ENABLED`로 차단된 decision-only 상태로
   추정되나, 이번 세션에서 현재 서버 설정을 직접 재확인하지는 않았다 — 별도 확인 권장.
3. 시드별 전용 risk sidecar까지 포함한 재검증(현재 단순화가 결과를 왜곡했는지 확인) — ETH와 동일한
   미해결 축.

## 산출물

- 스크립트: `scripts/btc_live_promotion_seed_robustness_{prefix_snapshot,canonicaldata,precheck,
  seed_variant,eval_5seed}_20260819.py`
- 번들(서버, `/home/llewyn/crypto-scalping/` 기준 상대경로): `tmp/causal_regen_20260516/
  btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_
  swingtransition_livepromo_seedvariant_{750703416,160125165,626578270,179796523}/`
- 오버레이 재생성 산출물: `tmp/causal_regen_20260516/btc_live_promotion_seed_robustness_20260819/
  regime3_overlay_rebuild/`
- 평가 결과(로컬로 pull 완료): `tmp/causal_regen_20260516/btc_live_promotion_seed_robustness_
  20260819_eval/report.json`, 시드×창별 trade ledger CSV 30개(서버에만 있음, pull 안 함)

**fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false** (평가 스크립트 자체 report.json 필드로 명시).
