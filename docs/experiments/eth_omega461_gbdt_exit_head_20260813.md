# ETH Omega4.6.1 GBDT 기반 exit_head (2026-08-13, Odyssey2 #4)

상태: `tested_negative_closed` — **VAL 사전등록 게이트(컴포넌트+포트폴리오 레벨, PnL·MDD 넷 다
비악화) 실패로 OOS 미실행.** 포트폴리오 레벨만 보면 baseline(TabM 라이브ATR exit_head) 대비
PnL·MDD 둘 다 큰 폭으로 개선되지만(+46.59%→+101.27%, -21.70%→-19.81%), **컴포넌트 레벨(h48qual
단독)에서 PnL·MDD 둘 다 악화**(+9.23%→+2.72%, -7.59%→-7.69%)돼 사전등록 기준 4개 지표 중 2개를
충족하지 못한다. 규율대로 OOS는 절대 열지 않았다.

## 배경

Odyssey2 우선순위 큐 #4: h48qual의 exit_head를 현재 확정 베이스라인인 TabM(라이브 ATR 배리어
재라벨, `docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md`) 대신
**GBDT(LightGBM)**로 학습시키면 같은 라벨/데이터셋에서 다른 결과가 나오는가를 검증한다.
Odyssey(1)에서 GBDT/LightGBM은 `direction_head` 분류(진입 선택)에서 8시드×6구간 전패(0/48)로
확정 부정됐지만, exit_head는 이미 확정된 포지션의 청산 시점만 결정하는 post-entry 문제라
분류 실패가 그대로 전이된다는 보장이 없다 — Odyssey2 계약서의 "Phase 1 아이디어 → post-entry
재적용 트리아지"에서 "재시도 가치 있음"으로 분류된 항목이다. zig075는 이 실험에서 건드리지
않는다(direction/quality/encoder 동결과 동일한 이유로, 이 실험도 h48qual만).

비교 대상은 원본 라이브 h48qual 번들이 아니라, **Odyssey(1)이 만든 신규 exit_head 번들**
(`tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/h48qual/
true_3head_tabm_bundle.pt`, VAL PnL+46.59%/MDD-21.70% — 현재 Odyssey2가 채택한 확정 베이스라인)
이다. GBDT exit_head는 이 TabM exit_head를 대체하는 세 번째 후보다.

## 방법

### 데이터셋 — 재학습 아님, 기존 레시피 재현

`scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py`의
`_fast_timescale_checkpoint`/`_build_exit_dataset_entry_label_live_atr_barrier`를 그대로
import해서 재사용하고, TabM `full1500` 런과 동일한 시드(260813)·후보수(1500)로 데이터셋을
**재구축**했다(원본 런이 `x_exit_raw`/`y_exit`/`frame_exit`를 디스크에 저장하지 않고 학습된
번들+`report.json` 진단정보만 남겼기 때문). 재구축 후 원본 `report.json`의 `dataset` 블록과
직접 대조해 **행 수·양성 개수·사용된 후보 수 3개 지표 전부 정확히 일치**함을 확인했다(아래
표) — GBDT와 TabM이 100% 동일한 데이터셋으로 학습됐음을 재실행이 아니라 직접 대조로 보장한다.

| | 원본 TabM full1500 런 | 재구축(이번 GBDT용) |
|---|---:|---:|
| 행 수 | 1,234,431 | 1,234,431 (일치) |
| 양성(exit=1) 개수 | 245,600 | 245,600 (일치) |
| 사용된 후보 수 | 1,500 | 1,500 (일치) |
| 양성 비율 | 19.90% | 19.90% |

### 학습 — 레짐별 3개 분리 GBDT, TabM과 동일한 가중치 스킴

`hard.EXPERT_NAMES`(bull/bear/chop) 레짐마다 **별도의 LightGBM 분류기**를 학습했다(단일
GBDT+레짐-피처 방식이 아니라, 라이브 3-expert 라우팅 아키텍처와 구조적으로 대응시키기 위해).
세 모델 전부 **같은** `x_exit`(115차원: 102 base + 13 pos_*,
`parent._exit_input_from_position_rows`로 변환)/`y_exit`을 입력받되,
`train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622._fit_exit_head_only`가 TabM
전문가별 exit_head를 재학습할 때 쓰는 것과 **동일한** 샘플 가중치 스킴
(`compute_sample_weight(balanced) × 그 레짐의 소프트 Regime3 라우팅 확률`)으로 가중치를 줬다 —
"모델 클래스만" 다르고 데이터·라벨·가중치는 동일하게 맞춘 ablation. 하이퍼파라미터는 이
저장소의 기존 레짐-라우팅 LightGBM 컨벤션(`train_eval_eth_h48qual_final_boss_v2_regime_routed_
20260813.py`)을 그대로 따랐다: `n_estimators=400, num_leaves=31, learning_rate=0.05`. 85/15
시간순-내 분할(TabM과 동일 규칙)로 held-out 진단만 뽑았다(조기종료엔 미사용, 단순 고정
`n_estimators`).

| 전문가 | train/val 행수 | val AUC | val logloss | 학습 시간 |
|---|---:|---:|---:|---:|
| bull | 1,049,266 / 185,165 | 0.9983 | 0.0130 | 24.3s |
| bear | 1,049,266 / 185,165 | 0.9983 | 0.0167 | 24.2s |
| chop | 1,049,266 / 185,165 | 0.9981 | 0.0157 | 23.7s |

세 전문가 전부 held-out AUC≈0.998로 거의 완벽하게 분리된다 — 버그가 아니라 이 라벨 설계 자체의
특성으로 보인다: `exit=1`의 지배적 두 사유(`mfe_giveback_exit` 75.6%, `adverse_unreal_exit`
22.5%, 원본 데이터셋 다양성 통계 기준)가 각각 `pos_giveback≥0.65`, `pos_unrealized≤-0.010`라는
**입력 피처 자체에 대한 단순 임계값 규칙**으로 정의돼 있어, 표현력 있는 모델이면 TabM이든
GBDT든 거의 완벽히 학습 가능한 구조다. 두 모델의 실질적 차이는 판별력이 아니라 **어디서 정확히
문턱을 긋고 그게 거래 행동으로 어떻게 이어지는가**에서 갈린다(아래 결과 참고).

### 런타임 통합 — duck-typing 래퍼, 시뮬레이션 코드 무수정

`scripts/research_eth_omega461_gbdt_exit_head_val_20260813.py`의 `GBDTExitHeadWrapper`가
`train_eval_omega4_2_risk_sidecar_20260622._predict_exit_prob_one`의 호출 계약
(`torch.softmax(model(x)["exit"], dim=-1).mean(dim=1)`)을 흉내낸다: `__call__`이
`predict_proba`의 로그를 `(batch, k=1, 2)` 모양으로 반환하면, `softmax(log(p))==p`이므로 그
위의 TabM 전용 softmax/앙상블-풀링 로직이 그대로 `predict_proba` 값을 재현한다. mean/std는
항등(0/1)— GBDT는 표준화가 불필요하므로 원본 스케일 그대로 넘긴다. 이 설계 덕분에
`_predict_exit_prob_one`, `research_eth_omega461_exit_sweep_20260721.replay_exit_variant`(컴포넌트
레벨), `replay_omega4_6_1_greedy_router_20260706.greedy_replay`(포트폴리오 레벨) **전부 단
한 줄도 수정하지 않고** 그대로 재사용했다 — `exit_runtime`/`loaded_models` 딕셔너리에서
TabM 모델 자리만 래퍼로 바꿔치기.

### G0 자체검증 — 100% 기존 코드로 알려진 수치 재현

GBDT를 평가하기 전에, 이 스크립트의 하네스가 **기존에 발표된** baseline/TabM-liveATR 수치를
그대로 재현하는지 먼저 확인했다(재구현 없이 `h48cons._evaluate_val`,
`research_eth_omega461_exit_head_portfolio_asymmetric_20260813.run_variant`를 그대로 호출).

| | 컴포넌트 baseline(원본) | 컴포넌트 TabM 라이브ATR | 포트폴리오 baseline(원본) | 포트폴리오 TabM 라이브ATR |
|---|---:|---:|---:|---:|
| 발표된 수치 | +5.45% / -11.62% / 29건 | +9.23% / -7.59% / 63건 | +36.82% / -24.34% / 29건 | +46.59% / -21.70% / 35건 |
| 이 하네스 재현값 | +5.45% / -11.62% / 29건 | +9.23% / -7.59% / 63건 | +36.82% / -24.34% / 29건 | +46.59% / -21.70% / 35건 |

4개 지표 전부(PnL·MDD·거래수) **정확히 일치**(부동소수점 반올림 오차 수준) — G0 통과. 이후
GBDT 수치를 신뢰 가능한 것으로 취급했다.

## 결과

### 컴포넌트 레벨(h48qual 단독, VAL 2025-10-01~12-31) — 악화

| | TabM 라이브ATR(baseline) | GBDT |
|---|---:|---:|
| PnL | +9.23% | **+2.72%**(악화, -6.51pp) |
| MDD | -7.59% | **-7.69%**(악화, -0.10pp) |
| 거래수 | 63 | 71 |
| 승률 | 30.2% | 63.4% |
| 평균 보유기간 | 210.8bar | 144.9bar |
| exit_reasons | `exit_head:52, take_profit:8, stop_loss:3` | `exit_head:65, take_profit:6` |

### 포트폴리오 레벨(h48qual+zig075 단일계좌 우선순위, 동일 VAL) — 개선

| | baseline(둘 다 원본) | TabM 라이브ATR(현재 확정 베이스라인) | GBDT |
|---|---:|---:|---:|
| PnL | +36.82% | +46.59% | **+101.27%**(TabM 대비 개선, +54.68pp) |
| MDD | -24.34% | -21.70% | **-19.81%**(TabM 대비 개선, +1.89pp) |
| 거래수 | 29 | 35 | 38 |
| 승률 | 41.4% | 37.1% | 57.9% |
| 평균 보유기간 | 676.5bar | 551.2bar | 492.5bar |
| exit_reasons | `stop_loss:17, take_profit:12` | `take_profit:13, stop_loss:13, exit_head:9` | `exit_head:15, take_profit:13, stop_loss:10` |
| 슬롯 승자(source_component) | `zig075:22, h48qual:7` | `zig075:22, h48qual:13` | `zig075:22, h48qual:16` |

### 사전등록 게이트 판정

| 지표 | 판정 |
|---|---|
| 컴포넌트 PnL 비악화 | **FAIL** |
| 컴포넌트 MDD 비악화 | **FAIL** |
| 포트폴리오 PnL 비악화 | PASS |
| 포트폴리오 MDD 비악화 | PASS |
| **종합 게이트** | **FAIL** (4개 중 2개 미충족) |

`scripts/research_eth_omega461_gbdt_exit_head_oos_20260813.py`가 이 `gate_pass=False`를
`report.json`에서 직접 읽어 **OOS 데이터를 전혀 로딩하지 않고 즉시 `RuntimeError`로 중단**함을
실행으로 재확인했다 — "VAL 못 이기면 OOS 안 연다" 규율이 연구자 재량이 아니라 코드 레벨에서
강제됨.

## 해석 — 왜 컴포넌트와 포트폴리오가 반대 방향인가

GBDT exit_head는 TabM보다 **더 이르고 더 잦은** 청산을 학습했다(컴포넌트 평균 보유기간
210.8→144.9bar, -31%; exit_head 발동 비중 82.5%→91.5%, stop_loss는 사실상 0건으로 소멸). 이
행동 변화가 두 레벨에서 반대 효과를 낸다:

- **컴포넌트(h48qual 단독, 전액가상자본) 레벨**: 승률은 크게 오르지만(30.2%→63.4%, 잦은 조기
  소액 익절/손실회피) 개별 거래 규모가 줄어 복리 PnL 자체는 baseline보다 낮아진다 — h48cons
  실패 사례(문서: `eth_omega461_live_exit_head_h48cons_relabel_20260813.md`)처럼 큰 승리
  거래의 "머리를 자르는" 것과 유사한 메커니즘으로 보인다. `max_trade_pnl`도 TabM
  4.47%→GBDT 4.20%로 소폭 하락.
- **포트폴리오(단일계좌·우선순위 공유 슬롯) 레벨**: 짧아진 평균 보유기간이 공유 슬롯을 더
  자주 비워, h48qual 자신의 다음 신호가 슬롯을 잡을 기회를 늘린다 — 슬롯 승자 수가
  baseline 7건→TabM 13건→GBDT 16건으로 계속 증가하는 추세가 이를 직접 보여준다(zig075는
  22건으로 세 구성 모두 불변, h48qual 진입신호 자체의 원시 빈도도 direction/quality 동결로
  불변 — 순전히 "슬롯을 얼마나 자주 비워주는가"의 효과). `eth_omega461_live_exit_head_
  liveatr_relabel_20260813.md`의 "후속 2"가 TabM에서 처음 관찰한 이 슬롯-재순환 상호작용을
  GBDT가 더 강하게 밀어붙인 결과로 해석된다.

즉 GBDT는 "컴포넌트 단독 경제성은 희생하되 슬롯 회전율을 극대화"하는 방향으로 학습됐고, 이
트레이드오프가 포트폴리오 지표만 보면 유리해 보인다. 그러나 컴포넌트 단독 악화가 사전등록
기준에 포함된 이유가 바로 이런 "포트폴리오 상호작용에 가려진 개별 경제성 훼손"을 놓치지
않기 위함이었다 — `eth_omega461_regime_specific_quality_threshold_20260813.md`(Odyssey2 #1)의
"no_gate/with_gate가 반대 방향이라 어느 쪽도 깔끔한 승리가 아니다"와 같은 계열의 결론이다.

세 전문가 모두 held-out AUC≈0.998이라는 점도 참고할 만하다 — GBDT가 TabM보다 "덜 학습됐다"는
문제가 아니라(오히려 held-out 판별력은 극도로 높다), 거의 동일하게 잘 학습된 두 분류기가
**서로 다른 위치에 결정 경계**를 긋고 그 차이가 거래 행동으로 증폭된 결과로 보인다.

## 결론

**채택 불가.** VAL 사전등록 게이트(컴포넌트+포트폴리오, PnL+MDD 넷 다 비악화)를 통과하지
못해 규율대로 OOS를 열지 않았다. 포트폴리오 레벨 수치만 보면 인상적이지만(+46.59%→+101.27%),
컴포넌트 단독 economics가 뚜렷이 나빠진 상태에서 나온 결과라 "GBDT exit_head 자체가 더
낫다"는 결론을 지지하지 않는다 — 슬롯-재순환이라는 이미 알려진 포트폴리오 상호작용을 GBDT의
과발동 성향이 우연히 더 강하게 자극한 것에 가깝다. Odyssey2 계약서의 우선순위 큐 #4는
이것으로 **종결**한다(부정 결과).

## 미해결 / 다음 단계

- exit_head 임계값(현재 `EXIT_THRESHOLD=0.95` 고정, TabM/GBDT 동일 값 적용)을 GBDT에 맞게
  재보정하면 컴포넌트 레벨 악화가 완화될 가능성이 있으나, 사전등록 기준에 없던 사후 튜닝이라
  이번엔 시도하지 않았다 — 시도한다면 새로운 사전등록·새 VAL-then-OOS 사이클이 필요하다.
  (오케스트레이터 지시 범위 밖.)
- 컴포넌트 레벨의 "과발동" 자체가 GBDT 특유의 문제인지, 아니면 라벨의 임계값-규칙적 성격(위
  "해석" 절) 자체가 표현력 높은 모델 일반에 이런 경계를 유도하는지는 미확인 — 이번 실험
  범위 밖.
- 채택 가능한 변경 0건, 라이브 파일 미변경.

## 준수 확인

`fresh_forward_bar_by_bar=true`(데이터셋 구축은 TabM 런과 동일한 causal 배리어 시뮬레이션
재사용, VAL/컴포넌트 리플레이는 `replay_exit_variant`/`greedy_replay` 단일 순방향 루프
무수정 재사용). `trade_ledgers_used_as_input=false`. `saved_parent_exit_timestamps_used=false`.
`future_rows_used_for_entry=false`. direction_head/quality_head/encoder 전부 동결·미변경(GBDT는
exit_head만 대체). `EXIT_THRESHOLD=0.95` 고정 유지. **OOS(2026-01-01~03-31)는 게이트 실패로
전혀 로딩되지 않았다**(`research_eth_omega461_gbdt_exit_head_oos_20260813.py` 실행 시
`RuntimeError`로 즉시 중단됨을 직접 확인). zig075 미변경. 라이브 파일
(`trading_bot_modules/omega4_6_1_live.py`, `trading_bot.py`, `trading_bot_modules/
runtime_config.py`, `.env`) 무변경(`git diff` 기준 0줄, 작업 시작 전/후 모두 확인).

## 산출물

- 새 스크립트:
  - `scripts/train_eval_omega461_gbdt_exit_head_liveatr_20260813.py` — 데이터셋 재구축(레퍼런스
    대조 포함) + 레짐별 3개 GBDT 학습 + 번들 저장.
  - `scripts/research_eth_omega461_gbdt_exit_head_val_20260813.py` — G0 자체검증 +
    `GBDTExitHeadWrapper`(duck-typing 런타임 주입) + 컴포넌트/포트폴리오 VAL 비교 + 게이트 판정.
  - `scripts/research_eth_omega461_gbdt_exit_head_oos_20260813.py` — VAL 게이트 통과 시에만
    실행되는 1회용 OOS 확인 스크립트(이번엔 게이트 실패로 `RuntimeError` 중단, OOS 미실행).
- GBDT 번들: `tmp/causal_regen_20260516/eth_omega461_gbdt_exit_head_liveatr_20260813/h48qual/
  gbdt_exit_bundle.pkl`(레짐별 LightGBM 3개 + `base_cols`/`pos_cols` 계약).
- report.json: `tmp/causal_regen_20260516/eth_omega461_gbdt_exit_head_liveatr_20260813/report.json`
  (학습), `tmp/causal_regen_20260516/eth_omega461_gbdt_exit_head_val_20260813/report.json`(G0+VAL
  비교+게이트).
- 거래 원장(diagnostic, 참고용):
  `tmp/causal_regen_20260516/eth_omega461_gbdt_exit_head_val_20260813/portfolio_ledger_
  asymmetric_h48qual_gbdt_zig075_original.csv`.
- 인용 문서: `docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md`(TabM
  베이스라인 근거), `docs/experiments/eth_omega461_regime_specific_quality_threshold_20260813.md`
  (같은 계열의 "지표별 반대 방향" 판정 선례), `docs/model_contracts/
  odyssey2_eth_live_injection_contract_20260813.md`(서브 프로젝트 계약).
