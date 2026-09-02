# XRP 증거신호 + 레짐모델 구축 (2026-09-03)

사용자 goal: *"xrp 코인도 증거신호와 레짐모델 개발해줘. 어제 오늘 이더리움에서 진행한 것처럼
같은 작업을 진행해줘. 대신 **비트코인에서 문제가 있었던 부분을 잘 참고해서** xrp에서는 그런
문제가 발생하지 않도록 주의해줘."*

절차 기준: `docs/homer/evidence_signal_new_coin_port_protocol.md` (§5 체크리스트 11단계)

---

## 1단계 · 데이터 실사 (착수 전) ✅

BTC에서 데이터를 가정했다가 파이프라인 3건이 터졌으므로 먼저 실사했다.

| 항목 | 상태 |
|---|---|
| klines | `binance_data/klines/XRPUSDT/XRPUSDT-5m-api.csv` **272,490행**, 2024-01-01 ~ 2026-08-04 |
| 5분 갭 | **0개** ✅ |
| 필수 컬럼 | `taker_buy_base` 포함 전부 존재 ✅ |
| 펀딩 | `funding_rate_other/XRPUSDT-fundingRate-*.zip` **31개월**(2024-01~2026-07), BTC와 같은 디렉토리 ✅ |
| 인덱스 오프셋 | `START`(2024-01-01) == klines 시작 ⇒ **오프셋 0** (그래도 하류는 타임스탬프 매핑 강제) |

⚠️**klines가 ETH(08-28)보다 24일 뒤처져 있다**(마지막 2026-08-04).
HOLDOUT(2026-04-01~)은 4개월분이 확보되어 연구 파이프라인엔 충분하다.
라이브 스코어러/섀도우는 API에서 직접 받으므로 영향 없다.
⚠️펀딩이 2026-07까지라 마지막 ~4일은 `funding_z`가 NaN이다.

## 2단계 · 후보 Tier0 CSV ✅

`scripts/build_xrp_5m_evidence_signal_candidates_tier0_20260903.py`
— BTC 빌더의 **자산 상수만** 바꾼 포팅(로직 재구현 0줄).

결과: **272,490행**, 2024-01-01 ~ 2026-08-04. NaN은 전부 워밍업 구간.

| 트리거 | bottom | top |
|---|---|---|
| `liquidity_sweep` | 7,500 | 7,051 |
| `taker_delta_z_climax` | 7,840 | 5,648 |
| `short_term_return_z` | 3,843 | 3,981 |
| `orthogonal_combo` | 2,076 | 1,015 |
| `fib_extension_exhaustion` | 905 | 839 |
| `local_extreme` | 15,749 | 15,947 |
| **any** | **28,120** | **26,834** |

⚠️**포팅 중 실제로 터진 것 1건**: 펀딩 로더가 `.dt.as_unit("us")`인데 klines CSV 파싱은 `[ns]`라
`merge_asof`가 `MergeError`로 죽었다(BTC 빌더 작성 당시엔 안 터졌다 — 환경 차이).
XRP 빌더에서 klines 단위에 맞춰 캐스팅해 해결. **BTC의 "파이프라인 3건"과 같은 계열이다.**

## 3단계 · ⭐HIT_TYPE 그리드스크린 (자산별 재탐색) ✅

**이 단계를 건너뛰면 BTC의 최대 사고가 재현된다** — 같은 이름의 신호가 자산 간에 HIT_TYPE·H·K가
전부 다른데, 서빙 코드가 원본을 따라가 라이브 hit률이 2.6배 과대평가됐다.

5신호 각각에 BTC 그리드스크린(4 HIT_TYPE × 6 HORIZON × 5 K = 120셀)을 포팅해 재탐색했다.
`scripts/research_xrp_<signal>_gridscreen_hittype_20260903.py`

### ⚠️자동 선택을 그대로 쓰지 않았다 — 표본 두께 감사

XRP는 **5신호 중 4개가 `touch_giveback_sustained`를 골랐다.** 그런데 이 HIT_TYPE은
BTC에서 이미 명시적으로 거부된 전례가 있다("mechanical global argmax … explicitly distrusted
for having only 2-5 OOS hits"). 조건이 둘(fast_mult AND giveback)이라 hit이 희소해져
lift는 커 보이지만 **몇 건 위에 서 있는 lift인지**를 봐야 한다.

`scripts/audit_xrp_gridscreen_selection_thickness_20260903.py`로 family별 최선을 hit 수와 함께 비교:

| 신호 | 자동 선택 | hits(b/t) | 두꺼운 대안 | hits(b/t) | **최종** |
|---|---|---|---|---|---|
| `taker_delta_climax` | giveback H=9 K=1.5, lift 1.45 | **481/233** | close_at_h lift 1.11 | 534/344 | ✅**자동 유지** |
| `liquidity_sweep` | giveback H=15 K=2.0, lift 1.36 | **385/311** | touch_mfe lift 1.07 | 2477/2613 | ✅**자동 유지** |
| `short_term_return_z` | touch_mae_capped H=12 K=1.5 | **746/652** | — | — | ✅**자동 유지** |
| `orthogonal_combo` | giveback H=12 K=2.5, lift 2.03 | **79/34** ⚠️ | touch_mfe H=8 K=2.0, lift 1.56 | **348/211** | ⭐**기각 → touch_mfe** |
| `fib_extension_exhaustion` | giveback H=16 K=3.25, lift 2.06 | **39/33** ⚠️ | touch_mfe H=10 K=1.5, lift 1.38 | **327/328** | ⭐**기각 → touch_mfe** |

판정 기준(BTC 전례 그대로): 측면별 hit이 한 자릿수~수십 건이면 그 lift를 믿지 않는다.
`short_term_return_z`는 스크립트 자신이 이미 얇은 argmax(giveback H=3, 40/21)를 거부하고
`touch_mae_capped`로 폴백했다 — 같은 판단이 자동으로 적용된 사례다.

⚠️`taker`/`liquidity_sweep`의 giveback은 **표본이 두꺼워서**(481/233, 385/311) 유지했다.
"giveback = 무조건 기각"이 아니라 **두께로 판단**한다.

### XRP 확정 라벨 (2026-09-03)

| 신호 | HIT_TYPE | H | K | 해상 필요 봉수 |
|---|---|---|---|---|
| `taker_delta_climax` | `touch_giveback_sustained` | 9 | 1.5 | **18** (2×H) |
| `liquidity_sweep` | `touch_giveback_sustained` | 15 | 2.0 | **30** (2×H) |
| `short_term_return_z` | `touch_mae_capped` | 12 | 1.5 | 12 |
| `orthogonal_combo` | `touch_mfe` | 8 | 2.0 | 8 |
| `fib_extension_exhaustion` | `touch_mfe` | 10 | 1.5 | 10 |
| `demarker_extreme` | plain touch (격자 진행 중) | — | — | — |
| `kalman_deviation_meanrev` | plain touch (격자 진행 중) | — | — | — |

⭐**해상 필요 봉수가 H와 다른 신호가 2개**다(giveback 계열). BTC에서 이걸 놓쳐
`liquidity_sweep`을 절반 시점에 잘못 확정했다. 서빙 코드는 반드시 `full_window`를 쓴다.

### 자산 3종 대조 (포팅 프로토콜 §1 갱신용)

| 신호 | ETH | BTC | **XRP** |
|---|---|---|---|
| `taker_delta_climax` | touch / 24 / 2.00 | close_at_h / 6 / 2.0 | **giveback / 9 / 1.5** |
| `liquidity_sweep` | touch / 30 / 4.00 | giveback / 20 / 2.0 | **giveback / 15 / 2.0** |
| `short_term_return_z` | touch / 12 / 1.75 | touch_mae_capped / 6 / 2.0 | **touch_mae_capped / 12 / 1.5** |
| `orthogonal_combo` | touch / 24 / 3.571 | touch / 8 / 2.0 | **touch_mfe / 8 / 2.0** |
| `fib_extension_exhaustion` | touch_and_mae / 20 / 2.35 | close_at_h / 10 / 2.75 | **touch_mfe / 10 / 1.5** |

**3자산이 모두 일치하는 신호는 하나도 없다.** 재스크리닝이 필수임을 다시 확인했다.

## 4단계 · TabPFN 메타라벨 (VAL/OOS/HOLDOUT) ✅

`scripts/research_xrp_evidence_signals_metalabel_tabpfn_20260903.py` — **5신호 통합 1개 스크립트**.

⭐**개별 포팅하지 않은 이유**: XRP 확정 HIT_TYPE 중 2종이 BTC와 **계열이 다르다**
(taker: close_at_h→giveback, fib: close_at_h→touch_mfe). BTC 스크립트를 포팅하면 그 안의
hit 계산을 손으로 다른 계열로 고쳐야 하는데, 그게 정확히 "재구현" 함정이다.
대신 **그리드스크린이 실제로 쓴 hit 함수를 그대로 import**했다 — 선정과 학습이 같은 함수를 쓰므로
어긋날 수가 없다. 피쳐 목록과 파생 함수도 각 BTC 모듈에서 import(모듈마다 이름이 다르다:
`add_missing_features` / `add_derived_features` / `augment_features`).

⚠️포팅 중 2건 더 터짐: ① 트리거 컬럼명이 `bottom_taker_delta_z_climax`(내 SPEC은 `_z` 누락)
② 후보 CSV엔 원재료만 있어 `nyse_open_flag`/`er_24`/`realized_vol_ratio` 등을 파생 함수로
만들어야 했다. 둘 다 BTC 동결 컨텍스트에서 이미 겪은 지점이다.

### 결과 (HOLDOUT 1회 노출, 셀은 사전 확정)

| 신호 | HIT_TYPE | H | K | 해상봉 | TRAIN n | hit률 | VAL | OOS | **HOLDOUT** |
|---|---|---|---|---|---|---|---|---|---|
| `demarker_extreme` | touch MFE | 6 | 2.0 | 6 | — | — | — | — | **0.6651** |
| `kalman_deviation_meanrev` | touch MFE | 5 | 2.0 | 5 | — | — | — | — | **0.6223** |
| `short_term_return_z` | touch_mae_capped | 12 | 1.5 | 12 | 2,314 | 0.6016 | 0.6466 | 0.5753 | **0.6132** |
| `taker_delta_z_climax` | giveback | 9 | 1.5 | **18** | 6,742 | 0.0899 | 0.6142 | 0.5556 | **0.6091** |
| `orthogonal_combo` | touch_mfe | 8 | 2.0 | 8 | 1,446 | 0.4198 | 0.5979 | 0.5847 | **0.5599** |
| `liquidity_sweep` | giveback | 15 | 2.0 | **30** | 5,019 | 0.0875 | 0.5099 | 0.4601 | **0.4886** ❌ |
| `fib_extension_exhaustion` | touch_mfe | 10 | 1.5 | 10 | 999 | 0.6086 | 0.5221 | 0.5079 | **0.4738** ❌ |

⭐**`liquidity_sweep`과 `fib_extension_exhaustion`은 HOLDOUT AUC가 0.5 미만** —
무작위보다 나쁘다. **서빙에서 제외한다.**
(BTC에서도 `liquidity_sweep`이 0.5214로 유일하게 무작위였다 — 두 자산에서 반복되는 약점이다.)

⚠️**demarker 격자 경계 경고**: 1차 실행에서 K=2.0이 격자 **상단 경계**에서 선택돼 스크립트가
경고를 냈다. 이 저장소는 같은 실수로 ETH demarker의 진짜 최적값을 놓친 전례가 있어
(README 5.6) K 격자를 4.0까지 확장해 재실행 중이다.

## 5단계 · demarker 격자 경계 확장 (3회) ✅

스크립트가 경계 경고를 낼 때마다 그 방향으로 확장했다(README 5.6 규칙).

| 회차 | 격자 | 선택 | HOLDOUT AUC | 경고 |
|---|---|---|---|---|
| 1차 | H[6..20] K[0.4..2.0] | H=6 K=2.0 | 0.6651 | K 상단 경계 |
| 2차 | K를 4.0까지 확장 | H=6 K=2.0 | 0.6651 | H 하단 경계 |
| **3차** | H[2..20] K[0.4..4.0] | **H=2 K=1.5** | **0.6759** | H=2 하단 경계 |

⚠️3차의 H=2 경고는 **구조적 하한**이라 더 못 넓힌다(H=1은 다음 봉 하나뿐이라 "H봉 내 터치"
라벨이 성립하지 않는다). 미탐색 경계가 아니라 정의상 바닥이므로 여기서 확정한다.
⭐확장 덕에 AUC가 0.6651 → **0.6759**로 올랐다 — 경계 경고를 무시했으면 놓쳤을 이득이다.

## 6단계 · 레짐 라벨 (Phase 2 조건부 lift) ✅

`scripts/build_xrp_regime_inputs_20260903.py` — XRP엔 상류 입력 **3개 전부 없어서** 먼저 만들었다
(klines 1year 224,245행 갭0 / 펀딩 31개월 추출 / 캐노니컬 피쳐는 만들지 않고 klines 직접 사용 —
Phase 2는 피벗 계산에 OHLC만 쓴다).

⭐**교차자산 파트너 슬롯**: BTC-주체 스크립트는 `btc_df=eth`로 ETH를 넣었다. XRP-주체이므로
BTC를 넣었다 — 인자 이름이 `btc_df`인 건 시그니처일 뿐이다(FeatureEngineer `close_btc` 명명 함정과 같은 계열).

`scripts/research_xrp_regime_label_conditional_lift_20260903.py` — S×K 격자 재탐색 결과:

| 라벨 | 양쪽창 양수 | mean VAL | mean OOS |
|---|---|---|---|
| ⭐**S48_K6** | **10/16** | +0.0274 | **+0.0652** |
| S6_K6 | 8/16 | +0.0283 | +0.0420 |
| S12_K6 | 6/16 | +0.0035 | +0.0812 |
| S48_K3 | 6/16 | +0.0056 | +0.0607 |
| S12_K3 (ETH 승자) | 5/16 | +0.0422 | +0.0170 |
| S24_K3 (BTC 승자) | **3/16** | −0.0036 | −0.0121 |

### ⭐⭐세 자산의 레짐 라벨이 전부 다르다

| 자산 | 승자 |
|---|---|
| ETH | **S12_K3** |
| BTC | **S24_K3** |
| **XRP** | **S48_K6** |

BTC 승자 S24_K3은 XRP에서 **3/16으로 거의 최하위**, ETH 승자 S12_K3도 5/16이다.
"ETH 승자가 BTC에서 3/10 최하위"였던 것과 **정확히 같은 패턴**이 XRP에서 재현됐다.
⇒ 레짐 라벨은 자산별 재스크리닝이 선택이 아니라 필수다.

⭐XRP는 **디바운스(K=6)가 스케일보다 중요**하다 — K=6이 S6/S12/S48 모두에서 같은 스케일의
K=1/K=3보다 낫다. BTC에서 "디바운스가 스케일보다 중요"했던 관찰과 같은 방향이다.

## 남은 단계

- [x] ~~3b) demarker/kalman 격자~~ ✅ (demarker는 격자 경계 경고로 K 확장 재실행 중)
- [x] ~~4) TabPFN 메타라벨 → VAL/OOS/HOLDOUT~~ ✅ — 5종 중 2종 HOLDOUT AUC<0.5로 **서빙 제외**
- [ ] 5) 동결 컨텍스트 + 라이브 스코어러 + 섀도우 러너 (HIT_SPEC에 `full_window` 반영)
- [ ] 6) 연구 구현 대조 검증 (무작위 800건, 불일치 0)
- [ ] 7) 결함 스캐너 (`audit_live_shadow_paths_defect_classes_20260903.py`)
- [x] ~~8a) 레짐 라벨 재스크리닝~~ ✅ **S48_K6** (ETH S12_K3·BTC S24_K3과 전부 다름)
- [ ] 8b) 레짐 분류기 학습 (Phase 3)
- [ ] 9) 대시보드 배선
