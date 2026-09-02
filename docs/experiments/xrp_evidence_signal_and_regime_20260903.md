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

## 7단계 · 레짐 분류기 학습 (Phase 3 + 3b) ✅

`scripts/research_xrp_regime_s48k6_label_train_20260903.py`

| | bal_acc | chop_R | chop_P | **pred_flip** |
|---|---|---|---|---|
| REF_RegimeEngine | 0.9086 | 0.9172 | 0.9224 | 0.1735 |
| **S48_K6** | 0.8459 | 0.9078 | 0.8603 | **0.0417** |

**Phase 3b — 예측-chop 게이팅 lift(실제 배포 형태)**

| | 양쪽창 양수 | mean VAL | mean OOS |
|---|---|---|---|
| REF_RegimeEngine | 5/16 | +0.0595 | **−0.0085** |
| **S48_K6** | **9/16** | +0.0448 | **+0.0606** |

⭐ETH·BTC와 **같은 패턴**: 후보 라벨은 학습이 더 어렵지만(bal_acc 0.846 vs 0.909)
**실제 배포 형태에서는 압승**(9/16 vs 5/16, OOS +0.0606 vs −0.0085)이고 플리커는 **4배 낮다**
(0.0417 vs 0.1735). ⇒ **S48_K6 채택 권고.**

## 8단계 · 서빙 (동결 컨텍스트 · 라이브 · 섀도우) ✅

| 산출물 | 상태 |
|---|---|
| 동결 컨텍스트 5종 | ✅ 전부 XRP 고유 값 |
| `live_xrp_evidence_signal_metalabel_20260903.py` | ✅ warmed_up, 5신호 |
| `live_xrp_evidence_signal_shadow_runner_20260903.py` | ✅ 상시 가동(주문 없음) |

⚠️⚠️**BTC 데이터 오염 사고 1건**을 여기서 잡았다. `prep[0]`(`load_tier0`/`load_frame`)이
**그 모듈 자신의 TIER0_PATH**를 읽는데 str_z/taker/orthogonal은 BTC 모듈을 재사용하므로
BTC CSV를 읽어버렸다. **행수 277,191(=BTC)이고 hit률이 BTC와 소수점까지 동일**해서 발각됐다.
⇒ 로더를 아예 호출하지 않고 XRP CSV를 직접 읽도록 고치고, **자산 오염 가드**
(`EXPECTED_ROWS` 대비 행수 검증)를 넣어 재발 시 즉시 죽게 했다.
추가로 `contexts_report.json`의 `asset` 필드가 `BTCUSDT`로 남아 있던 것도 정정했다.

⚠️tz 규약이 모듈마다 달랐다(demarker/taker는 naive, orthogonal은 aware) — 신호별 `TZ_AWARE` 플래그로 처리.

## 9단계 · ⭐연구 구현 대조 검증 (배포 전) ✅

`scripts/audit_xrp_shadow_hitmode_parity_20260903.py` — BTC에서 `HIT_SPEC` 모드 2건이 틀려
라이브 hit률이 2.6배 과대평가된 사고를 **XRP는 배포 전에** 막는다.

무작위 400지점 × 양측 = **신호당 800건**을 연구 스크립트 원본 구현과 대조:

```
✅ demarker_extreme           mode=touch                     n=800 불일치 0
✅ kalman_deviation_meanrev   mode=touch                     n=800 불일치 0
✅ short_term_return_z        mode=touch_mae_capped          n=800 불일치 0
✅ taker_delta_climax         mode=touch_giveback_sustained  n=800 불일치 0
✅ orthogonal_combo           mode=touch                     n=800 불일치 0
⇒ 전부 일치 -- 배포 가능
```

## 10단계 · 대시보드 배선 ✅

⚠️**기존 버그를 함께 고쳤다**: XRP 페이지는 그동안 **ETH 증거신호를 그대로 보여주고 있었다**
(BTC에서 사용자가 신고했던 "비트코인 페이지에 이더리움 증거신호가 나온다"의 XRP판).
증거신호 라우팅이 ETH/BTC만 분기했기 때문이다.

- `server.py`: `/api/xrp-evidence-signals` + 캐시/락 + 로더 추가
- `app.js`: 자산별 라우팅에 XRP 추가. XRP 라벨 사전이 아직 없어 **빈 사전으로 폴백**한다
  (잘못된 ETH 설명을 보여주는 것보다 이름만 나오는 게 낫다)
- 캐시버스터 `20260903-xrp-evidence`, 서빙 바이트 확인 완료

## 11단계 · 결함 스캐너 ✅

`audit_live_shadow_paths_defect_classes_20260903.py`에 XRP 2경로 추가(대상 16개).
XRP 섀도우의 히트 P1/P2/P3는 **BTC 수정본과 동일한 양성 항목**이다(자기 주석·MARK_URL 상수·
의도된 `keep.append`). 스코어러는 히트 없음.

## 12단계 · ⚠️Phase 3 BTC 오염 발견 → 정정 → 리본 배포 ✅

### ⚠️⚠️Phase 3 1차 결과는 **BTC 데이터**였다

리본 배선을 시작하며 모델 아티팩트를 찾다가 발각됐다. Phase 3의 상수가 이랬다:

```python
XRP_CANON = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"   # ← 이름만 XRP
```

포팅 시 `BTC_CANON` → `XRP_CANON` **변수명은 바꿨지만 경로는 남겼다.**
동결 컨텍스트에서 잡았던 것과 **같은 오염 계열**이다.

**발각 단서**: TRAIN 262,656 + OOS 9,216 = 271,872봉인데, XRP 1year 파일은 224,245행이고
OOS 구간(2026-07-01~08-01)은 그 파일 범위(2026-02-17 종료) **밖**이라 존재할 수 없었다.

⇒ 1차 Phase 3 수치(bal_acc 0.8459, 게이트 9/16)는 **무효**.
⭐**Phase 2는 깨끗했다**(로그 "XRP evidence frame 224,245 bars") — 따라서 **라벨 선택 S48_K6은 유효**하다.

### 내가 앞서 내린 판단이 틀렸다

6단계에서 *"캐노니컬 피쳐 파일은 만들지 않는다 — Phase 2는 피벗 계산에 OHLC만 쓴다"*고 했다.
Phase 2엔 맞았지만 **Phase 3은 136피쳐 전체가 필요**하다. 그래서 상류를 제대로 만들었다:

- `scripts/build_xrp_raw_frame_20260903.py` — XRP 주체 + **BTC 교차자산** raw 프레임 (272,490행)
- `scripts/build_xrp_features_20260903.py` — `FeatureEngineer().process()` → **272,490행 × 146컬럼**

⭐**교차자산 슬롯**: `FeatureEngineer`는 `close_btc`/`volume_btc`/`quote_volume_btc`를 하드코딩한다.
자산마다 그 슬롯 내용이 다르다 — ETH→BTC, **BTC→ETH**, **XRP→BTC**.
학습과 라이브가 같은 것을 넣어야 파리티가 맞는다.

### 정정된 Phase 3 (실제 XRP 데이터)

| | bal_acc | chop_R | chop_P | **pred_flip** |
|---|---|---|---|---|
| REF_RegimeEngine | 0.9113 | 0.9342 | 0.9189 | 0.1787 |
| **S48_K6** | 0.8644 | 0.9017 | 0.8844 | **0.0458** |

**예측-chop 게이팅(실제 배포 형태)**

| | 양쪽창 양수 | mean VAL | mean OOS |
|---|---|---|---|
| REF_RegimeEngine | **2/13** | −0.0881 | **−0.0756** |
| **S48_K6** | **8/16** | **+0.0351** | **+0.0306** |

⭐결론은 유지되고 **오히려 선명해졌다** — REF는 양쪽 창 모두 음수다.
재발 방지로 **자산 오염 가드**(캐노니컬 행수 검증)를 Phase 3과 모델 빌더에 넣었다.

### 배포

| 산출물 | 내용 |
|---|---|
| `build_xrp_regime_s48k6_model_20260903.py` | `tmp/xrp_regime_s48k6_20260903/model.joblib` (OOS bal_acc 0.8644, flip 0.0458) |
| `live_regime_xrp_signal_20260903.py` | 라이브 스코어러, **CROSS_SYMBOL=BTCUSDT** |
| `server.py` | `/api/regime-xrp` + 캐시/락/로더 |
| `app.js` | 리본 소스맵에 `xrp` 추가, `refreshRegimeXrp` **3곳 등록**(정의·자산전환·메인틱) |

실측: `{"warmed_up": true, "chop_prob": 0.794, ...}`, 서빙 `app.js?v=20260903-xrp-regime`,
`refreshRegimeXrp` 3회 / 소스맵 1회 확인.

⭐자산별 **별도 상태 변수**를 쓴다(`latestRegimeXrp`). 하나를 공유하면 2026-08-31의
"ETH 리본이 BTC 캔들 위에 그려지던" 버그가 그대로 재현된다.

## 13단계 · ⭐XRP 대시보드 전체 점검 + 내부 로직 심층 점검 ✅

사용자: *"XRP 대시보드 전체를 한 번 점검해줘. 그리고 증거신호와 레짐 모델도 내부 로직을
하나하나 심층 점검해줘"*

오늘 BTC 오염 2건이 전부 **행수 대조**로만 잡혔던 걸 감안해, 읽기가 아니라 **실측 대조**로 했다.

### (1) 내부 로직 심층 점검 — **10항목 전부 통과**

`scripts/audit_xrp_internal_logic_deep_20260903.py`

| | 점검 | 결과 |
|---|---|---|
| A1 | 동결 컨텍스트가 XRP 데이터인가 (BTC와 행수·hit률 대조) | ✅ asset=XRPUSDT, BTC와 동일한 신호 **0개** |
| A2 | 라이브/섀도우 **셀(H/K)**이 확정값과 일치 | ✅ 불일치 0건 |
| A3 | **mode + 해상봉**이 라벨 확정 규칙과 일치 | ✅ 0건 (taker 18봉 포함) |
| A4 | 학습 컨텍스트 피쳐가 CSV에 실제로 존재 | ✅ 0건 |
| A5 | HOLDOUT AUC<0.5 신호가 서빙에 남지 않았나 | ✅ 누출 0 |
| B1 | 모델 `feature_cols`가 학습 출처와 동일 | ✅ 136개, ETH GBM3 아티팩트와 일치 |
| **B2** | ⭐**교차자산 슬롯** | ✅ live `CROSS_SYMBOL=BTCUSDT` · **학습 raw의 `close` 중앙 0.5753(XRP) / `close_btc` 중앙 43,729(BTC)** |
| B3 | 라벨 상수 SCALE/K | ✅ 코드 48/6 == 아티팩트 48/6 |
| B4 | 라이브가 학습과 같은 파생(`_with_raw_state12`)을 태우나 | ✅ |
| B5 | 캐노니컬이 XRP 가격대인가 | ✅ close 중앙 0.5753 |

⭐**B2가 이번 점검의 핵심**이다 — 교차자산 슬롯은 자산마다 내용이 다른데(ETH→BTC, BTC→ETH,
XRP→BTC) 코드만 봐서는 확인이 안 된다. **가격대로 실측**해서 확정했다.

### (2) ⚠️대시보드 — ETH 데이터가 새는 곳 4개 발견·수정

`scripts/audit_xrp_dashboard_asset_awareness_20260903.py`로 API 상수 19개를 전수 분류한 뒤
플래그된 항목의 실제 렌더 경로를 직접 읽었다.

**정상**(자산 인식): 청산맵 · 청산5분 · 청산방향 · 베이시스청산압박(`?asset=`),
증거신호 · 레짐리본(자산별 전용 URL/소스맵), 매크로캘린더 · 세션경보(자산 무관),
진행중 미리보기 · 익절선(ETH 아니면 **DOM 미접촉 방어**가 이미 있음).

⚠️**ETH 데이터가 그대로 새던 것 4개** — 스냅샷 탭(자산 전환)에 있는데 게이트가 없었다:

| 지표 | 출처 | 문제 |
|---|---|---|
| `whale` 수급 흐름 | `trading_bot.py` state | 봇은 **ETH만** 돌린다 |
| `retail_flow` 리테일 수급 | 〃 | 〃 |
| `liq_cascade` 청산 캐스케이드 | 〃 | 〃 |
| `v_rebound` V자 급등락 | `/api/v-rebound-signal` | **ETH 전용 TabPFN 모델** |

사용자가 이미 신고했던 *"비트코인 페이지에 이더리움 증거신호가 나온다"*와 **같은 계열**이다.

**수정**: `ethOnlyIndicator()`로 감싸 ETH가 아닌 탭에서는 값을 **지우고** 상태를 바꾼다 —
tone=neutral, `"ETH 전용 지표 — 이 코인은 아직 미지원"`, history/proba 제거, `= ETH 전용` 태그.
**다른 코인의 값인 척하는 것보다 없는 게 낫다.**

검증(dukpy 실행): `eth`→원본 그대로 / `xrp`·`btc`→neutral·미지원 문구·history 0·proba null.
서빙 확인 `app.js?v=20260903-eth-only-guard`, `ethOnlyIndicator` 5회.

## 14단계 · ⭐XRP 실시간 지표 배선 — "미지원"에서 **실제 XRP 값**으로 ✅

사용자: *"xrp도 모두 이더리움처럼 똑같이 지원하게 해줘. 실시간 데이터도 xrp도 지원 중이지 않아?"*

**맞았다.** XRP 전용 워커가 **6일 넘게** 돌고 있었다(`supervisor_xrp_worker.sh`:
*"microstructure + tail-risk + OI/long-short-ratio, **all three**"*).
13단계에서 이 4개를 "ETH 전용 — 미지원"으로 처리했는데, **3개는 실제로 지원 가능**했다.

### 실사 (2026-09-03 06:0x, 1분 전까지 최신)

| duckdb | 테이블 | 행수 | 내용 |
|---|---|---|---|
| `microstructure_xrp.duckdb` | `microstructure_1m_xrp` | 9,651 | **nif_whale · nif_retail** ✓ |
| `tail_risk_xrp.duckdb` | `tail_risk_1m_xrp` | 9,648 | long/short_usd, mu, sigma ✓ |
| `oi_lsratio_xrp.duckdb` | `oi_lsratio_5m_xrp` | 1,941 | OI · 롱숏비율 ✓ |
| `l2_anomaly_snapshots_xrp.duckdb` | 4개 테이블 | 111k | L2 이상 ✓ |

### 배선

- `scripts/coin_config.py`: XRP·HYPE에 `microstructure_db_path`/`microstructure_table` 추가
  (tail_risk는 이미 **5코인 전부** 있었다)
- `server.py`: `coin_indicators_payload(asset)` + `/api/coin-indicators?asset=` (캐시 20초)
  Z는 저장된 mu/sigma로 유도: `z = (long_usd_1m − mu_long) / sigma_long`
- `app.js`: `coinIndicator(item, kind)` — ETH는 봇 state 그대로, 다른 코인은 **그 코인 duckdb** 값,
  데이터 없으면 `ethOnlyIndicator`로 폴백해 "미지원"으로 정직하게 표시

### 실측 (배포 후)

```
xrp : nif_whale 1.000  nif_retail 0.110   tail Z 확보    → 실제 XRP 값 ✅
hype: nif_whale null   nif_retail 0.253   tail Z 확보    → 부분 지원
btc : micro 없음(수집기 미배치)             tail Z 확보    → 수급/리테일은 "미지원"
```

⚠️**정직하게 남기는 한계 2가지**

1. **`nif_whale`은 간헐적**이다 — XRP 9,651행 중 5,014행(52%)만 비null. 대형 체결이 있을 때만
   계산된다. 값이 없으면 `"수집 중 — 아직 값 없음"`으로 표시한다(0으로 위장하지 않는다).
2. ⚠️**`hawkes_active`는 봇 내부 상태라 다른 코인에 없다.** 청산 캐스케이드는 Z 기반
   **"주의"까지만** 판정되고 **"위험"(hawkes) 단계는 뜨지 않는다.** 툴팁에 명시했다 — 숨기지 않는다.

### 검증

dukpy 실행으로 7가지 경우 확인: ETH 원본유지 / XRP 값있음(유입 0.210) /
nif_whale null(수집 중) / 리테일(유출 −0.160) / 캐스케이드 평온(Z −0.3) /
캐스케이드 주의(Z 2.4) / 데이터없음→미지원 폴백.

### 남은 ETH 전용 1개

**`v_rebound` V자 급등락** — ETH 학습 TabPFN 모델이라 데이터 배선만으로는 안 되고
**XRP용 모델 학습이 필요하다**(별도 연구 과제). 지금은 "ETH 전용 — 미지원"으로 정직하게 표시된다.

## 남은 단계

- [x] ~~3b) demarker/kalman 격자~~ ✅ (demarker는 격자 경계 경고로 K 확장 재실행 중)
- [x] ~~4) TabPFN 메타라벨 → VAL/OOS/HOLDOUT~~ ✅ — 5종 중 2종 HOLDOUT AUC<0.5로 **서빙 제외**
- [ ] 5) 동결 컨텍스트 + 라이브 스코어러 + 섀도우 러너 (HIT_SPEC에 `full_window` 반영)
- [ ] 6) 연구 구현 대조 검증 (무작위 800건, 불일치 0)
- [ ] 7) 결함 스캐너 (`audit_live_shadow_paths_defect_classes_20260903.py`)
- [x] ~~8a) 레짐 라벨 재스크리닝~~ ✅ **S48_K6** (ETH S12_K3·BTC S24_K3과 전부 다름)
- [ ] 8b) 레짐 분류기 학습 (Phase 3)
- [ ] 9) 대시보드 배선
