# BTC 레짐 분류기 신설 — Phase 1~3b + 배포 (2026-09-02)

상태: 완료. **BTC의 첫 대시보드 레짐 분류기.** 라벨은 **S24_K3**(RegimeEngine 스케일 + 3봉 확인).
ETH의 승자 S12_K3을 그대로 옮기지 않고 **BTC에서 격자를 다시 스크리닝**해 고른 값이다.

- 스크립트: `research_btc_regime_scalping_label_geometry_20260902.py`(P1),
  `research_btc_regime_label_conditional_lift_20260902.py`(P2),
  `research_btc_regime_s24k3_label_train_20260902.py`(P3/3b),
  `train_btc_regime_s24k3_20260902.py`(배포 artifact), `live_regime_btc_signal_20260902.py`(라이브)
- 발단: 2026-09-02 사용자 "같은 논리로 btc 레짐도 최고 수준으로 만들어서 대시보드에 배포해줘"

## 출발점 — BTC는 레짐 분류기가 아예 없었다

BTC 스냅샷 차트의 레짐 리본은 **회색 고정 밴드**("모델 없음")였다. 2026-08-31에 `app.js`
`renderCandleSvg()`가 `activeSnapshotAsset === "eth"`로 게이팅되도록 고쳐진 결과인데, 그 수정은
**ETH의 GBM3 리본이 BTC 캔들 위에 그려지던 버그**를 막기 위한 임시 조치였다
(memory `eth-dashboard-btc-regime-classifier-not-trained-todo-20260831`). 즉 ETH처럼 라벨만
바꾸는 작업이 아니라 **라벨+모델+라이브 스코어러+대시보드 배선**을 전부 새로 만드는 일이었다.

## ⭐파라미터를 이식하지 않고 재스크리닝한 이유와 결과

이 저장소엔 직접적인 선례가 있다 — DeMarker/칼만 BTC 포팅에서 ETH의 GAP/문턱값이 **새 목적함수로
재검증된 적이 없었고 전이되지 않아** 칼만을 빼야 했다(`btc_v_rebound_feeder_gap_threshold_screen_20260901`).
그래서 S×K 격자를 BTC에서 전부 다시 돌렸고, 실제로 **결과가 달랐다**:

| 라벨 | BTC 양쪽창-양수 셀 (표본충분 n≥150) |
|---|---|
| **S24_K3** | **7/13** — 양수 7개가 전부 대표본(n 162~594) |
| S48_K3 | 6/13 |
| S6_K6 | 6/11 |
| **S12_K3 (ETH의 승자)** | **3/10** |

**ETH의 승자는 BTC에서 최하위권이다.** BTC의 승자 S24_K3은 **RegimeEngine과 같은 스케일**이고,
TRAIN 백분위 매칭 결과 임계값도 **T1=0.2000 / T2=0.1600으로 RegimeEngine 자기 값과 일치**한다
(S=24가 그 스케일이므로 당연한 귀결). 즉 **BTC가 원한 건 스케일 축소가 아니라 디바운스**였다.
ETH에서 얻은 "디바운스가 스케일보다 중요"가 BTC에서 더 극명하게 재현된 셈이다.

## Phase 1 — 전환 타이밍 축은 BTC에서도 닫힌다 (OOS 미접촉)

16개 S×K 조합 전부, 그리고 RegimeEngine 자신도 전환 엣지의 95% CI가 0을 관통한다.
**0/16** (ETH는 1/16이었고 그것도 우연 수준). RegimeEngine on BTC: h6 +0.04bp [−0.42,+0.55],
h12 +0.19 [−0.52,+1.01], h24 **−0.46** [−1.39,+0.75]. baseline |move| 23.6~46.9bp.
→ ETH와 같은 재프레이밍(조건부 게이팅)으로 이동.

## Phase 2 — 조건부 lift (OOS 미접촉)

BTC 전용 배선을 새로 만들어야 했다:
- **zigzag 피벗**: ETH 라벨은 ETH 전용. `build_wave3_action_labels_20260531`에 **ETH 파라미터
  verbatim**(min_reversal_pct=0.009, min_wave_bars=6, …)으로 **캐노니컬 BTC OHLC**에서 생성
  (기존 `build_btc_5m_zigzag_and_pivot_labels_20260806.py`의 causalfix_final vintage 산출물을
  재사용하지 않고 재생성 — 이 연구 전체의 데이터 vintage를 하나로 맞추기 위함).
- **교차자산**: BTC가 주체이므로 smt_divergence의 참조자산은 ETH.
- **펀딩**: BTC 자기 펀딩(`data/research/funding_extracted/BTCUSDT/`, 2024-01~2026-06),
  ETH 것이 아님. 동일한 rolling-90 z 레시피.

대조군은 ETH와 같은 **순환 이동 귀무분포**(B=200), VAL/OOS 분리 보고.

| 변이 | VAL | OOS | 양쪽창-양수 |
|---|---|---|---|
| REF_RegimeEngine | **−0.0131** | +0.2552 | 4/13 |
| **S24_K3** | +0.0565 | +0.2147 | **7/16** |
| S48_K3 | +0.1416 | +0.0745 | 8/16 |
| S12_K3 | +0.1680 | +0.1143 | 5/16 |

S48_K3의 8/16이 표면상 최고지만 표본충분 셀로 좁히면 6/13이고, S24_K3은 7/13이며 **OOS 귀무분포
통과 6/16으로 전 변이 중 최고**(S48_K3은 3, S6_K6은 4)다. 기준선은 VAL이 음수라 게이트 탈락.

## Phase 3 — 학습가능성 (⚠️BTC의 첫 OOS 조회)

GBM3 config·136피처·시드 고정, 라벨만 교체. OOS 2026-07-01~2026-08-01(9,141봉, ~32일 —
BTC 캐노니컬 피처 파일이 2026-08-01 17:40에서 끝남, ETH의 ~50일보다 짧다).
**이 창은 ETH와 달리 기존 레짐 연구로 소모된 적이 없어 비교적 신선하다** — 그래서 더더욱
이번 한 번만 조회했다.

| 라벨 | 피처 | bal_acc | **chop_R** | **chop_P** | bull_R | bear_R | pred_flip |
|---|---|---|---|---|---|---|---|
| REF_RegimeEngine | full136 | **0.9088** | 0.9208 | **0.9219** | 0.9084 | 0.8971 | 0.1748 |
| REF | ablated | 0.8786 | 0.9114 | 0.8915 | 0.8603 | 0.8642 | 0.1763 |
| **S24_K3** | full136 | 0.8687 | 0.9025 | 0.8827 | 0.8620 | 0.8417 | **0.0777** |
| **S24_K3** | ablated | 0.8359 | 0.8828 | 0.8528 | 0.8195 | 0.8054 | 0.0802 |

- 분류는 후퇴(bal_acc −4.0pp, chop_P −3.9pp)하나 **chop_R은 −1.8pp에 그친다**(ETH보다 잘 보존).
- **예측 flip_rate가 0.1748 → 0.0777로 절반 이하**.
- 방향프록시 ablation: REF −0.0302 vs S24_K3 −0.0328 — **거의 동일**한 의존도
  (ETH는 S12_K3이 배포판의 2배였는데, BTC는 이 점에서 더 깨끗하다).

## ⭐Phase 3b — 실제 배포형태(예측 chop 게이팅)에서 S24_K3 우세

| 라벨 | VAL | OOS | POOLED | 양수셀(POOLED) | 양쪽창-양수 |
|---|---|---|---|---|---|
| REF_RegimeEngine | +0.0116 | **+0.2668** | +0.0924 | 9/13 | 4/13 |
| **S24_K3** | **+0.1114** | +0.1976 | **+0.1403** | **13/16** | **7/16** |

REF는 OOS 수치만 크고 **VAL이 사실상 0(+1.2%)** 이라 뒷받침이 없다. S24_K3은 세 창 전부 양수이고
평가 가능한 셀 수도 더 많다(16 vs 13). ETH와 같은 구조 — 학습가능성 손실보다 조건화 이득이 크다.

## 배포

- artifact `tmp/btc_regime_s24k3_20260902/model.joblib`(gitignore, rsync 전송)
- 라이브 스코어러 `scripts/live_regime_btc_signal_20260902.py` — ETH 스코어러와 동일 계약
- `dashboard/server.py`: `/api/regime-btc` 신설(캐시·락·loader·handler·route, ETH 것과 동일 구조)
- `dashboard/live/app.js`: 자산별 소스 맵(`REGIME_SOURCE_BY_ASSET`)으로 전환 — **변수 하나를
  두 자산이 공유하지 않게** 한 것이 핵심(2026-08-31 버그가 정확히 그거였다). 분류기 없는 자산은
  기존 "unsupported" 회색 밴드로 그대로 폴백.

### ⭐교차자산 컬럼 네이밍 함정 (학습/추론 파리티)

`FeatureEngineer`는 교차자산 컬럼명을 `close_btc`/`volume_btc`/`quote_volume_btc`로 **하드코딩**
한다. 캐노니컬 BTC 학습 파일은 **BTC가 주체, ETH가 교차자산**으로 만들어졌음을 직접 확인했다 —
같은 2024-01 봉에서 `close`≈42,437(BTC)인데 `close_btc`≈2,290(ETH), `btc_corr_60` 평균 0.797
(자기참조면 ~1.0). 그래서 라이브 스코어러도 **ETH를 `*_btc` 슬롯에 넣는다**. 여기서 BTC를 자기
교차자산으로 넣으면 매 봉 자기상관 ~1.0인 조용한 파리티 붕괴가 된다.

### ⚠️ JS는 CI가 검사하지 않는다

`.github/workflows/ci.yml`의 syntax-check는 `.py`와 `.sh`만 본다. app.js 문법 오류는 대시보드
프런트를 통째로 깨뜨리는데 자동 검증이 없다. node는 로컬·서버 모두 없어서, `esprima`(순수 파이썬,
ES2017)로 **optional chaining 등을 구조보존 치환한 뒤** 파싱했고 **HEAD 원본을 대조군으로 함께
통과**시켜 검사법 자체의 유효성을 확인했다. 문법 한정 검증이며 런타임 보증은 아니다.

## 한계

1. Phase 3b의 레짐 예측은 **in-sample**(증거신호 창 ⊂ 레짐 TRAIN). 두 arm에 동일 적용이라
   비교는 공정하나 절대치는 라이브 추정이 아니다.
2. BTC OOS가 9,141봉(~32일)으로 ETH(~50일)보다 짧다.
3. **분류 정확도는 실제로 후퇴**한다(chop_P 0.9219 → 0.8827). 리본 표시 정확도와 안정성의
   트레이드오프이며, 안정성 쪽(flip 0.1748 → 0.0777)이 크게 개선된다.
4. BTC엔 증거신호가 대시보드에 노출돼 있지 않으므로 Phase 2/3b의 게이트 이득은 **현재 소비자가
   없다** — 리본은 표시 전용이다. 향후 BTC 증거신호를 붙이면 바로 쓰인다.
