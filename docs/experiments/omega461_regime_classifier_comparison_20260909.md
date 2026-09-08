# 전신 wide24 HMM vs 대시보드 S12_K3 GBM — 레짐 분류기 실측 비교 (2026-09-09)

상태: **완료.** 하위 프로젝트 `omega461_regimegbm_rebuild_20260909` 개설 근거.
계약: `docs/model_contracts/omega461_regimegbm_rebuild_contract.md`
스크립트: `scripts/research_omega461_regime_classifier_comparison_20260909.py`
산출물: `tmp/omega461_regimegbm_rebuild_20260909/{regime_classifier_comparison.json,regime_labels_side_by_side.csv}`

## 질문

사용자 질문(2026-09-09): "지금 레짐 분석을 이전 버전을 쓰고 있는데 현재 내 대시보드에 있는 레짐
분류기와 오메가 모델의 레짐 분류기를 비교해줘."

## 두 분류기 사양 (코드/아티팩트 직접 확인)

| | 전신 Omega4.6.1 | 대시보드 ETH |
|---|---|---|
| 모델 | 12-state `GaussianStateModel`(HMM) + RobustScaler + `state_class_matrix (12,3)` | `HistGradientBoostingClassifier` (depth 10, 400 iter, lr 0.04, l2 2.0, leaves 31) |
| 아티팩트 | `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/regime3_current_sensitive_hmm_wide24_2024.joblib` | `tmp/eth_regime_s12k3_20260902/model.joblib` |
| 선정/배포일 | 2026-05-30 | 2026-09-02 |
| 피쳐 | 24 (wide24) | 136 (wide24 24개를 포함하는 상위집합) |
| 라벨 | `balancedish_adx16_slope15_bb012`<br>ADX≥16 · slope≥0.00015 · tight_BB≤0.012 | **S12_K3**<br>`er_12`/`er_24` 효율비 + `net_24` 방향앵커 + `slope_12`, K=3봉 confirm |
| 학습 구간 | **2024년만** | **2024-01-01 ~ 2026-06-30** |
| 자체 검증 bal_acc | 0.7638 | 0.8550 |

⚠️ **두 bal_acc는 직접 비교할 수 없다.** 서로 다른 라벨을 학습했으므로 공통 정답이 없다.

## 전신에서 레짐이 쓰이는 3곳

1. 부모 TabM `base_cols` 102개 중 마지막 6개:
   `regime3_current_sensitive_wide24_{bull_prob,bear_prob,chop_prob,confidence,entropy,margin}`
2. 전문가 서브넷 라우팅 (`argmax` → bull/bear/chop 전문가 선택)
3. 리스크 사이드카 29피쳐 중 `parent_router_margin`
   (= `regime3_current_sensitive_wide24_margin`, `trading_bot_modules/omega4_6_1_live.py:198`)

대시보드 쪽은 스냅샷 탭 청산맵 오버레이 + `/api/regime-wide24`로 서빙되며,
`scripts/live_eth_fire_cont_shadow_runner_20260904.py`가 "최선 노력 태그"로 소비한다.
**라이브 매매 게이트에는 들어가지 않는다.**

## 방법

전신 출력은 사이드카 CSV에 이미 계산돼 있어 그대로 읽고, 대시보드 GBM은 라이브 스코어러와
**동일 경로**(`_with_raw_state12()` → `feature_medians` 대체 → `predict_proba`)로 같은 프레임에서
채점했다. `_with_raw_state12()`는 8개 `state7_*`/`state12_*` 컬럼을 만드는 필수 단계다 —
빠뜨리면 그 8개가 조용히 median으로 대체된다(2026-08-26에 실제로 발생했던 라이브 버그).

실측 결과 **median 대체 0개** — 136피쳐 전부 캐노니컬 프레임에서 실값으로 채워졌다.

## 결과 (2026-01-01 ~ 08-19, 공통 66,528봉)

| | bull | bear | chop | flip율 | 중앙 상태지속 |
|---|---:|---:|---:|---:|---:|
| omega wide24 HMM | 0.228 | 0.226 | 0.546 | **0.1213** | **3봉 (15분)** |
| dashboard S12_K3 GBM | 0.221 | 0.218 | 0.561 | **0.0944** | **7봉 (35분)** |

**일치율 69.3%.** 클래스 비중은 거의 동일한데도 세 봉 중 하나꼴로 다르게 판정한다.

교차표 (행=omega, 열=dash, 전체 대비 %):

| | bear | bull | chop |
|---|---:|---:|---:|
| **bear** | 14.98 | 0.11 | 7.49 |
| **bull** | 0.85 | 13.83 | 8.12 |
| **chop** | 5.95 | 8.19 | 40.47 |

## 해석

**불일치의 성격이 핵심이다.**

- **정면충돌(한쪽 bull ↔ 다른쪽 bear)은 641봉 · 0.96%뿐이다.** 두 모델은 방향 판단에 사실상
  합의한다.
- **불일치 30.7% 중 29.76%p가 전부 추세↔횡보 경계다.** 갈리는 지점은 "어느 방향인가"가 아니라
  "지금이 추세인가 횡보인가"이다.
- 전신 쪽이 **flip이 28% 잦고 상태 지속이 절반 이하**다(3봉 vs 7봉). 라우팅 신호가 15분마다
  바뀌므로 전문가 서브넷도 그만큼 자주 교체된다. 이는 선행 라인
  `eth_candidate_shared_trunk_regime_experts_20260817`이 지적한 "전문가별 유효표본 부족"
  (route_w 가중 bull 28.6%/bear 28.0%/chop 43.4%)과 같은 병목을 다른 각도에서 건드린다.

## 이 비교가 답하지 않는 것

- **어느 쪽이 더 정확한가** — 공통 정답이 없어 답할 수 없다. 답하려면 제3의 경제적 기준(예:
  각 레짐 판정으로 라우팅했을 때의 실현 PnL)이 필요하며 그게 이 라인의 Phase 1이다.
- **교체하면 좋아지는가** — 전혀. 이 문서는 "얼마나 다른가"만 잰다.

## 파생된 계약 이슈 (최우선)

대시보드 S12_K3 GBM의 `train_range`는 **2024-01-01 ~ 2026-06-30**으로, 이 라인의
validation(2025-10~12)과 OOS(2026-01~02)를 **전부 포함**한다. 라벨 정의 자체는 인과적(er/net/slope
전부 후방참조)이지만 **적합된 모델이 그 봉들의 라벨을 이미 봤으므로**, 그 예측을 부모 피쳐로
쓰면 downstream 평가 전체가 오염된다.

→ 계약의 "최우선 미해결 이슈"로 등록. Phase 0에서 컷오프 재학습(또는 워크포워드 레짐 예측)으로
해소하기 전까지, 이 GBM 아티팩트의 상태는 데이터 레지스트리에서 **`blocked`**다.

## 준수

신규 학습 없음. 신규 하이퍼파라미터/임계값 탐색 없음. 라이브 파일
(`trading_bot.py`, `omega4_6_1_live.py`, `runtime_config.py`, `dashboard/**`, `.env`) 미변경.
전신 아티팩트 읽기 전용. 인라인 실행 결과와 저장된 스크립트 실행 결과가 동일함을 확인했다.
