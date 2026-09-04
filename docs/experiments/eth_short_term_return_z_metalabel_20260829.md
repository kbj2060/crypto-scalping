# ETH short_term_return_z 메타라벨링 — v1 최종 확정 (2026-08-29)

호메로스 프로젝트(`docs/homer/README.md`) 3번째 신호. `docs/experiments/eth_taker_delta_climax_metalabel_20260829.md`의
"재사용 가능한 방법론 템플릿"을 그대로 따름 — 같은 Tier0급 23피쳐 빌더(재구현 아니라 import),
같은 TabPFN 패널/순열중요도 헬퍼, 같은 Fresh-Forward 분할.

## 배경

신호 정의(`scripts/live_evidence_signal_dashboard_20260823.py::SIGNAL_ORDER`): `ret3_z`
(3봉/15분 수익률 z-score)가 ±2.5를 넘으면 발동("3-bar (15m) return z-score beyond +-2.5").
이 신호가 발동했을 때 실제로 유의미한 방향 이동이 뒤따르는지를 TabPFN 메타라벨 분류모델로
예측하는 것이 목표.

## Phase 1 진단 (라벨 설계 전 필수 체크) — taker_delta_z_climax와 정반대 패턴

방법론 템플릿의 "라벨 설계 체크리스트"에 따라, 라벨을 정하기 전에 먼저 실측(scratchpad 진단,
저장소에 커밋 안 됨):

1. **발동봉↔실제극값 어긋남**: 진짜 국소극값의 **88~89%가 발동봉 시점에 이미 지났거나 그
   시점**(median lag -75분, -15봉)에 있음 — `taker_delta_z_climax`(70%가 발동 **이후**, median
   +20분)와 **정반대 방향**. 원인: `ret3_z` 자체가 과거 3봉을 돌아보는 수익률이라, ±2.5를
   넘는 시점엔 그 급격한 움직임이 대개 이미 끝나 있음.
2. **호라이즌 민감도**: 부호만 맞히는 방향적중률이 **짧은 호라이즌(15분~1h)에서 최고**(바닥
   56~57%/천장 53~54%)이고 **2h/4h로 갈수록 감소**(천장은 4h에서 49.6%로 기준선 이하로 떨어짐)
   — taker가 2h로 **넓혀야** 신호가 드러났던 것과 반대. taker의 창을 그대로 복붙했다면 이미
   감쇠가 시작된 구간에 라벨을 앉혔을 것.
3. **클러스터링**: 같은 방향 연속발동이 심함(46~50%가 인접 3~6봉 내 재발동) — taker v4와 동일한
   메커니즘(클러스터 앵커링, gap≤3봉, 가장 극단적인 `ret3_z`봉으로 앵커) 그대로 적용.

## 최종 채택 라벨 (v1)

**정의**: 클러스터 앵커링 후 발동봉 종가=entry → HORIZON=12(1h 순방향) intrabar MFE(고/저가
기준 최대유리이탈)가 발동봉 시점 `atr_pct`의 **1.75배** 이상이면 hit=1(터치기반, 지속성조건
없음 — taker v5가 이미 반증한 함정을 반복하지 않음).

**K 선택**: {0.5~3.5} 스윕 후 균형분포(51.1%/48.9%, 바닥52.2%/천장50.1%)가 나오는 1.75 채택.

**이벤트 수**: 원시 발동 → 클러스터 앵커링 → 4,522건(바닥 2,297/천장 2,225) → dropna 후
**4,510건** 사용(TRAIN 2,834 / VAL 526 / OOS 439 / HOLDOUT 711, `ts`로 분할).

**사용자 검증**: 20개 예시 캔들차트(scratchpad, 미보존)를 사용자가 검토, 이견 없이 채택.

## 무작위봉 대비 리프트 검증 (`liquidity_sweep`의 정반대 결과)

taker_delta_z_climax와 같은 방식으로, "`|ret3_z|>=2.5`라는 극값 조건 자체가 정보를 더하는가"를
확인하기 위해 **같은 방향-히트 정의(ret3_z 부호로 방향, MFE≥1.75×ATR로 히트)를 전체 279,332봉에
그대로 적용**해 기준선을 계산(`random_bar_baseline()`, `report.json` 직접 확인):

| | hit rate | 리프트 |
|---|---:|---:|
| 전체봉 기준선(항상 `ret3_z` 부호로 베팅) | 41.86% | 1.00x |
| 발동봉만(`\|ret3_z\|>=2.5`) | 51.18% | **1.22x** |

`liquidity_sweep`의 V_REBOUND 라벨이 무작위봉 대비 리프트가 **없었던 것**(0.91x/0.93x,
`docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md` 참조)과 달리, 이
신호는 극값 임계값 자체가 실질적인 방향정보를 더한다 — 메타라벨링 모델이 "이미 무의미한
이벤트 위에 껍데기만 씌우는" 상황이 아님을 확인.

## 최종 결과 (TabPFN, Tier0급 23피쳐, 4시드)

| 구간 | AUC (평균±표준편차) | 정확도 | naive 대비 |
|---|---|---|---|
| VAL (2025-09~12) | 0.6738 ± 0.0003 | 62.4% | +11.1%p |
| OOS (2026-01~03) | 0.6488 ± 0.0009 | 61.7% | +4.5%p |
| HOLDOUT (2026-04~, 1회성) | 0.6425 ± 0.0007 | 61.4% | +10.0%p |

시드: 20260829 / 141592 / 271828 / 577215 (서버 `llewyn` GPU, `quant_ai` conda env,
`TabPFNClassifier(device="cuda")`). 전부 `tmp/eth_short_term_return_z_metalabel_tabpfn_20260829/
report.json`에서 직접 확인됨(2026-08-30 재검증).

**다른 두 완료 신호와 비교**: `taker_delta_z_climax`(0.622/0.608/0.650), `liquidity_sweep`
(0.642/0.657/0.647)보다 **VAL이 더 높고, v1 한 번에 채택 성공** — 앞선 두 신호는 라벨을
4~5버전 갈아엎어야 했음(아래 "v1이 한 번에 성공한 이유" 참조).

## 필수 검증 3종 (매 신호 반복)

### 1) 룩어헤드 감사
`build_indicator_frame`은 taker 스크립트에서 **변경 없이 import**(이미 감사 완료). 신규 라벨
코드는 동일한 인과적 패턴 재사용 — 발동봉 **이후** 구간만 참조:
```python
fwd_high_max = high[::-1].rolling(window=HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
fwd_low_min = low[::-1].rolling(window=HORIZON, min_periods=HORIZON).min()[::-1].shift(-1)
```

### 2) 순열중요도 (VAL, 단일시드 20260829, baseline AUC 0.6741, 5회 반복 — `report.json` 전문)

| 순위 | 피쳐 | importance |
|---:|---|---:|
| 1 | `bb_pctb` | +0.05567 |
| 2 | `p_slow` | +0.03584 |
| 3 | `atr_percentile_864` | +0.02652 |
| 4 | `vol_z` | +0.00751 |
| 5 | `nyse_open_flag` | +0.00677 |
| 6 | `rsi` | +0.00438 |
| 7 | `is_bottom` | +0.00426 |
| 8 | `p_fast` | +0.00397 |
| 9 | `ret3_z` | +0.00317 |
| 10 | `realized_vol_ratio` | +0.00214 |
| 11 | `atr_pct` | +0.00199 |
| 12 | `ndi` | +0.00140 |
| 13 | `lower_wick_ratio` | +0.00111 |
| 14~23 | `bb_width_pctile`/`vwap_dev_z`/`adx14`/`hour_utc`/`cvd_roll_roc_48`/`weekday`/`upper_wick_ratio`/`er_24`/`delta_z`/`pdi` | ±0.001 이하(잡음 수준) |

`taker_delta_z_climax`(`atr_percentile_864`가 압도적 1위, 2위의 5배)와 달리 **오실레이터/
포지션 계열(`bb_pctb`/`p_slow`)이 상위** — `liquidity_sweep`의 `p_fast` 역U자 발견과 같은
계열("가격이 어디쯤 있는지"가 방향성 피쳐 자체보다 중요함을 시사).

### 3) 변동성 ablation
`atr_pct`/`atr_percentile_864`/`realized_vol_ratio` 3종을 제거한 20피쳐로 재학습(`scripts/
research_eth_short_term_return_z_metalabel_ablation_vol_20260829.py`):
**VAL 0.6629(−0.0109) / OOS 0.6441(−0.0047) / HOLDOUT 0.6335(−0.0090)** — taker의 손실폭
(−0.01~0.012)과 비슷한 수준, 변동성 레짐 하나에 몰빵된 신호가 아님을 시사(두 신호 모두 이
항목을 통과).

> ⚠️ **검증 상태 caveat (2026-08-30 확인)**: 위 ablation 수치는 세션 기록에서만 확인됨 —
> 생성 스크립트(`scripts/research_eth_short_term_return_z_metalabel_ablation_vol_20260829.py`)는
> 저장소에 존재하고 로직도 직접 확인했으나(기존 피쳐 CSV 재사용, `atr_pct`/`atr_percentile_864`/
> `realized_vol_ratio` 3종 제거 후 동일 TRAIN/VAL/OOS/HOLDOUT·4시드로 재평가하는 구조가 맞음),
> 그 출력 파일(`tmp/eth_short_term_return_z_metalabel_tabpfn_20260829/ablation_vol_regime_
> report.json`)은 현재 디스크에 없음(`tmp/` 정리 또는 저장 누락 추정). 정확한 수치가 필요하면
> 스크립트를 서버에서 재실행해 확인할 것 — 기존 피쳐 CSV를 그대로 재사용하므로 재빌드 불필요,
> 빠르게 재현 가능.

## v1이 한 번에 성공한 이유 (재사용 가능한 교훈)

`taker_delta_z_climax`(v1→v5, 5버전)와 `liquidity_sweep`(라벨 정의 3단계 개정,
43.9%로 확정까지)은 둘 다 라벨을 여러 번 갈아엎어야 했지만, 이 신호는 **phase1 진단(호라이즌
민감도 + 발동봉-실제극값 어긋남 방향)을 라벨 설계 전에 먼저 실측**하고 그 결과에 맞춰
호라이즌(1h, 넓히지 않음)과 지속성조건 없음(taker v5의 실패를 미리 알고 반복 안 함)을 처음부터
선택한 게 주효했음.

**재사용 가능한 원칙**: 다른 신호에 통했던 호라이즌/설계를 복붙하지 말고, 반드시 이 신호
자체의 타이밍 어긋남부터 측정할 것 — 이번에 발동봉-극값 어긋남 방향이 taker와 정확히
반대(선행 vs 후행)로 나왔는데, 만약 phase1 진단 없이 taker의 2h 창을 그대로 가져다 썼다면
이미 감쇠가 시작된 구간에 라벨을 앉혀 실패했을 가능성이 높음.

## 배포 (2026-08-30, 별도 세션에서 실행 — taker_delta_z_climax와 함께)

대시보드 증거신호 칩 자체를 이 모델로 **in-place 교체**(V_REBOUND처럼 별도 신규 칩 추가가
아님) — 사용자가 명시적으로 이 방식을 지정. 발동조건(`bottom_/top_short_term_return_z`,
`ret3_z<=∓2.5`)은 전혀 변경하지 않고, 발동 시 이 23피쳐를 TabPFN에 넣어 실시간 확률을 계산해
상태뱃지에 `"바닥 발동 · 68%"` 형태로 덧붙임. 기존 60초 캐시 사이클(klines fetch +
`compute_signals()`)에 얹혀가는 구조라 별도 fetch/캐시 없음
(`scripts/live_evidence_signal_metalabel_20260829.py::compute_evidence_signal_metalabels()`).
`SIGNAL_ORDER`/`net_score`/투표 로직은 무변경. 학습 컨텍스트는 이 문서의 TRAIN 2,834건에 동결.
서버 실제 재현 및 공개도메인(`thesan.xyz`) 확인 완료. 상세: `docs/homer/README.md` "배포 방식"
섹션.

## 하지 않은 것 / 캐비엇

- **TRAIN 정확도 비교(과적합 갭 직접 측정)** 미실행.
- **경제성(cost-gate) 미검증** — `docs/experiments/eth_evidence_signal_short_horizon_economic_
  gate_20260824.md`가 이 신호군 전체(11종)의 고정TP:SL 번역 자체를 이미 소진 REJECTED
  처리했으므로, 이 메타라벨도 자동매매 승격 전에는 반드시 별도 cost-gate 필요(`liquidity_sweep`의
  V자반등도 분류 AUC는 통과했지만 cost-gate 0/2 FAILED였음 — 통계적 엣지≠경제적 엣지, 매 신호
  재확인 필요). 대시보드 상세 텍스트에도 "자동매매 근거 아님" 명시됨.
- **지속성 체크(스무딩된 형태)** 미시도 — taker v5는 단일시점 지속성체크로 실패했지만, 마지막
  몇 봉 평균/과반 같은 스무딩된 형태는 아직 아무 신호에서도 시도 안 됨(재시도 여지 있음).
- provisional(진행중 봉 미리보기) 엔드포인트에는 메타라벨 미연결 — 폼bar 자체가 계속 바뀌는
  입력에 GPU 추론을 반복하는 건 낭비이고, 모델도 confirmed bar로만 학습됨.
- `data/ensemble/reports/`(승격 레지스트리) 무관.

## 파일 목록

- `scripts/research_eth_short_term_return_z_metalabel_tabpfn_20260829.py` — 최종 v1 코드
  (docstring에 phase1 진단 결과와 라벨 설계 근거 전문 기록).
- `scripts/research_eth_short_term_return_z_metalabel_ablation_vol_20260829.py` — 변동성
  ablation(기존 피쳐 CSV 재사용, 출력 파일은 현재 미존재 — 위 caveat 참조).
- `data/labels/eth_5m_short_term_return_z_metalabel_20260829/eth_5m_short_term_return_z_metalabel_features.csv`
  — 4,510건 최종 피쳐+라벨. 같은 디렉토리의 `tabpfn_train_context_frozen_20260829.csv`는 배포용
  동결 TRAIN 컨텍스트(2,834건).
- `tmp/eth_short_term_return_z_metalabel_tabpfn_20260829/report.json` — 최종 결과 원본(이
  문서의 모든 수치가 여기서 직접 확인됨, ablation 리포트만 예외).
- `scripts/live_evidence_signal_metalabel_20260829.py` — 대시보드 배포 코드(taker_delta_z_climax와
  공용).

## 다음 신호

**4. `volume_wick_climax`** — 위 방법론 템플릿 그대로, phase1 진단부터 이 신호 고유로 재측정.
배포 방식(증거신호 칩 교체 vs 신규 칩 추가)은 매번 사용자에게 새로 확인할 것 — 이번 방식이
자동으로 승계된다고 가정하지 말 것. 전체 진행상황은 `docs/homer/README.md` 참조.
