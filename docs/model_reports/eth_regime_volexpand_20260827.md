# ETH VolExpand — 변동성-확장 이산 레짐(돌파 조기경보 시도) — 2026-08-27

## Context

GBM2(추세/횡보 2-class, 같은 날 이전 배포)가 되돌림 잦은 그라인딩성 하락(2026-08-25/26,
2528→2417, 17시간, -4.4%)을 대부분 chop으로 놓치는 걸 발견 → "이 그라인딩성 움직임을 GBM2보다
먼저 잡아낼 수 있는, 크기(magnitude) 기반 변동성 이산 레짐"을 시도. 계획 문서:
`/home/kbj20/.claude/plans/pure-hugging-book.md`.

## 결론(요약)

**전체(pooled) 조기경보 주장은 기각. 다만 "그라인딩형 전환"이라는 원래 동기가 된 하위집합에서만
방향성 있는 신호가 보임 — 단 n=13으로 표본이 너무 얇아 확정 불가.** 자동배포/자동매매 근거로
쓰기엔 이르고, 추가 데이터 축적 후 재검정이 필요한 상태로 남긴다.

## Phase 0+1 (라벨 설계) — `scripts/research_eth_volexpand_regime_label_design_20260827.py`

- 라벨 원재료: `realized_vol_ratio`(`features/engineering.py:296-298`, `rv_short(12봉)/
  rv_long(288봉)`, 이미 존재하는 컬럼, 새 공식 없음).
- TRAIN(2024-01-01~2026-06-30, GBM2와 동일)에서 상위 20% 백분위 컷(qcut, 사후탐색 아님) →
  `is_expand_raw` → K=12봉 디바운스(GBM2와 동일 K, 직접비교 목적) → `is_expand_confirmed`.
- **가설 확인**: raw flip_rate=0.037(20%컷) — GBM2 raw(0.1997)·GBM3 라이브(0.1981, 자체 디바운스
  없음) 대비 4~5배 안정적. K=12에서 confirmed flip_rate=0.0129 — GBM2의 K=12(0.0122)와 사실상
  동률, 같은 안정성에 도달하는 데 더 큰 디바운스가 필요 없었음.
- 라이브 08-24~08-27 KST 차트(GBM2/GBM3/ratio 3패널)에서 눈으로도 08-25/26 구간에 ratio가
  1.5~1.95까지 튀는 게 보였으나, ratio 자체가 전체 기간 내내 0.4~2.0을 오르내려 이 구간만
  유별난 건 아니었음 — 정량 검증(Phase 2) 필요 판정.

## Phase 2 (모델 학습) — `scripts/train_eth_regime_volexpand_20260827.py`

- GBM2와 동일 구조(TRAIN/SEL/VAL/OOS 날짜, HistGradientBoostingClassifier, HP_CANDIDATES,
  hysteresis grid) 미러링. 데이터: 동일 3개 CSV.
- **피쳐 제외(15개)**: `realized_vol_ratio`(라벨 원본) + 근접-동어반복 변동성 컬럼 11개
  (`volatility_z`/`garman_klass_vol`/`rogers_satchell_vol`/`parkinson_vol`/`bb_width`/
  `bb_width_z`/`bb_width_pct_rank_288`/`atr_pct_rank_288`/`compression_score`/`garch_vol`/
  `garch_vol_z`) + `state7_volatility_state` + 신규발견 2개(`state12_garman_klass_vol`=이미
  제외한 컬럼의 단순변환, `state7_range_compression`=공식의 35%가 이미 제외한 `bb_width_z`).
  GBM2의 5개 제외목록(방향성 라벨용)은 상속하지 않음 — 이 라벨은 크기(magnitude) 기반이라
  순환관계가 다름.
- **136 → 121 피쳐**(15 근접동어반복 제외 + `compression_release_up/down` 2개는 별도 ablation
  arm으로 기본 제외).

### 결과 — 일반 분류 성능

| 구간 | balanced_accuracy | expanding 재현율 |
|---|---|---|
| VAL | 0.8375 | 0.700 |
| OOS | 0.7475 | 0.523 |

GBM2의 OOS 0.78과 비슷한 수준. **ablation**: `compression_release_up/down`(과거 압축→방출
임펄스) 포함 시 OOS 0.7832로 +3.6%p — 퍼뮤테이션 중요도 1-2위(0.040/0.028 상당, 3위의 3배 이상)
로 실제 기여가 큼, 그러나 제외해도 0.7475로 우연보다 훨씬 나은 분류력은 유지됨(다른 121개
피쳐만으로도 의미있는 신호).

### 결과 — 조기경보 이벤트 스터디(핵심, `event_study()` 재사용)

GBM2 `is_trend_confirmed` 0→1 전환(chop→trend 시작만)을 pivot으로, VAL+OOS 구간에 244건.
모델의 "확장" rising-edge(780건) vs trivial baseline(원시 ratio 컷+디바운스만, 780건과 별개로
262건)을 각각 lift로 비교.

**전체(pooled) — 기각**: 모델 lift 0.81~0.94(K=12/24/48/96 전 구간 1.0 이하 또는 근처) —
GBM2 전환을 앞서 예측한다는 근거 없음. trivial baseline이 짧은 horizon(K=12)에서 오히려 더
나음(1.42 vs 모델 0.94) — 121개 피쳐를 쓴 모델이 원시 ratio 직접 보는 것보다 조기경보 목적으로는
못하다는 뜻.

**ER 층화(`er_24` 기준 clean/grinding) — 그라인딩만 방향성 있음, 표본 부족**:

| K | clean(n=231) lift | grinding(n=13) lift |
|---|---|---|
| 12 | 0.88 | **1.85** |
| 24 | 0.85 | **2.56** |
| 48 | 0.74 | **2.04** |
| 96 | 0.86 | **1.57** |

clean(GBM2가 이미 깨끗하게 잡는 전환)에서는 모델이 전혀 도움 안 됨(lift<1 일관). **grinding
(원래 이 프로젝트를 촉발한 유형)에서만 lift>1 일관되게 나타남** — 방향은 가설과 정확히 일치.
**단 n=13은 통계적으로 신뢰 불가능한 수준**(사건 2~4건 차이로 숫자가 크게 흔들릴 표본) —
"확정된 효과"로 읽으면 안 됨.

## 종합 판정

1. **일반 조기경보 신호로는 REJECTED** — pooled lift가 1.0 근처/이하, trivial baseline보다도
   못함.
2. **그라인딩형 전환에 한해서는 방향성 있는 흥미로운 신호** — 가설이 맞다는 쪽으로 4개 horizon
   전부 일관되게 나왔으나 n=13이라 노이즈와 구분 불가. 자동 배포·매매 근거로 쓰기엔 이름.
3. **분류기 자체 성능(OOS bal_acc 0.75~0.78)은 GBM2와 비슷한 수준으로 준수** — 다만 이걸
   "조기경보"라는 원래 목적에 못 쓴다는 게 이번 결과의 핵심.
4. **배포 안 함.** `dashboard/server.py`/`app.js` 무변경.

## How to apply (재제안 시)

- "그라인딩형 전환 조기경보"를 다시 시도하려면, 그라인딩 표본 자체가 늘어날 때까지(추가 수개월
  데이터 축적, 또는 er_24 컷을 완화해 표본을 늘리는 재정의) 기다리는 게 맞다 — 지금 재도전해도
  같은 n=13 함정.
- "일반 조기경보"(clean+grinding 안 가리고)는 이 결과로 재확인 불필요 — trivial baseline도
  못 넘는 구조적 결과.
- `compression_release_up/down`이 이 모델 예측력의 상당 부분을 차지한다는 건 재사용 가치 있는
  발견 — 다른 변동성 관련 축을 다룰 때 우선 검토 대상.

산출물: `tmp/eth_regime_volexpand_20260827/model.joblib`,
`data/ensemble/reports/eth_regime_volexpand_20260827_report.json`,
`tmp/eth_volexpand_regime_label_design_20260827/`(Phase 0/1 차트+그리드).
