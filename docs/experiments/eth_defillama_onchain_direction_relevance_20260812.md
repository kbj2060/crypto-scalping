# ETH DefiLlama 온체인 지표(TVL/DEX거래량/수수료) 방향 relevance + 신규 방법론 함정 (2026-08-12)

## 배경

사용자 지시: 최근 리서치(퀀트펀드 데이터 소스 조사)가 짚은 5개 대체데이터 후보(CoinGlass,
Dune, DefiLlama, LunarCrush, Santiment) 재검토. 이 문서는 **완전 무료 + API키 불필요 + 전체
TRAIN/VAL/OOS 구간 백필 가능**함을 사전 확인한 **DefiLlama**만 다룬다 — 나머지 4개의 접근성
확인 결과는 `docs/experiments/eth_alt_data_source_feasibility_check_20260812.md`(같은 세션)
참고.

## 방법

- 스크립트: `scripts/verify_eth_defillama_onchain_direction_relevance_20260812.py`
- 지표 3종(전부 일별, API키 불필요, `api.llama.fi` 무료 엔드포인트 직접 확인):
  `historicalChainTvl/Ethereum`(TVL, 2017-09~), `overview/dexs/ethereum`(DEX 거래량,
  2018-11~), `overview/fees/ethereum`(가스+프로토콜 수수료/매출, 2018-03~) — 전부 TRAIN/VAL/
  OOS 전체 구간을 커버.
- 원값 + 파생 2종(day-over-day 변화율, 7일 이동평균 대비 편차) = 총 9개 컬럼.
- **표준 절차**: 신규 raw-level 피쳐는 학습 전 `corr(price)`/`corr(시간순번)` 오염도부터 확인
  (배제 기준 0.561), 그다음 튜닝 없는 LightGBM 홀드아웃으로 FINAL12 단독 vs +DefiLlama 대조.

## ⚠ 신규 방법론 함정 발견 — 일별 forward-fill 데이터의 `mutual_info_classif`가 degenerate함

경제적으로 서로 다른 9개 컬럼(TVL, DEX거래량, 수수료, 각각의 원값/diff1/ma7dev) 전부가
**소수점 4자리까지 동일한 MI=0.1066**을 반환했다 — 상관관계가 전혀 다른데도(`eth_chain_tvl`
corr(price)=+0.842 vs `eth_dex_volume_diff1pct` corr(price)=-0.008) MI만 완전히 같다는 건
계산 버그를 의심하게 만들었다.

**합성 데이터로 직접 재현·확정**: 일별 값을 288개 bar로 forward-fill하고 라벨도 zigzag_action처럼
여러 bar 동안 값이 유지되는 block 구조로 시뮬레이션하니, **완전 무관한 순수 랜덤 daily 시리즈도
실제 트렌드가 있는 시리즈와 거의 동일한 MI(0.1268 vs 0.1268)**를 반환했다. 원인: `mutual_info_
classif`의 k-NN(KSG) 기반 연속형 추정량이, 288개씩 반복되는 대량의 중복값(tie) 앞에서 내부적으로
추가하는 tie-breaking 노이즈가 실제 값 차이보다 **day-boundary와 라벨 block-boundary의 정렬
구조**에 더 크게 좌우돼 degenerate한(내용과 무관한) 숫자를 재현성 있게 반환하는 것으로 확인됨.

**영향 범위**: 이 문제는 corr(price)/corr(시간순번) 체크(Pearson/Spearman, 중복값에 강건함)와
LightGBM 홀드아웃 비교(실제 트리 분기, 이 버그의 영향 없음)에는 해당하지 않는다 — 아래 두
결과는 신뢰할 수 있다. **다만 앞서 완료한 Fear&Greed 실험
(`eth_fear_greed_backfill_direction_relevance_20260812.md`)의 MI 수치도 같은 계열(일별
forward-fill)이라 정도는 다를 수 있어도 오염 가능성이 있음** — 그 문서에 정정 노트를 추가함(3종
MI가 서로 달랐다는 점은 완전 동일 케이스는 아님을 시사하나, 안전하게 "참고용, GBM 홀드아웃이
결정적 근거"로 재해석함).

**일반 교훈(신규, 이 프로젝트 표준 절차에 추가 권고)**: 일별(또는 저해상도) 외부 데이터를
5분봉에 forward-fill해서 `mutual_info_classif`/`mutual_info_regression`으로 relevance를
잴 때는, 반드시 무관한 랜덤 daily 시리즈를 같은 방식으로 forward-fill해서 대조군 MI를 함께
계산하거나, 애초에 일별 해상도에서 직접 상관/MI를 재고(re-aggregate) 상태로 검증할 것 — bar
단위 MI만 믿으면 이번처럼 완전히 무관한 데이터도 "신호 있음"으로 보일 수 있다.

## 결과 — 오염도 체크 (신뢰 가능)

| 컬럼 | corr(price) | corr(시간순번) | 판정 |
|---|---:|---:|---|
| `eth_chain_tvl`(원값) | +0.842 | +0.591 | **오염** (0.561 기준 초과) |
| `eth_dex_volume`(원값) | +0.465 | +0.485 | 통과(여유 크지 않음) |
| `eth_fees_revenue`(원값) | +0.530 | +0.402 | 통과(여유 크지 않음) |
| `eth_chain_tvl_diff1pct` | +0.063 | +0.050 | 통과, 오염 거의 없음 |
| `eth_dex_volume_diff1pct` | -0.008 | +0.006 | 통과, 오염 거의 없음 |
| `eth_fees_revenue_diff1pct` | -0.007 | +0.008 | 통과, 오염 거의 없음 |
| `eth_chain_tvl_ma7dev` | +0.142 | +0.101 | 통과 |
| `eth_dex_volume_ma7dev` | +0.012 | +0.027 | 통과 |
| `eth_fees_revenue_ma7dev` | +0.008 | +0.006 | 통과 |

원값(특히 TVL)은 가격추세와 강하게 얽혀 있어 detrend 없이 쓰면 위험 — 이 프로젝트의 표준
패턴(`CapMVRVCur` 등)과 일치. Detrend 파생 6개는 전부 오염도 통과.

## 결과 — LightGBM 홀드아웃 (신뢰 가능, 결정적 근거)

| 구성 | VAL balanced_acc | VAL macro_f1 | OOS balanced_acc | OOS macro_f1 |
|---|---:|---:|---:|---:|
| FINAL12(패널가용 8개) 단독 | 0.469 | 0.448 | 0.466 | 0.446 |
| FINAL12 + DefiLlama(오염도 통과 8개) | 0.462 | 0.437 | 0.462 | 0.434 |

**VAL/OOS 둘 다, 두 지표 다 오히려 소폭 악화**(-0.4~1.2pp) — Fear&Greed 실험과 정확히 같은
패턴(추가해도 안 도움, 살짝 해가 됨).

## 결론

**부정 결과.** DefiLlama의 ETH 체인 지표(TVL/DEX거래량/수수료)는 일별 해상도 외부 지수를
5분봉에 forward-fill하는 방식의 구조적 한계(Fear&Greed에서 이미 확인된 문제)를 그대로
반복한다 — 오염도는 대체로 통과하지만 실제 홀드아웃 성능은 개선이 없다. 이 후보는 여기서
닫는다.

**부수 성과**: 일별 forward-fill 데이터의 `mutual_info_classif` degenerate 문제를 발견·재현·
확정 — 앞으로 이 계열(F&G류 저해상도 외부지수) 신규 후보를 검증할 때 재발 방지용 표준 절차로
추가할 가치가 있음(대조군 랜덤 daily 시리즈로 MI 베이스라인 확인, 또는 GBM 홀드아웃만 신뢰).

## 산출물

`tmp/eth_defillama_onchain_20260812/` — `contamination_and_mi_report.json`(MI 수치는
참고만, 아래 caveat 참고), `holdout_comparison.json`.
