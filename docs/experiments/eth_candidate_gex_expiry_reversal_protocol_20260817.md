# ETH/BTC Deribit 옵션 만기 감마 리버설 — 사전 등록 forward 테스트 프로토콜 (2026-08-17)

## 상태: **설계 완료, 데이터 축적 중 (Tier 0 스모크 테스트만 완료, 실제 판정 없음)**

## 배경

`docs/eth_direction_alpha_non_microstructure_research_20260817.md`의 1순위 후보. Weiss,
Gaudiosi, Zhou & Webb (2026), *Finance Research Letters*, DOI:10.1016/j.frl.2026.110340:
Deribit BTC 옵션 만기 전후 통계·경제적으로 유의한 일중 리버설. ATM open interest가 높은
날에 집중되고, 누적 감마 익스포저가 음수일 때(딜러 숏감마 → 동방향 강제 헤징 → 만기 후
반대방향 되돌림) 최강. 연 ~$50M 규모.

**중요한 한계**: 이 논문 원문은 ScienceDirect 유료 접근이라 본 세션에서 방법론 원문(정확한
ATM OI 임계값, 만기 전후 윈도우 폭, 감마 부호 산출 방식)을 확보하지 못했다. 아래 조작적
정의는 초록 수준 설명에 기반한 **이 리포 자체의 합리적 프록시**이지 논문 재현이 아니다 —
`feedback_modeling_needs_literature_grounding` 원칙에 따라 이 한계를 숨기지 않고 명시한다.

## 가설

Deribit ETH/BTC 옵션 일일 만기(08:00 UTC) 전후로, **만기 직전 front-month 누적 감마
익스포저가 음수 + ATM open interest가 높은 날**에는 만기 전 방향과 만기 후 방향이
반대(리버설)이고, 감마가 양수이거나 ATM OI가 낮은 날에는 이 패턴이 없거나 약하다.

## 데이터

| 자원 | 경로 | 상태 |
|---|---|---|
| GEX 요약 (시간별, `total_gex_usd`/`front_month_gex_usd`) | `data/live/deribit_gex.duckdb::gex_summary` | 2026-08-15~ 수집 중 (수집기 정상, [[eth_dev_collector_root_cause_and_fix_20260817]]) |
| 옵션 체인 스냅샷 (`open_interest`, `strike`, `days_to_expiry`, `gamma_bs`) | `data/live/deribit_gex.duckdb::option_chain_snapshot` | 동일 |
| ETH 5m 가격 | `binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv` | 이번 세션에 `scripts/extend_klines_20260713.py`로 2026-08-04→08-16 23:10까지 갱신. **본 테스트는 이 파일을 계속 갱신하며 사용** |

과거 백필 불가(기확인, [[eth_candidate_lob_microstructure_data_scoping_20260817]]와 동일
제약) → **순수 forward 전용**. 캐노니컬 VAL(2025-09~12)/OOS(2026-01~03)와 겹치지 않으므로
Fresh-Forward 규칙에 따라 새 split 경계를 여기 명시한다: **조건화 관측 시작 2026-08-15,
판정 시점은 아래 3-tier 문턱 도달 시.**

## 조작적 정의 (자체 프록시, 논문 원문 미확보 명시)

- **만기 시각**: Deribit ETH/BTC 옵션은 매일 08:00 UTC 정산. 월물(매월 마지막 금요일)은
  별도 표시하되 표본이 너무 적어(연 12회) 이번 프로토콜의 1차 대상에서 제외, 탐색적
  2차 arm으로만 기록.
- **감마 부호**: 만기 시각(08:00 UTC) **직전** `gex_summary` 관측치의 `front_month_gex_usd`
  부호. NEG = 음수, POS = 0 이상. (causal, 미래 데이터 없음)
- **ATM OI 고조 여부**: `option_chain_snapshot`에서 만기 직전 최신 스냅샷 기준
  `days_to_expiry <= 1.0` AND `|strike/underlying_price − 1| <= 0.02`인 계약의
  `open_interest` 합 = `atm_oi_frontexp`. "고조"는 절대 임계값이 아니라 **누적 표본의
  trailing 상위 tercile** (표본이 쌓일수록 재계산 — 논문의 절대 임계값을 모르므로 상대
  임계값으로 근사).
- **리버설 측정 윈도우 (2개, 사전 등록 — 이후 추가 금지)**:
  - Broad: pre = [07:00, 08:00) UTC, post = [08:00, 09:00) UTC
  - Tight: pre = [07:45, 08:00) UTC, post = [08:00, 08:15) UTC
  - (기존 20-23 UTC/펀딩정산 연구의 버킷 설계와 동일한 방법론적 일관성 유지)
- **리버설 판정**: pre 구간 로그수익률과 post 구간 로그수익률의 부호가 반대.

## 사전 등록 판정 기준 (falsification, 결과를 보기 전에 고정)

리포 표준 방법론(`eth_session_split_feature_price_correlation_20260817` 문서와 동일 골격)을
그대로 채택한다:

1. **부호 일관성**: NEG-감마 & 고조-OI 조건에서 broad·tight 두 윈도우 모두, 그리고 축적되는
   3개 롤링 서브기간(아래 tier별로 분할) 전부에서 "pre 부호 ≠ post 부호"의 평균 방향이
   일관되어야 한다.
2. **rotation null**: 만기 시각으로 취급하는 UTC 정각을 1~23시간 회전시킨 귀무분포 대비
   |z| ≥ 2 (세션 분할 연구와 동일 기법).
3. **pooled 초과**: NEG-감마&고조-OI 조건부 |리버설 크기|가 조건 없는 전체 평균보다 커야 한다.
4. **비용 게이트**: bp 환산 후 taker round-trip 10bp(WS-A 상수) 기준 breakeven 통과.
   Weiss et al.의 메커니즘(딜러 강제 헤징의 반대편에 서는 것)은 구조상 되돌림을 페이드하는
   성격이라 maker 체결 적합 가능성이 있음 — 20-23 UTC 연구와 같은 논리로, taker 실패 시
   maker 경로를 2차로 검토.
5. **월(일) 클러스터 기준 유의성**: 만기는 하루 1회뿐이라 유효 표본 단위는 "거래"가 아니라
   "NEG-감마&고조-OI 조건을 만족한 날"이다. 20-23 UTC 연구에서 월 단순평균/월 표준편차/
   t-stat을 썼던 것과 동일하게, **일별 리버설 크기의 평균/표준편차/t-stat**을 1차 통계로 쓴다.

**이 5개 중 하나라도 실패하면 그 시점 표본 크기와 무관하게 기각.** 사전 등록 이후 새 윈도우
정의나 새 임계값을 추가하는 것은 forking path이므로 금지 — 바꾸려면 이 문서를 새 버전으로
남기고 이전 버전을 무효 처리한다.

## 3-tier 데이터 축적 문턱 (L2 스코핑 문서와 동일 패턴)

| Tier | 조건 | 의미 | 예상 도달 시점 (일 1회 만기 기준, 현재 관측상 NEG-감마 발생률 0/2일이므로 보수적 추정) |
|---|---|---|---|
| **Tier 0 — 스모크 테스트** | 지금 | 파이프라인 동작 확인만, 통계적 주장 0 | **완료, 아래 결과 참조** |
| **Tier 1 — 탐색적** | NEG-감마&고조-OI 조건 만족일 ≥ 20일 누적 | 최초 사전 등록 판독 허용, 단 승격 주장 불가 (단일 윈도우 관측) | 미정 — 지금까지 2/2일이 POS-감마였다는 점을 반영해 축적 필요 (아래 참고) |
| **Tier 2 — 승격 후보 자격** | 조건 만족일 ≥ 60일 (3개 롤링 서브기간으로 쪼갤 수 있는 최소치) | 판정 기준 5개 전체 배터리 통과 시에만 후보 지위 | Tier 1 통과 후에만 진행 |

## Tier 0 결과 (2026-08-17, 파이프라인 스모크 테스트 — 통계적 의미 없음)

| 날짜 | front_gex (만기 직전) | ATM OI | pre_ret(07-08 UTC) | post_ret(08-09 UTC) | 리버설? |
|---|---|---|---|---|---|
| 2026-08-15 | **+2.05e6 (POS)** | 6,142 | −5.6bp | −17.3bp | no |
| 2026-08-16 | **+3.03e6 (POS)** | 5,269 | +8.7bp | −4.1bp | YES (그러나 POS-감마 조건이라 가설 대상 아님) |

- 조인 파이프라인(gex_summary ↔ option_chain_snapshot ↔ 5m 가격) 정상 동작 확인.
- **2일 모두 front_month_gex_usd가 양수(POS)였다** — 논문의 핵심 조건화 변수(NEG-감마)가
  아직 한 번도 관측되지 않았다. 즉 지금 당장 판독을 시도해도 가설의 조건절 자체가 공집합이라
  무의미하다. Tier 1 도달까지 **날짜 수뿐 아니라 "NEG-감마가 실제로 발생하는 빈도"에도
  좌우된다** — ETH 감마가 얼마나 자주 음전환하는지 자체가 미지수이므로, 위 표의 "예상 도달
  시점"은 이번 관측만으로는 추정 불가. 매주 재확인 권장.

## 다음 단계

1. 수집기 건강 유지 (완료, [[eth_dev_collector_root_cause_and_fix_20260817]]).
2. `binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv`를 주기적으로 `extend_klines_20260713.py`로
   갱신(현재 수동 — cron화는 별도 작업, 이번 세션 범위 밖).
3. 최소 주 1회, NEG-감마&고조-OI 발생일 카운트만 추적(판독 아님) → Tier 1 도달 확인.
4. Tier 1 도달 시 이 문서에 새 절을 추가해 사전 등록 기준 5개를 그대로 적용, 결과를
   덧붙인다 (판정 기준을 사후 변경하지 않는다).
5. BTC도 같은 프로토콜로 병행 관측 중(`gex_summary`에 이미 BTC 행 존재) — BTC는 감마가
   더 자주 음전환하는 경향이 Tier 0에서 이미 보였음(front_month_gex_usd 음수 관측 다수,
   본문 표 참조) → **BTC가 Tier 1에 먼저 도달할 가능성** 있음, 별도 추적.

## 이력 연속성 수정 — dev 고아 duckdb 병합 (2026-08-23)

2026-08-17 서버 이관 커밋이 `CREATE TABLE IF NOT EXISTS`만 하고 기존 dev duckdb를 복사하지
않아, dev 시절 이력(2026-08-15 06:03 ~ 08-17 04:00 UTC)이 dev 로컬 고아파일에만 남아
Tier 1 카운트에서 조용히 빠질 위험이 있었다(2026-08-20 발견, 미조치 상태였음). 2026-08-23
물리 병합으로 해소:

- **병합 내용**: gex_summary 94행(BTC/ETH 각 47) + option_chain_snapshot 70,560행을 서버
  라이브 DB(`data/live/deribit_gex.duckdb`)에 삽입. 시간 겹침 없음(dev max 04:00:02 UTC <
  server min 04:35:53 UTC), `recorded_at_utc < 서버최소` WHERE 가드 + 삽입 전후 카운트
  일치 + (ts,currency) 중복 0 검증 통과. 컬렉터 크론(매시 정각) 사이 안전창(06:32 UTC)에
  실행, 단일 writer 원칙 위반 없음.
- **스크립트/보존**: `scripts/ops/merge_deribit_gex_dev_orphan_20260823.py`. 병합 전 서버 DB
  백업 `data/live/deribit_gex.duckdb.bak_pre_dev_orphan_merge_20260823`, dev 원본 사본
  서버 `data/research/deribit_gex_dev_orphan_20260815_0817.duckdb`(dev 로컬 원본도 유지).
- **효과**: 병합으로 복원된 **2026-08-16 BTC가 현재까지 유일한 조건만족일(NEG-감마&고조-OI)**
  — 서버 이관 후 6일간(08-17~22) NEG-감마 pre-expiry 관측이 BTC/ETH 모두 0건이므로, 이
  병합이 없었다면 Tier 1 카운트가 0에서 시작할 뻔했다.

### 첫 전체이력 조건일 카운트 (2026-08-23, 판독 아님 — "다음 단계" 3번 항목의 첫 실행)

카운트 스크립트 신설: `scripts/count_gex_tier1_condition_days_20260823.py` (조작적 정의
그대로 구현: pre-expiry = 당일 08:00 UTC 이전 최신 스냅샷, ATM = days_to_expiry≤1.0 &
|strike/spot−1|≤0.02, 고조 = 누적 표본 상위 tercile). 서버에서 주 1회 실행 권장.

| | 관측일 | NEG-감마일 | 조건만족일(NEG&고조OI) | Tier1(20일)까지 |
|---|---:|---:|---:|---:|
| BTC | 8 (08-15~22) | 1 (08-16) | **1 (08-16)** | 19일 |
| ETH | 8 (08-15~22) | 0 | 0 | 20일 |

발생률 실측 BTC 1/8(12.5%), ETH 0/8 — 이 비율이 유지되면 BTC Tier 1은 대략 5개월 뒤
(2027-01경)로 추정되나, 표본 8일의 비율 추정은 극히 불확실(희귀사건 게이트라는 성격 불변).
"BTC가 먼저 도달" 원 가설은 이 카운트로 다시 뒷받침됨(유일 조건일이 BTC).
