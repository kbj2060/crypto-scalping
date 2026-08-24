# ETH 청산 히트맵 자석효과 신호 — 일리아스 적용 스코핑 (2026-08-22)

상태: **신규 프로젝트로 착수 비권고 — 기존 축(tail_risk_1m 09-15 게이트)으로 흡수, 핵심 개념(풀 스윕→반전)은 이미 구현·검증된 기존 신호로 대체**

## 배경

사용자가 제시한 방법론(코인글래스식 청산 히트맵에서 거리가중 비대칭 점수를 계산해 방향을 판단하고, 근거리 고레버리지 풀 스윕을 진입 타이밍으로, 종가 이탈을 무효화 조건으로 삼는 7단계 절차 + funding/OI/롱숏비율 교차확인)을 일리아스(ETH 방향/리스크 서브프로젝트)에 적용할 수 있을지 스코핑했다. 결론부터: 이 방법론이 요구하는 정확한 데이터 상품(코인글래스 청산 히트맵)은 이미 2회 벤더 검토에서 사망 판정을 받았고, 그 밑에 깔린 핵심 메커니즘(밀집된 스탑/청산 주문이 국소적 자석처럼 작동, 스윕 후 반전)은 이미 이 repo의 22개 리버설 신호 스코어카드에서 1위(`liquidity_sweep`, lift 3.01~3.40x, 독립 윈도우 검증 완료)로 구현돼 있다. 새로 만들 것이 거의 없다.

## 1. 데이터 실현가능성 — 코인글래스는 이미 2회 기각됨

`docs/experiments/eth_alt_data_source_feasibility_check_20260812.md`에서 코인글래스 API를 결제 없이 공식 문서/GitHub 레퍼런스로 직접 조사 완료(요금제 4단계, `futures/liquidation`·`futures/open-interest`·`futures/long-short-ratio` — 사용자가 제시한 히트맵·OI·롱숏비율 교차확인 3종 전부 포함 — 바이트 단위로 동일한 "플랜×인터벌→lookback" 제한 확인):

- 5분봉: 어떤 유료 등급도 불가. Professional($699/월)조차 최근 **60일**뿐 — 이 프로젝트의 TRAIN(2024-06~)/VAL/OOS 백필은 구조적으로 불가능.
- 시간봉: Professional 최대치(720일)도 TRAIN 시작보다 82일 모자람.
- 일봉만 all-time 커버 — 단, 이미 이 프로젝트가 2번(DefiLlama, CoinMetrics) 겪은 저해상도 forward-fill 오염 패턴([[feedback_forward_fill_mutual_info_degenerate]], [[feedback_raw_feature_price_trend_contamination]])과 동일 리스크.
- 판정 원문: "**CoinGlass 결제는 권장하지 않음**... 돈을 낸다고 풀리는 문제가 아님."

`eth_candidate_polymarket_eth_market_coverage_scan_20260817.md`에서 재확인: "Tier A('코인글래스 청산 히트맵')는 벤더 경유로는 막다른 길."

**즉 사용자가 첨부한 차트(레버리지 티어별 청산 히트맵)를 그대로 데이터 소스로 쓰는 프로젝트는 이미 두 번 닫힌 문이다.** 이번 조사로 세 번째 확인만 추가된 셈이다.

## 2. 문헌 그라운딩 — 방법론 자체는 학계 근거 없음, 밑에 깔린 메커니즘은 있음(단 다른 시장)

OpenAlex 9개 질의로 4개 축을 확인했다(제목/저자/인용수는 대화 로그 참고, 요지만 정리):

| 축 | 지지 수준 | 핵심 근거 |
|---|---|---|
| 청산 캐스케이드 → 가격충격 | 약함, 최신(2026, 대부분 미인용/비피어리뷰) | 가장 강한 결과도 **방향이 아니라 크래시확률**을 예측(OI변화·누적펀딩비·실현변동성으로 8시간내 ≥5%크래시, OOS AUROC 0.76). 오히려 한 논문(subcritical branching, arXiv 2026)은 2025-10 최대 캐스케이드조차 자기증폭 안 됨(branching ratio λ≈0.1-0.2)을 발견 — "캐스케이드=자석" 서사에 반증 |
| 스탑 클러스터링 → 스윕후반전/돌파 | **강함, 피어리뷰(단 FX)** | Osler(2003, *Journal of Finance*, 223회 인용) — 실제 은행 스탑로스/익절 주문데이터로, 익절주문은 라운드넘버에 뭉쳐 반전을, 손절주문은 라운드넘버 살짝 너머에 뭉쳐 가속(돌파)을 만든다는 것을 실증. **사용자 5단계(윅=유지, 종가이탈=반전) 로직과 메커니즘상 거의 동일**하나, 크립토 청산이 아니라 FX 스탑 주문 기준 데이터다 |
| OI/funding → 수익률 예측 | 약함~혼재 | 크래시확률 예측이 최선(위와 동일 논문). 방향을 직접 예측하는 강한 피어리뷰 결과는 찾지 못함 |
| 코인글래스식 **추정** 히트맵 방법론 자체 | **전무** | "Coinglass"로 직접 검색해도 무관한 논문뿐. 유일하게 코인글래스를 실제로 쓴 논문(**$176.6B, 2026, 비피어리뷰 Zenodo 자가아카이브**)조차 히트맵이 아니라 코인글래스의 "청산 히스토리"(실현된 기록)만 사용하고 "어떤 추정치·모델·프록시도 쓰지 않는다"고 명시 — 진지하게 다룬 유일한 사례도 이 히트맵 추정 상품 자체는 일부러 피했다는 뜻 |

## 3. 개념적 선행연구 — "풀 스윕 후 반전"은 이미 이 repo의 1위 신호

`docs/experiments/eth_broad_evidence_signal_sweep_20260814.md` — `liquidity_sweep` 정의: "low가 직전 48bar 스윙로우 하향 돌파 후 종가는 위로 복귀(스탑헌트)". 22개 신호 중 바닥반전 **1위**(3.01x, 독립 2차 윈도우 3.40x로 재확인, 순위안정성 Spearman 0.976/0.924), 상단반전 **1위**(2.78x).

이게 정확히 사용자 방법론 4~5단계(근거리 풀 스윕 → 반전, 종가 이탈은 방향전환 확인)와 같은 구조다 — 차이는 **"풀"을 무엇으로 정의하느냐**뿐이다. 이 repo는 이미 대안 앵커를 여러 번 테스트했고 전부 기존 방식(48bar 롤링 스윙 고/저)에 졌다:

- iFVG(Fair Value Gap) 앵커 + 스윕 조합: 0.65x/0.56x — 기존 3.01x보다 **악화**
- 전일 레벨 앵커(Yush Y7): 1.65x — 기존 3.01x보다 악화, 문서 원문: "스윕 개념은 유효하지만, 앵커를 '전일 레벨'로 바꾸면 오히려 나빠진다"

레버리지 티어 추정 청산가는 이 앵커 실험 목록에 없던 새 후보이긴 하나, (a) 데이터 자체가 없고(§1) (b) 지금까지 테스트한 모든 대안 앵커가 스윙 기반 원본보다 나빴다는 사전 증거가 쌓여 있어, 기대치를 낮게 잡아야 한다.

또한 이 신호의 **진입측 사용은 이미 금지된 anti-goal**이다(`eth_omega461_evidence_intervention_surface_ceiling_20260815.md`, 이전 quality-relabel OOS-reversal 사례가 근거) — 청산 기반으로 앵커를 재구성한다고 이 제약이 자동으로 풀리지는 않는다. 청산측(exit) 사용은 이미 6/6창에서 발화bar가 매칭랜덤bar와 통계적으로 구별 안 됨으로 기각됐다.

## 4. 사용자 7단계 방법론 vs 이 repo의 실제 상태

| 사용자 단계 | 이 repo 상태 |
|---|---|
| 1. 풀 정량화(가격대별 청산물량) | 데이터 없음 — 코인글래스 히트맵 접근 불가(§1) |
| 2. 거리가중 비대칭 점수 | 구현 대상 데이터가 없어 계산 불가 |
| 3. 경로저항(진공구간) | 동일 |
| 4. 근거리 풀 스윕 = 타이밍 | **이미 구현·검증됨**, 단 앵커는 청산레벨이 아니라 48bar 스윙고/저(§3) |
| 5. 윅=유지, 종가이탈=반전(무효화) | 개념적 근거 있음(Osler, §2) — 스윙 앵커 버전에는 미적용된 세부 로직. 신규 시도라면 여기가 유일하게 남은 저비용 후보(단, §3의 진입측 anti-goal 제약과 별개로 검토 필요) |
| 6. funding/OI/롱숏비율 교차확인 | funding만 수집 중(Binance 공식 아카이브, ETHUSDT 연속 ~2.5년). **OI·롱숏비율은 수집기 자체가 없고, 유일한 소스(코인글래스)가 §1에서 이미 막힘** — 이 교차확인 레이어는 지금 구축 불가 |
| 7. 확률분포/시나리오화 | 방법론 자체는 타당하나 입력(1~3, 6)이 없어 적용 대상이 없음 |

## 5. 그나마 살아있는 유일한 청산 관련 축

`docs/experiments/eth_candidate_liquidation_feed_features_cheap_gate_20260817.md` — `tail_risk_1m`(Binance `@forceOrder` **실현** 청산 이벤트, 히트맵 아님)에서 **컨트래리언** 스파크가 발견됐다(`liq_net_z_12`, h=3 IC +0.131 — 롱청산 뭉침→상방, 숏청산 뭉침→하방). 단, 두 가지 이유로 아직 미확정:

1. 유효 구간이 24일뿐(2026-07-18 WS 엔드포인트 버그 수정 이후만 유효) — 8주 누적 시점인 **2026-09-15 이후**에야 결정 게이트(B1 존재성/B2 증분성) 실행 가능. 사전등록 완료: 주 지평 h=3로 변경, `lag1_ret × large_(long|short)_recent` 상호작용항 추가.
2. 선형 증분 미확립 — ridge Δρ의 부트스트랩 CI가 0을 가로지름(+0.0015, [−0.042, +0.046]), "리버설의 재표현"일 가능성을 아직 배제 못함.

부가 발견: 방향 lift는 미미하나 **vol-lift 1.59x**는 확인됐다 — 방향이 실패하면 리스크/베토 레이어로 별도 제안될 예정. 이는 §2 문헌(크래시확률 예측이 최선)과 spot-perp basis 축이 인용한 문헌(He/Manela/Ross/von Wachter, "Fundamentals of Perpetual Futures": basis는 방향이 아니라 향후 변동성·숏청산밀집을 예측)과 정확히 같은 결을 가리킨다 — **청산/포지셔닝 데이터는 이 repo와 문헌 양쪽에서 일관되게 "방향"보다 "변동성/리스크" 쪽에서 더 그럴듯하다.**

## 6. 결론 및 권고

1. **사용자가 제시한 정확한 방법론(코인글래스 히트맵 기반)으로 신규 프로젝트 착수는 비권고.** 데이터 접근이 2회 기각으로 막혀 있고(§1), 방법론(추정 히트맵) 자체의 학술 근거도 없다(§2 축4).
2. **핵심 아이디어(밀집 스탑/청산 = 국소 자석, 스윕 후 반전)는 이미 구현·검증된 형태로 이 repo에 존재한다**(`liquidity_sweep`, 3.01~3.40x) — 별도 프로젝트가 아니라 기존 신호로 이미 답이 나와 있다. 새 앵커(청산레벨)를 시도할 데이터가 없다는 점, 지금까지 시도한 대안 앵커가 전부 스윙 앵커보다 나빴다는 사전 증거가 있다는 점에서 기대치를 낮춰야 한다.
3. **유일하게 열려 있는 진짜 청산 축은 `tail_risk_1m`의 09-15 게이트뿐** — 이미 설계·사전등록이 끝난 상태이며, 이번 조사로 추가할 신규 작업은 없다. 09-15 이후 세션에서 결정 게이트를 실행하면 된다.
4. **가장 근거 있는 재구성 방향은 "방향 신호"가 아니라 "리스크/사이징 레이어".** 문헌(크래시확률 예측이 최선, basis→변동성·청산밀집 문헌)과 이 repo 자체 관찰(vol-lift 1.59x, 방향 lift 거의 0)이 독립적으로 같은 결론을 가리킨다. 09-15 게이트에서 방향(B1/B2)이 실패하더라도, vol-lift 기반 리스크/베토 레이어 제안은 이미 사전등록돼 있다 — CLAUDE.md의 Futures Risk Sizing Contract(margin_fraction 예측 우선 원칙)와도 결이 맞는 방향이다.
5. **funding/OI/롱숏비율 교차확인 레이어는 지금 구축 불가** — OI·롱숏비율 수집기가 없고 유일한 소스(코인글래스)가 막혀 있다. funding만 있는 축은 이미 `global_funding_carry_contrarian`으로 소진 판정(registry).

## Open Issues

- (a) 09-15 이후 `tail_risk_1m` 결정 게이트 실행 — 이미 사전등록된 다음 단계, 별도 신규 설계 불필요.
- (b) 스윙 앵커 스윕 신호에 "윅 유지 vs 종가이탈 반전"(사용자 5단계) 세부 로직을 추가하는 저비용 실험은 미검증 상태로 남아 있으나, 진입측 사용이 anti-goal로 막혀 있다는 제약과 별개로 검토해야 하고, AMT/VSA/Yush의 반복된 "변형은 원본보다 나쁨" 패턴을 감안하면 우선순위는 낮다.
- (c) OI·롱숏비율은 코인글래스 외 무료/저비용 소스가 나타나면 재검토 여지가 있다 — Dune Analytics(계정 필요, 사용자 결정 대기 상태였던 항목, `eth_alt_data_source_feasibility_check_20260812.md` 참고)가 온체인 파생 근사치를 낼 가능성은 미확인.

## 추가 업데이트 (2026-08-22, 자체수집 가능성 재질문)

사용자가 "직접 데이터를 모아서 히트맵을 만들면 위 전략을 만들 수 있나"를 재질문. 확인 결과, §1 결론은 유지되나 실행 가능한 대안 경로가 하나 명확해졌다.

- **바이낸스 자체 OI 히스토리 API(`/futures/data/openInterestHist`)도 공식 문서상 "최근 30일치만" 제공**(WebSearch로 확인, `developers.binance.com` 공식 문서). 즉 코인글래스의 lookback 상한(§1)이 벤더 페이월 때문만이 아니라 상류 거래소 API 자체의 근본 제약임이 재확인됐다 — 자체수집도 "코인글래스보다 긴 과거"를 주지 못하고, 오늘부터 새로 쌓기 시작하는 것만 가능하다. §1의 백필 불가 결론은 자체수집으로도 풀리지 않는다.
- **청산가 계산 자체는 정확하다**(바이낸스가 `/fapi/v1/leverageBracket`로 심볼별 유지증거금 구간을 공개). 불확실한 것은 "그 가격·레버리지 조합에 실제 얼마나 포지션이 몰려있는가"이며, 이는 거래소만 아는 정보라 자체수집해도 우리가 만드는 값은 코인글래스의 추정치와 마찬가지로 **가정이지 데이터가 아니다**. §2 문헌결론(추정 히트맵 방법론 자체 무근거)이 자체제작 버전에도 동일하게 적용된다.
- **엔지니어링적으로 낮은 비용의 신규 축이 하나 있다**: 히트맵을 재구성하지 않고 원재료(OI, 롱숏비율)만 직접 폴링해 z-score 등으로 바로 피처화하는 경로. `tail_risk_interceptor.py`(`@forceOrder` WS push, 606줄, `data/live/tail_risk.duckdb`)와 메커니즘이 달라(OI·롱숏비율은 WS가 없고 REST 폴링만 가능) 기존 파일 확장이 아니라 같은 운영 패턴(`scripts/ops/supervisor_tail_risk_btc_sol_worker.sh` + systemd)을 따르는 신규 경량 poller가 필요하지만, 인증 불필요한 공개 market-data 엔드포인트라 구현 자체는 단순하다. 이 경로는 registry의 `global_funding_carry_contrarian` 항목이 재개 조건으로 명시한 "새 raw source"에 해당해 기존 소진 판정과 충돌하지 않는다.
- 단, forward-only 제약은 동일하게 적용된다 — 오늘 시작해도 `tail_risk_1m`과 같은 8주 축적 규칙을 적용하면 cheap-gate 실행은 10월 중순 이후에나 가능하고, TRAIN/VAL/OOS 캐노니컬 구간 백테스트에는 영구히 쓸 수 없다.
- 결론 불변(§6, 히트맵/비대칭점수 재구성은 여전히 비권고), 단 §Open Issues (c)가 구체화됨: OI·롱숏비율 raw poller는 낮은 비용·낮은 리스크의 실행 가능한 신규 축 — 착수 여부는 사용자 결정 대기.

## 추가 업데이트 (2026-08-22, 실제 스캔 실행 결과)

사용자가 "바이낸스 자체 OI API로 바로 지금 테스트할 수 있지 않나"를 재질문 — 실제로 `openInterestHist`/`globalLongShortAccountRatio`를 호출해 확인했다(`/tmp/.../scratchpad/oi_lsratio_quick_scan.py`, 스크립트 자체는 세션 scratchpad라 repo에는 없음, 재현 시 아래 조건대로 재작성 필요).

### 정정: "30일"은 해상도별로 다름 — 5분봉은 1.7일뿐

실측 결과 이 API는 해상도와 무관하게 정확히 **500개 데이터포인트만 보존**한다(문서의 "30일"은 사실상 1d 해상도 기준 수치였고, 앞선 업데이트에서 이를 그대로 인용한 건 부정확했다):

- period=5m: 500×5분 = **1.7일**(ETHUSDT 실측 2026-08-20 00:30 ~ 08-21 18:05)
- period=1h: 500×1시간 = **~21일**(실측 경계 2026-07-31 23:00 — day20 요청은 480개 부분수신, day25/29는 500으로 clamp, day35는 HTTP 400 `parameter 'startTime' is invalid`로 명시적 거부)
- period=1d: **31일**(문서 "30일"과 대략 일치)

프로젝트 표준 해상도(5분봉)에서는 사실상 못 쓴다. 1시간봉이 그나마 실사용 가능한 표본이며, `globalLongShortAccountRatio`도 동일 패턴 확인.

### 실제 스캔 결과 (ETHUSDT, 2026-08-22 실행, `tail_risk_1m` cheap-gate와 동일한 방법론 축소판: 원인과 causal 계산 → 오염게이트 → IC 전반/후반 부호일관성 → 벤치마크 대조)

**5분봉(n=418, ~1.7일)**: `oi_z`가 종가와 스피어만 상관 **+0.7067**로 오염게이트(0.5) 즉시 위반 — 짧은 단일방향 구간이라 OI레벨과 가격레벨이 그냥 같이 우상향한 것. 나머지 피쳐도 전후반 분할에서 부호·크기 극단적으로 불안정(예: `gls_z` h=12 전반 -0.0121 vs 후반 -0.3706). **표본이 너무 작아 아무것도 말할 수 없음** — 사전 예상이 실측으로 확인됨.

**1시간봉(n=459, ~21일, 이 구간 always-long +29.84% 단일 강세장)**: 오염게이트는 전부 통과. `tls_z`(상위 트레이더 포지션 롱숏비율)가 h=1/3/12 전 지평에서 부호 일관(+0.0599/+0.1214*/+0.1634*, `*`=탐지문턱 0.0915 통과)하고 지평이 길수록 커지는 형태 — 단, 전후반 분할에서 크기가 불안정하고(h=12: 전반 +0.0495 vs 후반 +0.2871, 최근 절반에 쏠림), **무료 벤치마크(`lag1_ret`)도 같은 구간에서 이미 유의**(h=3 -0.1055*, h=12 -0.0943*, 둘 다 전후반 부호일관)해서 `tls_z`가 이 되돌림의 재표현인지 진짜 증분정보인지는 미확인 상태다(tail_risk_1m의 B2 게이트에 해당하는 ridge 벤치마크-대비-증분 검정은 이번 스캔에서 실행하지 않았다). 21일 전체가 단일 강세장 레짐이라는 점도 `eth_tabm_label_logic_retest_initiative_20260819`의 long_frac↔PnL confound와 같은 구조의 위험을 안고 있다.

### 결론

탐색적으로 완전히 죽지는 않았다(`tls_z`가 유일하게 살아있는 후보) — 하지만 tail_risk_1m의 실제 CONFIRM 문턱(0.025)보다 훨씬 느슨한 문턱(0.09~0.13)에서, 그것도 증분 미확인·단일레짐·전후반 불안정 상태다. §6 결론(방향보다 리스크레이어 프레이밍, 신규 poller가 유일한 실질적 진전 경로)은 그대로 유지된다. 이번 스캔으로 새로 확인된 것: **이 무료 롤링윈도우는 시간이 지나도 자라지 않는다**(오늘도 21일, 내일도 21일) — 더 깊은/다중레짐 표본을 원하면 poller를 지금 시작해 직접 축적하는 것 외엔 방법이 없다는 게 실측으로 재확인됐다.

## 추가 업데이트 (2026-08-22, B2 게이트식 증분검정 실행)

앞선 업데이트의 미확인 항목(`tls_z`가 무료 벤치마크의 재표현인지 진짜 증분인지)을 마저 검정. 1시간봉 데이터(n=447)를 시간순 1st-half(fit, 223, 08-02~08-11)/2nd-half(eval, 224, 08-11~08-21)로 나누고, 벤치마크 4종(`lag1_ret`,`ret_12`,`abs_ret_12`,`taker_imbalance`, 전부 klines에서 causal 계산) 단독 ridge vs +후보피쳐 ridge를 eval에서 비교, eval 내부 1일블록(24시간) 부트스트랩(n_boot=2000, seed=20260822)으로 Δρ 95% CI 산출 — `tail_risk_1m` cheap-gate의 B2 게이트와 동일한 설계.

**결과: 6/6 셀 전부 실패**(CI 하한 > 0 기준 미달):

| 후보 | h=1 delta [CI] | h=3 delta [CI] | h=12 delta [CI] |
|---|---|---|---|
| `tls_z`만(primary) | −0.0056 [−0.015,+0.006] | −0.0402 [−0.080,−0.007] | −0.0870 [−0.240,+0.038] |
| oi_z+oi_delta_z+gls_z+tls_z(exploratory) | +0.0650 [−0.131,+0.185] | +0.0983 [−0.062,+0.187] | −0.0809 [−0.302,+0.037] |

`tls_z` 단독은 세 지평 전부에서 점추정치 자체가 **음수**(벤치마크 단독보다 악화) — 앞선 스캔에서 보인 단변량 부호일관성은 무료 벤치마크(되돌림) 통제 시 사라졌다. 4종 전부 투입한 exploratory 버전은 h=1/h=3 점추정치가 양수이나 CI가 넓게 0을 가로지름(`n_day_blocks=9`뿐이라 예상된 정도의 폭).

`tail_risk_1m`의 `liq_net_z_12`가 겪었던 것과 동일한 실패 패턴("단변량 IC는 그럴듯하나 벤치마크 대비 증분 미확인")이며, 여기서는 한발 더 나아가 점추정치 자체가 음수인 셀이 다수라 그보다 약한 근거다.

### 이 실험축의 최종 상태

오늘 받을 수 있는 데이터(21일, 단일 강세장 레짐)로는 tls_z를 포함한 OI·롱숏비율 raw 피쳐 전부 증분을 못 보였다. 단, day-block 9개뿐인 표본은 확실한 기각을 내리기에도 너무 작다 — "신호 없음"이 아니라 "지금 표본으로는 판단 불가"가 정확한 결론이다. §6/앞선 업데이트의 결론(신규 poller로 직접 축적하는 것이 유일한 실질적 진전 경로)은 이번 결과로 오히려 강화된다 — 21일 단발 시도로는 끝나지 않았고, 더 깊고 다중 레짐인 표본 없이는 이 축을 더 밀어붙일 근거가 없다.

## 추가 업데이트 (2026-08-22, OI/롱숏비율 poller 서버 배포 완료)

앞선 두 업데이트의 결론(더 깊은 표본을 원하면 poller로 직접 축적하는 것 외엔 방법 없음)에 따라 실제 배포함.

### 구현

- **`oi_lsratio_collector.py`**(신규, repo root) — `tail_risk_interceptor.py`와 동일한 클래스 인터페이스(`start()`/`stop()`, 자체 asyncio 루프)로 설계. `openInterestHist`/`globalLongShortAccountRatio`/`topLongShortPositionRatio` 3종을 5분마다 폴링(`limit=12`로 최근 1시간 트레일링 재조회 → self-heal), 이미 저장된 `MAX(ts)`보다 새로운 행만 필터링해 삽입 — 순수 수집·저장만 하고 신호 계산(z-score 등)은 하지 않음(그건 축적된 테이블 위에서 별도 research 스크립트가 담당, tail_risk_interceptor의 raw-persist/shadow-signal 분리와 동일 원칙).
- **자체 duckdb 파일**(`data/live/oi_lsratio.duckdb::oi_lsratio_5m`) — `tail_risk.duckdb`(ETH, trading_bot.py 소유)·`tail_risk_btc_sol.duckdb`(BTC/SOL) 어느 쪽과도 안 겹침. 2026-08-17 BTC/SOL 첫 배포 때 겪은 "다른 프로세스가 같은 duckdb 파일에 동시쓰기" 장애를 처음부터 구조적으로 차단([[feedback_duckdb_single_writer_per_file]]).
- **`scripts/duckdb_persist_worker.py`**에 surgical 추가(신규 import 1줄 + `COLLECT_OI_LSRATIO` env flag, 기본값 **false**) — 이 파일은 서버에서 이미 살아있는 BTC/SOL 수집기가 같이 쓰는 공유 파일이라, 기존 배포가 새 flag를 명시적으로 켜지 않는 한 완전히 기존 동작 그대로임을 로컬에서 코드 검토로 확인 후 진행.
- **`scripts/ops/supervisor_oi_lsratio_worker.sh`** + **`scripts/ops/systemd/oi-lsratio-worker.service`**(참고용 템플릿) — `supervisor_tail_risk_btc_sol_worker.sh`와 동일 패턴(`_supervise.sh`+crontab @reboot, systemd는 이 호스트 sudo 권한 밖이라 미사용). `BOT_SYMBOLS=ETHUSDT` 기본값(Ilias가 ETH 전용이므로) — env var만 바꾸면 BTC/SOL로 코드 수정 없이 확장 가능.

### 로컬 검증

스크래치 duckdb로 부트스트랩→poll→재poll(중복 0건 확인)→status_line까지 기능 테스트 통과. `py_compile`로 신규/수정 파일 문법 확인.

### 서버 배포 (2026-08-22, `scripts/ops/handoff.sh` 사용)

1. `handoff.sh push server`로 4개 파일 전송.
2. push 직후 BTC/SOL 수집기 헬스체크(원격 1회성 명령) — 프로세스 정상, duckdb 최신행 방금 시각까지 기록됨 확인 → **push가 기존 라이브 수집기에 영향 없음 확인**.
3. `handoff.sh launch server oi_lsratio_worker`로 실행 시작 → 실제 로그(`logs/supervisor/oi_lsratio_worker_20260822.log`) 확인: `symbols=ETHUSDT, microstructure=False, tail_risk=False, oi_lsratio=True`, `sources_ok=3/3 new_rows=12` — 정상 동작.
4. crontab에 `@reboot` 항목 추가(기존 항목 grep으로 먼저 제거 후 재추가 — 재실행해도 중복 안 쌓이는 idempotent 방식) — 기존 크론잡 8개(주간앙상블·parity드리프트·altdata·watchdog·로그정리·백업·deploy_watcher·GEX수집·BTC/SOL tail-risk) 전부 그대로 확인, 새 줄만 추가됨.
5. 재확인: 워커 여전히 RUNNING(동일 pid), 진단용 임시 job 디렉토리 정리.

### 알려진 사소한 버그 (기능에 영향 없음, 우선순위 낮음)

poll 주기 동기화 로직(`sleep_sec = interval - (now%interval) + 90`)이 의도한 "경계+90초 버퍼"를 보장 못하고 0~300초 사이 아무 값이나 나올 수 있는 산술 실수가 있음(실제로 시작 26초 만에 첫 poll 발생). `fetch_limit=12`(1시간 트레일링) self-heal 덕에 데이터 누락·중복 없이 다음 사이클에서 자동 보정되므로 데이터 무결성에는 영향 없음 — 재배포까지 할 정도는 아니라 기록만 남기고 다음에 이 파일을 만질 때 같이 고치는 것으로 보류.

### 추가: BTC/SOL 확장 (같은날, 사용자 요청)

`BOT_SYMBOLS`를 `ETHUSDT`→`ETHUSDT,BTCUSDT,SOLUSDT`로 변경해 재배포. 재시작 시 실제로 위험한 지점을 하나 발견·회피함: **`handoff.sh stop`이 `_supervise.sh` 래퍼 프로세스만 죽이고, 그 자식인 실제 python 워커는 orphan으로 살아남았다**(pid 4122350, 여전히 `BOT_SYMBOLS=ETHUSDT`로 폴링 지속 중이었음 — `/proc/<pid>/environ`로 직접 확인). 그대로 재실행했다면 신·구 두 프로세스가 같은 `oi_lsratio.duckdb`에 동시쓰기 — 2026-08-17 BTC/SOL 첫 배포 때 실제로 터진 것과 동일한 장애 재현 위험이었음. orphan을 명시적으로 SIGTERM(필요시 SIGKILL)한 뒤 프로세스 목록에서 완전히 사라진 걸 확인하고 나서 재기동.

결과: ETH는 `resuming after ts=2026-08-22 03:30:00`으로 기존 13행 유실 없이 이어받음, BTC/SOL은 새 테이블(`oi_lsratio_5m_btc`/`_sol`)로 정상 부트스트랩. 코드 변경 없음(`oi_lsratio_collector.py`는 애초에 symbol-parametrized로 설계) — `supervisor_oi_lsratio_worker.sh`의 env var 한 줄만 변경.

**교훈**: 이 repo의 `_supervise.sh`+`handoff.sh stop` 조합으로 재시작할 때는 `stop` 이후 반드시 실제 자식 프로세스가 완전히 죽었는지 `pgrep`/`/proc/<pid>/environ`로 확인 후 재기동해야 한다 — supervisor wrapper가 죽었다고 자식까지 죽었다고 가정하면 안 됨. 같은 duckdb 파일에 쓰는 다른 신규 poller를 재시작할 때도 동일하게 적용.

### 추가: 축적 데이터 감사 + 실제 버그 수정 (같은날, 사용자 요청)

배포 후 실제 라이브 데이터를 직접 감사(`data/live/oi_lsratio.duckdb` 3테이블 + 비교용 `tail_risk.duckdb`/`tail_risk_btc_sol.duckdb`, 갭·중복·소스별 결측률·컬럼 degenerate 여부·파일간 컬럼중복 전부 확인).

**결과 요약:**
- 갭·중복: 0건(3테이블 전부). 컬럼 degenerate: 0건(8개 메트릭 컬럼 전부 실제 분산 있음). `tail_risk`와 컬럼 중복: 0건(청산이벤트 계열 vs OI/포지셔닝 계열, 완전히 다른 축).
- **실제 버그 발견**: OI/글로벌롱숏/상위트레이더롱숏 3개 소스가 같은 5분 타임스탬프를 항상 동시에 발행하지 않음(BTC는 12/12행 완전, ETH는 14행중2행·SOL은 13행중2행이 `sources_ok<3`인 부분NULL행). 기존 dedup 로직("이미 저장된 ts보다 새것만 삽입")은 먼저 도착한 불완전한 행을 저장한 뒤 워터마크를 전진시켜서, 나중에 도착하는 소스의 값을 **영구적으로** 못 채워 넣는 구조였음.
- (참고, 오늘 작업과 무관) 기존 ETH `tail_risk_1m`(5월~, trading_bot.py 소유)은 90.2% 커버리지·224개 갭(최대 2일+)·중복 16건 — WSL2 불안정 시기와 겹치는 이력성 문제.

**수정**: `oi_lsratio_collector.py`를 워터마크-INSERT 방식에서 **read-merge-delete-insert 업서트**로 변경 — 매 poll이 트레일링 윈도우(fetch_limit=12) 전체를 다시 병합해, 컬럼별로 `새 값이 있으면 새 값, 없으면 기존 값 유지`(COALESCE 방향, 좋은 기존값을 새 NULL로 덮어쓰지 않음) 후 `sources_ok`는 union으로 갱신. `schema_version=2`로 표시. 겸사겸사 poll 동기화 타이밍 산술버그(이전 업데이트에서 발견)도 같이 수정(불필요한 wraparound 제거).

**로컬 검증**: 4가지 시나리오 유닛테스트 전부 통과 — (1) 부분poll 저장, (2) 다음poll이 NULL 컬럼 채움(행 중복 안 생김), (3) 이후poll에서 한 소스가 다시 빠져도 기존 좋은값 안 지워짐, (4) 신규 타임스탬프는 정상 추가.

**재배포**: [[feedback_handoff_stop_leaves_orphaned_child]]에서 문서화한 절차 그대로 재적용 — push → stop → `pgrep`+`/proc/<pid>/environ`로 orphan 확인(**이번에도 실제로 orphan 발견**, pid 다름·같은 패턴) → 명시적 kill → 재확인 → relaunch. 재시작 후 ETH/BTC/SOL 전부 기존 타임스탬프에서 정상 resume(유실 없음) 확인됨.

**라이브 재확인 (완료)**: 재배포 후 첫 poll 사이클(03:46:49, 전 심볼 `sources_ok=3/3`)이 실제로 기존 부분NULL행을 채웠는지 직접 조회 — **ETH `03:30:00`행(원래 OI·글로벌롱숏 NULL)이 `sources_ok=3`로 완전히 채워짐, SOL `03:30/03:35`도 동일하게 패치됨을 확인**. 남은 `sources_ok<3` 행: ETH 1건(방금 들어온 최신 03:45 행, 다음 사이클에 자연 완성 예정, 정상), BTC 0건, SOL 2건(그중 1건은 픽스 배포 이전·컬렉터 최초 가동 첫 몇 분 시점이라 1시간 트레일링 self-heal 창을 이미 벗어남 — 초기 테스트성 데이터라 무시 가능한 수준, 별도 조치 안 함).

### ⚠️ 2026-08-23 갱신 — 10월 체크포인트 명분(연구용) 소멸, poller는 계속 유지

동시 세션이 `data.binance.vision` metrics 아카이브(2021-12~전일, 5분, 무료)를 조사해 이 poller가 모으는 것과 사실상 동일한 데이터(OI, 톱트레이더/글로벌 계정 롱숏비)가 이미 다년치로 존재함을 확인·백필 완료(`data/TOTAL_ETHUSDT_metrics_2024_2026.csv`, 2024-01~2026-08, 277,497행). 캐노니컬 2024/2025 빌드의 동일 컬럼과 완전일치(같은 원천) — 즉 이 컬럼들은 이미 154피쳐 chance-level 스크리닝에 포함돼 있었음. **아래 "다음 체크포인트(10월 중순)"는 연구 목적으로는 더 이상 유효하지 않음** — 다년치 데이터가 지금 바로 있으므로, 재개하려면 poller 축적을 더 기다릴 필요 없이 이 아카이브로 바로 진행 가능(단, 이 신호 자체는 §추가3에서 이미 증분검정 6/6 실패로 판정된 상태이므로 "데이터가 생겼다"가 "신호가 있다"를 의미하진 않음 — 재시도한다면 위 §6 리스크레이어 프레이밍과 아카이브 데이터를 함께 써야 함).

**poller 자체는 계속 유지**(중단하지 않음) — 아카이브는 일 단위 지연이라, poller의 남은 가치는 실시간성(라이브 서빙용)으로 재정의됨.

**별도 발견(더 중요할 수 있음)**: 같은 조사에서 캐노니컬 `training_features_2026_rebuilt.csv`의 동일 컬럼들이 **2026-02~07 구간에서 아카이브와 완전히 다른 시계열**(상대오차 13~40%)임을 발견 — VAL(2026Q2)·OOS(7월)가 이 기간의 OI/롱숏비 피쳐(+wide24 레짐분류기의 state12_oi_change_rate 포함)를 오염된 값으로 소비했을 가능성. 근본원인 미조사. 상세: `docs/experiments/eth_binance_metrics_archive_backfill_and_canonical_divergence_20260823.md`, [[eth_binance_metrics_archive_backfill_canonical_divergence]].

### 다음 체크포인트 (2026-08-23 기준 연구용으로는 위 갱신으로 대체됨, poller 운영 관점에서만 참고)

지금부터 5분마다 데이터가 실제로 쌓인다. `tail_risk_1m`이 결정 게이트에 8주를 요구한 전례(§ 앞선 업데이트)를 그대로 적용하면 판정 가능한 시점은 대략 10월 중순 — 단, 이건 참고 기준일 뿐 고정 약속은 아니고, 실제로는 그때 가서 (a) 얼마나 결측 없이 쌓였는지, (b) 여러 레짐을 걸쳤는지 보고 재판단해야 한다. 그 전까지 이 축은 조용히 데이터만 쌓는 상태로 둔다.

## 출처

- 리포지토리: `docs/experiments/eth_alt_data_source_feasibility_check_20260812.md`, `eth_candidate_liquidation_feed_features_cheap_gate_20260817.md`, `eth_broad_evidence_signal_sweep_20260814.md`, `eth_omega461_evidence_intervention_surface_ceiling_20260815.md`, `eth_amt_vsa_footprint_ifvg_strategy_absorption_study_20260815.md`, `eth_yush_orderflow_strategy_absorption_study_20260815.md`, `eth_candidate_gex_expiry_reversal_protocol_20260817.md`, `eth_candidate_polymarket_eth_market_coverage_scan_20260817.md`, `ilias_eth_label_fusion_combined_model_research_20260821.md`, `docs/model_contracts/research_line_registry.json`(id: `eth_candidate_spot_perp_basis_direction_cheap_gate_20260820` 항목의 인용문헌 포함, liquidation/coinglass 전용 항목은 없음을 grep으로 확인).
- 문헌(OpenAlex, 접근일 2026-08-22): Osler (2003) *Journal of Finance* DOI:10.1111/1540-6261.00588; "Systemic Risk from Financial Leverage in Digital Asset Markets" (2026, IJEBMR) DOI:10.51505/ijebmr.2026.10315; "Measuring the engine of a liquidation cascade" (2026, arXiv preprint); "Forced Liquidation Cascades in Unregulated Perpetual Futures Markets" (2026, Zenodo, 비피어리뷰) DOI:10.5281/zenodo.19975149; "Perpetual Futures Pricing" (NBER WP 2024) DOI:10.3386/w32936; "Trading behavior in bitcoin futures: Following the smart money" (2022, *Journal of Futures Markets*) DOI:10.1002/fut.22332.
