# ETH 데이터 스플릿 재설계 — "최근 데이터를 OOS로만 소비하는 낭비" 문제의 문헌조사 (2026-08-20)

## 배경 및 문제의식

사용자 문제제기: 현재 VAL/OOS-Q1/OOS-Q2 고정 캘린더 스플릿(`.claude/CLAUDE.md` Fresh-Forward
Validation/OOS/Test Rule 기본값 — VAL 2025-09-01~12-31, OOS 2026-01-01~03-31, 확장판
OOS-Q2 2026-04-01~06-30)에서는 **라이브에 가장 가까운(=현재 레짐과 가장 가까운) 최근 데이터가
영원히 학습에 안 쓰이고 OOS 테스트로만 소비된다.** 실제 배포되는 모델은 항상 VAL 종료 시점
기준으로 학습되므로, 배포 시점 기준 데이터가 최대 9개월 이상 뒤처진 채로 라이브에 나간다 —
이게 아깝다는 문제의식. 이 문서는 (1) 실제 데이터가 정말 24년1월~26년6월 구간에 결측 없이
존재하는지 감사하고, (2) 이 "최근 데이터 낭비" 문제를 다루는 외부 문헌을 조사해 이 저장소의
기존 제약(Fresh-Forward Rule, DSR/PBO 게이트)과 충돌하지 않는 구체적 데이터 split 재설계안을
제시한다.

이 문서는 리서치 문서다 — 일리아스 계약(`docs/model_contracts/
ilias_eth_human_direction_risk_management_contract_20260817.md`)에 실제로 반영할지/어떻게
반영할지는 별도 세션·사용자 결정 사항이며, 이 문서는 그 결정을 위한 근거자료다
([[feedback_contract_vs_experiment_docs]] 컨벤션: 계약문서=결과요약, 전체과정은 별도 문서).

## 1. 데이터 완전성 감사 (2026-08-20 실측)

캐노니컬 ETH 학습피쳐(`data/splits/year_oos/training_features_{2024,2025,2026_rebuilt}.csv`,
오디세이4/일리아스/DC 154피쳐 작업 등이 공통으로 쓰는 소스)를 5분봉 그리드와 직접 대조했다.

- **24년1월~26년6월 기대 bar 262,656개 중 실제 결측 47개(0.018%)** — 전부 거래소 순간단절급
  미세 gap(최대 17bar=85분, 대부분 1~11bar), 2024-02/06/08·2025-02/04/07에 흩어져 있고 특정
  구간에 몰려있지 않음.
- 142개 전 컬럼의 NaN 비율이 전 구간(포함 각 연도파일의 최근 30일 꼬리) 0.5% 미만 — 조용히
  깨진 피쳐 없음.
- **2026년 파일은 내부 gap 0건**이고, 실제로는 6월이 아니라 **2026-07-20까지** 이미 확장돼
  있음(`.bak_pre_extend_20260720` 백업 확인). raw 5분봉(`ETHUSDT-5m-spot.csv`)은 확인시점
  기준(2026-08-20) 사실상 라이브.

**결론: "채워넣을" 결측은 없다.** 이 정도 미세 gap은 백필 투자 대비 실익이 없다. 단, curated
피쳐가 raw보다 약 1개월 뒤처져 있다는 점([[eth_canonical_data_date_range_verification_20260820]])은
아래 4절 제안의 실행 시점에 영향을 준다.

## 2. 문헌조사 방법

독립 리서치 에이전트가 arXiv(q-fin/cs.LG), OpenAlex, Crossref, Semantic Scholar, SSRN 및
실무자 문헌을 대상으로 6개 축으로 조사했다(74회 조사, DOI/arXiv-ID 상호검증). 아래는 그 결과를
이 저장소 맥락에 맞춰 재구성한 것이다.

## 3. 문헌 정리 (6개 축)

### 3.1 Combinatorial Purged Cross-Validation (CPCV)

López de Prado, *Advances in Financial Machine Learning*(Wiley, 2018) Ch.7(purging+embargo)·
Ch.12(CPCV)가 원전. 라벨의 path-dependency로 인한 누수를 purge+embargo로 제거하고, 전체
구간을 N개 그룹으로 나눠 모든 C(N,k) 조합을 test로 순회 — **이론상 최근 데이터를 포함한 전체
구간이 대부분의 조합에서 train으로도 쓰인다.**

- Bailey, Borwein, López de Prado & Zhu(2014), "Pseudo-mathematics and financial
  charlatanism", *Notices of the AMS* 61(5), [DOI:10.1090/noti1105](https://doi.org/10.1090/noti1105) — PBO의 이론적 근거(이 저장소가 이미 쓰는 지표).
- Bailey et al.(2016), "The probability of backtest overfitting", *J. Computational
  Finance* 20(4), [DOI:10.21314/JCF.2016.322](https://doi.org/10.21314/JCF.2016.322).
- Arian, Norouzi Mobarekeh & Seco(2024), "Backtest overfitting in the machine learning
  era", *Knowledge-Based Systems*, [DOI:10.1016/j.knosys.2024.112477](https://doi.org/10.1016/j.knosys.2024.112477) — K-Fold/Purged K-Fold/CPCV/Walk-Forward를 synthetic
  데이터로 직접 4파전 비교, **CPCV가 PBO/DSR 기준 최우수**.

**⚠️ 이 저장소와의 관계 — 조문 자체가 아니라 목적에서 나오는 제약(정정, 2026-08-20 대화 중
정밀화)**: `.claude/CLAUDE.md` Fresh-Forward Rule의 실제 조문(원장 재사용 금지, 미래 row
조인 금지, bar마다 그 시점 feature만 사용)은 **리플레이/백테스트 메커니즘**에 관한 것이지
"학습 데이터가 test 구간보다 미래여선 안 된다"를 문면으로 명시하진 않는다 — CPCV의 test 구간
안에서 원장 없이 bar-by-bar causal 리플레이를 제대로 하면 조문 자체는 어기지 않는다. 그러나
이 룰의 목적("원장 대신 라이브처럼 매매결정을 평가하라", 사용자 재확인)에서 논리적으로 따라
나오는 제약이 있다: 이 시스템은 온라인학습이 아니라 한 번 학습한 고정 가중치를 배포하므로,
"라이브처럼"이 성립하려면 그 순간 배포된 모델은 그 시점 이전 데이터로만 학습된 것이어야 한다
— 학습데이터에 그 시점 이후 데이터가 섞이면, 그 모델은 애초에 그 순간 라이브로 실재할 수
없었던 모델이다. **CPCV는 조합 경로별로 갈린다**: test 그룹이 자기 학습그룹 전부보다 시간상
나중인 경로(예: 맨 마지막 그룹을 test)는 "라이브처럼"이 성립하지만, test 그룹 앞뒤로 학습그룹이
걸쳐있는 경로(대다수, 그리고 CPCV가 단일 holdout보다 데이터 활용도가 높은 이유 그 자체)는
그 순간 존재할 수 없었던 모델을 시뮬레이션하는 것이라 성립하지 않는다. **"라이브처럼 되는
경로만 쓰는 CPCV"는 결국 walk-forward와 수학적으로 같아진다** — combinatorial의 이점을 포기하는
셈이므로 실질적으로 3.2절과 다른 제안이 아니다. **프로모션/라이브후보 증거로는 full CPCV
채택 불가.** 대신 **purged walk-forward**(combinatorial은 버리고 순수 walk-forward 구조를
유지하되 CPCV의 purge+embargo 경계위생만 가져오는 절충안)를 3.2절 제안에 더할 가치가 있다 —
특히 DC/CUSUM처럼 forward-looking(K bar 앞을 보는) triple-barrier 라벨을 쓰는 경우, 단순
캘린더 컷만으로는 TRAIN/OOS 경계 바로 앞 학습샘플의 라벨 윈도우가 OOS 쪽으로 살짝 넘어가는
미세누수 가능성이 있다 — 이 저장소 라벨 빌더가 이 경계 purge를 실제로 하는지는 별도 코드
확인이 필요(미확인, 이 문서 범위 밖). CPCV 자체는 "이 모델링 접근이 역사 전체에 걸쳐
일반화되는가"를 묻는 research-stage 스크리닝(아키텍처·HP 비교, 기존 Seed-Diversity 게이트처럼
프로모션 증거와 분리)에서는 여전히 유효한 도구다.

### 3.2 Walk-forward / rolling-origin 재학습 (가장 직접적인 해법)

Anchored(확장) vs rolling(고정길이 슬라이딩) 윈도우로 스플릿 경계를 주기적으로 전진시키고
재학습 — **OOS가 다음 세대의 TRAIN으로 "졸업"하고, 새로 도래한 미터치 구간이 새 OOS가 된다.**

- Pesaran & Timmermann(2003/2004), "How costly is it to ignore breaks when forecasting
  the direction of a time series?", *International Journal of Forecasting*,
  [DOI:10.1016/S0169-2070(03)00068-2](https://doi.org/10.1016/S0169-2070(03)00068-2) — **방향예측**(이 저장소의 direction head와 정확히 같은 과제) 대상으로 stale window
  사용 비용을 정량화.
- Cerqueira, Torgo & Mozetič(2020), *Machine Learning* 109, [DOI:10.1007/s10994-020-05910-7](https://doi.org/10.1007/s10994-020-05910-7) — 비정상성 하에서는 holdout/walk-forward가 CV보다 더 신뢰할 만한 추정량.
- Bergmeir & Benítez(2012), *Information Sciences* 191, [DOI:10.1016/j.ins.2011.12.028](https://doi.org/10.1016/j.ins.2011.12.028) — 반대로, 모델이 올바르게 특정되고 오차가 무상관이면 blocked CV도 walk-forward만큼
  타당할 수 있음(비정상성이 클수록 walk-forward 우위가 커짐 — 이 저장소 상황에 부합).
- **크립토 직접 사례**: Mroziewicz & Ślepaczuk(2026), arXiv:[2602.10785](https://arxiv.org/abs/2602.10785) — BTC/BNB/ETH에 walk-forward 윈도우 81조합 테스트 후 상위2개를 단일터치 21개월
  OOS에 적용, single-touch 규율까지 이 저장소와 사실상 동일.
- Jung(2026), SSRN [10.2139/ssrn.6727738](https://doi.org/10.2139/ssrn.6727738) — BTC 방향예측에 timing×**재학습주기**×피쳐버전×holdout 180셀 설계를
  Bonferroni/Holm+DSR+PBO(CSCV)로 스크리닝 — 126조합 중 3개만 생존(반월 재학습 주기 1종),
  재학습 주기가 실제로 결과를 좌우한다는 직접 증거이자 "무분별하게 자주 재학습"의 위험 경고.

### 3.3 "전체 데이터로 최종 재학습" 관행

문헌 근거가 가장 얇은 축이다. AFML 본문·CPCV/walk-forward 실무 설명 어디에도 이 관행 자체를
공식화한 곳은 못 찾았다("일반적이지만 공식화 안 된 관행"이라는 사용자쪽 전제와 일치). 실무
근거: backtrex.com(hedge fund backtesting 가이드)의 "가장 엄격한 퀀트팀은 배포 직전 딱 한 번만
소비하는 최종 holdout을 둔다"는 서술 — 그러나 그 최종holdout을 재학습에 되접는지까지는 언급
안 함. 이론적 배경은 다시 Bailey et al.(2014) — 백테스트는 재시도 횟수가 늘수록 증거가치가
줄어드므로, 배포 후 **라이브 자체가 진짜 OOS**가 된다는 실무자 논리의 근거로 흔히 인용된다.

**안전한 해석(에이전트 자체 종합)**: "전체 재학습"은 이미 DSR/PBO를 통과한 뒤 딱 한 번의
gated unlock으로만 정당화되고, 반복적 관행이 되면 Fresh-Forward Rule이 막으려는 바로 그
p-hacking이 재발한다. 즉 이 관행은 **3.2의 walk-forward 롤의 마지막 단계**로 흡수시키는 게
안전하지, held-out 테스트 자체를 없애는 독립 관행으로 채택하면 안 된다.

### 3.4 Recency/time-decay 가중 학습

전량을 학습에 남기되 sample별 가중치를 최신성에 따라 지수감쇠 — split 자체는 안 건드리고
손실함수만 바꾸는 **가장 저위험 레버**.

- López de Prado, AFML Ch.4 "Sample Weights" — Uniqueness+Sequential Bootstrap 뒤
  Time Decay 서브섹션이 정확히 이 메커니즘.
- Wong & Barahona(2023), arXiv:[2303.07925](https://arxiv.org/abs/2303.07925) — 이 저장소와 같은 tabular(XGBoost) 세팅에서 시간대별 스냅샷 앙상블로 recency와
  전체이력을 동시에 반영.

**단독으로는 사용자 불만을 해결 못 함** — TRAIN 윈도우 자체가 그대로면 OOS-Q1/Q2는 여전히
안 쓰인다. 3.2와 결합해야 실효.

### 3.5 Concept drift / 비정상성 적응 재학습 문헌

**흥미로운 공백**: arXiv에 `"concept drift" AND cryptocurrency`류 쿼리 4종 전부 **0건** — 크립토
트레이딩ML 문헌은 "concept drift" 대신 "regime"/"regime shift"/"non-stationarity" 용어를 쓴다
(용어 공백이지 연구 공백은 아님).

- Gama et al.(2014), *ACM Computing Surveys* 46(4), [DOI:10.1145/2523813](https://doi.org/10.1145/2523813) — 표준 서베이. 적응전략 3부류(재학습기반/앙상블기반/증분·가중기반)가 정확히
  3.2/3.4/스냅샷앙상블에 대응.
- **가장 직접적으로 이 문제를 프레이밍한 논문**: Huang, Liu, Deng & Li(2024), arXiv:[2401.03865](https://arxiv.org/abs/2401.03865) — "기존 방법들은 최근데이터의 신흥패턴이나 과거데이터의 반복패턴 중 하나를
  무시하는데, 둘 다 미래예측에 실증적으로 유용하다"(초록 인용) — recent/historical을 배타적
  버킷으로 안 보고 동적으로 블렌딩. 주식 대상이지만 이 저장소 문제의식과 프레이밍이 가장 근접.
- 크립토+오버피팅 프레이밍: Gort et al.(2022), arXiv:[2209.05559](https://arxiv.org/abs/2209.05559) — 크립토 10종, 2022년 두 차례 폭락장 관통 검증.

### 3.6 크립토 특화 CV 비교 문헌

Jaquart, Köpke & Weinhardt(2022), *J. Finance and Data Science*, [DOI:10.1016/j.jfds.2022.12.001](https://doi.org/10.1016/j.jfds.2022.12.001) — 시총상위100 방향예측, 무조건53%→최고확신10분위 57.5~59.5%(이 저장소의
direction+quality head 게이팅 구조와 정성적으로 유사). Cocco, Tonelli & Marchesi(2021),
*PeerJ CS* 7, [DOI:10.7717/peerj-cs.413](https://doi.org/10.7717/peerj-cs.413)는 반례로 유용 — plain k-fold만 쓰는 낮은 엄밀성 사례라, 이 저장소의
기존 Fresh-Forward 규율이 이미 업계 중앙값보다 훨씬 엄격함을 보여준다.
**공백**: CPCV/walk-forward/expanding-window/fixed-holdout을 크립토 데이터로 직접 4파전
비교한 논문은 못 찾음(3.1의 synthetic 비교만 존재) — 이 특정 질문은 문헌에 아직 없다.

## 4. 이 저장소 고유 제약과의 교차분석

**핵심 발견 — OOS-Q1/Q2는 이미 여러 번 "소진"됐다.** 메모리 기록만으로도 최소 다음 시도들이
독립적으로 OOS-Q1/Q2를 조회했다: 오디세이4 G0 원 프로모션, 일리아스1 단일시드,
일리아스1 dual N=5(6시드), 일리아스1 zig075단독 N=5(6시드×always벤치마크), ETH 라이브
프로모션 N=3→N=5, BTC 라이브 N=5, SOL 라이브 N=5, 오디세이4 섀도우 풀리시드 N=6,
섀도우 exit_head N=5, veto+guard BTC/SOL 이식 N=3, 5-way 라벨로직 N=3 — **최소 10회 이상의
독립 시도가, 각각 N=3~6개 시드로 같은 두 창을 반복 조회**했다. 이건 3.1이 인용한
"재시도 횟수가 늘수록 백테스트 증거가치가 준다"(Bailey et al. 2014)는 바로 그 상황이며,
**[[eth_live_stack_never_passed_dsr_pbo_20260819]]의 DSR=0.915(선 0.95 미달) 첫 FAIL이
정확히 이 누적비용이 수치로 드러난 사례로 읽힌다.**

즉 지금 시점에서 OOS-Q1/Q2를 "아직 안 쓴 신선한 창"으로 취급하고 새 프로모션 주장을 얹는 건
이미 통계적으로 정직하지 않다 — 반대로 말하면, **이 두 창을 은퇴시켜 학습에 편입하는 게
문헌상으로도(3.2) 이 저장소 실측상으로도(DSR fail) 지금 해야 할 일**이라는 두 근거가 수렴한다.

Fresh-Forward Rule과의 정합성 확인: 3.2(walk-forward 롤)는 "그 시점까지 확정된 feature/state만
사용하는 bar-by-bar causal 평가"를 전혀 건드리지 않는다 — 바뀌는 건 어느 캘린더 구간이
train/val/OOS로 라벨링되는지뿐이다. single-touch 규율도 유지된다: 매 세대 OOS는 **그 세대
기준으로 진짜 미터치인 구간**이기만 하면 된다. 3.1(CPCV)은 위에서 이미 배제.

## 5. 제안 — 분기별 롤링 walk-forward 정책

기존 분기 경계(VAL/OOS-Q1/OOS-Q2)를 그대로 재사용하되, **고정이 아니라 매 분기 전진**시킨다.

| 세대 | TRAIN | VAL | OOS(single-touch) |
|---|---|---|---|
| 현재(이미 다회 소진) | ~2025-09-30 | 2025-10~12 | Q1 2026-01~03, Q2 2026-04~06 |
| 제안 다음 세대 | ~2026-03-31(구 VAL+Q1 편입) | 2026-04~06(구 Q2, 재사용해도 무방 — VAL은 애초에 튜닝용) | **2026-07-01~09-30**(진짜 미터치, 완전 데이터는 09-30에야 확정) |
| 그 다음 세대 | ~2026-06-30 | 2026-07~09 | 2026-10~12 |

**적용상 걸림돌과 대안**: 오늘(08-20) 기준 curated 피쳐가 07-20까지뿐이라
([[eth_canonical_data_date_range_verification_20260820]]) 다음 세대 OOS(Q3)는 아직
완성되지 않았다. 두 갈래:
- **(A) 09-30까지 대기** — 통계적 검정력 최대(분기 전체), 이번 세대 재학습(TRAIN~03-31,
  VAL=구Q2)은 지금 바로 시작 가능, curated 피쳐 파이프라인을 매일/매주 갱신해 09-30에
  즉시 검증 가능하도록 준비.
- **(B) 짧은 구간(예 07-01~07-20 3주치)으로 저전력 중간 체크** 후 09-30에 정식 재확인 — 3.2가
  인용한 window-length/검정력 트레이드오프(Pesaran & Timmermann류)를 감안하면 이건 예비신호일
  뿐 프로모션 근거로는 약함.

**부가 권고**: 3.4(time-decay 가중)를 위 롤과 별개로, 지금 당장 **기존 고정 split 위에서
격리 실험**으로 먼저 시험 가능(split 자체를 안 건드리므로 최저위험) — TabM 학습루프에 지수감쇠
가중치만 추가, 나머지는 그대로.

**North Star**: 이 정책이 정착하면 "배포모델이 최근데이터를 못 쓴다"는 원 불만이 구조적으로
해소된다 — 매 세대 staleness가 최대 9개월이 아니라 최대 1분기로 캡된다.

## 6. 채택 보류 항목과 이유

- **CPCV(3.1)**: 프로모션 근거로는 4절에서 이미 배제. 아키텍처/HP 스크리닝용 별도 research-stage
  도구로는 향후 검토 가치 있으나, 이번 문서의 범위 밖(프로모션 게이트와 완전히 분리 문서화 필요).
- **"전체 재학습"을 독립 관행으로(3.3)**: 문헌 근거 자체가 얇고, 5절의 롤링 정책의 마지막
  단계로 이미 흡수됨 — 별도 정책으로 만들 필요 없음.

## 7. 열린 질문

- 재학습 주기를 분기(현재 컨벤션과 정합)로 할지 더 촘촘히(월 단위, Jung 2026이 경고한 위험
  감안 필요) 할지.
- 5절 (A)/(B) 중 어느 쪽으로 시작할지 — 사용자 판단 필요.
- 이미 소진된 Q1/Q2를 은퇴시키는 걸 어떻게 "공식화"할지(예: research_line_registry.json에
  OOS-사용이력 등록?) — 별도 설계 필요, 이 문서 범위 밖.
- 3.6이 지적한 공백(크립토데이터 직접 4파전 비교 부재)을 이 저장소가 직접 메울지(예: 154피쳐
  DC축 데이터로 fixed-holdout vs walk-forward 직접 비교) 여부.

## 8. 최종 결정 (2026-08-20, 대화 종료 시점)

위 탐색을 종합해 다음 방법으로 확정한다 — 5~7절의 탐색 기록은 삭제하지 않고 그대로 두되(과정
보존), 실제 적용 시 아래를 우선한다.

**방법: 분기 앵커드(확장) Walk-Forward + 경계 Purge/Embargo + TRAIN 내부 Time-Decay 가중.**

| 구성요소 | 결정 |
|---|---|
| TRAIN 구조 | Anchored(확장) — 시작점 고정, 매 세대 끝점만 전진. Rolling(고정길이 슬라이딩)은 채택 안 함 — lookback 길이라는 또 다른 임의결정이 늘어나고, 3.4의 time-decay 가중이 같은 문제(오래된 레짐 희석)를 split을 안 건드리고 해결한다 |
| 경계 처리 | forward-looking 라벨(DC/CUSUM류)의 라벨윈도우가 VAL/OOS 경계를 넘는 학습샘플 purge + 경계 직후 짧은 embargo(purged walk-forward, 3.1절 정밀화 참고). 이 저장소 라벨빌더가 이미 이걸 하는지는 미확인 — 적용 전 코드 확인 필요 |
| TRAIN 내부 가중 | 3.4절 time-decay(AFML Ch.4) — 별도 실험으로 기존 고정split 위에서 먼저 격리 테스트 가능 |
| VAL/OOS 세대교체 | 매 세대 OOS는 그 세대 기준 진짜 미터치 분기 1개만, 사용 즉시 소진 기록 후 다음 세대부터 TRAIN 편입 — 재사용 금지. 이 사용이력을 추적하는 ledger가 지금 없어서 Q1/Q2가 이미 10회+ 재사용됐음(4절) — 신규 ledger 신설 필요 |
| CPCV | 프로모션 증거에서 배제(3.1절 정밀화 논리). Research-stage(아키텍처/HP 스크리닝) 전용 옵션으로만 남김, Seed-Diversity 게이트와 동일하게 프로모션 증거와 분리 문서화 |
| 온라인학습(장기) | 지금 범위 밖. 이 구조는 재학습주기를 줄여가는 축소판이라 전환 시 재작업 최소화(대화 중 별도 논의, 라벨지연/prequential evaluation/재튜닝-가중치업데이트 방화벽 등 후속 고려사항 있음 — 미착수) |

**적용 순서**:
1. (즉시 가능) Q1/Q2를 TRAIN으로 편입 — 다음 세대 TRAIN ~2026-06-30. OOS 사용이력 ledger 신설.
   Purge/embargo 코드 확인·구현. Time-decay 가중 격리 프로토타입.
2. (2026-09-30까지 대기) 완전한 새 OOS 분기 2026-07-01~09-30 확정 후 그 세대의 single-touch
   프로모션 판정 — 3주치 부분데이터로 조기체크하지 않는다(그 자체가 또 한 번의 "조회"로
   집계될 위험, 4절 참고). 대기 기간 동안 1의 준비작업을 완료해 09-30에 즉시 실행 가능하게
   한다.
3. 이후 매 분기 1~2 반복.

이 결정은 이 문서 범위의 리서치 결론이다 — 실제 코드/파이프라인 구현과 일리아스 계약 반영
여부·시점은 별도(1절 참고).

**⚠️ 갱신(같은 날, 사용자 지시)**: `docs/model_contracts/
ilias_eth_human_direction_risk_management_contract_20260817.md`에 "## 데이터 Split 재설계
제안 — 2026-08-20 리서치 결론 (제안됨, 미실행)" 절로 정식 등록 완료 — 기존 "## Dataset
Split"(Baseline v1 프로덕션 split)은 대체하지 않고 병기. 같은 세션에서 다른 세션이 구축한
154피쳐 데이터셋(2024-01~2026-06-30)도 독립 재검증 완료 — 대체로 일치했으나 계약이 보고
안 한 gap 1건(2026-02-28 16:00→03-01 00:00, 8시간) 신규 발견, 계약에 추가 기록됨.
