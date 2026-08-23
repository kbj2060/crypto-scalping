# 크립토 트레이딩 전략 — 외부 논문·문헌 조사 (2026-08-23)

## 조사 개요

일리아스 프로젝트에서 "학습모델 아키텍처로 아직 신호를 잡을 수 있다"는 사용자 가설을 검토하는
과정에서, 이 저장소 내부 실증([[eth_ilias_regime_direct_exposure_seed_stable_direction_20260822]]
등)과 별개로 외부 학계·업계 문헌이 크립토 트레이딩 전략 전반에 대해 뭐라고 말하는지 확인하기
위해 진행했다. 4개 하위 클러스터(오더북 미시구조/오더플로우/마켓메이킹, 모멘텀/평균회귀,
차익거래/펀딩레이트-베이시스, 온체인/센티먼트/이벤트드리븐)로 나눠 병렬 리서치 에이전트 4개를
띄워 arXiv(q-fin 카테고리), Semantic Scholar, OpenAlex, Crossref를 조사했다(paper-lookup
스킬 사용, 모든 인용은 API로 직접 검증된 DOI/arXiv ID만 포함, 환각 인용 배제).

각 논문마다 **실제 out-of-sample/live 검증이 있는지, 아니면 in-sample·단일-split 백테스트에
그치는지**를 명시했다 — 이 저장소의 [Fresh-Forward 규칙](../.claude/CLAUDE.md)이 요구하는
기준으로 걸러 읽기 위함이다. 이건 확정적 학술 서베이가 아니라 **타겟 조사**(각 하위주제 상위
5~10편, 전수조사 아님)임을 밝힌다.

**핵심 결론을 먼저 말하면**: 독립적인 최신(대부분 2023~2026) 문헌이 이 저장소가 이미 자체
실증으로 도달한 결론들 — 아키텍처는 병목이 아니다, gross 신호는 종종 실재하지만 비용이
집어삼킨다, 크립토의 "공짜 점심"류 차익거래는 시장이 성숙하며 붕괴했다 — 을 상당히 강하게
재확인한다. 다만 두 갈래는 이 저장소에서 아직 안 닫힌 축과 맞물려 있어 따로 짚을 가치가 있다
(§4.3 GEX, §1.2 자산매칭 마켓메이킹). 클러스터별 결과와 저장소 교차검증은 마지막
"종합 판단" 절 참고.

---

## 1. 오더북 미시구조 / 오더플로우 / 마켓메이킹

### 1.1 LOB(오더북) 기반 방향예측 아키텍처

- **Zhang, Zohren & Roberts (2019)**, *DeepLOB*, IEEE Trans. Signal Processing,
  DOI:10.1109/TSP.2019.2907260 (arXiv:1808.03668). CNN+LSTM, FI-2010+LSE 1년치 호가. 미학습
  종목까지 일반화되는 안정적 OOS 정확도 보고 — 이 분야의 기초 아키텍처(주식시장).
  **검증: OOS 검증됨.**
- **"Exploring Microstructural Dynamics in Crypto LOBs" (2025)**, arXiv:2506.05764.
  DeepLOB/Conv1D-LSTM vs 로지스틱회귀/XGBoost, Bybit BTC/USDT 호가. **적절한 필터링 후
  단순모델이 딥넷과 OOS 정확도에서 동급이거나 능가.** **검증: OOS 검증됨 — 아키텍처
  무관성의 크립토 특화 증거.**
- **"Explainable Patterns in Cryptocurrency Microstructure" (2026)**, arXiv:2602.00776.
  CatBoost+SHAP, Binance 선물 5종(BTC/LTC/ETC/ENJ/ROSE), 시계열CV, taker+maker 백테스트,
  플래시크래시 구간 스트레스테스트. 피쳐 중요도가 자산 간 안정적. **검증: OOS 검증됨.**
- **T-KAN (2026)**, arXiv:2601.02310. DeepLOB의 LSTM을 학습가능 B-spline으로 대체, FI-2010
  k=100에서 F1 +19.1%p, 비용반영 백테스트에서 T-KAN +132% vs DeepLOB -83%(드로다운) —
  **DeepLOB 자체의 알파 감가상각을 문서화**(주식시장 벤치마크). **검증: 백테스트뿐.**
- Hawkes 프로세스 기반 크립토 예측(2023, arXiv:2312.16190; Raffaelli et al. 2026,
  DOI:10.1007/s10203-026-00570-z, BTC/USD 호가) — 둘 다 몬테카를로/시뮬레이션 수익성 보고뿐,
  **검증: 백테스트뿐.**
- Gould & Bonart (2015, arXiv:1512.03492) — 큐임밸런스가 다음 틱 방향을 유의하게 예측한다는
  기초 결과(NASDAQ 10종목, 주식시장). **검증: in-sample fit, 백테스트 아님.**

### 1.2 오더플로우임밸런스(OFI) & 마켓메이킹

- **Kolm, Turiel & Westray (2023)**, *Deep Order Flow Imbalance*, Mathematical Finance,
  DOI:10.1111/mafi.12413. 오더플로우 피쳐로 학습한 신경망이 원시 호가상태 학습보다 우수
  (NASDAQ 115종목, 주식시장). **검증: 불명확(검증방식 미기재).**
- **"The Market Maker's Dilemma" (2025)**, arXiv:2502.18625. **Binance BTC 무기한선물 실거래
  실험** — 메이커 체결확률과 체결후 수익률이 음의 상관, 즉 유효한 메이커 전략은 오더북
  임밸런스 반대편에 서야 함(역선택 구조). **검증: 실거래 검증됨** — 이 저장소가 배포한
  peg-maker 섀도우([[eth_maker_fill_simulation_l2_20260822]])와 직결되는 경고: 메이커 체결이
  "공짜"가 아니라 체결 자체가 역선택 신호일 수 있음.
- **"Trading in the Sunshine or in the Shade" — Barone & Lillo (2026)**, arXiv:2606.15715.
  Hyperliquid 온체인 무기한선물 호가창에서 실제 대형주문 430만 건 vs 공개TWAP 46.5만 건
  재구성 — sunshine-trading 이론 확인(공개된 흐름이 더 좋은 체결가를 받고, 비공개자에게
  역선택이 전가). **검증: 실데이터 검증됨.**
- **"Funding-Aware Optimal Market Making for Perpetual DEXs" — Le (2026)**, arXiv:2605.06405.
  펀딩레이트를 재고 리스크에 결합한 HJB 최적제어, **Hyperliquid ETH/BTC/SOL 실데이터로
  캘리브레이션**(이 저장소가 거래하는 자산과 정확히 일치). 100시드 홀드아웃 시뮬레이션에서
  ETH/BTC는 Avellaneda-Stoikov 능가, SOL은 혼재. **검증: 백테스트뿐(이론+시뮬레이션, 실거래
  아님)** — 그래도 자산이 일치한다는 점에서 눈여겨볼 가치가 있음(§종합판단).
- **"Microstructure alpha" — Pindza (2026)**, Frontiers in Blockchain,
  DOI:10.3389/fbloc.2026.1811716. OFI 포함 9개 미시구조 피쳐, 크립토 6종, Binance
  현물+선물, **purged walk-forward CV**. **신호 자체는 실재하지만 실제 수수료를 못 이길
  만큼 약하고, 자산 간 전이도 안 됨.** **검증: OOS 검증됨 — 이 저장소의 RDE/정보원천병목
  패턴과 거의 동일한 결과(§종합판단).**
- Optimal Adaptive Market Making for perpetual futures(2026, arXiv:2607.11888) — Avellaneda-
  Stoikov/GLFT/Glosten-Milgrom을 통합한 이론적 HJB 프레임워크, 실증 백테스트 없음.
  **검증: 이론뿐.**

**탐색 공백**: arXiv가 세션 중 15~20분 429 락아웃에 걸렸고 "order flow imbalance"+crypto
조합 쿼리는 끝내 응답을 못 받음(빈 결과 확정 아님, 요청 실패). Semantic Scholar도 429가 잦아
"Modelling Crypto Asset Order-Flow Imbalance..."(SSRN), Koutmos & Wei(2023, Rev. Quant.
Finance Acc., "Nowcasting bitcoin's crash risk with order imbalance") 2편은 존재를
확인했으나 초록을 못 가져와 요약에서 제외(추측 요약 대신 정직한 누락 선택).

---

## 2. 모멘텀 / 평균회귀

### 2.1 모멘텀·추세추종

- **Liu & Tsyvinski (2021)**, Review of Financial Studies, DOI:10.1093/rfs/hhaa113. 기초
  논문 — 시계열모멘텀+투자자관심 예측력 문서화. **검증: 백테스트뿐(전체표본 통계적
  유의성).**
- **Grobys & Sapkota (2019)**, Economics Letters, DOI:10.1016/j.econlet.2019.03.028. 143개
  코인, 2014~2018 — **유의미한 모멘텀 수익 없음**, 선행연구와 반대. **검증: 백테스트뿐 —
  부정적 결과.**
- **Rozario et al. (2020)**, arXiv:2009.12155. 10년치 추세추종, **워크포워드 연 255%
  수익률**, 코드 공개, 상품형/약세장 분산 프로파일. **검증: OOS(워크포워드) 검증됨.**
- **Bui & Nguyen (2026)**, "AdaptiveTrend", arXiv:2602.11708. 150개+ 페어, **명시적
  36개월(2022~2024) OOS 백테스트**, TSMOM/buy-hold 대비 Sharpe 2.41. **검증: OOS
  검증됨** — 이 클러스터에서 가장 최신·명시적인 긍정 결과.
- **Zaremba et al. (2021)**, Int'l Review of Financial Analysis,
  DOI:10.1016/j.irfa.2021.101908. 3,600개+ 코인 — **유동성 낮은 코인은 1일 반전, 유동성
  높은 코인은 모멘텀** — 부호 자체가 유동성에 좌우됨. **검증: 백테스트뿐** — ETH처럼
  유동성 높은 자산엔 반전이 아니라 모멘텀 쪽 예측을 시사(이미 [[eth_weekly_tsmom_bias_
  closed_20260817]]에서 테스트·기각된 방향).
- Shen, Urquhart & Wang(2021, DOI:10.1111/fire.12290, BTC 일중 TSMOM), Gerritsen et
  al.(2020, DOI:10.1016/j.frl.2019.08.011, 브레이크아웃 규칙) — 둘 다 **검증: 불명확.**

### 2.2 평균회귀·페어트레이딩

- **Fischer, Krauss & Deinert (2019)**, J. Risk Financial Management,
  DOI:10.3390/jrfm12010031. RF기반 롱숏 40개 코인, **명시적 OOS 구간(2018-06~09)**, 비용반영
  +7.1bp/일, 다만 한계 자체를 저자가 지적(차익거래 제약·용량 한계). **검증: OOS 검증됨.**
- **Fil & Krištoufek (2020)**, IEEE Access, DOI:10.1109/access.2020.3024619. 거리/공적분
  페어트레이딩, Binance 26종 — **일봉 기준 buy-and-hold보다 부진**(-0.07%/월)하나 **5분봉
  기준 +11.61%/월**. **검증: 백테스트뿐 — 저빈도에선 부정적, 파라미터에 극도로 민감.**
- **Tadi & Kortchemski (2021)**, DOI:10.1108/SEF-12-2020-0497 (arXiv:2109.10662). OU모델
  공적분 페어, 형성기간/거래기간 분리, 현실적 매수-매도 호가 반영 — buy-and-hold 능가.
  **검증: OOS 검증됨.**
- **Caporale & Plastun (2019)**, J. Economic Studies, DOI:10.1108/jes-09-2018-0310.
  트레이딩로봇 실측 — **과잉반응 후 반전 전략은 수익성 없음**, 모멘텀/관성은 수익성
  있어보이나 무작위와 통계적으로 구분 안 됨. **검증: 백테스트뿐 — 부정적 결과.**
- Makarov & Schoar(2019, 피인용 833 — §3.1에서 상술), Figà-Talamanca et al.(2021,
  DOI:10.1007/s10203-021-00318-x, 레짐의존적 수익성 — §3.1과 중복), Leung & Nguyen(2019,
  DOI:10.1108/sef-08-2018-0264), Lebiedź & Ślepaczuk(2026, arXiv:2606.04574, RL오버레이
  페어트레이딩, 10% 유의수준에서만 OOS 우위 — 약한 긍정)도 확인.

**모멘텀/반전 사인 뒤집힘 관련(별도 요청 항목)**: 진짜 라이브 트레이딩에서 부호가 뒤집힌
사례는 못 찾았으나(전부 과거 홀드아웃 기준 "OOS"), Grobys&Sapkota(모멘텀 자체 무의미),
Zaremba(유동성 따라 반전↔모멘텀 전환), Fil&Krištoufek(같은 전략이 일봉 부정/5분봉 긍정),
Caporale&Plastun(반전전략 실패)까지 **부정적·부호전환 결과가 이 클러스터 안에서 이례적으로
많다** — 크립토 모멘텀/평균회귀는 학계에서도 안정적 컨센서스가 없는 논쟁적 영역.

**탐색 공백**: arXiv 반전 관련 쿼리 2/4건 지속 429(진짜 빈 결과 아님). Semantic Scholar는
3/8건만 성공(무료키 부재). OpenAlex가 무장애로 주력 소스가 됨.

---

## 3. 차익거래 / 펀딩레이트-베이시스

### 3.1 거래소간·삼각 차익거래

- **Makarov & Schoar (2019)**, Journal of Financial Economics, DOI:10.1016/j.jfineco.
  2019.07.001 (피인용 833). 기초 논문 — 거래소간 BTC 가격 격차가 크고 반복적이며, 자본통제
  마찰과 결합. **검증: 실데이터 검증됨(관측연구, 전략 백테스트 아님).**
- **Hautsch, Scheuch & Voigt (2024)**, Review of Finance, DOI:10.1093/rof/rfae004. 실제
  블록체인+호가데이터 — 거래소간 스프레드가 온체인 정산 지연과 연동. **검증: 실데이터
  검증됨.**
- **Saggese et al. (2023)**, J. Economic Behavior & Organization, DOI:10.1016/j.jebo.
  2023.07.025. **유출된 Mt.Gox 체결데이터**로 실재 삼각차익거래자 440명 식별 — 숙련 트레이더의
  실현수익이 압도적. **검증: 실거래 검증됨.**
- **Wang et al. (2022)**, WWW'22, DOI:10.1145/3487553.3524201. 실제 Uniswap V2 데이터 —
  11개월간 순환차익거래 292,600건 체결·1.38억달러+, 그런데도 **가장 큰 미실현 기회가
  꾸준히 4,000달러를 초과** — 완전효율 아님. **검증: 실거래 검증됨.**
- **Crépellière, Pelster & Zeisberger (2023)**, J. Financial Markets, DOI:10.1016/
  j.finmar.2023.100817. 차익거래는 실재하나 **"2018년 4월 이후 규모가 크게 감소했고 그 뒤로
  거의 수확 불가"** — 명시적 감가상각 증거. **검증: 실데이터 검증됨.**
- **Seo, Koo & Yang (2024)**, Economic Modelling, DOI:10.1016/j.econmod.2024.106726. 김치
  프리미엄 임계회귀 — 임계값 이상일 때만 평균회귀(큰 프리미엄만 차익거래됨). **검증:
  실데이터 검증됨.**

### 3.2 펀딩레이트 차익·베이시스 트레이딩

- **Zhivkov (2026)**, Mathematics, DOI:10.3390/math14020346. **3,570만 관측치, 26개
  거래소** — 스냅샷의 17%가 20bp+ 격차를 보이지만, 그 중 가장 큰 것들도 **비용+반전 반영시
  40%만 수익성 생존**, 델타뉴트럴 포지션 시뮬레이션의 95%가 강제청산. **검증: 백테스트뿐
  (실데이터 기반 시뮬레이션)** — 가장 최신·가장 큰 표본, §종합판단에서 상술.
- **Lorig (2026)**, arXiv:2605.11263 (SSRN). Ethena(대형 실거래 스테이블코인 캐리 전략)
  최적제어 모델링 — **차익거래자 자신의 자금흐름이 영구적 시장충격을 만들어 베이시스를
  압축, 미래 펀딩수익을 갉아먹음**(내생적 감가상각). **검증: 불명확(제어이론, 실거래
  손익 미보고)** — 다만 수십억달러 규모의 실제 라이브 전략을 모델링한다는 점에서 의미있음.
- Krestenko et al.(2026, arXiv:2605.05089, DeFi 베이시스 담보관리), Ackerer/Hugonnier/
  Jermann(NBER WP32936/arXiv:2310.11771)·He/Manela/Ross/von Wachter(arXiv:2212.06888,
  무기한선물 무차익 가격이론) — 이론 위주, **검증: 불명확/이론뿐.**
- Zou(2022, DOI:10.2991/aebmr.k.220307.316, 낮은 등급 학술대회 — 엄밀성 미검증),
  Nimmagadda & Ammanamanchi(2019, arXiv:1912.03270, BitMEX 펀딩-가격 GARCH 상관분석)도
  확인.

**펀딩/베이시스 엣지 압축 관련(별도 요청 항목)**: 문헌이 명확히 다룬다 — Crépellière et
al.이 2018년 이후 거래소간 차익거래 붕괴를, Zhivkov(2026, 최신 대규모 데이터)가 현재
펀딩스프레드 대부분이 비용 반영시 수익성 없음을, Lorig·Krestenko가 지금도 라이브인 전략
(Ethena 등) 내부에서 규모 자체가 엣지를 갉아먹는 내생적 압축을 각각 정량화한다 — 이 저장소가
RDE 실증에서 찾은 "2024년엔 유효했을 무기, 2026년엔 이론상한 분기 0.68% 미만으로 붕괴"
([[eth_ilias_regime_direct_exposure_seed_stable_direction_20260822]])와 놀랍도록 일치.

**탐색 공백**: Semantic Scholar 5회 중 4회 429(무료키 없음), 단건 DOI 조회는 성공. arXiv
"delta neutral"+crypto 쿼리는 진짜 0건(에러 아님). CORE/Unpaywall은 호출 안 함(전문 요청
없었음).

---

## 4. 온체인 / 센티먼트 / 이벤트드리븐(GEX)

### 4.1 온체인 분석 / 고래추적

- **Herremans & Low (2022)**, arXiv:2211.08281. CryptoQuant+고래알림 트윗 → 익일 BTC
  변동성 급등 예측(방향이 아닌 변동성), 백테스트에서 드로다운 축소 보고. **검증:
  백테스트뿐.**
- **Ciaian et al. (2026)**, arXiv:2602.08429. ARDL, 2019~2024 일별 — **오프체인 수요가
  장기가격을 지배, 온체인(고래거래 포함)은 단기·동시성에서만 유의**. **검증: 계량회귀,
  전략 백테스트 아님.**
- Kim/Shin/Choi/Lim(2022, DOI:10.1109/ACCESS.2022.3177888), Zhang/Garg/Golden/Brockett
  (2024, DOI:10.3390/jrfm17030128, 2011~2017 데이터로 투자자유형별 세그멘테이션),
  Bubolz et al.(2026, arXiv:2607.15258, 온체인+가격+트위터로 "가격"이 아닌 "센티먼트"
  자체를 분류, F1≈0.84) — **검증: 전부 백테스트뿐.**
- Saggu(2025, arXiv:2501.05232) — Tether 발행/소각 이벤트 스터디, "Whale Alert" 트윗과
  결합시 반응 증폭. **검증: 불명확(이벤트 스터디, 적합된 전략 아님).**

### 4.2 소셜/뉴스 센티먼트

- **Kraaijeveld & De Smedt (2020)**, J. Intl Financial Markets Inst. Money, DOI:10.1016/
  j.intfin.2020.101188 (피인용 422). 렉시콘 센티먼트+그레인저 인과검정, 9개 코인 — BTC/BCH/
  LTC에서 센티먼트가 수익률을 그레인저-인과. 관련 트윗의 1~14%가 봇이라는 점도 발견.
  **검증: 불명확(OOS 분할 없음).**
- **Noguchi et al. (2026)**, IEEE Access, DOI:10.1109/ACCESS.2026.3691381. "CARVS" —
  레딧 상대거래량 센티먼트 규칙기반 전략, 6개 코인, **명시적 2024~2025 OOS 구간**, 비용반영
  수익률·Sortino가 buy-and-hold와 단순센티먼트 베이스라인 모두 능가. **검증: OOS 검증됨** —
  이 클러스터에서 가장 명시적인 긍정 결과 중 하나.
- **D'Amario & Ciganovic (2022)**, arXiv:2210.00883. LASSO-VAR+센티먼트, 10개 코인, 30일
  롤링 예측 — 방향정확도는 벤치마크 대비 +10%p 개선되지만 **post-double-LASSO 그레인저
  검정에서 센티먼트→수익률 인과관계 자체는 기각**. **검증: OOS 검증됨 — 눈에 띄는 부정적
  인과관계 결과(정확도 개선과 인과관계는 별개임을 보임).**
- **Haroon (2026)**, arXiv:2607.23370. 변동성 레짐에 따라 레딧 FinBERT 센티먼트와 OHLCV를
  게이팅 결합, 3,491개 시간봉(2024-07~2025-09) — **macro-F1 0.55로 동전던지기에 근접**(정적
  결합 베이스라인은 능가). **검증: 불명확 — 센티먼트 엣지 과신에 대한 경고성 결과.**
- Valencia et al.(2019, DOI:10.3390/e21060589), Vella Critien et al.(2022, DOI:10.1186/
  s40854-022-00352-7, 방향+크기 예측 63%), Kyriazis et al.(2022, DOI:10.1016/
  j.qref.2022.09.004, COVID기간 비선형 그레인저) — **검증: 백테스트뿐/불명확.**

### 4.3 옵션만기·감마노출(GEX) 이벤트드리븐

- **Weiss, Gaudiosi, Zhou & Webb (2026)**, Finance Research Letters, DOI:10.1016/
  j.frl.2026.110340. **Deribit 일일 옵션만기 전후 BTC 현물의 통계적·경제적으로 유의한
  일중 반전**을 문서화, 딜러 누적감마가 음수일 때 가장 강함(딜러 헤징압력과 일치), 연간
  ~5천만달러 규모의 부(富) 이전 추정. **검증: 실데이터 이벤트 스터디** — 이 조사 전체에서
  이 하위주제 중 가장 신뢰할 만한 실증. 이 저장소가 아직 Tier0 문턱을 못 채우고 대기 중인
  [[eth_gex_status_and_next_direction_candidates_20260820]] 축을 계속 열어둘 근거가 됨.
- **Lachowicz (2025)**, SSRN, DOI:10.2139/ssrn.5782822. "Do Gamma Walls Actually Move
  Bitcoin Prices at Deribit?" — 제목상 정확히 이 주제이나 OpenAlex/Semantic Scholar/Crossref
  전부 초록을 못 가져와 **결과 방향을 검증 못 함** — 존재만 확인, 판단 보류.
  **검증: 불명확(검증 불가).**
- Regan & Xie(2026, DOI:10.3390/jrfm19060382) — 본론은 LLM 추론능력 검증 논문이고 크립토
  옵션 GEX 패턴(2020~2025)을 테스트 도메인으로만 사용, 다만 **0DTE 옵션 확산에 따라 GEX기반
  Sharpe가 1.8→0.1로 붕괴**했다는 부수적 결과를 보고 — 이 축도 시간이 갈수록 감가상각될
  수 있음을 시사. **검증: 불명확(전략논문 아님).**

**탐색 공백**: arXiv가 6개 쿼리 성공 후 지속 429에 걸려 GEX 전용 쿼리 3건(각각 "gamma
exposure"/"options expiry"/"dealer gamma") 전부 실패 — 이 하위주제의 진짜 공백.
Semantic Scholar 9건 중 7건 429(OpenAlex가 25건 무장애로 대부분 커버). GEX는 예상대로
가장 얇은 클러스터 — 위 3편 중 실제로 GEX 가격효과를 다루는 건 1편뿐.

---

## 종합 판단 — 이 저장소 기존 결론과의 교차검증

**압도적으로 재확인되는 쪽**:

1. **아키텍처는 병목이 아니다.** arXiv:2506.05764(단순모델이 크립토 LOB에서 DeepLOB와
   동급/우위)는 [[eth_candidate_lob_ofi_pipeline_smoke_test_20260822]](DeepLOB vs
   Transformer 유의차 없음, 둘 다 정보하한 벽)와 정확히 같은 결론이다. 두 결과 모두 "더
   나은 딥러닝 아키텍처"가 크립토 방향예측에서 병목이 아님을, 서로 다른 방법론으로
   독립적으로 가리킨다.
2. **gross 신호는 종종 실재하지만 비용이 집어삼킨다.** Pindza(2026) "Microstructure
   alpha"의 "신호는 실재하나 수수료를 못 이기고 자산 간 전이도 안 됨"은 이 저장소 RDE의
   "gross +4.8~9.7%지만 breakeven 0.4~1.5bp vs 실제비용 7bp/leg" 패턴,
   그리고 이 저장소의 지배적 메타발견([[repo_label_methodology_meta_finding]])과 문자
   그대로 같은 모양이다.
3. **크립토의 "공짜 점심"류 차익거래·베이시스 전략은 시장성숙과 함께 붕괴했다.**
   Crépellière(2018년 이후 거래소간 차익거래 급감)와 Zhivkov(2026, 3,570만 관측치 기준
   펀딩스프레드 대부분 비용반영시 수익성 없음)는 이 저장소가 RDE에서 실측한 "펀딩
   델타뉴트럴이 2024년엔 +12.9%/yr였으나 2026년엔 이론상한 분기 0.68% 미만으로 붕괴"와
   시점·방향까지 일치한다. 이건 우연한 일치라기보다, **2025~2026년 크립토 시장 자체가
   구조적으로 그렇게 변했다**는 여러 독립 소스의 수렴이다.
4. **모멘텀/평균회귀는 학계에서도 논쟁적이다.** [[eth_weekly_tsmom_bias_closed_20260817]]
   (TRAIN+2663bp→OOS−3207bp 완전반전)이 유별난 실패가 아니라, Grobys&Sapkota(모멘텀
   무의미)·Zaremba(부호가 유동성에 좌우)·Caporale&Plastun(반전전략도 실패) 등 문헌 자체가
   안정적 컨센서스 없이 갈라져 있다는 걸 이번 조사가 보여준다.

**아직 안 닫힌 축과 맞물리는, 좁지만 진짜인 여지**:

1. **GEX/옵션만기 반전** — Weiss et al.(2026)이 Deribit BTC 옵션만기 반전을 실데이터로
   문서화한 건, [[eth_gex_status_and_next_direction_candidates_20260820]]에서 로드맵 4종
   중 유일하게 안 닫힌 GEX 축을 계속 열어둘 외부 근거가 된다. 다만 같은 클러스터의
   Regan&Xie가 보고한 "0DTE 확산에 따른 Sharpe 1.8→0.1 붕괴"는 이 엣지도 시간이 지날수록
   빠르게 감가상각될 수 있다는 경고다. 그리고 이건 **BTC/Deribit** 증거이지 ETH 증거가
   아니다 — 이 저장소 로드맵이 "BTC가 먼저 Tier0 도달 예상"이라 적어둔 것과 방향이 맞다.
2. **자산매칭 마켓메이킹(방향예측이 아닌 스프레드+펀딩 전략)** — Le(2026)의 펀딩인지
   HJB 마켓메이킹은 정확히 이 저장소가 거래하는 ETH/BTC/SOL을 Hyperliquid 데이터로
   캘리브레이션했다는 점에서 우연찮게 자산이 일치한다. 다만 이건 이론+시뮬레이션뿐이고,
   "포지션 방향을 예측"하는 게 아니라 "양방향 호가로 스프레드+펀딩을 수확"하는 완전히
   다른 전략 축이다 — 이 저장소의 peg-maker 인프라([[eth_maker_fill_simulation_l2_
   20260822]])는 "방향성 베팅의 체결비용을 낮추는" 용도였지, "포지션 방향 자체가 필요없는
   양방향 마켓메이킹"은 시도된 적이 없다. 같은 클러스터의 "The Market Maker's Dilemma"
   (실거래 검증, 메이커체결이 역선택 신호)는 이 축을 시도한다면 반드시 짚어야 할 리스크로
   같이 발견됨. **⚠️ 같은 날 후속 — 수수료 산술 cheap-gate 실행, REJECTED_FEE_STRUCTURE**:
   최대관용 수익상한 1.39bp/RT vs Binance VIP0 수수료만 4.0bp/RT(3배), breakeven maker
   수수료 0.007%/leg는 딥VIP/타 venue 영역이고 제로수수료 venue조차 역선택 ≥0.7bp/leg면
   음수 — 재개는 수수료 현실 변경(운영 결정)+resting AS 실측 둘 다 충족 시에만.
   상세: [eth_candidate_maker_mm_spread_capture_fee_cheap_gate_20260823.md](
   experiments/eth_candidate_maker_mm_spread_capture_fee_cheap_gate_20260823.md),
   registry `eth_mm_spread_capture_fee_arithmetic_cheap_gate_20260823`.

**결론**: 이번 조사는 애초 질문("AI가 트레이더와 같은 데이터로 같은 로직을 더 빠르게 할
수 있지 않을까")에 대한 회의적 입장을 독립 문헌으로 강화한다 — 특히 아키텍처·비용·차익거래
붕괴 세 갈래에서. 동시에 완전히 새로운 축을 찾진 못했지만, **방향예측이 아닌 마켓메이킹**과
**GEX 이벤트드리븐**은 "재탕이 아닌 진짜 좁은 여지"로 남는다.

## 조사의 한계

- 전수조사가 아닌 타겟서베이(하위주제당 상위 5~10편). arXiv/Semantic Scholar 양쪽 모두
  세션 중 반복적인 HTTP 429(무료키 미설정)를 겪어 일부 쿼리가 실패·미완료로 남았다 — 각
  클러스터 절 말미에 구체적으로 기록.
- 초록/본문을 못 가져와 요약을 못 한 논문(Lachowicz 2025 등)은 추측하지 않고 "검증
  불가"로 명시했다.
- CORE/Unpaywall/PubMed계열은 이 주제와 무관해 조회하지 않음(금융/CS 문헌이 arXiv+
  Semantic Scholar+OpenAlex에 집중돼 있음).
- 검증 여부(OOS/live vs 백테스트뿐)는 각 논문이 스스로 보고한 내용에 근거한 것이며, 이
  저장소의 Fresh-Forward 기준(진짜 causal bar-by-bar walk-forward)만큼 엄격하지 않은
  경우도 "OOS 검증됨"으로 표시했을 수 있다 — 구체적 방법론은 원문 확인 필요.
