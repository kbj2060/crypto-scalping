# Yush(@TraderYush) 오더플로우 전략 조사 및 흡수 가능성 검증 (2026-08-15)

## 성격

외부 트레이더 전략의 **데스크 조사 + 우리 저장소 대조 + 미보유 구성요소의 증거 측정**.
retrospective evidence study이므로 **promotion/live 후보 근거가 아니다**. Fresh-Forward
bar-by-bar 규칙은 적용 대상이 아니다(모델 선택/승격 주장 없음, 기존 evidence sweep 문서들과
동일한 성격). 라이브 파일 변경 없음.

## 요청

"yush라는 데이 트레이더가 오더플로우 전략으로 트레이딩하고 있다. 전략을 모두 조사해서 내
전략에 흡수 가능한지 연구해달라."

---

## 1부 — Yush 전략 조사 결과

### 인물 특정

- **Yush (@TraderYush)**, Yush Capital 운영. ICT(Inner Circle Trader) 방식에서 오더플로우로
  전환한 뒤 프랍펌 페이아웃 누적 **$2M+** 를 기록했다고 공개된 선물 트레이더.
- **거래 대상: NQ / ES (미국 지수 선물)**. 암호화폐가 아니다.
- 여러 프랍펌(Tradeify, Apex, TPT, FundedNext, Lucid, E8)에 분산 운용.
- 거래 시간: **뉴욕 개장 초반 몇 시간만** 거래.

### 전략 = LAF Model

Auction Market Theory + Volume Profile + 오더플로우를 결합한 프레임워크. 공개 자료 기준 전체
구조는 아래와 같다. (LAF 약어 자체의 원문 정의는 유료 커뮤니티 내부 자료라 공개 확인 불가.)

#### 1단계 — AOI(Area of Interest) 설정: 4개 도구 중 **최소 2개 이상 정렬** 필수

| 도구 | 내용 |
|---|---|
| Market-Generated Levels | 전일 고가/저가, 오버나이트 고가/저가, 전일 종가, 오프닝 레인지(32초 ORB) |
| Volume Profile | Value Area(거래량 약 70%), Low Volume Node(LVN) = 불균형 구간 |
| Big Trades | NQ 75 lot 이상, ES 200 lot 이상 체결 |
| Delta Profile | 양(+)델타=매수자가 ask 타격, 음(-)델타=매도자가 bid 타격. 핵심 관찰: **흡수(absorption)** 와 **갇힌 트레이더(trapped traders)** |

#### 2단계 — 두 개의 모델

**Model 1: Range (균형 시장)**
- Value Area High(저항) / Value Area Low(지지) **가장자리에서만** 거래.
- "레인지 중앙은 예측 불가능하므로 피해야 한다."
- 공격적 진입: 가장자리에서 조기 거부 신호(흡수 = 활동은 격렬한데 가격이 안 움직임, 델타는
  강한데 연속성 없음, 대량 체결이 실패) 발생 시.
- 확인 진입: 돌파가 실패하고 가격이 레인지 안으로 재진입한 뒤.
- 목표: 1차 = 레인지 중간, 2차 = 반대편 Value Area.
- 손절: 가장자리 바로 바깥.

**Model 2: Trend (불균형 시장)**
- 방향성 움직임 중 **LVN으로의 되돌림**에서 진입.
- 반대편이 공격적으로 나오는데 가격이 이어지지 않고 흡수가 보일 때.
- "약세가 아니라 강세에 스케일 인(scale into strength, not weakness)."
- 이전 고/저에서 부분 익절, 나머지는 구조 기준 트레일링.

#### 3단계 — 실행 규칙

- **예측 금지.** 모든 진입은 확인(confirmation) 기반.
- 확인 **최소 2개** 필수.
- 가격이 레벨로 돌아오지 않으면 **거래 없음**.
- **하루 2~3개** 고품질 거래만 (스캘핑).
- 수익 중인 거래를 절대 손실로 전환시키지 않음.
- 강한 모멘텀에 직접 진입 금지.
- 리스크는 **장 시작 전에 확정**, 장중 협상 없음.

### 핵심 통찰

한쪽이 움직임을 시도했다가 **실패하는 순간**이 기회다. 반전을 예측하는 것이 아니다.

### 증거 수준에 대한 솔직한 평가

- 세부 규칙의 출처는 대부분 전략 아카이브 사이트(TradeZella, ChartFanatics)의 2차 정리
  기사이지 Yush 본인의 1차 문서가 아니다. 유료 커뮤니티 밖에서 확인 가능한 범위가 여기까지다.
- **$2M+는 "프랍펌 페이아웃 누적"이다.** 다계좌 eval 스케일링 결과이지 위험조정 수익률
  (Sharpe/MDD)이 검증된 수치가 아니다. 트랙 레코드로서의 강도는 제한적으로 봐야 한다.

---

## 2부 — 우리 저장소와의 대조

전략을 구성요소로 분해해 저장소를 실제로 grep/파일 확인한 결과다.

### 이미 보유 중인 구성요소 (신규 알파 아님)

| Yush 구성요소 | 우리 쪽 대응물 | 위치 |
|---|---|---|
| Trapped traders (갇힌 트레이더) | **유동성 스윕(스탑헌트)** — 마스터 순위 **2위, lift 3.01배** | `docs/experiments/eth_deep_evidence_signal_sweep_round2_20260814.md` |
| Absorption (긴 꼬리형) | **거래량 폭증 + 긴 아랫꼬리 (Wyckoff)** — **3위, 2.94배** | 동일 문서 |
| Delta (매수/매도 공격성) | **오더플로우 순매도 급증** — **5위, 2.75배**. 라이브 102 feature에 `taker_buy_base`/`taker_buy_quote` 포함 | 동일 문서 / live TabM 번들 |
| Value Area / POC / VAH-VAL | **`core/cvp.py`** — K-means 클러스터 볼륨 프로파일, 실제 70% Value Area 구현. `cvp_poc_dist`, `cvp_vah_val_width`, `cvp_cluster_position`, `cvp_volume_imbalance` | `core/cvp.py:121` |
| Big Trades (대량 체결) | **고래 체결 임계값** `whale_usd_th` → `nif_whale`(고래 순유입) | `microstructure_scanner.py:459` |
| Absorption (호가 기준) | **`_compute_shadow_queue_absorption`** — 체결 방향 vs 호가 방향 divergence | `microstructure_scanner.py:523` |
| Range vs Trend 이원 모델 | regime HMM 라우팅 (regime3/regime4 state) | `data/ensemble/supervised/regime3_*` |
| 오더북 불균형 / 스푸핑 | `_compute_obi`, `_detect_spoofing`, `_compute_shadow_toxicity` | `microstructure_scanner.py:452`, `:545` |

즉 **LAF Model의 4개 확인 도구 중 3개(Volume Profile / Big Trades / Delta)는 이미 어떤 형태로든
저장소에 존재한다.** 그것도 대부분 우리 쪽 버전이 이미 lift 2.7~3.0배로 측정까지 끝나 있다.

### 보유하지 않은 구성요소

1. **Market-Generated Levels** — 전일 고/저, 오버나이트 고/저, 전일 종가, 오프닝 레인지.
   `features/`, `core/` 전체에서 `prev_day_high|pdh|overnight_high|opening_range|session_high`
   grep 결과 **0건**. 기존 유동성 스윕은 **롤링 N봉 스윙** 기준이라 다른 객체다.
2. **LVN (Low Volume Node)** — `core/cvp.py`는 POC/VA폭/클러스터위치/거래량불균형은 내주지만
   "저거래량 노드" 개념 자체가 없다.
3. **"흡수 = 격렬한 활동 + 가격 무변동"의 문자 그대로의 정의** — 우리 Wyckoff 신호는 **긴 꼬리를
   요구**한다(= 가격이 움직였다가 돌아옴). Yush의 정의는 애초에 가격이 안 움직이는 경우다.
4. **"확인 최소 2개" 규칙의 카운트 인코딩** — 개별 신호가 아니라 정렬 개수 자체의 사다리.
5. **대량 체결 필터** — 5분봉에는 틱 테이프가 없다. 최근접 대용치는 평균 체결 규모
   (`quote_volume / trades`).

### 구조적 이식 장벽 (측정 이전의 문제)

- **시간 해상도**: Yush는 footprint/DOM 틱 단위(32초 ORB, 호가 레벨별 흡수)로 본다. 우리 라이브
  모델은 **5분봉** 결정이다. 5분봉이 가진 오더플로우 정보는 `taker_buy_quote` 하나로 집계된
  거친 델타뿐이다.
- **틱 데이터 백테스트 불가**: `microstructure_scanner.py`가 `depth20@100ms` + `aggTrade`를
  실시간 수집하긴 하지만 (a) 섀도/분석 경로이고 실제 매매결정에 개입하지 않으며
  (`trading_bot.py:13320` 이하, Playbook Router는 "분석/대시보드 전용"), (b)
  `data/live/microstructure.duckdb`(91MB, 최종 기록 2026-08-11)는 과거 아카이브로서 백테스트
  기간을 못 덮는다. **즉 Yush식 호가 레벨 흡수를 과거 검증할 데이터가 없다.**
- **시장 구조**: NQ/ES는 정규장/오버나이트가 분리돼 전일 종가·오버나이트 레인지가 실제 의미를
  갖는다. 크립토는 24/7이라 이 레벨들의 의미가 약하다.
- **의사결정 주체**: 하루 2~3개 재량 거래 + 인간의 confluence 판단 vs ML 모델. 우리는 이미
  confluence 방식 자체를 별도로 테스트한 이력이 있다
  (`docs/experiments/eth_evidence_signal_top6_confluence_standalone_backtest_20260814.md`).

---

## 3부 — 미보유 구성요소의 실증 측정

의견 대신 숫자를 내기 위해, **저장소에 없는 것들만** 기존 evidence sweep과 **동일한 하네스**로
측정했다.

### 방법

- 스크립트: `scripts/analyze_eth_yush_orderflow_component_evidence_20260815.py`
- 하네스 재사용(재구현 아님): `event_study` / `excess_move` / `load_zigzag_pivots`
  (`analyze_eth_confluence_oscillator_bottom_top_evidence_20260814.py`),
  `compute_indicators` (`backtest_eth_slowk_williamsr_persistence_confluence_20260814.py`).
- 정답(ground truth): 저장소의 zigzag_action 스윙 피벗. 기존 마스터 순위와 **동일 척도**라 직접
  비교 가능하다.
- 창: VAL 2025-09-01~2025-12-31 + OOS 2026-01-01~2026-02-17(원데이터 실제 커버리지).
  창 내 33,939봉, 피벗 1,015개(bottom)/1,020개(top).
- 레벨 인과성: 모든 레벨은 **완료된 전일 / 완료된 전일 아시아 세션**에서만 생성해 전방 적용.
  같은 봉 lookahead 없음. 레벨 커버리지 전일 99.9% / Value Area 99.9% / 아시아 99.9%.
- 볼륨 프로파일: UTC 일 단위, 60 price bin, HLC3 가중, POC에서 양방향 확장하는 표준 70%
  Value Area 구성. LVN = 구간 내 평균 bin 거래량의 50% 미만인 bin.
- "레벨 터치" 허용오차: 해당 봉 ATR%의 0.35배.

### 결과 — 1시간(K12) 기준 lift

**바닥(bottom)**

| 신호 | n | 정밀도 | lift | 초과이동(%) |
|---|---|---|---|---|
| **Y6 흡수(거래량↑ + 몸통 없음 + 델타 역방향)** | 263 | 30.8% | **2.46배** | -0.39 |
| **Y7 전일레벨 스윕 후 재진입** | 489 | 20.7% | **1.65배** | -0.75 |
| Y8 대량체결 대용치(평균 체결규모 z) | 3,880 | 17.0% | 1.36배 | -0.55 |
| Y2 전일 아시아 세션 고/저 터치 | 949 | 14.1% | 1.13배 | -0.61 |
| Y1 전일 고/저·종가 터치 | 3,215 | 13.0% | 1.04배 | -0.59 |
| Y4 Value Area **중앙**(Yush의 안티신호) | 7,074 | 12.3% | 0.98배 | -0.67 |
| Y3 Value Area **가장자리** | 1,418 | 12.0% | 0.96배 | -0.61 |
| Y5 LVN 터치 | 7,323 | 11.0% | 0.88배 | -0.58 |
| Y9 확인 ≥1 / ≥2 / ≥3 | 14,603 / 2,203 / 229 | | 1.03 / **1.25** / **1.68**배 | |

**천장(top)**

| 신호 | n | 정밀도 | lift |
|---|---|---|---|
| **Y6 흡수** | 165 | 27.3% | **2.32배** |
| Y8 대량체결 대용치 | 3,880 | 13.9% | 1.19배 | 
| Y9 확인 ≥3 / ≥2 / ≥1 | 237 / 2,189 / 14,243 | | 1.08 / 1.06 / 1.01배 |
| Y4 Value Area 중앙 | 7,074 | 11.7% | 0.99배 |
| Y2 / Y1 / Y5 / Y7 / Y3 | | | 0.98 / 0.96 / 0.97 / 0.91 / **0.81**배 |

### 발견 1 — Market-Generated Levels는 ETH 5분봉에서 작동하지 않는다

Yush 프레임워크의 1번 도구인 전일 고/저/종가는 lift **1.04(바닥) / 0.96(천장)** 으로 사실상
무작위와 구분되지 않는다. 아시아 세션 레벨도 1.13 / 0.98이다. **24/7 시장에서 "세션 레벨"이라는
개념 자체가 약하다는 구조적 예상이 숫자로 확인됐다.**

단, **Y7(전일 레벨을 뚫었다가 안으로 되돌아온 실패 돌파)** 은 바닥에서 1.65배로 유의하다.
그러나 이는 **우리가 이미 가진 롤링 스윙 기반 유동성 스윕(3.01배)보다 명백히 약하다.**
즉 스윕 개념은 유효하지만, **앵커를 "전일 레벨"로 바꾸면 오히려 나빠진다.**

### 발견 2 — Value Area 기하학이 통째로 전이되지 않는다

Yush 규칙의 핵심 중 하나는 "가장자리에서만 거래하고 중앙은 피하라"다. ETH 5분봉에서는:

- Value Area **가장자리** = 0.96(바닥) / 0.81(천장)
- Value Area **중앙** = 0.98(바닥) / 0.99(천장)

**가장자리가 중앙보다 나은 게 아니라 오히려 같거나 나쁘다.** 천장 쪽 가장자리 0.81은 방향이
반대라는 뜻이다. LVN도 0.88 / 0.97로 근거 없음. **Volume Profile 기하학은 NQ/ES의 정규장
오클러 구조에 붙어 있는 것이지, 24/7 ETH에 그대로 옮겨지지 않는다.**

### 발견 3 — Yush의 "흡수"는 우리가 이미 가진 신호에 흡수된다 (결정적)

Y6이 유일하게 강했으므로(2.46 / 2.32배) 절제 실험과 VAL/OOS 분할을 돌렸다.

**바닥, 1시간 기준**

| 변형 | n | lift (VAL+OOS) | VAL | OOS |
|---|---|---|---|---|
| 거래량 z만 (`vol_z>=1.5`) | 3,075 | **2.10배** | 2.17 | 1.94 |
| 작은 몸통만 | 21,723 | 0.93배 | 0.93 | 0.93 |
| 델타만 | 20,670 | 1.19배 | 1.19 | 1.17 |
| 거래량 z + 작은 몸통 | 608 | 2.12배 | 2.21 | 1.92 |
| **Y6 전체 (거래량+몸통+델타)** | 263 | **2.46배** | 2.65 | **2.04** |
| **기존 Wyckoff (거래량+긴꼬리)** | 622 | **2.58배** | 2.66 | **2.40** |

**천장, 1시간 기준**

| 변형 | n | lift | VAL | OOS |
|---|---|---|---|---|
| 거래량 z만 | 3,075 | 1.69배 | 1.67 | 1.72 |
| **Y6 전체** | 165 | **2.32배** | 2.37 | 2.20 |
| **기존 Wyckoff** | 535 | 2.08배 | 2.08 | 2.08 |

세 가지가 동시에 드러난다.

1. **Y6 lift의 대부분은 그냥 "거래량 급증"이다.** 거래량 z 단독으로 이미 2.10배(바닥). "몸통이
   작다"는 조건은 바닥에서 +0.02배밖에 못 보탠다. Yush 정의의 정수라고 할 "가격이 안 움직인다"가
   **거의 아무 정보도 추가하지 않는다.**
2. **겹침이 결정적이다. Y6 바닥 신호 263개 중 217개(82.5%)가 이미 기존 Wyckoff 신호 봉이다**
   (천장 74.5%). 새 신호가 아니라 기존 신호의 **더 작은 부분집합**이다.
3. **바닥에서는 기존 Wyckoff가 전부 이긴다** — lift 2.58 vs 2.46, 표본 622 vs 263, OOS 안정성
   2.40 vs 2.04(Y6은 VAL 2.65→OOS 2.04로 감쇠). 천장에서만 Y6이 2.32 vs 2.08로 앞서지만
   OOS 표본이 52개에 불과하다.

이는 이 저장소가 반복해서 만난 패턴 그대로다 — **신호는 진짜지만 더 단순한 무료 벤치마크에
흡수된다.**

### 발견 4 — "확인 최소 2개" 규칙은 바닥에서만, 그것도 자기 최고 부품을 못 넘는다

Yush의 핵심 실행 규칙을 사다리로 측정했다.

- 바닥: ≥1 → 1.03배, ≥2 → **1.25배**, ≥3 → **1.68배** (단조 증가, 규칙의 방향성은 지지됨)
- 천장: ≥1 → 1.01배, ≥2 → 1.06배, ≥3 → 1.08배 (사실상 평평). 4시간 기준으로는 오히려 역전
  (0.95 → 0.92 → 0.89).

그런데 **≥3(1.68배)조차 단일 부품 Y6(2.46배)에 못 미친다.** 즉 이 신호 집합에서는 confluence
카운팅이 최고 부품을 개선하는 게 아니라 **희석한다**. 규칙 자체가 틀렸다기보다, 부품들의 품질
편차가 클 때 "개수 세기"는 잘못된 결합 방식이라는 뜻이다.

---

## 결론 — 흡수 가능성 판정

**신규 알파로서의 흡수: 불가.** Yush의 LAF Model을 구성요소로 분해했을 때 각 요소는 둘 중
하나였다.

- (a) **우리가 이미 더 강한 형태로 보유** — trapped traders(3.01배 vs 우리 앵커로 1.65배),
  absorption(기존 Wyckoff 2.58배가 Y6 2.46배를 82.5% 포함하며 상회), delta(2.75배),
  volume profile(`core/cvp.py`), big trades(`nif_whale`), range/trend 이원화(regime HMM).
- (b) **ETH 5분봉에서 lift ≈ 1.0으로 근거 없음** — market-generated levels(1.04/0.96),
  value area edge(0.96/0.81), LVN(0.88/0.97).

여기에 **구조적으로도 이식이 막힌다**: Yush의 실제 엣지는 호가 레벨별 footprint 흡수 판독인데,
우리는 (i) 5분봉 결정 주기이고 (ii) 그 해상도의 과거 데이터가 아예 없어 검증 자체가 불가능하다.
공개된 서술만 옮기면 남는 것은 이미 우리가 가진 것들의 열화판이다.

**부분적으로 취할 가치가 있어 보였던 것 — 둘 다 후속 실험(2026-08-15)에서 종결됨**

1. ~~평균 체결 규모 z-score~~ — **CLOSED**([[eth_yush_avg_print_size_zscore_redundancy_check_20260815]]):
   가격 추세 오염은 없었지만, 채택 전 필수라고 명시했던 `spearmanr(avg_print_z, volume_z)` 사전점검을
   탈락(ρ=0.74, 문턱 0.6 초과) — 순수 거래량 z-score와 사실상 같은 정보였다. 미채택.
2. ~~브레이크이븐 스톱 exit 오버레이~~ — **REJECTED**
   ([[eth_yush_breakeven_stop_exit_overlay_20260815]]): 6창 게이트(참고 3창+VAL+OOS-Q1/Q2)에서 실제
   판정 대상인 OOS-Q1/OOS-Q2 둘 다 악화(OOS-Q2는 PnL 부호반전+MDD 3배 악화). MFE 대비 되돌림을
   트리거로 쓰는 exit 계열이 이 자산/모델에서 구조적으로 불리하다는 세 번째 독립 사례.

**추천하지 않는 것**: Value Area 가장자리 게이트, LVN 되돌림 진입, 전일/오버나이트 레벨 피처
추가, confluence 카운트 게이트. 모두 위에서 근거가 없거나 역방향으로 측정됐다.

## 알려진 한계

- Y6/Y7의 표본이 작다(263 / 489, OOS는 75 / 그 이하). 절제 실험의 OOS 수치는 신뢰구간이 넓다.
- 대량 체결은 틱 테이프가 없어 **평균 체결 규모라는 대용치**로만 측정했다. 진짜 "75 lot 이상
  단일 프린트" 필터의 성능은 이 문서로 판정되지 않는다.
- 호가 레벨 흡수(Yush의 진짜 도구)는 과거 depth 데이터 부재로 **측정 자체를 못 했다.** 이 문서의
  "흡수" 결론은 5분봉 대용 정의에 한정된다.
- Yush 전략 서술의 출처가 2차 정리 기사다. 유료 커뮤니티 내부의 세부 규칙이 공개분과 다를 수 있다.
- retrospective event study이므로 거래 비용/체결 슬리피지가 반영되지 않았다. lift는 수익성이
  아니라 "피벗 근처일 확률의 배수"다.

## 산출물

- `scripts/analyze_eth_yush_orderflow_component_evidence_20260815.py`
- `tmp/eth_yush_orderflow_component_evidence_20260815/{yush_component_evidence_table,absorption_ablation_table}.csv`

## 출처

- [Yush's Trading Strategy: How He Uses Order Flow to Read the Market — TradeZella](https://www.tradezella.com/strategies/order-flow-strategy)
- [Yush's Strategy: Trading Trapped Buyers & Sellers with Order Flow — ChartFanatics](https://www.chartfanatics.com/strategies/order-flow-strategy)
- [Yush Capital | Live Futures Trading & The LAF Model](https://yushcapital.co/)
- [The ICT Trader Who Made $2 Million After SWITCHING To Orderflow — Words of Rizdom](https://open.spotify.com/episode/57VfH4eRQYCNwT0Wj1Pjhx)
- [Why Your ICT Strategy is Failing: The $1.5 Million Orderflow Reality Check — Medium](https://medium.com/@wernerotto/why-your-ict-strategy-is-failing-the-1-5-million-orderflow-reality-check-5abf91de7775)
