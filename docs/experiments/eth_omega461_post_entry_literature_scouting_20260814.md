# ETH Omega4.6.1 — post-entry(exit/사이징) 문헌 스카우팅 (2026-08-14, Odyssey2 #6)

상태: `literature_research_complete_no_implementation` — **순수 문헌 리서치 + 랭킹만. 학습/구현/코드
변경 0건, 라이브 파일 무변경.** Odyssey2 계약서 우선순위 큐의 6번째(마지막) 항목("1~5가 전부
소진되면 최신 논문 기반 신규 아이디어 탐색")을 실행한 결과.

## 목적과 범위

Odyssey2 재적용 우선순위 큐(레짐별 threshold→앙상블 불일치→오토인코더 latent→GBDT exit_head→TCN
exit_head) 5개가 전부 부정 결과로 소진된 시점에서, **post-entry(이미 열린 포지션의 청산·사이징)
컨텍스트에 실제로 적용 가능한 2024~2026년 문헌**을 4개 방향으로 조사한다: (1) 자본 기회비용을
명시적으로 반영하는 청산/사이징(최우선), (2) TabM 앙상블이 아닌 다른 방식의 불확실성 인식
사이징, (3) 메타라벨링/2차 필터의 2024~2026 발전(De Prado 이후, Odyssey1이 이미 다룬 두 논문과
비중복), (4) RL 기반 exit/포지션관리(낮은 우선순위, 가벼운 확인만). **direction/quality(진입
선택)의 무스킬 결론은 재검증 대상이 아니다** — 아래 모든 후보는 이미 확정된 direction_head/
quality_head를 그대로 둔 채 exit_head 또는 사이징 레이어에만 적용 가능한지를 기준으로 스크리닝했다.

- 방법: 리드 세션(나) 직접 다수 웹서치 + 핵심 인용 전부 arXiv abstract/PDF 직접 fetch로
  저자명·날짜·초록 원문 검증(Odyssey1의 "리드 세션 검증" 관례를 그대로 적용).
- 중복 배제 확인: Odyssey1의 두 리서치 문서(`eth_h48qual_tabm_backbone_replacement_model_
  research_20260812.md`, `eth_h48qual_oracle_label_design_literature_research_20260812.md`)를
  먼저 읽고 그 문서들이 이미 인용한 논문(Label Horizon Paradox arXiv:2602.03395, Spurious
  Predictability arXiv:2604.15531, Drift-Resilient TabPFN, TabICL, ModernNCA, AEDL 등)은 이
  문서에서 재인용하지 않았다.

## 이 프로젝트가 이미 확정한 배경 (재검증 대상 아님)

- **entry는 죽었다, exit만 유일 생존**: 7개 이상 모델 계열·3개 라벨이 always-short를 못 이김;
  h48qual exit_head의 라이브 ATR 배리어 재라벨만 VAL+OOS 둘 다 생존(섀도우 배포 중).
- **오늘 밤(GBDT/TCN exit_head) 핵심 패턴**: exit를 더 공격적으로 만들수록 컴포넌트 단독 PnL은
  악화(GBDT +9.23%→+2.72%, TCN +9.23%→**-7.74%** 부호반전)하는데 공유 슬롯을 자주 비워 포트폴리오
  지표는 개선(GBDT +46.59%→+101.27%, TCN +46.59%→+60.24%) — **포지션 자본의 기회비용을 명시적으로
  값매기지 않고 exit만 최적화하면 나쁜 포지션 관리를 회전율로 가리는 결과가 나올 수 있다.**
- **Evidential/Dirichlet(quality_head 대체 candidate B)은 이미 "신중 취급" 상태**였다
  (`eth_h48qual_quality_scalar_alternatives_research_20260811.md`) — 원 논문(Sensoy 2018)의 핵심
  주장이 "Is EDL uncertainty a Mirage?"에서 반박된다는 점까지는 Odyssey1이 이미 인용함. 이 문서의
  임무는 그 반박 **이후** 문헌(2025~2026)까지 확인해 최신 상태를 파악하는 것.
- **conformal 방법이 entry 쪽에서 이미 실패·금지된 전례가 있다**: `research_line_registry.json`의
  `eth_overnight_generic_feature_entry_filter_20260809`(17개 아이디어 전부 부정) 중 하나가
  "conformal-prediction(APS) abstention on h48qual's own direction probabilities"였고, 계약 문서가
  conformal abstention을 **hard gate**로 쓰는 것을 이미 명시적으로 금지한다(`eth_h48qual_quality_
  scalar_alternatives_research_20260811.md` "하지 말아야 할 것" 절). **아래 2·3번 후보는 전부
  "hard gate 아님"(연속 사이징 스케일이거나, entry가 아니라 exit 임계값 재보정)임을 확인하고
  골랐다** — 이 구분이 무너지면 후보 전체가 이미 닫힌 라인의 반복이 된다.
- **RL/bandit 게이트는 포트폴리오 레이어에서 이미 실패**: `docs/model_contracts/
  portfolio_online_bandit_gate_native_20260709.md` — CQL(2301.01298)/IQL(2110.06169)/Decision
  Transformer(2305.14550)/PAC-Bayesian Offline Contextual Bandits(2210.13132) 문헌 기반 보수적
  contextual bandit, `oos_extended` PnL -18.03%, decisions 1668건 중 skip 1612건(96.64%) — 표본
  극소·과도한 보수성으로 실패.

## 리서치 결과

### 1. 자본 기회비용을 명시적으로 반영하는 청산/사이징 (최우선 질문)

**핵심 발견: 이 문제는 OR(Operations Research)에서 "retirement formulation"이라 불리는, 이미 수학적으로
정식화된 문제다.** 여러 경쟁 옵션(h48qual 신호, zig075 신호)이 하나의 자원(공유 슬롯)을 놓고 경쟁할 때
"지금 계속 쥐고 있을 가치 vs 놓아주고 은퇴(retire)했을 때 받는 고정가치"를 비교하는 것이 Gittins index의
핵심 구성인데, 그 "은퇴 가치"가 정확히 이 프로젝트가 명시적으로 값매기지 않고 있는 **자본 기회비용**이다.

| 후보 | 출처 | 핵심 아이디어 | 이 프로젝트 적용 방법 | 검증 비용 | 기존 결론과 상충 |
|---|---|---|---|---|---|
| **Gittins Index Retirement Formulation의 Deep RL화(QGI/DGN)** | **Dhankhar, Mishra, Bodas, arXiv:2405.01157** (2024-05-02 v1, 2025-08-25 v4) | "계속 참여 vs 은퇴(retirement value 받고 이탈)"의 무차별점을 학습 — 은퇴가치 = 그 자원을 다른 곳에 썼을 때의 기대가치, 즉 기회비용을 지표(index) 하나로 압축. 응용 사례로 "배치로 도착하는, 서비스시간을 모르는 job들의 mean flowtime 최소화"를 직접 다룸 — "배치로 도착하는 트레이드 신호, 알 수 없는 최적 보유기간"과 구조적으로 동형 | (전체 재학습판) exit_head를 "continuation value 회귀"가 아니라 "retirement index 회귀"로 재정의 — 이미 계약서에 설계만 있는 Deep Optimal Stopping 경로(`design_only_not_implemented`)의 자연스러운 확장 | **높음** — 새 MDP/bandit 정식화, 신규 학습 필요 | 없음(exit_head만 건드림, direction/quality 동결 유지) |
| **(저비용 실전판) "대기압력(queue pressure)" 후처리 규칙** | 위 논문의 개념을 재사용한 이 문서의 제안(신규 논문 아님, retirement-value 개념의 실전 근사) | 재학습 없이: 기존 exit_head 확률 + "지금 이 순간 반대 컴포넌트가 진입 대기 중인가"를 결합해 `EXIT_THRESHOLD=0.95` 고정값을 동적으로 낮춤(대기압력 높을 때만) | GBDT/TCN exit_head VAL 실험이 이미 포트폴리오 리플레이에서 `source_component`/슬롯 승자를 bar 단위로 추적하고 있어(예: `eth_omega461_gbdt_exit_head_20260813.md`의 "슬롯 승자" 집계), 같은 리플레이 하네스에서 "그 순간 반대 컴포넌트가 진입조건을 만족했는데 슬롯이 막혀 있었는가"를 diagnostic으로 뽑는 것은 재학습 불필요, 순수 후처리 | **최저** — 기존 저장 예측/리플레이 재사용, 신규 학습 없음 | 없음 — exit_head의 **정책**만 조건부로 바꾸는 것이라 h48cons/zig075 라벨·direction·quality 전부 동결 유지. GBDT/TCN이 "무조건 더 빨리" 나가서 실패했다면, 이 규칙은 "반대쪽이 실제로 대기 중일 때만" 더 빨리 나가므로 같은 실패 메커니즘(무조건적 과잉회전)을 원칙적으로 피할 수 있음(가설, 미검증) |
| **Multiple Optimal Stopping의 Deep Learning화** | **Laurière, Talbi, arXiv:2512.22961** (2025-12-28) | Becker/Cheridito/Jentzen(2019, 이미 이 프로젝트 exit_head 설계가 인용 중)의 단일 stopping을 다중 행사(exercise)로 확장 — Dynamic Programming Principle + NN이 value surface `V(x,k)`(`k`=남은 행사 기회)를 직접 학습. 미국식 바스켓 옵션·비선형 효용 최적화로 검증 | **주의**: 이 논문 자체는 "한 자원을 여러 경쟁자가 놓고 다툰다"는 하드 제약을 명시적으로 모델링하지 않는다(같은 프로세스의 반복 행사가 대상) — 이 프로젝트의 "h48qual vs zig075가 슬롯 하나를 다툰다"는 구조에 그대로 맞지 않고 재해석이 필요함(`k`=슬롯 수=1로 두고 "행사"를 "진입"으로 재정의하는 시도는 가능하나 미검증) | 높음 — 이미 계획된 Deep Optimal Stopping 경로의 v2 격, 신규 학습 필요 | 없음 |
| **(참고, 더 무거움) Restless Multi-Armed Bandit / GINO-Q** | Zhang et al. 계열, GINO-Q arXiv:2408.09882(2024) 등 | h48qual/zig075를 "arm" 두 개, 슬롯을 "자원 1개"로 보는 restless bandit 정식화 — indexability 불필요라는 장점 | 완전히 새로운 포트폴리오 레벨 정책 학습 필요, 이 프로젝트의 "라이브 파일 무변경/섀도우만" 원칙과 결합하려면 설계가 커짐 | **매우 높음** | 포트폴리오 online bandit gate(2026-07-09) 이미 실패 전례 — 같은 계열 리스크, 표본 부족 문제가 재현될 가능성 높음 |

**정직한 평가**: 이 방향의 이론적 근거(Gittins retirement formulation)는 이 프로젝트가 오늘 밤 직접
관찰한 현상("exit를 공격적으로 하면 포트폴리오는 좋아지고 컴포넌트는 나빠진다")을 **정확히 설명하는
수학적 언어**를 제공한다 — GBDT/TCN이 사실상 "무조건적 은�퇴 성향"을 과하게 학습한 것이라면, 진짜
필요한 건 "은퇴가치(기회비용)가 실제로 높을 때만 은퇴"하는 조건부 정책이다. 다만 **문헌 자체가
트레이딩·단일슬롯 사례를 직접 다루지 않아 재해석이 필요**하고, 전체 재학습판(QGI/DGN)의 검증 비용은
낮지 않다. **저비용 실전판(대기압력 후처리 규칙)은 신규 논문이 아니라 이 문서가 retirement-value
개념을 이 프로젝트 구조에 맞게 근사한 제안**이라는 점을 명시한다 — 문헌적 근거는 개념적 정당화이지
직접 이식 가능한 알고리즘은 아니다.

### 2. 불확실성 인식 사이징 (TabM 앙상블이 아닌 다른 메커니즘)

| 후보 | 출처 | 핵심 아이디어 | 적용 방법 | 검증 비용 | 상충 여부 |
|---|---|---|---|---|---|
| **Conformal Kelly** | **Ryan, arXiv:2608.01494** (2026-08-02, 이 리서치 시점 기준 12일 전 게재) | conformal prediction 구간(75%)의 폭을 fractional Kelly 사이징의 스케일로 직접 사용 — 구간이 넓어지면(불확실) 포지션 축소, 좁아지면(확신) 확대. 구간 자체는 **모델 앙상블이 아니라 자산별 롤링 분위수(가장 단순한 방법이 최고 성능)**로 구성 — TabM의 파라미터공유 앙상블과 메커니즘이 완전히 다름 | 이 프로젝트의 사이징 GBM(`train_eval_omega4_2_risk_sidecar_20260622.py`, 이미 `--risk-context-feature-dir` 확장점 사용 경험 있음) 출력(margin_fraction)에 곱하는 스케일로 통합 가능 — CLAUDE.md의 Futures Risk Sizing Contract("margin_fraction 예측, TP/SL은 notional 도출 후 재승산 금지")와 정합적으로 margin_fraction 자체를 스케일하면 이중계산 위험 없음. 구간은 저장된 `oos_predictions_qXXX.csv` 등 기존 예측의 realized residual로 롤링 계산 가능(원 논문도 "느린 무가중 롤링 분위수가 최고"라고 명시) | **낮음~중간** — 기존 사이징 GBM 재학습 없이 후처리 스케일만 추가 가능, 다만 conformal calibration set 구성(어떤 잔차를 캘리브레이션 소스로 쓸지) 설계는 필요 | 앙상블 불일치 사이징 피처(Odyssey2 #2)가 "신호 분산 거의 0"으로 null result였던 전례가 있다 — Conformal Kelly의 구간폭은 **모델 내부 분산이 아니라 실현 오차 이력**에서 나오므로 같은 실패 원인을 자동으로 상속하지 않지만, 검증 전까지는 낙관 근거 없음. 원 논문도 정직하게 "development window 성공이 2022+ true OOS에서 완전히 재현되지 않았다"(40개 설정 중 다수가 pre-registered holdout에서 저조)고 보고 — 이 프로젝트의 반복된 VAL승리→OOS반전 패턴과 같은 계열의 경고 |
| **Evidential Deep Learning — 반박 이후(2025~2026) 최신 상태** | 반박: **Shen, Ryu, Ghosh, Bu, Sattigeri, Das, Wornell, NeurIPS 2024, arXiv:2402.06160**("Is EDL a Mirage?", Odyssey1이 이미 인용). **반박의 반박/수리 시도(2024~2026, 신규 확인)**: Jürgens/Meinert/Bengs/Hüllermeier/Waegeman(ICML 2024, Mirage 논문이 직접 인용하는 선행 비판), Wu et al. "evidence contraction issue"(AAAI 2024), Duan et al. "variance-based perspective"(WACV 2024), Wang & Ji "Beyond Dirichlet-based models"(UAI 2024), Chen/Gao/Xu(TPAMI 2025), "Generalized EDL: Bayesian Perspective"(arXiv:2605.25599, 2026), "Plug-in Losses for EDL"(arXiv:2605.22746, 2026), "Density-Informed Pseudo-Counts"(arXiv:2602.01477, 2026) | 원 논문(Sensoy 2018)의 핵심 주장(epistemic uncertainty가 데이터 증가에 따라 소멸)이 Mirage 논문에서 반박된 뒤, **2025~2026까지도 field가 계속 "고치는 중"** — 즉 2년 가까이 지나도 vanilla EDL을 신뢰할 만하다고 되돌리는 합의된 수정판이 없다 | (권장하지 않음, 참고용) | 매우 높음(전용 loss 재설계 + 재학습) | **Odyssey1의 "신중 취급" 판단이 2026-08 현재도 유효하다는 게 이 리서치의 결론** — 오히려 강화됨: 2년간 활발한 수리 시도(6편 이상)에도 "표준 답"이 안 나왔다는 것 자체가 아직 프로덕션에 넣을 성숙도가 아니라는 신호. **이 축은 순위表에서 제외한다** |

### 3. 메타라벨링/2차 필터의 2024~2026 발전 (De Prado 이후, Odyssey1 두 논문과 비중복)

**중요한 스크리닝 원칙**: entry 쪽 meta-labeling/필터는 quality_head 자체가 이미 그 역할이고 9개 후보로
소진됐다(Odyssey1). 아래 후보는 **entry 필터의 대체가 아니라, 이미 살아있는 exit_head의 고정 임계값
(`EXIT_THRESHOLD=0.95`)을 재보정하는 post-entry 전용 적용**으로만 스크리닝했다.

| 후보 | 출처 | 핵심 아이디어 | 적용 방법 | 검증 비용 | 상충 여부 |
|---|---|---|---|---|---|
| **Risk-Controlled Post-Processing of Decision Policies** | **Joshi, Wang, Hassani, Dobriban, arXiv:2605.06479** (2026-05-07) | "이해관계자가 바꾸길 꺼리는 기존 결정론적 baseline 정책"을 전제로, **위험 제약(chance constraint)을 위반할 위험이 큰 컨텍스트에서만** 대체 정책(oracle fallback)으로 전환하는 후처리 — population-level 최적해가 threshold 구조를 가짐을 증명하고, 유한표본에서도 O(log n/n) 초과위험 보장. **재학습 불필요, 기존 baseline 정책의 출력 위에서만 작동** | 이 프로젝트의 정확한 상황(TabM 라이브ATR exit_head 고정 임계값 0.95를 "바꾸길 꺼리는 baseline"으로 두고, fallback으로 GBDT/TCN exit_head 또는 대기압력 규칙을 후보로 넣어, "위험(컴포넌트 PnL 악화)이 클 것으로 예측되는 bar에서만" 전환)에 **개념적으로 가장 정확히 대응** — GBDT/TCN을 "전면 교체"가 아니라 "위험이 높다고 판단되는 좁은 조건에서만 개입하는 fallback"으로 재활용할 길을 열어줌 | **낮음** — calibration-only, VAL 데이터로 임계값만 선택, 재학습 0 | 없음. **오히려 GBDT/TCN 실험의 "전부 아니면 전무" 채택 방식(사전등록 게이트가 4개 지표 중 2개만 실패해도 전체 기각)의 대안**이 된다 — 두 실패한 exit_head를 "위험이 높은 좁은 상황에서만 쓰는 fallback"으로 재활용 가능한지 검토할 근거가 생김(재검증 필요, 이 리서치는 가능성만 제시) |
| **Selective Conformal Risk Control (SCRC)** | **Xu, Guo, Wei, arXiv:2512.12844** (2025-12-14 v1, 2026-04-27 v2) | conformal risk control을 selective classification과 결합한 2단계: 1단계가 "확신 있는 샘플만" 선별, 2단계가 선별된 부분집합에만 conformal 위험 제어 적용 — SCRC-T(정확한 유한표본 보장)/SCRC-I(계산 효율적 PAC 보장) 두 변형 | exit_head 확률을 1단계 선별 기준으로, 선별된 "확신 있는 exit 신호"에만 위험 제어된 sizing/청산강도를 적용 — 위 Risk-Controlled Post-Processing과 유사 계열이나 "선별 후 제어"라는 명시적 2단계 구조가 추가됨 | **낮음~중간** — 계산비용은 SCRC-I 기준 낮음, 다만 2단계 캘리브레이션 설계·튜닝 필요 | 없음. **"conformal abstention을 hard gate로 쓰지 말라"는 기존 금지와 다름** — 이건 entry gate가 아니라 exit 강도 조절이고, 위험제어 실패 시 "일부만 abstain"이 아니라 "보수적 fallback으로 완만히 전환"되는 설계라 과거 실패한 hard-gate abstention(2026-08-09)과 실패 메커니즘을 공유하지 않을 가능성 — **단, 이는 가설이며 직접 검증 전까지 확정 아님** |

### 4. RL 기반 exit/포지션관리 (낮은 우선순위 — 가볍게만 확인)

기존 실패(portfolio online bandit gate, 2026-07-09, CQL/IQL/DT/PAC-Bayesian offline bandit 기반,
`oos_extended` skip rate 96.64%)와 **질적으로 다른 강한 신규 후보를 찾지 못했다.** 2024~2026
문헌에서 decision transformer/offline RL/world model 계열 트레이딩 적용 다수 확인했으나, 전부 (a)
일반 주식/암호화폐 방향 트레이딩 맥락이지 "단일 슬롯 exit 재보정" 같은 좁은 문제에 특화된 사례가
아니고, (b) 이 프로젝트의 근본 제약(VAL 20~70건대 극소 표본)은 알고리즘을 CQL→IQL→DT 무엇으로
바꿔도 해소되지 않는다. **결론: 이 축은 낮은 우선순위 유지가 타당** — 새 RL 알고리즘을 찾는 것보다
표본 크기 자체(예: 레짐별 대신 통합 학습, 더 긴 학습기간)를 늘리는 게 먼저 필요하다는 게 기존
bandit gate 문서가 이미 시사하는 바와 일치. 이 축은 아래 순위表에서 제외한다(재확인만, 신규 후보
없음).

## 검증비용 낮은 순 우선순위 랭킹 (1~5위)

**사용자 지정 기준**: 검증비용 낮은 순, 단 "자본 기회비용" 방향에서 찾은 후보는 최우선으로.

1. **대기압력(queue pressure) 후처리 exit 규칙** (자본 기회비용, 저비용 실전판, Gittins
   retirement-value 개념 기반) — **최우선.** 재학습 불필요, 기존 GBDT/TCN 실험의 포트폴리오
   리플레이 하네스가 이미 bar 단위 `source_component`/슬롯 점유 상태를 추적하고 있어 "반대
   컴포넌트가 그 순간 진입 대기 중이었는가"를 diagnostic으로 뽑아내는 것만 필요. 오늘 밤
   발견한 "컴포넌트 vs 포트폴리오 괴리"를 구조적으로 해명하는 유일한 후보 — GBDT/TCN이
   "무조건 빨리 나간" 것과 달리 "필요할 때만 빨리 나가는" 조건부 정책을 시험할 수 있다.
2. **Risk-Controlled Post-Processing** (Joshi/Wang/Hassani/Dobriban 2026)으로 `EXIT_THRESHOLD=
   0.95` 자체를 위험 제어된 임계값으로 재보정, 또는 이미 실패 판정된 GBDT/TCN exit_head를
   "위험이 높은 좁은 조건에서만 개입하는 fallback"으로 재활용 가능한지 검토. Calibration-only,
   재학습 0. 1번 후보의 "언제 개입할지" 조건을 형식적 위험 보장과 함께 정하는 보완 관계로 묶을
   수 있음(1번의 규칙을 fallback으로, 이 논문의 알고리즘을 전환 시점 결정 메커니즘으로).
3. **Conformal Kelly 스타일 구간폭 사이징 스케일** (Ryan 2026) — 사이징 GBM 출력(margin_fraction)에
   후처리 스케일로 통합, 재학습 불필요하나 conformal calibration set 설계 필요. 사이징 축이라
   1·2번(exit 타이밍)과 다른 레버, 병행 검토 가능.
4. **Selective Conformal Risk Control** (Xu/Guo/Wei 2025/2026) — 2번과 유사 계열이나 2단계
   선별 구조가 추가돼 설계 복잡도가 약간 높음. 2번이 통하면 다음 단계로 검토.
5. **Gittins Index Deep RL 전체 재학습판(QGI/DGN, Dhankhar+ 2024) 또는 restless bandit
   재정식화** — 1번(저비용 근사판)이 방향성을 보이면 다음 단계로 고려할 수 있는 "제대로 된
   버전". 검증 비용이 이 목록에서 가장 높아 5위.

**순위 밖(제외)**: Evidential Deep Learning(2번 항목의 "권장하지 않음" 후보, 2년째 미해결
반박 상태) — 순위表에 넣을 만큼 검증비용 대비 기대치가 낮다. RL 기반 exit(4번 리서치 방향) —
신규 후보 자체를 못 찾음.

## 정직한 결론

이 리서치의 가장 중요한 발견은 개별 논문이 아니라 **문제의 재정식화**다 — 오늘 밤 GBDT/TCN
실험이 우연히 드러낸 "exit를 공격적으로 할수록 포트폴리오는 좋아지고 컴포넌트는 나빠진다"는
현상은, OR 문헌에서는 이미 "자원 하나를 놓고 경쟁하는 옵션들의 최적 정지 문제"로 반세기 넘게
연구된 구조다(Gittins 1979의 retirement formulation). 이 프로젝트가 GBDT/TCN에서 시도한 건
사실상 "청산 분류기를 더 예민하게 만들기"였는데, retirement-value 관점에서 보면 진짜 필요한
축은 "예민함"이 아니라 **"기회비용이 실제로 높을 때만 예민해지는 조건부 정책"**이다. 1위
후보(대기압력 후처리 규칙)는 이 재정식화를 재학습 없이 가장 싸게 시험하는 방법이고, 2위
(Risk-Controlled Post-Processing)는 그 조건부 전환을 형식적 위험 보장과 함께 하는 더 엄밀한
버전이다.

다만 **모든 후보가 미검증 가설이다** — 이 문서는 "구현하면 반드시 통한다"가 아니라 "이 방향이
문헌적으로 정당하고 검증비용이 낮다"만 주장한다. 특히 Conformal Kelly(2위/3위 후보 모두의
방법론적 기반)조차 원 논문 스스로 development window 성공이 진짜 OOS에서 부분적으로만
재현됐다고 정직하게 보고한다는 점은, 이 프로젝트의 반복된 VAL→OOS 반전 패턴에 대한 경고로
읽어야 한다 — 어떤 후보든 이 서브 프로젝트 표준(VAL 사전등록 게이트 → 실패 시 OOS 금지, N≥5
시드 등)을 그대로 적용해야 한다.

## 출처

리드 세션이 arXiv abstract/PDF 직접 fetch로 저자·날짜·초록 검증:
- [Tabular and Deep Reinforcement Learning for Gittins Index (arXiv:2405.01157)](https://arxiv.org/abs/2405.01157) — Dhankhar, Mishra, Bodas, 2024-05-02(v1)/2025-08-25(v4)
- [Deep Learning for the Multiple Optimal Stopping Problem (arXiv:2512.22961)](https://arxiv.org/abs/2512.22961) — Laurière, Talbi, 2025-12-28
- [GINO-Q: Learning an Asymptotically Optimal Index Policy for Restless Multi-armed Bandits (arXiv:2408.09882)](https://arxiv.org/pdf/2408.09882) — 2024
- [Conformal Kelly (arXiv:2608.01494)](https://arxiv.org/abs/2608.01494) — Ryan, 2026-08-02
- [Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage? (arXiv:2402.06160, NeurIPS 2024)](https://arxiv.org/abs/2402.06160) — Shen, Ryu, Ghosh, Bu, Sattigeri, Das, Wornell(Odyssey1이 이미 인용, 이 문서는 그 이후 상태만 추가 확인)
- [Risk-Controlled Post-Processing of Decision Policies (arXiv:2605.06479)](https://arxiv.org/abs/2605.06479) — Joshi, Wang, Hassani, Dobriban, 2026-05-07
- [Selective Conformal Risk Control (arXiv:2512.12844)](https://arxiv.org/abs/2512.12844) — Xu, Guo, Wei, 2025-12-14(v1)/2026-04-27(v2)

리드 세션 웹서치로 확인(원문 미직접 fetch, 2025~2026 EDL 수리 시도 최신 상태 파악용):
- Jürgens, Meinert, Bengs, Hüllermeier, Waegeman, "Is epistemic uncertainty faithfully represented by evidential deep learning methods?" ICML 2024
- Wu et al., "The evidence contraction issue in deep evidential regression", AAAI 2024
- Duan et al., "Evidential uncertainty quantification: A variance-based perspective", WACV 2024
- Wang & Ji, "Beyond Dirichlet-based models: when Bayesian neural networks meet evidential deep learning", UAI 2024
- Chen, Gao, Xu, "Revisiting essential and nonessential settings of evidential deep learning", TPAMI 2025
- [Generalized Evidential Deep Learning: From a Bayesian Perspective (arXiv:2605.25599)](https://arxiv.org/pdf/2605.25599) — 2026
- [Plug-in Losses for Evidential Deep Learning (arXiv:2605.22746)](https://arxiv.org/pdf/2605.22746) — 2026
- [Density-Informed Pseudo-Counts for Calibrated Evidential Deep Learning (arXiv:2602.01477)](https://arxiv.org/pdf/2602.01477) — 2026

이 프로젝트 내부 인용:
- `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md` (서브 프로젝트 계약)
- `docs/experiments/eth_omega461_gbdt_exit_head_20260813.md`, `docs/experiments/eth_omega461_tcn_exit_head_20260813.md` (오늘 밤 "컴포넌트 vs 포트폴리오 괴리" 발견)
- `docs/experiments/eth_h48qual_quality_scalar_alternatives_research_20260811.md` (Evidential candidate B "신중 취급" 원 판단, conformal abstention hard-gate 금지)
- `docs/model_contracts/research_line_registry.json`의 `eth_overnight_generic_feature_entry_filter_20260809` (conformal APS entry abstention 실패 전례)
- `docs/model_contracts/portfolio_online_bandit_gate_native_20260709.md` (RL/bandit 게이트 실패 전례)
- `docs/experiments/eth_h48qual_tabm_backbone_replacement_model_research_20260812.md`, `docs/experiments/eth_h48qual_oracle_label_design_literature_research_20260812.md` (Odyssey1 선행 리서치, 중복 배제 기준)
