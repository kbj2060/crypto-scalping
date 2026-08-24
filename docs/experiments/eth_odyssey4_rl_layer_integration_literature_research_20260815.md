# ETH Odyssey4(Omega4.6.1 스택) 강화학습 레이어 통합 조사 (2026-08-15)

상태: **문헌조사 + 내부 이력 감사 완료, 구현/학습 0건.** 순수 리서치 문서 — 코드 변경 없음.

## 요청

사용자: "지금 오디세이4 모델에 강화학습 레이어를 추가하고 싶어. 어떻게 넣는게 좋을지 어디에 넣는게
좋을지 최신논문을 이용해서 확실한 방법을 연구해줘."

두 갈래로 조사했다: (A) Explore 에이전트로 Odyssey4/Omega4.6.1 파이프라인 전체를 코드 기준으로
재확인 — 특히 이 스택에 RL을 시도한 기존 이력이 있는지, (B) 별도 에이전트로 2023~2026년
RL-트레이딩 문헌을 삽입 지점별로 조사. (A)에서 나온 사실이 (B)의 결론을 크게 바꿨다 — 0절에
그 이유를 먼저 적는다.

## 0. 선행 확인: 이 파이프라인에 RL을 시도한 이력이 이미 3건 있다 (재탕 방지)

**RL을 "새로 추천"하기 전에 반드시 확인해야 하는 사실**: 이 프로젝트는 Omega4.x/Odyssey 계열의
세 지점에서 이미 RL을 실제로 학습·평가했고, 셋 다 기존 baseline을 못 이겼거나 VAL 게이트에서
죽었다.

| 날짜 | 삽입 지점 | 시도한 알고리즘 | 결과 | 문서 |
|---|---|---|---|---|
| 2026-06-23 | 리스크 사이징 사이드카(margin_fraction/leverage) | `bandit_qnet`, `iql_awac`, `td3_bc_continuous`, `dsac_contextual` (16-action 이산 그리드: margin_fraction×leverage) | VAL선택 `iql_awac` OOS `+15.40%`(MDD -5.54%) vs 기존 HGB 사이드카 baseline OOS `+22.21%` — **RL이 기존 지도학습 회귀보다 못함** (부호반전은 아님, "미달") | `docs/model_contracts/omega4_4_rl_risk_sidecar_v1_full_20260623_contract.md` |
| 2026-07-09 | 포트폴리오 레벨 진입 게이트 | CQL(arXiv:2301.01298)/IQL(arXiv:2110.06169)/Decision Transformer(arXiv:2305.14550)/PAC-Bayesian Offline Contextual Bandit(arXiv:2210.13132) 문헌 기반 보수적 contextual bandit | `oos_extended` PnL `-18.03%`, 결정 1668건 중 skip 1612건(96.64%) — **표본부족+과보수로 실패** | `docs/model_contracts/portfolio_online_bandit_gate_native_20260709.md` |
| 2026-08-14(어제) | h48qual exit_head(청산 시점) | Gittins Index Deep RL(QGI/DGN, arXiv:2405.01157) — TD-bootstrap 기반 "은퇴가치(retirement value)" 재정식화 | `REJECTED_VAL_GATE` — 모든 임계값이 컴포넌트 PnL 부호반전 또는 퇴화 모드(원본과 동일 재현) | `docs/experiments/eth_omega461_gittins_index_exit_head_20260814.md` |

같은 날(2026-08-14) 작성된 문헌 스카우팅 문서
(`docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md`)도 독립적으로 RL을
"낮은 우선순위"로 이미 강등해뒀다 — 2026-07-09 bandit gate 실패를 근거로 들며 "새 RL 알고리즘을
찾는 것보다 표본 크기 자체를 늘리는 게 먼저 필요"라고 명시하고, 검증비용 순위표(1~5위)에서 RL
축 자체를 제외했다.

**왜 이게 중요한가**: 아래 2절의 최신 문헌 조사가 "가장 근거가 탄탄하다"고 꼽는 지점(포지션
사이징)과 알고리즘(offline RL — IQL/CQL 계열)은, 이 코드베이스에서 **이미 정확히 그 조합으로
시도된 적이 있다.** 즉 이번 조사는 "새 아이디어를 찾는 조사"가 아니라 "이미 3전 3패한 축을 다른
레시피로 재시도할 근거가 있는가"를 판단하는 조사여야 한다. direction-head 축이 40개 이상의
라벨링 방법론으로 재시도되다 결국 닫힌 것과 같은 패턴이 반복될 위험을 항상 염두에 둬야 한다
([[repo_label_methodology_meta_finding]]).

## 1. Odyssey4/Omega4.6.1 파이프라인 현황 (코드 확인, Explore 에이전트)

L0~L10 계약(`docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`) 기준, 실제
코드에서 재확인:

- **L0 피처**: `features/engineering.py` — `dual_momentum`(`:957`) 등 102 base cols + 13
  POS_COLS.
- **L1 레짐 라우팅**: HMM bull/bear/chop 3-expert,
  `scripts/train_omega1_regime3_expert_direction_head_volpca_20260602.py`.
- **L2 방향 생성**: 3-Head TabM(direction/quality/exit),
  `scripts/train_eval_omega1_2_tabm_3head_20260603.py`. h48qual(q=0.50)·zig075(q=0.75) 두
  컴포넌트가 같은 아키텍처를 각자 학습.
- **L3 진입 게이트**: quality_threshold 이상일 때만 통과
  (`trading_bot_modules/omega4_6_1_live.py:176-179`).
- **L4 Odyssey4의 유일한 신규 레이어 — zig075 SHORT 상승추세 진입veto**: **섀도우 스크립트에만
  존재**(`scripts/live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py:256-280`), 실거래
  `trading_bot.py`/`omega4_6_1_live.py`는 무수정 — 정적 규칙
  (`rolling(2016).mean(dual_momentum>0) > 0.8026`), 학습 모델이 아님.
- **L5 우선순위 조정**: 공유 슬롯 1개, `priority=("h48qual","zig075")` 순서로 먼저 조건 만족한
  쪽이 승자 (`omega4_6_1_live.py:330-349`).
- **L6 TP/SL**: ATR 기반 고정 공식(`atr_pct * mult`, min/max 클립) — 세 레짐 모두 2025년 내내
  floor에서 saturate(기존에 알려진 이슈).
- **L7 리스크 사이징**: **이미 학습된 모델**(HGB 회귀, `scripts/train_eval_omega4_2_risk_sidecar_20260622.py`)
  → sigmoid 매핑으로 margin_fraction/leverage 변환(Stage A) → 컴포넌트 스케일/캡(Stage B) →
  `finalize_sizing()` 최종 불변식 검증(Stage C, `trading_bot_modules/omega4_6_1_runtime_contract.py:18-73`
  — CLAUDE.md Futures Risk Sizing Contract의 코드 구현체).
- **L8/L9 청산**: TP/SL 우선, 그 다음 exit_head 확률 vs `EXIT_THRESHOLD=0.95`. zig075는
  exit_head가 사실상 발동한 적이 없음(0/86 bar-관측) — TP/SL만 실질적으로 작동.
- **L10 원장**: 실거래는 `binance_execution.py`/`position_accounting.py`, 섀도우는
  `closed_trades.jsonl`.

**삽입 지점 후보(구조상 가능한 지점, 순수 서술)**: `trading_bot.py:9005`(진입 결정 직후
`finalize_sizing()` 호출 전 — 사이징 개입) / `trading_bot.py:9139` 부근(`evaluate_exit` 호출
지점 — 청산 타이밍 개입). 두 지점 모두 같은 `Omega461EntryDecision`/`evaluate_exit` 계약을
공유해서, 그 계약 위에서 동작하는 레이어는 두 지점 어디에도 재사용 가능하다.

## 2. 최신 문헌 조사 요약 (2023~2026, 원 출처는 부록)

> 주의(신뢰도 보정): Odyssey1/Odyssey2가 지켰던 "리드 세션이 arXiv 원문을 직접 fetch로
> 검증"하는 관례와 달리, 오늘 웹 리서치 에이전트는 대부분의 인용을 검색 스니펫 기준으로
> 수집했고 원문 직접 fetch로 재검증하지 않았다(ACM "Margin Trader" 한 건만 페이월로 막혀
> 미검증임을 자진 명시). 아래 목록은 "탐색됨"이지 "이 프로젝트 기준으로 전수 검증됨"이
> 아니다 — 실제 구현 착수 전 핵심 논문(특히 재시도 레시피의 근거가 되는 논문)은 원문
> 직접 fetch로 재확인할 것.

### 2.1 삽입 지점별 근거 강도 순위 (문헌 기준)

1. **실행/헤징(집행 최적화)** — 가장 근거가 두텁고 실제 자본이 걸린 사례(JPMorgan의 Deep
   Hedging 계열, Buehler/Gonon/Teichmann/Wood 2019)가 있음. 다만 Odyssey4는 주문 집행
   레이어가 아니라 신호/사이징 레이어라 이 항목은 구조적으로 적용 대상이 아님.
2. **진입/청산 타이밍 정교화** — EarnHFT(AAAI2024, arXiv:2309.12891), Hierarchical
   Reinforced Trader(arXiv:2410.14927), DeepScalper(2022, arXiv:2201.09058) 등 계층형 RL.
   다만 세 논문 모두 **같은 연구실(NTU, Bo An)** 계열이라 독립 재현이 아니라 하나의 연구
   프로그램으로 취급해야 함. **이 지점은 이미 Gittins 실험(0절)으로 이 코드베이스에서
   REJECTED — 재확인 완료.**
3. **포지션/베팅 사이징(기존 신호 위에 얹는 사이징)** — Odyssey4의 margin_fraction
   사이징과 구조가 가장 가까움. FinPos(arXiv:2510.27251), FineFT(arXiv:2512.23773, 레버리지
   선물 특화), Zhang/Zohren/Roberts(arXiv:1911.10107, 변동성-스케일 사이징의 원조). **이
   지점도 이미 omega4_4 실험(0절)으로 이 코드베이스에서 시도 — 미달(not-better) 결과.**
4. **포트폴리오/자본배분** — RL-in-finance 문헌에서 가장 많이 연구됐지만 2025 메타분석
   (Hoque et al.)은 "정체(plateaued), 고전적 방법 대비 개선 미미"라 평가. **이 지점도 이미
   2026-07-09 bandit gate 실험으로 이 코드베이스에서 시도 — 실패.**
5. **End-to-end 정책(신호 생성 자체를 RL로 대체)** — 근거 최약. 이 프로젝트의 40개 이상
   방향 라벨링 실패 이력과 정확히 같은 실패 양식이 재현될 것으로 예상됨(2026-07-07
   Omega4.7-RL, discrete SAC 전체 스택 교체, 이미 `rejected_research_not_live_wired`).

핵심 메타분석: Hoque/Ferdaus/Hassan, "Reinforcement Learning in Financial Decision
Making"(arXiv:2512.10913, 2025) — 167편 이상 리뷰 대상 중 **"알고리즘 정교함은 성과와 유의한
상관관계 없음"**(policy-gradient vs DQN p=0.640), **구현 품질·도메인 전문성이 지배적
요인**(가중치 0.92/0.85 vs 알고리즘선택 0.45), 순수 RL 대비 hybrid ML+RL이 15~20% 우위,
2020→2025 사이 hybrid 채택 비중 15%→42% 상승. **이 프로젝트가 이미 하고 있는 구조(고정된
지도학습 신호 + 얇은 RL 오버레이)가 문헌이 수렴하는 방향과 일치한다** — 다만 "그 얇은
오버레이가 실제로 이긴다"는 것까지 보장하진 않는다(0절 참고).

### 2.2 오프라인/안전 RL (라이브 탐색 불가 전제 필수)

- **Conservative Q-Learning**(Kumar et al., NeurIPS2020, arXiv:2006.04779), **Implicit
  Q-Learning**(Kostrikov et al., arXiv:2110.06169) — IQL은 OOD 행동을 아예 쿼리하지 않아
  (expectile regression) 고정 데이터셋에 가장 보수적. **omega4_4 실험이 이미 "iql_awac"
  버킷으로 테스트 — 순수 IQL만 분리된 ablation은 아니었음(재시도 시 분리 검토 가치 있음).**
- **CVaR-제약 RL**: Chow/Ghavamzadeh/Janson/Pavone(JMLR2018, arXiv:1512.01629), CPPO
  (IJCAI2022, arXiv:2206.04436). 천연가스 선물 특화: Hêche et al.(arXiv:2501.04421, C51이
  CVaR 최적화로 baseline 대비 +32%).
- **무거래 함정(no-trade trap)**: 보상이 거래량으로 정규화 안 되면 "아예 안 함"이
  최적정책이 되는 실패양식 — 서로 다른 두 도메인(태양광 인트라데이 트레이딩
  arXiv:2510.16021, HFT arXiv:2511.02136)에서 독립 재현. **omega4_4 사이징 실험의
  보상식(`log(1+net_per_notional·margin·leverage) - 0.5·tail_excess -
  0.25·liquidation_excess`)에는 이 함정을 막는 명시적 baseline-상대항이 없었다** — 재시도
  시 반드시 보강해야 할 지점.

### 2.3 부정적 결과 / 비판 — 신뢰도 교정용

- Gort/Liu/Sun/Gao/Chen/Wang, "Deep RL for Crypto Trading: Addressing Backtest
  Overfitting"(arXiv:2209.05559) — 백테스트 과최적화를 예외가 아니라 **기본값**으로
  취급해야 한다는 입장.
- Henderson/Islam/Bachman/Pineau/Precup/Meger, "Deep RL that Matters"(AAAI2018,
  arXiv:1709.06560) — 5-seed 평균조차 진짜 알고리즘 성능을 왜곡할 수 있음을 증명한 원조
  논문. 이 프로젝트의 Sigma3-1h 5-seed 사고([[tabm_hp_low_signal_pattern]])와 **정확히
  같은 경고**를 RL 일반에 대해 먼저 해뒀던 셈 — FinRL Contest 후기(arXiv:2501.10709)는
  "크립토 시장은 이 변동성이 더 심하다"고 명시.
- "Optimal Stop-Loss/Take-Profit Parameterization"(arXiv:2604.27150) — RL 논문은 아니지만,
  저자들이 전쟁 등으로 왜곡된 시계열 OOS를 만나자 **시간순서를 섞은 랜덤 데이터로 대체
  평가**한 사례 — 이 프로젝트의 Fresh-Forward 원칙이 막으려는 바로 그 패턴의 실제 사례로
  반면교사.

### 2.4 보상 설계 (margin_fraction 사이징 기준)

- Moody & Saffell(2001)의 **differential Sharpe ratio**가 여전히 표준 참조.
- CFA Institute 실무 가이드(Halperin/Kolm/Ritter, 2025): `Reward = Profit − Costs −
  λ×Risk`(Risk=CVaR/drawdown, 분산 아님), "risk-neutral RL은 불충분" — 이 프로젝트의 기존
  notional/margin_fraction/TP-SL price-move 파이프라인 그대로 보상을 계산하고, 별도 스케일을
  발명하지 말 것(CLAUDE.md Futures Risk Sizing Contract와 정합).
- 2.2의 "무거래 함정" 방지책 — **고정 기준 사이징 정책 대비 상대 보상**(절대 PnL이 아니라)으로
  설계해야 size→0가 공짜 승리가 되지 않음.

### 2.5 검증 방법론

- **Combinatorial Purged K-Fold를 RL로 확장**(Gort & Yang, AI4Finance 2022) — Deflated
  Sharpe Ratio(Bailey & López de Prado, 2014)로 여러 시드/설정 스윕에 따른 선택편향 보정.
- 시드 개수에 대한 금융 특화 권고치는 문헌에 없음(일반론만 존재) — 이 프로젝트의 N≥5
  무작위시드 규정([[tabm_hp_low_signal_pattern]])이 오히려 문헌 평균보다 엄격한 편이라는
  점이 재확인됨. **바닥선으로 유지할 것.**
- 재학습 주기(비정상성 대응)에 대한 정량적 근거는 문헌에 없음 — 정성적 권고("드리프트
  감지기")뿐. 이 프로젝트의 Fresh-Forward 규정이 이 공백을 메꾸는 자체 기준으로 계속
  기능해야 함.

## 3. 종합 판단

### 3.1 어디에 넣지 말아야 하는가 (배제)

- **direction_head/quality_head**: End-to-end RL로 방향 자체를 학습하는 건 40개 이상 실패한
  지도학습 방향예측과 근본적으로 같은 축 — 문헌도 이 지점을 근거 최약으로 꼽고(2.1-⑤), 이
  코드베이스도 이미 Omega4.7-RL로 시도해 `rejected_research_not_live_wired`. **재시도 근거
  없음.**
- **zig075 진입veto 자체(L4)**: 지금은 학습모델이 아니라 정적 임계값 규칙이고, 이게 이
  프로젝트 전체에서 유일하게 CONFIRMED된 메커니즘이다
  ([[eth_odyssey4_zig075_long_entry_veto_downtrend_confirmed_20260815]]). 이걸 RL로
  대체하는 건 "작동 확인된 단순 규칙"을 "학습 필요·seed-분산 위험 있는 정책"으로 바꾸는 것
  — 문헌도(Hoque et al.) 알고리즘 정교화가 성과와 무관하다고 명시. **배제.**
- **exit_head(L8/L9)**: 바로 어제(2026-08-14) Gittins/DGN으로 재시도했고
  REJECTED_VAL_GATE. 진단된 실패 메커니즘("경쟁 컴포넌트 상태를 비교하지 않는 진짜 index
  policy 부재")은 다른 RL 알고리즘으로 바꿔도 그대로 남는 구조적 문제일 가능성이 높다.
  **최소 새로운 가설(zig075 상태를 실제 index 비교에 포함하는 진짜 restless-bandit
  재정식화) 없이는 재시도 금지.**
- **포트폴리오 레벨 게이트**: 2026-07-09에 이미 CQL/IQL/DT/PAC-Bayesian bandit로 시도,
  표본부족으로 실패. **표본 자체를 늘리지 않는 한 재시도 무의미.**

### 3.2 어디에 넣을 만한가 (유일하게 완전히 닫히지 않은 지점)

**L7 리스크 사이징(margin_fraction/leverage)만.** 이유:

1. 문헌이 가장 수렴적으로 추천하는 지점과 일치(2.1-③).
2. 이 프로젝트의 유일한 CONFIRMED 메커니즘(zig075 veto)과 구조적으로 같은 "이미 결정된
   신호 위에 얹는 조정 레이어"이지 새 신호 생성이 아니다.
3. 2026-06-23 시도가 실패가 아니라 **"미달"**(iql_awac OOS +15.40% vs baseline +22.21% —
   부호반전도 아니고 가드레일 붕괴도 아님)이었다는 점에서, Gittins/bandit-gate의 "구조적
   실패"와는 다른 급이다. 재시도 여지가 있다.

### 3.3 어떻게 넣을 것인가 (재시도 레시피 — 2026-06-23판과 반드시 달라야 할 지점)

1. **RL보다 먼저, 더 싼 걸로 이 사이드카를 이길 수 있는지부터 확인**: fractional-Kelly
   또는 direction/quality 확률의 단조 함수로 margin_fraction을 스케일하는 **비-RL 규칙**을
   먼저 만들어 기존 HGB 사이드카와 비교한다. 이 비교는 2026-06-23 실험에도, 오늘 조사한
   어떤 논문에도 없었다(웹 리서치가 "아무 논문도 이 비교를 안 한다"고 스스로 명시) — **가장
   싸고, 가장 먼저 해야 할 단계.** 이걸 못 이기면 RL을 볼 이유가 없다.
2. 그래도 RL을 볼 가치가 있다면: **연속 행동공간**(2026-06-23의 4×4=16 이산 그리드보다
   세밀하게) + **IQL 단독**(AWAC 항 없이 순수 expectile regression — OOD 행동 전혀 쿼리 안
   함, 6월 시도는 iql_awac로 뭉뚱그려 순수 IQL 단독효과 미분리) 또는 TD3+BC.
3. **보상에 무거래함정 방지 항 명시 추가**: 절대 PnL이 아니라 고정 기준 사이징 정책(예:
   현재 배포된 HGB 사이드카 자체) 대비 상대 보상 + CVaR 또는 differential Sharpe.
4. **N≥5 무작위시드**(2026-06-23은 알고리즘당 1회 실행뿐 — 이 프로젝트 현재 기준 미달) —
   VAL 전용 선택 → 사전등록 가드레일 → OOS는 통과시에만.
5. **Fresh-Forward bar-by-bar 재검증** — 2026-06-23 계약에는 fresh_forward 체크리스트가
   명시돼 있지 않았다(Gittins 문서와 달리); 재시도 시 명시적으로 채울 것.
6. bandit_qnet의 OOS 단일수치(+22.48%, baseline과 거의 동률)가 VAL 선택 기준(iql_awac
   선정) 때문에 묻혔다는 점도 흥미롭지만 **단일 실행이라 노이즈일 수 있음 — 재확인 없이
   근거로 쓰지 말 것.**

## 4. 정직한 결론

- **문헌에 "확실한 방법"은 없다.** 웹 리서치 자체가 이 프로젝트의 검증 기준(causal
  walk-forward + N≥5 무작위시드 + OOS 부호일치)을 충족하는 논문을 단 하나도 찾지 못했다고
  명시했다. "최근 논문 다수가 수렴하는 방향"은 있지만(신호 고정 + 사이징만 RL), "이게
  이긴다는 증거"는 없다.
- **이 코드베이스 자체의 실측 이력이 문헌보다 더 강한 증거다**: RL 3전 0승(사이징 미달,
  포트폴리오 실패, exit 구조적 실패). 문헌이 추천하는 정확히 그 지점(사이징)에서 이미 한
  번 졌다.
- 그럼에도 사이징 축은 "구조적으로 죽었다"기보다 "아직 이길 레시피를 못 찾았다"에 가깝다
  (3.2-③) — 재시도할 유일한 후보로 남겨두는 게 합리적이나, **RL이 필요조건이 아닐 수 있다는
  가능성(3.3-①의 비-RL 벤치마크)을 먼저 닫아야** 진짜 RL의 증분가치를 주장할 수 있다.
- 이 세션은 구현을 하나도 하지 않았다 — 사용자가 이 방향(3.1~3.3)에 동의하면, 다음 세션에서
  (a) 비-RL 사이징 벤치마크 → (b) 그걸 넘지 못하면 종료, 넘으면 (c) 위 레시피로 RL 재시도,
  순서로 진행을 제안한다.

## 5. 후속: 3.3절 1단계(비-RL 벤치마크) 실행 결과 (같은 세션, 2026-08-15)

사용자가 "비-RL 사이징 벤치마크부터 만들어줘"로 3.3절 1단계를 승인해 즉시 실행했다. Fractional
Kelly(`f = p - (1-p)/b`, p=quality_score, b=TP/SL비)로 zig075의 HGB 사이드카 스코어를 대체 —
**컴포넌트 단독·VAL(선정 기준 구간)에서 Kelly가 HGB를 못 이겼다**(PnL 29.02% vs 40.31%). 포트폴리오
레벨 6구간 게이트는 기계적으로 `CONFIRMED`가 찍혔으나 정확히 VAL에서 포트폴리오도 악화되고
컴포넌트-포트폴리오 괴리(Gittins/GBDT/TCN과 같은 패턴)가 나타나 신뢰하지 않는다. 상세:
[[eth_omega461_fractional_kelly_sizing_benchmark_20260815]].

**이게 3절의 판단에 미치는 영향**: "RL이 사이징에서 이길 근거가 약할 수 있다"는 우려가 한 단계
더 구체화됐다 — 닫힌형 규칙조차 HGB를 못 이겼으므로, RL이 이 축에서 이기려면 HGB가 아니라
**"HGB보다 나은 무언가"**를 만들어야 하는데, 지금까지의 증거(HGB 자체가 이미 이 feature set의
꽤 촘촘한 근사)는 그 여지가 크지 않을 수 있음을 시사한다. 다만 이는 확정이 아니다 — Kelly는
그리드 5개 축 전부에서 탐색 경계 끝에 승자가 위치해(부록 참고 없음, 실험 문서의 "그리드 경계
캐비엇" 참고) 더 넓은 탐색이나 다른 비-RL 규칙(예: 문헌이 언급한 posterior-probability 기반
변형)이 남아 있을 가능성은 열려 있다. RL 재시도를 결정한다면, 3.3절의 원 권고(연속 행동공간·
순수 IQL·무거래함정 방지 보상·N≥5 시드·Fresh-Forward)에 더해 "HGB를 넘는 비-RL 대안도 결국
못 찾았다"는 이 결과를 반드시 재검토 근거로 인용해야 한다.

## 6. 축 종결 (같은 세션, 2026-08-15)

5절의 사이징 벤치마크(fractional Kelly)가 HGB를 못 이긴 뒤, 사용자 요청으로 진단에 근거한
2번째 비-RL 후보(변동성-스케일 Kelly, [[eth_omega461_volatility_scaled_kelly_sizing_20260815]])도
테스트했으나 **plain Kelly보다도 나빴다**(VAL PnL 29.10% vs 33.95% vs HGB 40.31%). 사용자가
"여기서 사이징 축 접고 다른 문제로 넘어가자"로 결정해 이 서브프로젝트를 종결한다.

**최종 상태 — 이 조사가 다룬 5개 삽입 지점 전부 부정 또는 배제**:

| 지점 | 상태 |
|---|---|
| direction_head/quality_head | 배제(40+ 라벨링 실패 이력과 같은 축, Omega4.7-RL도 rejected) |
| zig075 진입veto(정적 규칙) | 배제(유일한 CONFIRMED 메커니즘, 학습모델로 바꿀 이유 없음) |
| exit_head | 부정(Gittins/DGN, `REJECTED_VAL_GATE`, 2026-08-14) |
| 포트폴리오 게이트 | 부정(contextual bandit, 2026-07-09, OOS -18.03%) |
| **사이징(margin_fraction/leverage)** | **부정**: RL(iql_awac 등, 2026-06-23, HGB 대비 미달) **+** 비-RL 대안 2종(plain Kelly, 변동성-Kelly, 둘 다 2026-08-15에 HGB 못 이김) |

**정직한 결론**: "오디세이4에 RL을 어디에 넣을까"라는 원 질문에 대해, 이 저장소 자체의 실측
증거로 답할 수 있는 모든 삽입 지점을 소진했다 — 지금 시점에 양(+)의 근거를 가진 지점은 하나도
없다. 사이징 축은 RL·비-RL 둘 다 시도했고 둘 다 기존 HGB 사이드카를 못 이겨, "HGB가 이 12개
남짓 `parent_outputs` 피쳐 조합에서 상당히 촘촘한 근사"라는 가설이 3개의 독립 실패
(RL 사이드카, plain Kelly, 변동성-Kelly)로 뒷받침된다. 이 결론을 뒤집으려면 새로운
가설이나 새로운 정보원(예: `parent_outputs` 바깥의 피쳐, 또는 근본적으로 다른 문제
정식화)이 필요하다 — 같은 정보원으로 사이징 규칙의 형태만 바꾸는 시도는 이 세션 기준으로
소진됐다.

## 부록: 원 출처 (에이전트 웹 리서치 원문에서 추출, 대부분 원문 직접 재검증 안 됨)

**서베이/실무 가이드**
- Pippas, Ludvig, Turkay, "The Evolution of RL in Quantitative Finance" (ACM Computing
  Surveys 2025) — arXiv:2408.10932
- Hoque, Ferdaus, Hassan, "RL in Financial Decision Making: A Systematic Review" (2025) —
  arXiv:2512.10913
- Bai, Gao, Wan, Zhang, Song, "A Review of RL in Financial Applications" (2024) —
  arXiv:2411.12746
- Halperin, Kolm, Ritter, "RL and Inverse RL: A Practitioner's Guide" (CFA Institute
  Research Foundation, 2025)

**삽입 지점 / 계층형**
- Qin, Sun, Zhang, Xia, Wang, An, "EarnHFT" (AAAI2024) — arXiv:2309.12891
- Zhao, Welsch, "Hierarchical Reinforced Trader" (2024/2026) — arXiv:2410.14927
- Sun, Xue, Wang, He, Zhu, Li, An, "DeepScalper" (2022) — arXiv:2201.09058
- Buehler, Gonon, Teichmann, Wood, "Deep Hedging" (Quantitative Finance 2019) —
  doi:10.1080/14697688.2019.1571683

**베팅 사이징**
- Liu, Dang, "FinPos" (2025/2026) — arXiv:2510.27251
- Macrì, Jaimungal, Lillo, "Deep RL for optimal trading with partial information" (2025) —
  arXiv:2511.00190
- Zhang, Zohren, Roberts, "Deep RL for Trading" (2019/2020) — arXiv:1911.10107
- Borrageiro, Firoozye, Barucca, "RL for Systematic FX Trading" (2021/2022) —
  arXiv:2110.04745
- Qin, Cai, Li, Xia, Zong, Sun, Wang, An, "FineFT" (2025) — arXiv:2512.23773

**오프라인/안전/리스크인지 RL**
- Kumar, Zhou, Tucker, Levine, "Conservative Q-Learning" (NeurIPS2020) — arXiv:2006.04779
- Kostrikov, Nair, Levine, "Implicit Q-Learning" (2021) — arXiv:2110.06169
- Chen et al., "Decision Transformer" (2021) — arXiv:2106.01345
- Chow, Ghavamzadeh, Janson, Pavone, "Risk-Constrained RL with Percentile Risk" (JMLR2018)
  — arXiv:1512.01629
- Hêche, Nigro, Barakat, Robert-Nicoud, "Risk-averse policies... natural gas futures" (2025)
  — arXiv:2501.04421

**부정적 결과/비판**
- Gort, Liu, Sun, Gao, Chen, Wang, "Deep RL for Crypto Trading: Addressing Backtest
  Overfitting" (2022/2023) — arXiv:2209.05559
- Henderson, Islam, Bachman, Pineau, Precup, Meger, "Deep RL that Matters" (AAAI2018) —
  arXiv:1709.06560
- FinRL Contests 후기(시드/분산 코멘트) — arXiv:2501.10709

**보상 설계**
- Moody, Saffell, "Learning to Trade via Direct Reinforcement" (IEEE TNN 2001)
- Srivastava, Aryan, Singh, "A Risk-Aware RL Reward for Financial Trading" (2025) —
  arXiv:2506.04358

**검증 방법론**
- Gort, Yang, "Combinatorial PurgedKFold CV for Deep RL" (AI4Finance 2022)
- Bailey, López de Prado, "The Deflated Sharpe Ratio" (2014) — SSRN:2460551

**이 프로젝트 내부 인용**
- `docs/model_contracts/omega4_4_rl_risk_sidecar_v1_full_20260623_contract.md`
- `docs/model_contracts/portfolio_online_bandit_gate_native_20260709.md`
- `docs/experiments/eth_omega461_gittins_index_exit_head_20260814.md`
- `docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md`
- `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`
- `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md`
