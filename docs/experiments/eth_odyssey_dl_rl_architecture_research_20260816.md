# ETH Odyssey — 딥러닝/강화학습 아키텍처 조사 (레이어가 아닌 모델 구조 자체) (2026-08-16)

상태: **조사 완료, 결론 명확. 내부 이력과 외부 최신 문헌이 독립적으로 같은 결론에 수렴한다 —
아키텍처 교체는 지금 이 프로젝트에서 시간 투자 가치가 낮다.** 순수 리서치 문서 — 구현 없음.

## 요청

사용자: "레이어보다 딥러닝과 강화학습 아키텍쳐에 관해 좀 더 연구해봐." 이번 세션 내내 시도한
"레이어"(entry veto, exit trigger, 우선순위, 동시-슬롯 등 TabM 위에 얹는 메커니즘) 전부가
실패한 뒤, 핵심 예측 모델 아키텍처(TabM) 자체를 바꾸는 방향은 검토된 적이 없어서 이걸 조사했다.

두 갈래로 조사했다: (A) 이 저장소가 TabM 대신/위에 이미 시도해본 딥러닝 아키텍처들과 그 결과,
(B) 2024~2026년 외부 문헌에서 tabular/시계열 예측 아키텍처의 최신 동향과 RL 재확인.

## A. 내부 이력 — TabM 대안이 전부 이미 시도되고 전부 실패했다

| 아키텍처 | 스크립트 | 타깃 | 결과 |
|---|---|---|---|
| TabM + VSN(변수선택 게이트) | `train_eval_omega1_2_tabm_3head_vsn_20260707.py` | direction/quality/exit | validation 소폭 개선하나 OOS PnL/MDD/WR 전부 악화(n=13~14, 근거 약함). 미승격 |
| TabM + Conditional Diffusion 리스크 정책 | `train_eval_omega1_2_tabm_diffusion_risk_20260603.py` | TP/SL/leverage/notional(사이징) | validation +23.03%지만 OOS −7.08%(고정템플릿보다 악화) — **"Do not promote"** |
| Mamba(SSM), 3가지 용도 | `build_omega1_dir3_tabm_cryptomamba_direction_20260601.py`(direction), `train_regime3_cryptomamba_pred_20260531.py`(regime), `train_eval_omega1_2_mamba_sac_coordinator_20260603.py`(exit lifecycle) | direction sidecar / regime / exit lifecycle | direction sidecar는 기존 CryptoMamba보다 약함(미승격). lifecycle 컨트롤러는 OOS +16.07%지만 **단일시드**(N≥5 게이트 미충족) |
| Transformer / FT-Transformer | `eval_alpha7_shared_backbone_ft_contract_test_20260601.py`, `train_omega1_transformer_teacher_20260531.py` | direction/quality lifecycle parent, teacher 소프트라벨 | FT-Transformer validation Cost3 −14.04%(음수) → 미승격. Transformer teacher는 결과 문서 자체를 못 찾음 — 사실상 방치 |
| TCN(dilated causal), 2가지 용도 | `verify_eth_h48qual_tcn_sequence_model_20260812.py`(direction), `research_eth_omega461_tcn_exit_head_val_20260813.py`(exit_head) | direction 직접예측 / exit_head | **가장 철저하게 검증됨**: direction TCN은 N=5 시드 VAL에서 이 세션 유일하게 완패는 아니었으나(2~3/5가 always-short를 이김) **OOS는 5/5 결정적 패배**(Wilcoxon p=1.0), 이후 150-trial HP서치×5피처셋 재검증에서 **OOS 0/75로 최종 반증**. exit_head TCN도 포트폴리오 레벨은 개선했지만 컴포넌트(h48qual) 레벨 PnL이 양전→음전해 게이트 실패, 종결 |
| (참고, 재조사 불필요) JEPA/contrastive embedding, MuZero/AlphaZero | — | — | 이미 이번 세션/이전 세션에서 닫힘(JEPA 0/9, MuZero overlay reject) |

**요약**: VSN·diffusion·Mamba·Transformer·TCN — TabM에 얹거나 대체할 수 있는 합리적인 아키텍처
선택지는 사실상 전부 시도됐고, 전부 baseline을 못 이기거나 OOS에서 반증됐다. TCN은 특히
h48qual과 가장 직결되는 시도였고(direction 직접예측) 가장 넓고 깊게(N=5 시드 + 150-trial
HP서치 + 5피처셋) 검증됐는데도 최종적으로 0/75.

## B. 외부 문헌 — RL은 여전히 막다른 길, 지도학습 아키텍처는 오히려 "지금 구조 유지"를 지지

### B-1. RL/시퀀스 정책 (짧게)

2025~2026 신규 논문(bilevel offline RL 포트폴리오, hierarchical RL pair trading, Gittins/restless
bandit 확장 등)을 확인했지만, `docs/experiments/
eth_odyssey4_rl_layer_integration_literature_research_20260815.md`가 이미 커버한 5개 삽입지점·
알고리즘(CQL/IQL/Decision Transformer/Gittins/restless bandit)에 진짜로 없던 새 패러다임은
없었다. **결론 유지: RL 재추천 근거 없음.**

### B-2. 지도학습 tabular/시계열 아키텍처 (본론)

1. **TabReD 벤치마크(arXiv:2406.19380, 원문 확인)** — TabM이 실제로 검증받은 그 벤치마크
   자체에서, **진짜 시간순 split**(random split 아님)에서는 단순 MLP·GBDT가 attention/retrieval
   기반 복잡 tabular DL을 이긴다. **이건 아키텍처 교체가 아니라 현재 선택(TabM=단순 구조+
   파라미터 효율 앙상블)을 지지하는 증거다.**
2. **TabPFN v2/v2.5/v3(arXiv:2501.02945 일부 확인)** — in-context 학습 패러다임이라 주로
   iid 벤치마크에서 검증됐고, causal 시계열 드리프트·저-SNR 금융 데이터에서의 강건성은
   미확인. 스케일이 맞는 버전(v2.5/v3, 최대 100만행 주장)은 원문 검증이 너무 최신이라 신뢰
   어려움. **패러다임 자체가 이 프로젝트 세팅에 안 맞음.**
3. **Tabular foundation model 앙상블의 "다양성 천장"(arXiv:2605.18696, 원문 확인)** — 153개
   OpenML 태스크에서 최신 tabular foundation model 6개의 pairwise 예측 상관이 0.961(거의
   중복), 최선의 앙상블도 최고 단일모델 대비 +0.18%p 얻는 데 연산량 253배. **앙상블 고도화
   방향은 실증적으로 거의 무익.**
4. **BatchEnsemble(=TabM 핵심 메커니즘)의 알려진 약점(arXiv:2601.16936, 원문 확인)** —
   이미지 벤치마크 기준이지만, BatchEnsemble 멤버들이 함수공간에서 거의 동일하고 단일모델
   베이스라인을 그대로 따라간다는 결과. 보완책 LoMETab(arXiv:2605.14365)은 데이터셋별 튜닝이
   필요해 drop-in 개선이 아님. **유일하게 값싸고 실행 가능한 진단**: TabM k=8 앙상블 멤버
   간 예측 상관을 실측해서 실제로 붕괴돼 있는지 확인하는 것 — 단 ③의 다양성 천장 결과와
   이 저장소의 Seed-Diversity 게이트 정책을 감안하면 기대치는 낮게 잡아야 한다.
5. **Mamba/SSM** — MambaStock 등 2024~2025 사례는 전부 가격 레벨 회귀·추세추종이고, 이
   프로젝트처럼 causal 5분봉 zigzag+ATR-barrier 방향/품질 분류에 SSM을 적용한 선례를 못 찾음.
   새 근거 없음.
6. **정보량 병목 vs 아키텍처(가장 중요한 질문)** — Nonstationarity-Complexity
   Tradeoff(arXiv:2512.23596)는 **신호가 약하고 비정상적일 때 모델 복잡도를 높이면 OOS
   성과가 악화**된다(나중에 깨지는 허위 패턴을 학습)고 보고. Spurious Predictability in
   Financial ML(arXiv:2604.15531, 원문 확인)은 falsification-audit 없이는 **아키텍처
   탐색 자체가 순수 무작위 데이터에서도 유의한 "개선"을 만들어낼 수 있음**을 보임.

## 종합 판단

내부 이력(전 아키텍처군 시도·전패)과 외부 문헌(TabM의 원 검증 벤치마크 자체가 단순구조 손,
TabPFN 부적합, 앙상블 고도화 무익, 복잡도 증가가 약신호·비정상성 하에서 알려진 실패 패턴)이
**독립적으로 같은 결론에 도달한다**: 이 프로젝트의 병목은 모델 아키텍처가 아니라
피처/라벨의 정보량 자체다(이미 이 저장소가 `repo_label_methodology_meta_finding`으로 라벨
쪽에서 확인한 것과 정확히 같은 결론이 아키텍처 쪽에서도 재확인됨). **아키텍처 교체 방향도
"레이어" 방향과 마찬가지로 지금 시점에서는 막힌 것으로 판단한다.**

유일하게 남은 값싼 실행 항목(④): TabM k=8 앙상블의 실제 예측 상관 진단 — 하고 싶다면
저비용이지만, 기대치는 낮게(문헌상 다양성 천장이 이미 알려진 패턴).

## 추록 (2026-08-16): 구현 충실도 감사 — "논문 이름만 빌리고 실제로는 다른 걸 짠 게 아닌가?"

TabM(`ThreeHeadTabM`)이 "TabM-style BatchEnsemble"이라고 자칭하면서 실제로는 R 어댑터만 있고
논문의 S·per-layer B가 통째로 빠진 자체 변형이었다는 게 드러난 뒤(→
`eth_candidate_faithful_tabm_batchensemble_contract_20260816.md`), 사용자가 "위 표의 다른
아키텍처들도 같은 패턴 아니냐"고 질문해서 5개 에이전트를 병렬로 붙여 위 표에 있는 VSN·
Diffusion·Mamba(3용도)·Transformer/FT-Transformer(3파일)·TCN(3파일) 구현을 각각의 원 논문/공식
메커니즘과 코드 레벨로 직접 대조했다.

**결론: 패턴은 균일하지 않다 — TabM만 진짜 문제였고, 나머지는 대부분 충실하다.**

| 아키텍처 | 판정 | 근거 |
|---|---|---|
| TabM(`ThreeHeadTabM`) | **불충실, 미고지** | (기존 발견) R만 있고 S·B 없음, 논문에 없는 residual 추가, 이걸 밝히는 주석/문서 전무 |
| VSN(`train_eval_omega1_2_tabm_3head_vsn_20260707.py`) | **불충실하지만 자체 고지됨** | 실제 구현은 GRN+softmax 변수선택이 아니라 단일 sigmoid 게이트(Squeeze-and-Excite에 가까움). 단, 파일 자체 docstring이 "Deliberately lightweight... not the full per-variable GRN+softmax design"이라고 날짜까지 박아 명시적으로 고지함 — TabM과 달리 은폐된 근사가 아니라 문서화된 근사 |
| Diffusion(`train_eval_omega1_2_tabm_diffusion_risk_20260603.py`) | **충실** | 진짜 cosine β-schedule forward noising, ε-예측 네트워크, 24-step DDIM 역샘플링 확인. 24-step/critic rerank 등은 정당한 저차원 설정 적응이지 이름만 빌린 게 아님 |
| Mamba/SSM(3용도: direction sidecar, regime, SAC coordinator) | **충실** | 셋 다 자체 근사가 아니라 공식 `mamba_ssm.Mamba` 패키지(Tri Dao/Gu 원 구현)를 직접 import해서 씀 — selective B/C/Δ가 패키지 자체 보장. 단 usage 1의 기반 모듈(`build_omega1_dir3_cryptomamba_direction_20260531.py`)이 이후 커밋에서 삭제돼 현재 import가 깨져 있고, `mamba_ssm`이 이 dev 머신에 설치조차 안 돼 있어 GPU/서버에서만 재현 가능 — **아키텍처 충실도 문제가 아니라 재현성/인프라 문제** |
| FT-Transformer 파일 1·2(`eval_alpha7_shared_backbone_ft_contract_test_20260601.py`, `eval_alpha3_ft_transformer_mtl_parent_20260515.py`+v2) | **충실** | 진짜 피처별 개별 토큰화(`x_j*W_j+b_j`, 시퀀스 길이=피처 수) + CLS 토큰 + 진짜 `nn.TransformerEncoder` 확인 |
| Transformer teacher(`train_omega1_transformer_teacher_20260531.py`) | **이름만 다름 + 죽은 코드** | 이건 FT-Transformer가 아니라 시퀀스=피처가 아닌 시퀀스=과거 72bar인 표준 시계열 Transformer(CLS 없이 마지막 timestep 풀링) — "FT-Transformer"로 표에 묶인 게 잘못된 분류. 게다가 삭제된 모듈을 import해서 현재 실행 불가, 1회 실행 결과도 OOS 근사-무작위(overfit)였고 다운스트림 소비자 0개 — 원 문서의 "사실상 방치" 서술과 정확히 일치, 결론에 영향 없음 |
| TCN(3파일: direction, HP서치, exit_head) | **충실, 미래정보 누수 없음** | causal padding을 인덱스 대수로 직접 검증(symmetric-pad-then-trim이 left-pad와 수학적으로 동일함을 확인), 지수적 dilation(1,2,4,8,16 또는 2^i) 확인, per-bar 윈도우도 `row_i`까지만 사용 확인. **"OOS 0/75" 반증 결과는 누수 아티팩트가 아니라 진짜 아키텍처 성능 결과로 신뢰 가능** |

**이 감사가 위 "종합 판단"을 바꾸는가— 아니오.** TabM 자체의 불충실성은 이미 별도 후보
(`eth_candidate_faithful_tabm_batchensemble_contract_20260816.md`)로 분리돼 재검증 중이고, 그
외 5개 아키텍처(VSN/Diffusion/Mamba/FT-Transformer/TCN)의 원래 부정적 결론은 구현 충실도
문제로 무효화되지 않는다 — 특히 이 프로젝트에서 가장 깊게 판 TCN(N=5시드+150-trial HP서치)의
OOS 0/75는 이제 "정말 안 되는구나"로 더 확실히 믿을 수 있게 됐다. 유일한 문서 정정 사항은
Transformer teacher를 "FT-Transformer" 계열로 분류한 것 — 실제로는 다른 아키텍처(시계열
Transformer)이자 죽은 코드였다는 점.

## 아티팩트

- 두 조사 에이전트의 원 결과는 이 문서에 통합됨(별도 저장 스크립트 없음, 순수 리서치).
- 관련: `docs/experiments/eth_odyssey4_rl_layer_integration_literature_research_20260815.md`(RL,
  전일), `repo_label_methodology_meta_finding`(메모리, 라벨 쪽 동일 결론), 
  `eth_candidate_faithful_tabm_batchensemble_contract_20260816.md`(TabM 자체 불충실성은 여기서
  별도 재검증 중).
