# Odyssey4 레이어 추가/파라미터 개선 제안 (2026-08-16)

상태: **제안 단계.** 참고자료(`docs/deep_learning_layer_design_and_training_reference_20260816.md`) +
레이어 감사(`docs/experiments/eth_odyssey4_tabm_layer_design_review_20260816.md`) + 이 프로젝트의
전체 실험 이력(TabM 대안 5종·R+S+B 완성형·GCE/ELR/mixup)을 종합해서 구체적 실행 항목으로 정리했다.
**어느 것도 바로 라이브에 반영하지 않는다** — 이 저장소 표준 게이트(N≥5 진짜 랜덤 시드,
fresh-forward VAL/OOS, 신규 축이면 registry 중복확인+cheap_gate)를 통과해야 승격 검토 대상이 된다.

## 3줄 요약

1. ~~지금 당장 반영할 것: GCE(q=0.7)~~ — **정정(2026-08-16, §A1 끝부분 참고): 취소.** GCE
   단독도, LR/옵티마이저/선정기준까지 다 고친 4종 묶음도, 같은 시드로 기존 레시피와 N≥5 재현
   짝비교하면 전부 기존 레시피(AdamW+순정CE+flat lr=2e-3+patience=8)한테 진다. **지금 당장
   반영할 학습-레시피 항목 없음.**
2. **가장 실질적인 신규 제안**: 3개 레짐전문가 독립 트렁크 → 공유 트렁크+전문가별 헤드 재설계
   (데이터효율 근거로만, 정정 참고 — §C3).
3. **하지 말아야 할 것**: 레이어/hidden/k를 늘려서 용량 키우기, 그리고 이제 GCE/AdaBelief/
   cosine/Prechelt 학습-레시피 묶음도 — 전부 실패 확인됨.

---

## A. 즉시 반영 — 이미 검증 완료, 게이트 이미 통과

### A1. GCE(q=0.7)를 direction_head·quality_head에 적용
- **무엇을**: `scripts/train_eval_omega1_2_tabm_3head_20260603.py` 라인 274/279의 plain
  `cross_entropy`를 `research_eth_candidate_faithful_tabm_batchensemble_regularizer_isolation_20260816.py`의
  `gce_loss(q=0.7)` 구현으로 교체. exit_head(라인 285)는 그대로 plain CE 유지 — GCE isolation
  테스트가 exit_head는 범위 밖으로 뒀던 것과 동일한 스코프.
- **왜**: 이 정확히 같은 3-head 분류 구조·같은 라벨(zigzag_action)로 이미 N≥5시드 isolation
  테스트가 나왔고 GCE 단독이 baseline CE를 이겼다(val bacc 0.5758 vs 0.5740). ELR·mixup 단독은
  졌고, 셋 다 조합도 졌다 — 그러니 **GCE만** 넣는다, 나머지는 넣지 않는다.
- **게이트**: 이미 통과 — 단, **정정(2026-08-16, 병행 세션 발견)**: GCE만 단순 이식하면 안 된다.
  GCE 손실은 유계(`(1-p^q)/q`)라 confidence가 오르면 일찍 saturate돼서, 현재 canonical
  스크립트의 combined val_loss 조기종료 기준으로는 진짜 정확도 정점 대비 gap이 0.046~0.054까지
  벌어진다(plain CE의 gap 0.015~0.023 대비 약 2배) — "GCE가 별로다"라는 초기 인상 자체가 이
  결함 있는 선정기준의 착시였다는 게 병행 세션에서 N≥5시드로 확정됐다. **A1은 이제 4종 묶음이다**:
  GCE + plain class-balanced CE로 분리된 선정기준(Prechelt UP_4 strip 중단, `k=5`에폭
  strip·`s=3~4` 연속 악화 시 중단) + cosine LR(2e-4→2e-6, T_max는 실제 학습길이에 맞출 것) +
  AdaBelief 옵티마이저(최고 조합; AdamW만으로도 괜찮지만 순수 AdamW+flat-lr+기존선정기준 조합
  하나만으로는 GCE 효과가 과소평가된다는 게 확인된 유일한 조합이니 그걸로 결론내지 말 것).
  **RAdam+GCE 조합은 반드시 피한다** — 병행 세션에서 두 아키텍처 모두 파국적으로 실패.
  자세한 내용은 §E 및 `feedback_modern_dl_training_checklist` 메모리 참고.
- **최종 정정(2026-08-16, §E 끝부분에서 재확인 완료)**: 그 "4종 묶음"도 기존 레시피(순정
  AdamW+CE+flat lr=2e-3+patience=8)와 동일 시드·동일 프로토콜로 N≥5 짝비교하면 3개 expert
  전부 진다(bull -0.0123, bear -0.0053, chop -0.0150). **A1은 이제 "반영 안 함"으로 최종
  종결** — GCE 단독도, 4종 묶음도 둘 다 시도했고 둘 다 기존 레시피를 못 이겼다.

---

## B. 저비용 진단 — 구현 전에 먼저 실측

### B1. 레짐 전문가별 유효표본수
- **무엇을**: `route_w = _route_probs(route_frame)[:, expert_idx]`의 `sum()`을 bull/bear/chop
  3개 전문가에 대해 각각 재보고, `len(route_w)`(하드 카운트)와 비교.
- **왜**: 현재 3개 전문가는 완전 독립 모델인데 학습 데이터는 하드 분할이 아니라 소프트가중치라서,
  "각 전문가가 실질적으로 몇 개의 유효 샘플을 보는가"가 코드 어디에도 측정돼 있지 않다. 이 숫자가
  C안(아래 C2)의 우선순위를 정한다 — 전문가 간 유효표본수 격차가 크면 공유트렁크 제안의 근거가
  더 강해진다.
- **비용**: 학습 없이 실측만, 몇 분.

### B2. zigzag_action 피벗 확정까지 걸리는 bar 수 분포
- **무엇을**: `scripts/build_zigzag_action_labels_v2_20260604.py`의 pivot 로직으로 실제 라벨
  타임스탬프를 훑어서, 각 피벗이 "확정"되기까지(가격이 min/max_reversal_pct만큼 되돌아가는 데)
  걸린 bar 수의 분포(중앙값/95th percentile)를 실측.
- **왜**: 지그재그 라벨은 본질적으로 미래 가격움직임을 봐야 피벗이 확정되는 구조라, 내부
  85/15 검증분할 경계 바로 앞 학습샘플의 라벨이 검증쪽 정보를 살짝 담고 있을 수 있다(레이어 감사
  §6에서 지적). 정확한 갭 크기를 임의로 고르지 않고 이 분포의 95th percentile 정도로 정하는 게
  근거 있는 선택이다.
- **비용**: 낮음, 코드 새로 안 짜도 기존 라벨 산출물에서 바로 집계 가능.

---

## C. 신규 제안 — cheap_gate/본실험 필요

### C1. 내부 85/15 분할에 purge/embargo 갭 추가
- **무엇을**: `_fit_expert_3head`(라인 239-241)의 `train_idx=arange(split)`,
  `val_idx=arange(split,n)` 사이에 B2에서 실측한 갭만큼 인덱스를 비운다.
- **게이트**: cheap_gate로 갭 추가 전/후 val loss 곡선·조기종료 시점이 실질적으로 달라지는지부터
  본다(달라지면 기존 early-stop 판단이 낙관 편향이었다는 뜻). 달라지면 N≥5시드로 본실험.
- **실행 완료, CLOSED(2026-08-16)**: B2 실측(p95=54bar)으로 갭 크기를 정하고 cheap_gate 실행 —
  갭 추가 전/후 early_stop_epoch 완전 동일(둘 다 epoch 1), val_loss/direction_balanced_accuracy
  델타 무시할 수준 → 실질적 변화 없음, N≥5시드 본실험 진행 안 함. 진짜 102피처 파이프라인으로
  재실행은 안 했음(모델이 매번 epoch 1에서 조기종료되는 구조상 갭이 드러날 여지가 애초에 없다는
  설명이 정합적이라 재확인 우선순위를 C2에 먼저 배정). 전체 내용:
  `docs/experiments/eth_odyssey4_purge_embargo_gap_20260816.md`.

### C2. quality_loss_weight(0.80)/exit_loss_weight(1.15)를 nuisance 파라미터로 서치
- **무엇을**: 지금 진행 중인 N-HiTS/ModernTCN용 Optuna 인프라를 재사용해서 이 두 상수를
  탐색공간에 포함(로그스케일 아니라 선형, 0.3~2.0 범위 정도로 시작). scientific 파라미터는 이
  두 값, 나머지는 nuisance로 고정.
- **왜**: Tuning Playbook 분류법상 이 둘은 근거 없이 고정된 nuisance 파라미터다.
- **게이트**: 20trial 내외 quasi-random, best trial N≥5시드 재확인.
- **실행 완료, CLOSED(2026-08-16)**: Optuna(20trial, 단일시드)로 qw=0.451/ew=0.598을 찾았고,
  185피처 프록시 파이프라인 N=5시드 재현에서 평균Δ=+0.0037/std=0.0038(std≈mean, 경계선) —
  일단 승격 보류. **같은 날 진짜 102피처 파이프라인으로 재확인하자 평균Δ=+0.0009/std=0.0040으로
  악화**(std가 mean의 4배 초과, 개선 시드 4/5→2/5로 하락) — 경계선이 아니라 명확한 노이즈로
  격상, 캐노니컬 기본값(0.80/1.15) 유지. 전체 내용:
  `docs/experiments/eth_odyssey4_loss_weight_optuna_search_20260816.md`(1단계 프록시 결과),
  `docs/experiments/eth_odyssey4_loss_weight_true_features_reconfirm_20260816.md`(2단계 진짜
  피처 재확인, 최종 판정).

### C3. [신규, 가장 실질적] 레짐 전문가 독립 트렁크 → 공유 트렁크 + 전문가별 헤드
- **무엇을**: 현재 `bull`/`bear`/`chop` 각각 완전히 별개인 `ThreeHeadTabM` 인스턴스(자기만의
  `in_proj`+`blocks`+`norms` 전부)를 처음부터 학습하는 구조를, **트렁크(encode())는 3개
  전문가가 공유**하고 `direction_head`/`quality_head`/`exit_head` 3세트만(또는 헤드 앞에
  레짐 원-핫/확률 임베딩을 하나 더 붙여서) 전문가별로 분리하는 구조로 재설계.
- **왜**:
  - 지금 구조는 트렁크 파라미터가 3배인데, 각 트렁크가 보는 실질 학습신호는 `route_w`
    소프트가중치 때문에 오히려 희석된다 — B1 실측 전이라도 방향성은 명확: 트렁크당 유효
    데이터/파라미터 비율이 통합모델보다 나쁘다. 이건 그 자체로 독립적인 데이터효율 근거다.
  - **정정(2026-08-16)**: 원래 이 항목을 "R+S+B 완성형(용량+6.5%)이 N=5시드 전부 악화됐다는
    결과와 정반대 방향이라 성공확률이 높다"는 memorization/용량-데이터 경계 논리로 뒷받침했는데,
    병행 세션의 후속 N≥5시드 최종검증에서 그 설명 자체가 틀렸다는 게 확인됐다 — true-peak
    정확도(이상적 상한)는 두 아키텍처가 0.003~0.009 차이로 사실상 동일했고, 실제 격차
    (+0.042~0.058)는 R+S+B의 학습초반 손실이 더 시끄러워서 조기종료가 80%의 경우 1에폭만에
    터지는 **신뢰성 문제**였다(용량-데이터 경계/암기 문제 아님). 그러니 이 제안을 "이미 증명된
    용량-해악 메커니즘의 반대 방향"이라고 주장할 근거는 없어졌다 — 위 첫 번째 이유(데이터효율)
    만으로 독립적으로 판단해야 한다. 제안 자체를 접을 이유는 아니지만, 성공확률을 과대주장하지
    않는다.
  - registry 확인: `btc_zigzag_as_entry_model_component`(트렁크 공유가 아니라 지그재그를 별도
    헤드/피쳐/라우터로 쓰는 BTC 엔트리모델 이슈, "모든 조합모드가 오히려 나빠짐"으로 닫힘)와는
    질문이 다르다 — 이건 **ETH 3-전문가 구조 자체를 다시 묻는 것**이라 겹치지 않는, 진짜 새
    질문이다.
- **리스크**: 레짐별 특이 패턴이 트렁크 표현력 안에 실제로 존재한다면(즉 bull/bear/chop이
  근본적으로 다른 함수를 필요로 한다면), 트렁크 공유가 오히려 그 차이를 뭉갤 수 있다 — 이건
  실증으로만 판단 가능. 헤드만 분리하고 트렁크를 공유하는 게 "레짐별로 다른 게 얼마나 되는가"에
  대한 실증적 답을 준다는 점에서, 실패해도 유용한 정보다.
- **게이트**: 이건 아키텍처 축이라 cheap_gate로는 못 거른다(구조 자체를 바꾸는 것이라 무료신호
  프록시가 없음) — 바로 N≥5시드 본실험 + fresh-forward VAL/OOS, TabM 원본(현재 3-독립모델)
  대비 direction_balanced_accuracy로 비교. GPU 자원은 현재 N-HiTS/ModernTCN 작업 완료 후
  순서대로 진행 권장(같은 서버 GPU 하나를 공유 중).
- **실행 완료, CLOSED(2026-08-17)**: N=5 진짜무작위시드 본실험(진짜 102피처 파이프라인) 결과,
  direction_balanced_accuracy는 bull/bear/chop 전부 표준편차가 평균 델타와 같거나 훨씬 커서
  (1.22~78배) 노이즈와 구분 안 됨, 부호도 불일치. VAL/OOS PnL/MDD도 평균적으로 무개선이거나
  악화(5시드 중 1~2개만 shared_trunk가 더 나음). B1의 데이터효율 논거 자체는 반박되지 않았지만
  측정 가능한 개선으로 이어지지 않았다 — 이걸로 A(A1)/C(C1/C2/C3) 전 항목이 CLOSED로 마감,
  이 제안 문서의 후보 중 캐노니컬에 실제 반영되는 것은 없음. 전체 내용:
  `docs/experiments/eth_candidate_shared_trunk_regime_experts_n5seed_result_20260817.md`.

### C4. EMA 가중치
- **무엇을**: 학습 루프에 `θ_ema = β·θ_ema + (1-β)·θ`(β≈0.999 근방) 섀도우 가중치를 추가,
  추론/평가는 θ_ema로.
- **왜**: 참고자료 modern-DL 체크리스트 항목, 라벨 노이즈가 있는 상황에서 실제 개선 효과가
  있는지는 GCE/ELR/mixup 연구에서 테스트 안 된 별개 축(ELR의 소프트타깃 EMA와 다름, 이건
  가중치 자체의 EMA).
- **게이트**: cheap 저비용 재현 가능(학습곡선만 보면 됨) → N≥5시드.

### C5. 공유 선형층 명시적 초기화
- **무엇을**: `in_proj`/`blocks`/3개 헤드에 SiLU에 맞는 스케일(예: He 근사치 `σ=√(2/n_in)`,
  또는 SiLU 전용 미분값 기반 스케일)을 명시적으로 적용.
- **우선순위**: 낮음 — 실전 영향이 불확실하고, PyTorch 기본값이 실무에서 크게 문제되는
  경우는 드묾. 다른 항목들 다 끝나고 여유 있을 때.

---

## D. 하지 말아야 할 것 — 이미 닫힌 이유가 있음

| 하지 말 것 | 이유 |
|---|---|
| layers/hidden/k를 늘려서 용량 키우기 | R+S+B 완성형(용량 +6.5%)이 N=5시드 전부에서 실전 악화 — 단, 정정(08-16): true-peak 상한은 거의 동일, 실전 격차는 용량 문제가 아니라 학습 초반 손실이 시끄러워 조기종료가 자주 너무 일찍 터지는 신뢰성 문제였음(§C3 정정 참고). 그래도 "늘린 용량을 안정적으로 학습해낼 방법"이 아직 없는 한 실전에서 그대로 쓰긴 어려움 — 권장 안 함은 유지, 이유만 교정 |
| VSN/diffusion/Mamba/Transformer/TCN을 트렁크 전체 교체로 재시도 | 전부 이미 닫힌 axis(`eth_odyssey_dl_rl_architecture_axis_closed_20260816`) |
| label smoothing/ELR/mixup 단독 재시도 | GCE 단독만 이기는 것으로 이미 N≥5시드 검증됨, 재검증 불필요 |
| 배치 크기를 검증성능 튜닝 대상으로 서치 | Tuning Playbook 규칙 — 배치크기는 처리량 레버일 뿐 |

---

## E. 병행 세션에서 나온 추가 발견 (2026-08-16) — A1(GCE)과 별개 축, LR/옵티마이저/선정기준

다른 세션에서 진행한 `eth_candidate_faithful_tabm_batchensemble` 후보 조사(전체 기록:
`docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md`) 중 A1(GCE)과는 독립적인
학습 레시피 개선 3가지를 추가로 확인했다. A1과 마찬가지로 **아직 라이브 미반영, 게이트 대기.**

### E1. Cosine LR 스케줄 (기존 flat lr=2e-3 대신)
- lr을 2e-4(피크)→2e-6(바닥)로 단일 사이클 cosine 감쇠. **T_max는 반드시 실제 예상 학습
  길이에 맞춰야 한다** — 처음에 T_max=100을 썼다가 실제로는 patience 때문에 epoch 30~50에서
  멈추는 걸 원문(Loshchilov & Hutter, SGDR, arXiv:1608.03983) 재확인 후 발견, T_max=60으로
  수정. Warm restart(SGDR의 R)는 근거 약함(원 논문 자체 ablation이 단일 사이클 승리) — 안 씀.
  OneCycle(Smith, arXiv:1803.09820)은 "높은 peak LR이 유리하다"는 전제가 이 프로젝트의 실측
  결과(낮은 LR이 유리)와 정반대라 배제.
- 단독 효과: lr=2e-4가 lr=2e-3 대비 정점은 거의 유지하면서 붕괴를 크게 늦춤(epoch40 시점 정확도
  0.492→0.536, bull expert 단일시드).

### E2. 옵티마이저: AdaBelief 또는 RAdam (AdamW 대신)
- **AdaBelief**(Zhuang et al., arXiv:2010.07468): `E[g²]` 대신 `E[(g-momentum)²]` 기준으로 step
  크기 조절 — 예측 밖(노이즈성) 그래디언트에 덜 반응. PyTorch 코어에 없어서 논문 Algorithm 1대로
  직접 구현(`scripts/research_eth_candidate_faithful_tabm_batchensemble_optimizer_sweep_20260816.py`
  의 `AdaBelief` 클래스, decoupled weight decay, eps=1e-16). GCE와 조합 시 가장 높은 정점을 냄.
- **RAdam**(`torch.optim.RAdam`, PyTorch 코어 내장): Adam 초반 unstable variance를 보정 — 정점은
  살짝 낮지만 훨씬 넓고 안정적인 "좋은 구간"을 만듦(정점 epoch 12→27로 지연, epoch40 정확도
  0.536→0.562).
- **주의**: `RAdam+GCE` 조합은 두 아키텍처(baseline_R_only, full_R_S_B_embed) 모두에서 크게
  실패함(격차 최대 0.158) — 이 특정 조합은 피할 것.

### E3. 체크포인트 선정 기준 — combined val_loss는 임베딩류 아키텍처에서 위험할 수 있음
- 현재 캐노니컬 스크립트의 `_fit_expert_3head`가 쓰는 `combined val_loss`(direction+quality+exit
  가중합) 기준 patience는, **임베딩이 없는 현재 아키텍처(ThreeHeadTabM)에서는 실측 격차
  0.0000~0.0008로 사실상 문제없음** — 즉 A1(GCE)을 그대로 반영해도 이 축에서는 안전하다.
  다만 향후 어떤 이유로든 입력 표현(임베딩 등)을 바꾸는 시도를 하게 되면, direction/quality/exit
  loss 궤적이 서로 달라지면서 combined val_loss가 진짜 direction_balanced_accuracy 정점을 크게
  놓칠 수 있다(격차 최대 +0.053 실측) — 그때는 **GCE 학습 loss와 분리된 plain class-balanced CE
  + Prechelt UP_4 strip 기준**(원문: Prechelt, "Early Stopping — But When?", 1998; GCE 논문
  arXiv:1805.07836 자체가 자기 loss가 아니라 validation accuracy로 체크포인트를 고름)으로 바꿔야
  한다. C3(공유 트렁크) 실험 시 특히 유의.

### 결론 — **정정(2026-08-16, 재확인 완료): 4종 묶음도 기존 레시피를 못 이긴다**

`baseline_R_only+AdaBelief+GCE+cosine+Prechelt`(NEW) 조합을 3개 expert × 5개 무작위시드로
재현했고, 이어서 **같은 시드·같은 프로토콜로 기존 레시피(OLD: AdamW+순정CE+flat lr=2e-3+
patience=8)도 재현해서 직접 짝비교**했다:

| Expert | OLD(기존 레시피) | NEW(A1+E1+E2+E3 4종 묶음) | 격차 |
|---|---:|---:|---:|
| bull | **0.5657 ± 0.0098** | 0.5534 ± 0.0230 | OLD가 +0.0123 앞섬 |
| bear | **0.5623 ± 0.0049** | 0.5570 ± 0.0044 | OLD가 +0.0053 앞섬 |
| chop | **0.5767 ± 0.0041** | 0.5617 ± 0.0191 | OLD가 +0.0150 앞섬 |

**3개 expert 전부 OLD가 이긴다**, 2개는 표준편차도 더 작아 더 안정적이다. **위 "A1은 이제 4종
묶음이다"라는 권고는 이 재확인으로 철회한다** — GCE 캐노니컬 이식이 실패했던 것과 정확히 같은
패턴이다: 개별 요소(LR/옵티마이저/loss/선정기준)는 각자 따로 봤을 때 좋아 보였지만, 실제로 다
합쳐서 N≥5 시드로 공정 비교하면 단순한 기존 레시피를 못 이긴다. **A1(GCE) 자체도 이 4종 묶음의
구성요소이므로 함께 재보류한다** — 지금 시점에서 이 전체 학습-레시피 축(A1+E1+E2+E3)은
**라이브 반영 근거 없음**으로 정정. 경량 파이프라인(185피처) 한정 결과이며, 진짜 102피처
파이프라인은 여전히 저장소 전체 영구 제약(`eth_omega4_quality_threshold_alpha67_pipeline_
irreproducible_20260815`)으로 막혀있다는 caveat도 그대로 유지.

## 실행 순서 제안

1. B1·B2 (저비용 실측, 오늘 내로 가능)
2. A1 (GCE 반영, 이미 검증됨)
3. C1·C2 (B 결과에 따라 cheap_gate → 본실험)
4. C3 (공유트렁크, N-HiTS/ModernTCN 서버작업 끝난 뒤 GPU 순서 확보)
5. C4·C5 (여유 있을 때)
