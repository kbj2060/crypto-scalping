# ETH DC154(154피쳐 엔지니어링셋) tabular 트랜스포머 축 — CLOSED (2026-08-22)

## 배경

"3라벨 TabM 축이 chance-level인 건 DL 테크닉 부재 때문 아니냐"는 가설을 검증하기 위해,
[[eth_dc_engineered154_feature_set_20260820]] 154피쳐 엔지니어링셋에 FT-Transformer식(Gorishniy
et al. 2021 feature tokenization) tabular 트랜스포머를 적용해 이진 방향 타겟으로 재시도한 축.
같은 세션에서 진행한 raw L2/OFI 축([[eth_candidate_lob_ofi_pipeline_smoke_test_20260822]])과
아키텍처(TLOB 블록)는 공유하지만 입력 데이터가 다른 별개 실험.

## 아키텍처

`scripts/eth_candidate_dc154feat_tabular_transformer_smoke_test_20260822.py`:
- `FeatureTokenizer`: 피쳐별 affine 임베딩 + CLS 토큰
- `TabularBlockBody`: TLOB 블록에서 temporal attention 제거(표 데이터엔 시간축 없음) +
  StochasticDepth
- `DC154TabularTransformer`: 위 둘을 쌓은 본체

Split은 일리아스 계약 유일 규약(TRAIN 2024-01~2026-03-31, VAL 2026Q2, OOS 2026-07-01~)을
그대로 따름.

## 속도 디버깅 여정 (완료)

1. CPU에서 batch=256으로 30분+ "멈춘 것처럼" 보임 → `ps`로 482% CPU 확인해 행 아님 확인 →
   batch=4096으로 해결(262k행 TRAIN에 배치가 너무 작았음).
2. GPU 전환 후에도 3epoch에 508.7s — `.item()` sync 오버헤드를 의심해 `FastStochasticBlock`
   (skip 로직을 `.item()` 게이팅 대신 `torch.where`로 교체)으로 수정 → 재측정 537.5s, **개선
   없음**(가설 기각, 문서에 정직하게 기록).
3. CPU-vs-GPU 재대조(CPU 1332.5s vs GPU 299.8s, 3epoch)로 "GPU가 오히려 느린 것 아니냐" 가설도
   기각.
4. raw 연산 마이크로벤치마크(matmul/SDPA 단독 58회 호출=0.05s/0.7s)는 빠른데 실제 학습루프는
   느림 → `nn.MultiheadAttention`이 `attn_dropout>0`일 때 fused 커널 경로를 타지 않는 것이
   원인으로 특정됨 → 수동 QKV projection + `F.scaled_dot_product_attention`으로 교체 →
   **165.4s/3epoch(508s 대비 약 3배 개선)** — 이게 실제 근본원인 수정이었음.

이 수정 이후에도 Optuna 탐색은 trial마다 극단적으로 불균일했다: trial 0 완료까지 40분,
trial 1 49분인데, trial 2는 **3시간 37분+ 동안 끝나지 않았음**(2026-08-22 16:51 시작 →
20:28 시점까지도 미완료, 사용자 지시로 이 시점에 강제 종료). 같은 GPU(RTX 3070 Ti 1장)에서
TabM 계열 작업(`regime_expert_full`, `qdir_optuna`)이 동시에 돌고 있던 시간대와 겹쳐 GPU
경합이 유력한 원인으로 보이나, 추가 조사 없이 이 시점에 축을 닫아 확정 원인규명은 하지 않았다.

## 결과

| 실행 | val_bce | 이론하한(0.6927) 대비 | 비고 |
|---|---|---|---|
| 기본 HP 단일런 | 0.6920 | −0.0007 | val_acc=0.517, best@epoch16, 이후 과적합 재상승(~epoch79에 0.696~0.697) |
| Optuna trial 0 | 0.6918 | −0.0009 | n_blocks=3, d_token=8, batch=8192 |
| Optuna trial 1 | 0.6925 | −0.0002 | n_blocks=4, d_token=16, batch=4096 |

Optuna는 20trial 목표 중 **2trial만 완료된 채 중단**됐다(trial 2가 3시간 37분+ 끝나지 않아
"너무 오래 걸려서 안 되겠다"는 판단으로 종료, `dc154_optuna3` 서버 잡 stop).

## 판정

완료된 3개 실행(단일런+trial0+trial1) 전부 이론하한과 사실상 구분 안 되는 범위(차이
0.0002~0.0009)에 몰려 있다 — HP를 바꿔도 결과가 흩어지지 않고 한 점에 수렴한다는 것 자체가
탐색공간 안에 유의미한 신호가 없다는 정황증거다. trial을 더 기다린다고 이 결론이 뒤집힐
근거는 약하다: 같은 결론이 이미 정보이론적으로 확인돼 있다(154피쳐 개별모델 BCE가 절편전용
이론하한과 동일, [[eth_label_fusion_combined_model_research_20260821]]).

속도 문제(근본원인 규명+3배 개선까지는 성공)와 별개로, 아키텍처를 TabM→DeepLOB→FT-Transformer로
바꿔가며 시도해도 154피쳐 데이터셋 자체가 병목이라는 이 세션 전체의 결론([[eth_dc_engineered154_
feature_set_20260820]])이 다시 한번 확인된다.

**CLOSED.** 재오픈하려면 154피쳐를 대체할 질적으로 다른 정보원(raw LOB — 이미 별도 축에서
동일 결론, 청산 tail-risk 등)이 필요하다는 기존 결론이 그대로 유지된다. Optuna 미완료 잔여
18trial을 재개할 경우를 대비해 study는 `tmp/dc154_ilias_split_20260822/optuna_study.db`
(study_name=`dc154_tabular_transformer_20260822_v2`)에 그대로 남아 있다.
