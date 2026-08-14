# ETH Omega4.6.1 오토인코더 latent → 사이징 GBM 피처 (2026-08-13, Odyssey2 #3)

## 배경

Odyssey(1)의 오토인코더 실험(`eth_h48qual_autoencoder_latent_mi_r2_gate_20260812.md`)은 139개
원시 피처를 16차원으로 압축한 latent가 FINAL12보다 분류 지표(balanced_acc, macro_f1)를 이
세션 최대폭으로 개선했지만, 검증된 거래 시뮬레이션에서는 always-short를 못 이겼다 — "신호는
있으나 강한 레짐베타 기준선을 못 넘는다"는 미묘한 결론이었다. Odyssey2는 이 latent를
**direction/quality 재분류가 아니라 사이징 GBM 피처**로 재적용해 다른 질문("사이징 정밀도를
높이는가")을 던진다.

## 방법

Odyssey(1)과 동일 아키텍처(디노이징 오토인코더 139→64→32→16, 노이즈 std=0.05, Adam
lr=1e-3/wd=1e-5, patience=8)와 동일 원시 풀(`data/splits/year_oos/eth_features_2024_2026_analysis.csv`,
139컬럼)을 재사용하되, 사이징 GBM의 프레임 구성(`train_eval_omega4_2_risk_sidecar_20260622.
_prepare_frames` 산출 train/val/oos)에 맞춰 TRAIN 윈도우를 재정렬해 재적합했다(원본은
2024-06~2025-09 고정 윈도우, 이번엔 사이드카 자체 TRAIN 78,568행 기준 85/15 시간순 분할).
h48qual/zig075는 학습 데이터 소스가 동일해(같은 trade-candidate CSV) 오토인코더 자체는
공유(재구성 MSE: train=0.2733, val=0.2780, oos=0.3501 — OOS에서 다소 벌어짐, 일반화 경고
수준). Odyssey2 #2와 동일하게 `--risk-context-feature-dir`로 사이징 GBM 재학습.

## 결과 — 개선 없음, zig075는 뚜렷한 악화

| | baseline VAL | latent 추가 VAL | baseline OOS | latent 추가 OOS |
|---|---:|---:|---:|---:|
| h48qual PnL/MDD/거래 | +5.08% / -10.56% / 29 | +4.77% / -10.56% / 29 | +11.10% / -6.60% / 9 | +11.07% / -6.70% / 9 |
| zig075 PnL/MDD/거래 | +44.04% / -11.30% / 28 | +37.43% / -10.35% / 28 | +31.70% / -6.77% / 13 | +25.77% / -8.59% / 13 |

h48qual은 거의 무변화(소폭 악화). **zig075는 OOS에서 PnL -5.93%p, MDD -1.82%p 둘 다
뚜렷하게 악화** — VAL은 PnL 악화·MDD 개선으로 엇갈렸지만 OOS는 양쪽 다 나쁘다. 16개 추가
차원을 겨우 28건(VAL)/13건(OOS) 트레이드로 학습하는 GBM에 붙인 것이 과적합 위험을 키웠을
가능성이 가장 유력한 메커니즘 — Odyssey(1)의 앙상블 불일치 신호(거의 무변화, 과적합할
분산 자체가 없었음)와 대조적으로, latent는 진짜 분산이 있는 만큼 노이즈에 과적합할 여지도
있었다는 해석과 일치한다.

## 결론

**채택 불가.** Odyssey(1)의 "분류 지표는 개선해도 always-short를 못 이긴다"는 패턴이 사이징
맥락에서는 "GBM이 새 피처를 실제로 활용은 하지만 소수 표본에서 과적합해 오히려 해친다"는
사이징 특유의 실패 모드로 재현됐다 — 같은 결론(이 latent는 실전 가치가 없다)에 다른 경로로
도달. 원시 풀 자체를 압축하는 방향은 이걸로 direction/quality(Odyssey1)·사이징(Odyssey2)
양쪽에서 소진.

## 미해결 / 다음 단계

- 표본이 적어(28/13건) 과적합 가설을 직접 검증하려면 정규화를 강하게 건 GBM 변형이나 latent
  차원을 줄인 버전(예: 16→4)을 시도해볼 여지는 있으나, 이미 두 방향(direction/quality,
  사이징)에서 부정적이라 우선순위 낮음.
- 채택 가능한 변경 0건, 라이브 파일 미변경.

## 준수 확인

- `git diff` 기준 라이브 파일 무변경. 오토인코더는 순수 비지도 학습(라벨 미사용), 사이징
  GBM만 실질적 재학습 대상. 스크립트:
  `scripts/build_eth_autoencoder_latent_context_features_20260813.py`,
  `train_eval_omega4_2_risk_sidecar_20260622.py`(기존). 산출물:
  `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_odyssey2_{h48qual,zig075}_ae_latent_20260813/`.
