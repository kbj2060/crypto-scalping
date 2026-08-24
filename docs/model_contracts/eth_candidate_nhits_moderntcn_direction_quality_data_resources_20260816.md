# ETH ModernTCN/N-HiTS 백본 교체 후보 — 데이터 및 리소스 관리 (2026-08-16)

이 문서는 `docs/model_contracts/eth_candidate_nhits_moderntcn_direction_quality_contract_20260816.md`에서 실제로 만지거나 검토한 모든 데이터/코드/외부 리소스를 한 곳에 모은 목록이다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것** — 나중으로 미루지 않는다. 상태 값 컨벤션: `활성`, `인프라 확인됨-미착수`, `인프라 차단`, `검증 완료 — 긍정 결과`, `검증 완료 — 부정 결과`.

## 참조 선례 (읽기 전용, 컨벤션 원본)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| direction 시퀀스 TCN 참조 스크립트 | `scripts/verify_eth_h48qual_tcn_sequence_model_20260812.py` | TRAIN 2024-06~2025-09/VAL 2025-10~12/OOS 2026-01~02 | 데이터로딩·윈도우·split·라벨 관례 원본, SEQ_COLS(8컬럼) 출처 | 활성 | 이번 실험의 window/split/feature-source 규칙을 여기서 그대로 계승 |
| TCN HP서치 참조 스크립트 | `scripts/tune_eth_h48qual_tcn_sequence_model_hpsearch_20260812.py` | 동일 | Optuna 30-trial→VAL 재평가→N=5시드 관례 원본, window={48,96,192} 탐색범위 출처 | 활성 | 5개 피처셋 변형 중 raw_lite만 재사용(나머지 4개는 그 스크립트 자신의 탐색) |
| 닫힌 아키텍처 축 리서치 | `docs/experiments/eth_odyssey_dl_rl_architecture_research_20260816.md` | - | registry 중첩 근거(VSN/Diffusion/Mamba/Transformer/TCN 전부 반증), 이 후보가 그 축과 왜 다른지의 출발점 | 활성 | 계약 문서 "Registry 중첩 근거" 절에서 인용 |
| TabM 불충실성 사건 계약 | `docs/model_contracts/eth_candidate_faithful_tabm_batchensemble_contract_20260816.md` | - | "구현이 논문과 다르면 반드시 고지" 규율의 근거, 이번 계약의 "Known limitations" 절 형식 원본 | 활성 | - |
| TabM 정칙화 연구(GCE/ELR/mixup) | `scripts/research_eth_candidate_faithful_tabm_batchensemble_combo_regularizer_20260816.py`, `..._regularizer_isolation_20260816.py` | TabM+zigzag_action, bull expert, 단일시드 40epoch | GCE q=0.7/ELR λ=3.0 β=0.7/mixup α=1.0 하이퍼파라미터·구현 패턴 원본, isolation 4-way 설계 원본 | 활성 | 이 저장소 최초 소스 — 이번 스크립트의 `_cls_loss`/`_elr_term`/mixup 로직이 여기서 이식됨 |
| 918실험 벤치마크 문헌 리뷰 | `docs/experiments/eth_literature_review_cryptogat_and_918experiments_dl_architecture_20260816.md` | - | ModernTCN/N-HiTS를 candidate로 지목한 근거 문서(arXiv:2603.16886) | 활성 | 사용자가 "시도해보라"고 지시한 근거 |

## 외부 논문/코드 (2026-08-16, 원문/공식 구현 직접 확인)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| ModernTCN 논문 | Luo & Wang, ICLR 2024 Spotlight, OpenReview `id=vpJMJerXHU` | - | 아키텍처 설계 원본 확인 | 검증 완료 | arXiv 단독 ID 미확정(OpenReview/ICLR proceedings로 확인), 문헌 리뷰는 "arXiv" 표현을 썼지만 실제로는 ICLR 게재본이 1차 출처 |
| ModernTCN 공식 구현(classification) | `github.com/luodhhh/ModernTCN` — `ModernTCN-classification/models/ModernTCN.py`, `ModernTCN_Layer.py`, `layers/RevIN.py` | - | `ModernTCNBackbone`의 stem/multi-stage downsample/ReparamLargeKernelConv/ConvFFN1·2 이식 원본 | 검증 완료 — 긍정 결과(충실 이식) | **코드 대조로 두 죽은 컴포넌트 발견**: `revin_layer`가 생성만 되고 `forward_feature()`에서 호출 안 됨, `stem_ratio`/`dw_dims` 생성자 인자가 어디서도 참조 안 됨 — 계약 문서 "Known limitations" 4·5 참고 |
| ModernTCN classification 스크립트 예시 | `ModernTCN-classification/scripts/classification.sh` | UEA 9개 데이터셋 | `dims`/`large_size`/`patch_size` 등 실전 하이퍼파라미터 스케일 참고(특히 patch_size=1/stride=1 짧은 시퀀스 사례) | 검증 완료 | 이번 구현의 window(48~192, UEA 사례보다 짧음)에 patch_size=1 기본값을 정당화하는 근거로 사용 |
| N-HiTS 논문 | Challu et al., AAAI 2023, arXiv:2201.12886 | - | 계층적 multi-rate pooling + basis expansion + doubly-residual 설계 원본 확인 | 검증 완료 — 부분적(PDF 텍스트 추출 실패, 공식 참조 구현으로 대체 확인) | `WebFetch`가 PDF 텍스트를 못 읽어(이미지/압축 스트림) 직접 수식 인용은 못 함 — 대신 neuralforecast의 충실 구현으로 메커니즘 확인 |
| N-HiTS 참조 구현(neuralforecast) | `github.com/Nixtla/neuralforecast` — `neuralforecast/models/nhits.py` (`NHITSBlock`, `_IdentityBasis`) | - | `NHiTSBackbone`의 pooling→MLP→theta→basis expansion→doubly-residual 메커니즘 이식 원본 | 검증 완료 — 긍정 결과(충실 이식, 이 저장소가 이미 신뢰하는 라이브러리) | 원 논문을 arXiv:2201.12886로 명시 인용하는 유지보수 라이브러리 — `data/nf`의 죽은 `NHITS_0.ckpt`와 같은 계열이지만 이번 구현은 그 체크포인트를 재사용하지 않고 백본을 새로 학습 |

## 데이터/라벨 소스 (기존 자산 재사용, 신규 생성 없음)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| 피처 패널 | `data/splits/year_oos/eth_features_2024_2026_analysis.csv` | 2024-06-01~2026-08-04, 228,714행 | SEQ_COLS(8컬럼) 시퀀스 입력 원본 | 활성(서버에 이미 존재, 2026-08-08 기준 513MB, git 커밋 대상 아님) | dev/server 양쪽에 이미 존재 확인(2026-08-16) — 이번 실행에서 추가 sync 불필요 |
| direction 라벨(zigzag_action) | `tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531/zigzag_action_labels_{2024,2025,2026}.csv` | 2024~2026 | direction_head 타깃 | 활성(서버에 이미 존재) | 참조 TCN 스크립트와 동일 로딩 |
| quality 라벨(h48_conservative) | `tmp/eth_h48_conservative_orig_padded_to_zigzag_timestamps_20260811/zigzag_action_labels_{2025,2026}.csv` | 2024-01~2025-12(파일명 "2025") + 2026-01~02-28(파일명 "2026") | quality_head 타깃 | 활성(서버에 이미 존재) | 컬럼명이 파일 내부에서 `zigzag_action`(패딩 스크립트의 명명 관례) — 로드 시 `h48_conservative`로 rename. 2026-02-28 이후 미존재가 이 실험 OOS 경계를 03-31→02-28로 당긴 원인 |
| quality 라벨 원본 배리어 | `tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619/{train,validation,oos}_triple_barrier_labels.csv`, 컬럼 `tb_action_h48_conservative` | - | 위 패딩 라벨의 1차 출처(읽기 전용, 이번 실행이 직접 로드하진 않음) | 확인됨(패딩 스크립트 코드 대조) | horizon=48bar/tp_mult=1.2/sl_mult=0.8/min_tp=0.006/min_sl=0.004 |

## 신규 구현 산출물 (2026-08-16)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| 학습/평가 스크립트 | `scripts/train_eval_eth_direction_quality_nhits_moderntcn_20260816.py` | TRAIN/VAL/OOS 전체 | `TabMControlBackbone`/`ModernTCNBackbone`/`NHiTSBackbone` + 체크리스트 통합 학습루프 + isolation/hpsearch/final 스테이지 | 활성 — 로컬 sanity 통과, 서버 전체 파이프라인 실행 중 | `--stage {sanity,isolation,hpsearch,final,all,tabm_control}` |
| 로컬 sanity 로그 | (일회성 실행, 저장 안 됨 — 재현 시 `--stage sanity --device cpu` 재실행) | 2025-06~09 서브샘플, 1시드 2epoch | 두 아키텍처 모두 shape/NaN/crash 없이 학습됨을 확인 | 검증 완료 — 긍정 결과 | ModernTCN 269s/2epoch(CPU), N-HiTS 4s/2epoch(CPU) — CPU 기준, GPU에서는 훨씬 빠름(서버 첫 시드 실측 213s/**12**epoch, 전체 TRAIN 구간) |
| 서버 handoff 잡 | `tmp/handoff_jobs/eth_nhits_moderntcn_direction_quality/`(서버, `llewyn@<server>:2222:/home/llewyn/crypto-scalping`) | 전체 파이프라인(`--stage all`) | isolation→hpsearch→final(ModernTCN, N-HiTS) + tabm_control | **실행 중**(2026-08-16, pid는 `handoff.sh status server eth_nhits_moderntcn_direction_quality`로 재확인) | 첫 seed(none, moderntcn) es_loss=1.9034 es_bacc_peak=0.4662 (213s) 확인 후 계속 관찰 필요. 스테이지별 리포트: `tmp/eth_candidate_nhits_moderntcn_direction_quality_20260816/{isolation,hpsearch,final}_{moderntcn,nhits}.json`, `final_tabm_control.json` (서버 로컬, pull 필요) |
| 서버 GPU 확인 | `llewyn@<server>` — RTX 3070 Ti, ~7.4GB/8.6GB 여유(2026-08-16 launch 직전 실측) | - | 실행 가능성 확인 | 확인됨 | 다른 GPU 점유 작업 없음(`_watch_deploy*`류 4개만 실행 중, GPU 미사용 모니터링 잡) |

## 미검증 후보 / 보류

- Optuna HP서치·final N=5시드·tabm_control 결과 전부 — 서버 작업 완료 후 `docs/experiments/eth_candidate_nhits_moderntcn_direction_quality_20260816.md`에 전체 과정, 이 계약 문서 상태 표에 결과 요약 기록 예정.
- VAL/OOS PnL 비교(always_long/always_short 대비) — 위와 동일, `--stage final` 완료 후.
