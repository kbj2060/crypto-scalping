# Omega4.6.1-RegimeGBM Rebuild — 데이터 및 리소스 관리 (2026-09-09)

계약: `docs/model_contracts/omega461_regimegbm_rebuild_contract.md`

한 리소스를 새로 만지거나 상태가 바뀌면 **같은 턴에** 이 표의 행을 추가/갱신한다.
상태 값: `active` / `blocked` / `closed-negative` / `미검증`.

## 라벨/예측 데이터

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| 캐노니컬 피쳐 2024 | `data/splits/year_oos/training_features_2024.csv` | 2024 | 부모 학습 확장분 | active | 234MB |
| 캐노니컬 피쳐 2025 | `data/splits/year_oos/training_features_2025.csv` | 2025 | 부모 TRAIN/VAL | active | 234MB. 전신 부모가 실제로 쓴 프레임 |
| 캐노니컬 피쳐 2026 | `data/splits/year_oos/training_features_2026_rebuilt.csv` | 2026-01-01~08-30 | OOS | active | 145MB. `ou_halflife`/`kel`/`evt_excess_z`/`btc_corr_60`/`dual_momentum`가 alpha6/7 vintage와 어긋남(ou_halflife corr **-0.03**) |
| wide24 레짐 사이드카 (전신) | `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_{2024,2025,2026_rebuilt}_regime3_current_sensitive_hmm_wide24.csv` | 2024~2026-08 | **전신** 레짐 6컬럼 | active(비교 기준선) | 이 라인이 교체하려는 대상. 삭제·수정 금지 — 전신이 라이브에서 사용 중 |
| **레짐 사이드카 (채택)** | `data/ensemble/supervised/omega461_balgbm_cut2509_20260909/training_features_{2024,2025,2026_rebuilt}_regime3_balgbm_cut2509_sidecar.csv` | 2024~2026-08 | **이 라인의 레짐 6컬럼** | **active(채택)** | balancedish 라벨 + HGB. 컷오프 ≤2025-09-30, TRAIN 창은 purged 5-fold OOF. 모델 `tmp/omega461_regimegbm_rebuild_20260909/regime_balgbm_cut2509_model.joblib` |
| 레짐 사이드카 (s12k3) | `data/ensemble/supervised/omega461_regimegbm_cut2509_20260909/...regime3_s12k3_cut2509_sidecar.csv` | 2024~2026-08 | 대조 arm | **closed-negative** | OOS h288 변동성비 0.924 CI[0.897,0.967] — chop 태그 역전이 유의. 2026-09-09 종료 |
| 전신 부모 예측 | `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_*/{train,validation,oos}_predictions_q{050,075}.csv` | TRAIN 78,510 / VAL 26,490 / OOS 16,832 | 비교 기준선 | active | `validation`은 `_oof_` 접두사, `oos`는 없음 — 두 네이밍 모두 흡수해야 함 |

## 모델 아티팩트

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| 전신 zig075 번들 | `tmp/causal_regen_20260516/..._zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt` | — | 전신 부모(비교) | active(동결) | 전문가당 103,992 파라미터, `epochs_ran=2`, `base_cols` 102 / `pos_cols` 13 |
| 전신 h48qual 번들 | `tmp/causal_regen_20260516/..._zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt` | — | 전신 부모(비교) | active(동결) | 동상 |
| 전신 리스크 사이드카 ×2 | `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_{h48qual_q050,zig075_q075}_precomputed_20260630/risk_sidecar.pkl` | — | 사이징(비교) | active(동결) | `model_kind=hgb`, `side_split_model=True` → long/short 각 1개씩 총 2개. 29피쳐 **전부 부모출력+결정층값**, 원시 시장피쳐 0개 |
| wide24 HMM | `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/regime3_current_sensitive_hmm_wide24_2024.joblib` | 학습 2024 | 전신 레짐(비교) | active(동결) | 12-state `GaussianStateModel` + RobustScaler + `state_class_matrix (12,3)`. 검증 bal_acc 0.7638 |
| **S12_K3 레짐 GBM** | `tmp/eth_regime_s12k3_20260902/model.joblib` | **학습 2024-01-01~2026-06-30** | 이 라인의 레짐 원천 | **blocked** | 🔴 **학습구간이 이 라인의 VAL(2025-10~12)·OOS(2026-01~02)를 전부 포함 → 그대로 쓰면 미래참조.** Phase 0에서 컷오프 재학습 필요. HGB(depth10/400iter/lr0.04), 136피쳐, OOS bal_acc 0.8550 |
| S12_K3 트레이너 | `scripts/train_eth_regime_s12k3_20260902.py` | — | 컷오프 재학습에 재사용 | 미검증 | 재구현하지 말고 이 스크립트를 컷오프 인자만 바꿔 재사용할 것 |
| S12_K3 라이브 스코어러 | `scripts/live_regime_gbm3_signal_20260826.py` | — | offline↔live 파리티 기준 | active | `MODEL_PATH` 한 줄이 아티팩트를 가리킴. `_with_raw_state12()` 호출이 8개 state7_*/state12_* 컬럼을 만든다 — **빠뜨리면 median 대체로 조용히 틀린다(2026-08-26 실제 버그)** |

## 재사용할 코드 (재구현 금지)

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| greedy 리플레이 엔진 | `scripts/replay_omega4_6_1_greedy_router_20260706.py::greedy_replay` | Phase 1 A/B, Phase 3 평가 | active | 단일계좌 greedy 우선순위 라우팅. `prepare_component()`로 컴포넌트 구성 |
| 프레임 로더 | `scripts/retest_omega4_6_1_extended_oos_20260706.py::load_frame_current` | 프레임+오버레이 병합 | active | 2026 전용 하드코딩 — 2025용은 별도 필요 |
| ATR TP/SL | `scripts/eval_omega4_1_atr_safety_sltp_20260622.py::_atr_pct` + `omega4_6_1_live.py::_ComponentConfig` | 배리어 | active | `atr_window=192, tp_mult=12, sl_mult=6, min_tp=.075, min_sl=.040` — ETH 실제 ATR은 대부분 floor 미만이라 사실상 7.5%/4.0% 고정 |
| pin된 부모 트레이너 | `scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py` | Phase 2 재학습 | active | ⚠️ pin 없이 원 트레이너를 돌리면 피쳐 자동탐지가 102→**172**로 오염된다(문서화된 함정) |
| 아키텍처 워크벤치 | `pipeline/architecture_workbench.py` | 계약 검증·preflight·피쳐 분석 | active | `validate`가 Seed-Diversity(N≥5, 비클러스터)·sizing·fresh-forward를 **코드로 강제** |
| 선택 통계 | `core/selection_stats.py` | DSR/PSR/PBO-CSCV/falsification_audit | active | `falsification_audit`는 `n_periods>=10` 하드 요건 |

## 인프라

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| 서버 GPU | `server` (192.168.1.89), RTX 3070 Ti 8GB | 학습/추론 | active | 라이브 스택과 **공유** — 실측 7,086/8,192 MiB 사용 중. 대시보드 타임아웃 유발 전례 있음 |
| dev/서버 핸드오프 | `scripts/ops/handoff.sh` | 작업 분산 | active | `launch <host> <job> --sync <path> -- <cmd>`. ⚠️`push`는 git을 거치지 않아 서버에 미커밋 서빙코드를 남긴다 — 머지 전 `check_deploy_drift.sh` 필수 |
| TabPFN / TabICL | `quant_ai` env, tabpfn 8.5.0 / tabicl 2.2.0 | 대체 모델 실험 | active | 로컬은 CUDA·가중치·토큰 모두 없음 → **서버에서만 실행 가능**. 컨텍스트 상한 관례 18,000행 |

## 미검증 후보 / 보류

- **레짐 GBM 컷오프 재학습안 (a)/(b)/(c)** — 계약의 "최우선 미해결 이슈" 참조. Phase 0 결정사항.
- **신규 레짐 6컬럼의 `confidence`/`entropy`/`margin` 유도식** — wide24 사이드카 생성 코드와 대조 필요.
- **전신 `epochs_ran=2`** — 의도인지 미확인. Phase 2 대조군 포함 여부 미결정.
- **BTC/SOL 포팅** — 이 라인은 ETH 전용으로 시작한다. BTC는 자체 레짐(S24_K3), XRP는 S96_K9로
  이미 다른 라벨을 쓰므로 포팅 시 자산별 재선정이 필요하다(ETH의 S12_K3는 BTC에서 10점 만점에 3점).
