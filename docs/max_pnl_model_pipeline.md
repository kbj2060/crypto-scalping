# Max PnL Model Pipeline

> Current rank-1 baseline update, 2026-05-06 KST:
> The current highest-return main candidate is no longer the DSAC model described below. The canonical current rank-1 model is `current_top_muzero_az_stage2_azexit_2026`: MuZero Entry Planner -> AZ Risk Overlay -> Stage2 MuZero Sleeve Overlay (`g0.55 / p0.00 / d1 / score_floor0.12`) -> AZ Exit Governor (`threshold 0.45`) -> Execution Accounting. Verified 2026 OOS: `+752.65%`, MDD `-18.76%`, trades `353`, trades/day `6.02`, cost 2x `+279.36%`, cost 3x `+75.84%`. `Stage3 exit arbiter` and `Stage4 regime overlay` are excluded. See [2026-05-06_current_top_muzero_az_stage2_azexit.md](model_contracts/2026-05-06_current_top_muzero_az_stage2_azexit.md).

이 문서는 현재 워크스페이스에서 확인된 **최대 수익률 메인 DSAC 모델**의 전체 파이프라인을 정리한 것이다.

## 1. 최종 우승 모델

- 체크포인트: [best_dsac_agents.pth](/home/kbj20/crypto-scalping/data/ensemble/ckpt/best_dsac_agents.pth)
- 체크포인트 메타:
  - `epoch = 130`
  - `best_pnl = 448.8811959052901`
  - `state_dim = 29`
- 학습 설정: [dsac_train_config_latest.json](/home/kbj20/crypto-scalping/data/ensemble/ckpt/dsac_train_config_latest.json)
- 2026 OOS 평가: [eval_2026_oos_main.json](/home/kbj20/crypto-scalping/data/ensemble/reports/eval_2026_oos_main.json)
  - `training_env pnl = +349.72%`
  - `closed_loop pnl = +325.30%`
  - `closed_loop trades = 743`
  - `closed_loop sharpe = 18.197`
  - `closed_loop mdd = -4.04%`

## 2. 핵심 결론

- 메인 모델은 **29-state DSAC**다.
- 엔트리 프라이스 `price` 피처는 **메인 state에 직접 들어가지 않는다**.
- 하지만 메인 학습 CSV는 **최신 M7 앙상블 출력이 반영된 데이터**다.
- 즉 이 모델은:
  - `최신 M7 데이터`
  - `원본 29-state DSAC 구조`
  - `기존 강한 학습 파라미터`
  조합으로 만들어진 메인 라인이다.

## 3. 파이프라인 전체 흐름

### Step 1. 원천 피처 프레임 준비

- 원천 features 기본 경로:
  - [training_features_5m.csv](/home/kbj20/crypto-scalping/data/training_features_5m.csv)
- 통합 빌더:
  - [build_rl_dataset.py](/home/kbj20/crypto-scalping/pipeline/build_rl_dataset.py)

이 단계에서:
- 피처 CSV를 읽고
- RL용 베이스 CSV를 만들 수 있으며
- 연도별로 학습/평가 구간을 나눈다.

### Step 2. 연도 분리

- 지도/비지도 M7 앙상블 학습 연도: `2024`
- DSAC 재학습 연도: `2025`

생성 아티팩트 예:
- [training_features_2024.csv](/home/kbj20/crypto-scalping/data/splits/year_oos/training_features_2024.csv)
- [rl_base_2024.csv](/home/kbj20/crypto-scalping/data/splits/year_oos/rl_base_2024.csv)
- [training_features_2025.csv](/home/kbj20/crypto-scalping/data/splits/year_oos/training_features_2025.csv)
- [rl_base_2025.csv](/home/kbj20/crypto-scalping/data/splits/year_oos/rl_base_2025.csv)

### Step 3. 2024 데이터로 M7 앙상블 학습

관련 러너:
- [train_all_ensemble.py](/home/kbj20/crypto-scalping/scripts/train_all_ensemble.py)
- 또는 [train_all_ensemble_optuna.py](/home/kbj20/crypto-scalping/scripts/train_all_ensemble_optuna.py)

실제 M7 앙상블 추론기:
- [seven_model_ensemble.py](/home/kbj20/crypto-scalping/ensemble/seven_model_ensemble.py)

여기서 나오는 대표 출력:
- `m7_trend_xgb_*`
- `m7_mtl_*`
- `m7_quant_*`
- `m7_q10/q50/q90/qwidth`
- `m7_quality_pred`
- `m7_hold_pred`
- `m7_entry_*`
- `m7_tp/sl_*`
- `m7_gmm_*`
- `m7_iso_*`
- `m7_vae_*`
- `m7_expected_ret`
- `m7_tail_risk`

### Step 4. 2025 RL 데이터에 M7 출력 주입

주입 스크립트:
- [augment_m7_dataset.py](/home/kbj20/crypto-scalping/pipeline/augment_m7_dataset.py)

이 단계에서 하는 일:
- `rl_base_2025.csv` 로드
- `training_features_2025.csv`를 timestamp 기준으로 merge
- 누락된 파생 피처를 보강
- elite/high-order state 피처 재계산
- `SevenModelEnsemble.predict_batch()`로 M7 출력 생성
- deprecated M7 컬럼 제거
- RL keep-set 기준 passthrough 컬럼을 합쳐 최종 DSAC 학습 CSV 생성

메인 학습 CSV:
- [rl_training_2025_m7.csv](/home/kbj20/crypto-scalping/data/splits/year_oos/rl_training_2025_m7.csv)

### Step 5. DSAC 학습

메인 학습기:
- [train_rl_dsac_agent.py](/home/kbj20/crypto-scalping/ensemble/train_rl_dsac_agent.py)

핵심:
- `DSAC_STATE_DIM = 29`
- `include_entry_price = False`
- M7 core 필수 컬럼만 검증
- compact state 29개로 학습
- best checkpoint를 `data/ensemble/ckpt/`에 저장

### Step 6. 2026 진짜 OOS 평가

평가 스크립트:
- [eval_2026_oos.py](/home/kbj20/crypto-scalping/scripts/eval_2026_oos.py)

평가 데이터:
- [rl_training_2026_m7_supervised_redesign_clean.csv](/home/kbj20/crypto-scalping/data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv)
- [training_features_2026_rebuilt.csv](/home/kbj20/crypto-scalping/data/splits/year_oos/training_features_2026_rebuilt.csv)

## 4. 메인 학습 설정

설정 파일: [dsac_train_config_latest.json](/home/kbj20/crypto-scalping/data/ensemble/ckpt/dsac_train_config_latest.json)

- `csv_path = data/splits/year_oos/rl_training_2025_m7.csv`
- `train_ratio = 0.8`
- `episodes = 500`
- `fresh_start = true`
- `val_interval = 10`
- `gamma = 0.99`
- `adaptive_pessimism = false`
- `pessimism_weight_min = 0.55`
- `pessimism_weight_max = 0.75`
- `critic_var_weight = false`
- `side_balance_lambda = 0.12`
- `anti_flat_lambda = 0.08`
- `state_dim = 29`

## 5. 메인 학습 CSV 전체 컬럼

파일:
- [rl_training_2025_m7.csv](/home/kbj20/crypto-scalping/data/splits/year_oos/rl_training_2025_m7.csv)

총 컬럼 수:
- `117`

### 5.1 Market / OHLCV / 거래소 기본

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `quote_volume`
- `trades`
- `taker_buy_base`
- `taker_buy_quote`
- `sum_open_interest_value`
- `sum_toptrader_long_short_ratio`
- `count_long_short_ratio`
- `last_funding_rate`
- `close_btc`
- `volume_btc`
- `quote_volume_btc`

### 5.2 Alpha / Microstructure / Volatility / Market Context

- `smart_money_flow`
- `oi_change_rate`
- `taker_acceleration`
- `log_return`
- `mtf_trend_1h`
- `mtf_trend_4h`
- `rogers_satchell_vol`
- `amihud_illiquidity_z`
- `ofti`
- `kel`
- `pred_patchtst`
- `conf_patchtst`
- `whale_retail_ratio`
- `squeeze_power`
- `net_taker_ratio`
- `trade_intensity`
- `big_trade_ratio`
- `volatility_z`
- `rsi`
- `wick_ratio`
- `garman_klass_vol`
- `btc_corr_60`
- `eth_btc_ratio_change`
- `fvg_dist`
- `hour_cos`
- `cvp_poc_dist`
- `cvp_cluster_position`
- `cvp_volume_imbalance`
- `breakout_strength`
- `funding_roc_288`
- `long_squeeze_risk`
- `funding_price_divergence`
- `ofi_acceleration`
- `whale_conviction`
- `funding_abs`
- `funding_pressure`
- `hurst_48`
- `cvp_vah_val_width`

### 5.3 Regime / High-order / Event / Elite State

- `regime_bull`
- `regime_bear`
- `regime_chop`
- `regime_whipsaw`
- `regime_normal`
- `garch_vol_z`
- `ou_funding_z`
- `ou_halflife`
- `jump_flag`
- `jump_z`
- `evt_tail_flag`
- `evt_excess_z`
- `sig_volume_confirm`
- `sig_liquidity_trap`
- `sig_trend_health`
- `sig_whale`
- `sig_oi_divergence`
- `sig_ai_squeeze`
- `regime_persistence`
- `cross_scale_curvature`
- `liquidity_vacuum`
- `crowding_pressure`
- `execution_quality`

### 5.4 M7 Ensemble Output

- `m7_trend_xgb_dn`
- `m7_trend_xgb_up`
- `m7_mtl_dn`
- `m7_mtl_up`
- `m7_quant_dn`
- `m7_quant_up`
- `m7_confidence`
- `m7_action`
- `m7_size`
- `m7_q10`
- `m7_q50`
- `m7_q90`
- `m7_qwidth`
- `m7_quality_pred`
- `m7_hold_pred`
- `m7_target_hold`
- `m7_entry_long_offset`
- `m7_entry_short_offset`
- `m7_entry_long_price`
- `m7_entry_short_price`
- `m7_tp_offset`
- `m7_sl_offset`
- `m7_tp_price`
- `m7_sl_price`
- `m7_gmm_cluster`
- `m7_gmm_conf`
- `m7_gmm_vol_rank`
- `m7_iso_pred`
- `m7_iso_score`
- `m7_vae_error`
- `m7_iso_anom`
- `m7_vae_anom`
- `m7_gate_block`
- `m7_expected_ret`
- `m7_tail_risk`
- `m7_composite_score`

## 6. DSAC가 실제로 state에 쓰는 29개

메인 trainer는 117개 컬럼을 모두 직접 state로 쓰지 않는다.  
실제 compact state는 [train_rl_dsac_agent.py](/home/kbj20/crypto-scalping/ensemble/train_rl_dsac_agent.py)의 `_build_state()`에서 29개로 만들어진다.

### Block A. Market Prediction Meta (17)

1. `up * _M7_DIR_SCALE`
2. `dn * _M7_DIR_SCALE`
3. `fl * _M7_DIR_SCALE`
4. `trend_entropy * _M7_DIR_SCALE`
5. `quality_norm`
6. `hold_norm`
7. `q_mid_norm`
8. `q_uncertainty_norm`
9. `q_skew`
10. `gmm_cluster_norm`
11. `gmm_conf`
12. `vol_rank`
13. `anomaly_score`
14. `tp_offset_norm`
15. `sl_offset_norm`
16. `mtf_1h_norm`
17. `mtf_4h_norm`

주 원천 컬럼:
- `m7_trend_xgb_dn`
- `m7_trend_xgb_up`
- `m7_quality_pred`
- `m7_hold_pred`
- `m7_q10`
- `m7_q50`
- `m7_q90`
- `m7_qwidth`
- `m7_gmm_cluster`
- `m7_gmm_conf`
- `m7_gmm_vol_rank`
- `m7_iso_score`
- `m7_iso_anom`
- `m7_vae_error`
- `m7_vae_anom`
- `m7_tp_offset`
- `m7_sl_offset`
- `mtf_trend_1h`
- `mtf_trend_4h`
- `jump_z`
- `evt_excess_z`
- `garch_vol_z`
- `jump_flag`
- `evt_tail_flag`

### Block B. Immediate Tick Context (6)

18. `spread_norm`
19. `rs_vol_norm`
20. `micro5_norm`
21. `amihud_norm`
22. `smart_flow_norm`
23. `taker_accel_norm`

주 원천 컬럼:
- 내부 파생 `spread`
- 내부 파생 `micro_vol5`
- `rogers_satchell_vol`
- `amihud_illiquidity_z`
- `smart_money_flow`
- `taker_acceleration`

### Block C. Agent Private State (6)

24. `current_position`
25. `unrealized_norm`
26. `time_in_trade_norm`
27. `hold_vs_expected`
28. `margin_usage`
29. `drawdown_norm`

주 원천:
- 환경 내부 포지션 상태
- 현재 미실현손익
- hold count
- `m7_hold_pred`
- 내부 leverage / drawdown 상태

## 7. 메인 라인에서 실제로 중요한 M7 컬럼

메인 29-state가 직접 영향을 크게 받는 컬럼은 주로 아래다.

- `m7_trend_xgb_dn`
- `m7_trend_xgb_up`
- `m7_quality_pred`
- `m7_hold_pred`
- `m7_q10`
- `m7_q50`
- `m7_q90`
- `m7_qwidth`
- `m7_gmm_cluster`
- `m7_gmm_conf`
- `m7_gmm_vol_rank`
- `m7_iso_score`
- `m7_iso_anom`
- `m7_vae_error`
- `m7_vae_anom`
- `m7_tp_offset`
- `m7_sl_offset`

반면 아래는 CSV에는 있지만 메인 29-state에 직접 안 들어간다.

- `m7_entry_long_price`
- `m7_entry_short_price`
- `m7_entry_long_offset`
- `m7_entry_short_offset`
- `m7_target_hold`
- `m7_tp_price`
- `m7_sl_price`
- `m7_mtl_*`
- `m7_quant_*`
- `m7_gate_block`
- `m7_expected_ret`
- `m7_tail_risk`
- `m7_composite_score`

## 8. 한 줄 요약

현재 최대 수익률 메인 모델은  
**“최신 M7 앙상블 출력이 반영된 2025 학습 CSV + 29-state DSAC + 기존 강한 하이퍼파라미터 + 2026 OOS closed-loop +221.68%”**  
조합이다.
