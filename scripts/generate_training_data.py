"""
generate_training_data.py
학습용 앙상블 CSV 생성 스크립트
  input : data/training_features_5m.csv   (update_features.py 출력)
  output: data/ensemble/rl_training_data_full.csv

사용법:
  python scripts/generate_training_data.py
  python scripts/generate_training_data.py --input data/training_features_5m.csv \
                                            --output data/ensemble/rl_training_data_full.csv
"""
import os, sys, logging, gc, argparse

import numpy as np
import pandas as pd
import torch

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR,
           os.path.join(_ROOT_DIR, 'ensemble'),
           os.path.join(_ROOT_DIR, 'strategies')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── 상수 ─────────────────────────────────────────────────────────────────────
MODEL_PRED = ['pred_timesfm', 'pred_chronos', 'pred_ttm', 'pred_patchtst',
              'pred_tide', 'pred_mdjd', 'pred_ridge']
MODEL_CONF = ['conf_timesfm', 'conf_chronos', 'conf_ttm', 'conf_patchtst',
              'conf_tide', 'conf_mdjd', 'conf_ridge']

ELITE_COLS = [
    'sig_whale', 'sig_orderblock', 'sig_oi_divergence', 'sig_ai_squeeze',
    # 변동성 모델 전략 신호
    'sig_garch_regime', 'sig_ou_mean_rev', 'sig_jump_rebound', 'sig_evt_tail',
]

ALPHA_7_COLS = [
    'session_us', 'hour_cos', 'cvp_poc_dist',
    'cvp_volume_imbalance', 'fvg_dist', 'breakout_strength', 'oi_change_rate',
]

REGIME_COLS = ['regime_chop', 'regime_whipsaw', 'regime_bull', 'regime_bear', 'regime_normal']

TARGET_COL = 'log_return'

SYNTHETIC_ALPHA_COLS = [
    'ofti', 'kel', 'mta_funding', 'svps', 'cada', 'mshd', 'fvci',
    'wpad', 'fdlv', 'vsdi', 'vebr', 'tlad', 'mtmb', 'fcsz',
]

VOLATILITY_MODEL_COLS = [
    'garch_vol', 'garch_vol_z',
    'ou_funding_z', 'ou_halflife',
    'jump_flag', 'jump_z',
    'evt_tail_flag', 'evt_excess_z',
]

RL_REQUIRED_COLS = (
    ['timestamp', 'close']
    + MODEL_PRED + MODEL_CONF
    + ELITE_COLS + ALPHA_7_COLS + REGIME_COLS + SYNTHETIC_ALPHA_COLS
    + VOLATILITY_MODEL_COLS
    + [TARGET_COL]
)

# ─── Ridge 선형 퀀트 워크포워드 ───────────────────────────────────────────────
_RIDGE_FEATURES = [
    'log_return', 'rsi', 'macd_hist', 'bb_width', 'volatility_z',
    'garman_klass_vol', 'last_funding_rate', 'oi_change_rate',
    'net_taker_ratio', 'taker_acceleration', 'trade_intensity',
    'big_trade_ratio', 'hurst_12', 'hurst_48', 'hurst_288',
    'realized_vol_ratio', 'amihud_illiquidity_z',
]
_RIDGE_SAVE_PATH = 'data/ridge_model.pkl'


def _ridge_walkforward(df: pd.DataFrame):
    """워크포워드 Ridge 회귀 — 룩어헤드 없는 선형 퀀트 시그널.

    walk-forward 규칙:
      - train_end 시점에서 관측된 행 0..train_end-1 로 피팅
      - rows train_end..train_end+RETRAIN-1 를 예측
      → pred[t] = r_{t+1} 추정값; 예측 시 y[t]=r_{t+1} 미사용

    Returns:
        pred  (np.ndarray float32, shape N) — 수익률 방향 강도 (raw)
        conf  (np.ndarray float32, shape N) — 시그널 신뢰도 [0, 1]
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    feat_cols = [f for f in _RIDGE_FEATURES if f in df.columns]
    N = len(df)
    if not feat_cols or N < 1000:
        logger.warning("⚠️ Ridge: 피처 부족 또는 데이터 부족 — pred_ridge=0 대체")
        return np.zeros(N, np.float32), np.full(N, 0.5, np.float32)

    X = df[feat_cols].fillna(0).values.astype(np.float32)
    # y[t] = log_return[t+1] (다음 봉 수익률) — shift(-1), 마지막 행은 0
    y = df['log_return'].shift(-1).fillna(0).values.astype(np.float32)

    pred    = np.zeros(N, dtype=np.float32)
    WARMUP  = 1000   # 최소 워밍업 행 수 (~3.5일) — Ridge는 선형 모델이므로 1000봉으로 충분
    RETRAIN = 500    # 확장 윈도우 재학습 주기 (~1.7일) — 더 잦은 재학습으로 최신성 유지

    final_ridge = final_scaler = None
    for train_end in range(WARMUP, N, RETRAIN):
        scaler = StandardScaler()
        ridge  = Ridge(alpha=0.01)
        ridge.fit(scaler.fit_transform(X[:train_end]), y[:train_end])
        pred_end = min(train_end + RETRAIN, N - 1)
        pred[train_end:pred_end] = ridge.predict(scaler.transform(X[train_end:pred_end]))
        final_ridge, final_scaler = ridge, scaler

    # 라이브 트레이딩용 최종 모델 저장
    if final_ridge is not None:
        import pickle as _pkl
        os.makedirs('data', exist_ok=True)
        with open(_RIDGE_SAVE_PATH, 'wb') as f:
            _pkl.dump({'ridge': final_ridge, 'scaler': final_scaler,
                       'features': feat_cols}, f)
        logger.info(f"💾 Ridge 모델 저장: {_RIDGE_SAVE_PATH} ({len(feat_cols)}개 피처)")

    # 신뢰도 = tanh(|pred| / rolling-MAD_500)
    pred_s = pd.Series(pred)
    mad500 = pred_s.abs().rolling(500, min_periods=10).median().replace(0, 1e-8)
    conf   = np.tanh(pred_s.abs() / mad500).fillna(0.5).clip(0, 1).values.astype(np.float32)

    logger.info(f"✅ Ridge 워크포워드 완료 | range=[{pred.min():.5f}, {pred.max():.5f}]")
    return pred, conf


# ─── 출력 억제 (NF 모델 verbose 제거) ────────────────────────────────────────
class SuppressOutput:
    def __enter__(self):
        import io
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = io.StringIO()
    def __exit__(self, *_):
        sys.stdout, sys.stderr = self._orig_stdout, self._orig_stderr


# ─── 메인 마이닝 함수 ─────────────────────────────────────────────────────────
def generate_training_csv(input_csv: str, output_csv: str):
    from strategies.elite_strategies import BaseStrategy  # type: ignore  # noqa: F401
    from strategies.elite_builder import (  # type: ignore
        EliteSignals, row_to_market_row,
        compute_synthetic_alphas, compute_regime, compute_volatility_models,
    )

    logger.info("🚀 [단계 1] 원본 피처 데이터 로드 및 전처리...")
    df = pd.read_csv(input_csv).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if 'close' in df.columns:
        if 'mtf_trend_1h' not in df.columns:
            df['mtf_trend_1h'] = df['close'].ewm(span=12, adjust=False).mean().pct_change().fillna(0.0)

    if 'smart_money_flow' in df.columns:
        # bfill 제거: 워밍업(288봉) NaN을 미래로 채우는 룩어헤드 편향 방지
        df['smf_std'] = df['smart_money_flow'].expanding(min_periods=288).std().ffill().fillna(1.0)
    else:
        df['smf_std'] = 1.0

    logger.info("🚀 [단계 1.2] 합성 알파 피처 계산 (OFTI / KEL / MTA / SVPS 등)...")
    df = compute_synthetic_alphas(df)
    logger.info("✅ 합성 알파 피처 계산 완료")

    logger.info("🚀 [단계 1.5] 적응형(Adaptive) 레짐 스캔 중...")
    df = compute_regime(df)

    logger.info("🚀 [단계 1.6] 변동성 모델 피처 계산 (GARCH / OU / Jump / EVT)...")
    df = compute_volatility_models(df)
    logger.info("✅ 변동성 모델 피처 계산 완료")

    L            = len(df)
    # Ridge WARMUP(1000) 이후부터 의미 있는 예측 가능 → 최소 시작점을 1000으로 설정
    # NF 모델도 256봉 컨텍스트가 필요하므로 max(1000, 256) = 1000
    resume_start = 1000
    abs_output   = os.path.abspath(output_csv)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    if os.path.exists(abs_output):
        try:
            existing_df = pd.read_csv(abs_output, usecols=['timestamp'])
            if not existing_df.empty:
                last_ts = str(existing_df['timestamp'].iloc[-1])
                df['timestamp_str'] = df['timestamp'].astype(str)
                match = df[df['timestamp_str'] == last_ts]
                if not match.empty:
                    resume_start = match.index[-1] + 1
                    logger.info(f"♻️ 이어하기: {last_ts} 이후부터 재개 (인덱스: {resume_start})")
                df.drop(columns=['timestamp_str'], inplace=True)
        except Exception:
            pass

    if resume_start >= L:
        return logger.info("✅ 마이닝이 이미 완료되었습니다.")

    logger.info("📐 [단계 2] Ridge 선형 퀀트 시그널 워크포워드 계산 중...")
    _ridge_pred, _ridge_conf = _ridge_walkforward(df)
    df['pred_ridge'] = _ridge_pred
    df['conf_ridge']  = _ridge_conf

    logger.info("🧠 [단계 3] 앙상블 모델 적재...")
    from ensemble.ensemble_router import (  # type: ignore
        TimesFMForecaster, ChronosForecaster, TTMForecaster,
        PatchTSTForecaster, ITransformerForecaster, NHITSForecaster, TiDEForecaster,
    )

    elite_extractor = EliteSignals()
    CHUNK_SIZE      = 1024

    nf_models_list = [m for m in ['patchtst', 'itransformer', 'nhits', 'tide']
                      if f'pred_{m}' in MODEL_PRED]
    nf_forecaster = None
    if nf_models_list:
        from neuralforecast import NeuralForecast  # type: ignore
        nf_forecaster = NeuralForecast.load('data/nf')

    ttm_model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'pred_ttm' in MODEL_PRED:
        try:
            from tsfm_public import TinyTimeMixerForPrediction  # type: ignore
            ttm_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-r1"
            ).to(device).eval()
        except Exception as e:
            logger.warning(f"TTM 로드 실패: {e}")

    fallback_models = {}
    if 'pred_timesfm' in MODEL_PRED: fallback_models['timesfm'] = TimesFMForecaster()
    if 'pred_chronos' in MODEL_PRED: fallback_models['chronos']  = ChronosForecaster()

    def get_direction(traj):
        if len(traj) < 2: return float(np.sign(np.mean(traj)))
        slope = float(np.polyfit(np.arange(len(traj)), traj, 1)[0])
        delta = traj[-1] - traj[0]
        if slope > 0 and delta > 0: return  1.0
        if slope < 0 and delta < 0: return -1.0
        return 0.0

    def get_conf(traj):
        if len(traj) < 2: return 0.5
        slope = float(np.polyfit(np.arange(len(traj)), traj, 1)[0])
        std   = float(np.std(traj)) + 1e-8
        return float(np.tanh(abs(slope) / std))

    logger.info(f"🚀 [단계 4] 하이브리드 배치 마이닝 시작 (CHUNK: {CHUNK_SIZE})")
    df_records   = df.to_dict('records')
    np_closes    = df['close'].values
    alpha_matrix = df[ALPHA_7_COLS + SYNTHETIC_ALPHA_COLS].values

    _precomputed_pred = {'pred_mdjd', 'pred_ridge'}
    _precomputed_conf = {'conf_mdjd', 'conf_ridge'}

    for chunk_start in range(resume_start, L, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, L)
        chunk_len = chunk_end - chunk_start
        logger.info(f"⏳ 청크 처리 중: [{chunk_start} ~ {chunk_end-1}] ({chunk_len}개)")
        chunk_data = []

        for i in range(chunk_start, chunk_end):
            current_row = df_records[i]
            prev_row    = df_records[i - 1] if i > 0 else current_row
            row_res = {
                'timestamp':  current_row['timestamp'],
                'close':      current_row['close'],
                'log_return': current_row['log_return'],
            }
            for col in ALPHA_7_COLS:          row_res[col] = float(current_row.get(col, 0.0))
            for col in REGIME_COLS:           row_res[col] = float(current_row.get(col, 0.0))
            for col in SYNTHETIC_ALPHA_COLS:  row_res[col] = float(current_row.get(col, 0.0))
            for col in VOLATILITY_MODEL_COLS: row_res[col] = float(current_row.get(col, 0.0))

            all_sigs = elite_extractor.compute_all(
                current=row_to_market_row(current_row),
                prev=row_to_market_row(prev_row),
                smf_std=float(current_row.get('smf_std', 1.0)),
            )
            row_res.update({k: float(v) for k, v in all_sigs.items() if k in ELITE_COLS})

            for m in MODEL_PRED:
                row_res[m] = float(current_row.get(m, 0.0)) if m in _precomputed_pred else 0.0
            for c in MODEL_CONF:
                row_res[c] = float(current_row.get(c, 0.0)) if c in _precomputed_conf else 0.5
            chunk_data.append(row_res)

        # ── TTM ──────────────────────────────────────────────────────────────
        if ttm_model is not None:
            ttm_ctx      = ttm_model.config.context_length
            needed_start = chunk_start - ttm_ctx + 1
            if needed_start < 0:
                closes_slice = np.pad(np_closes[0:chunk_end], (abs(needed_start), 0), 'edge')
            else:
                closes_slice = np_closes[needed_start:chunk_end]

            windows = np.lib.stride_tricks.sliding_window_view(closes_slice, ttm_ctx)
            means   = windows.mean(axis=1, keepdims=True)
            stds    = windows.std(axis=1,  keepdims=True) + 1e-8
            scaled  = (windows - means) / stds

            inp = torch.tensor(scaled, dtype=torch.float32).unsqueeze(-1).to(device)
            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
                out = ttm_model(past_values=inp).prediction_outputs.squeeze(-1).cpu().numpy()

            out_unscaled = out * stds + means
            for i, traj in enumerate(out_unscaled):
                traj6 = traj[:6]
                chunk_data[i]['pred_ttm'] = get_direction(traj6)
                chunk_data[i]['conf_ttm'] = get_conf(traj6)

        # ── NeuralForecast (PatchTST / iTransformer / NHITS / TiDE) ─────────
        if nf_forecaster is not None:
            nf_rows      = []
            dummy_dates  = pd.date_range(end=pd.Timestamp.now(), periods=256, freq='5min')
            nf_exog_cols = ALPHA_7_COLS + SYNTHETIC_ALPHA_COLS

            for i, end_idx in enumerate(range(chunk_start, chunk_end)):
                start_idx = end_idx - 255
                uid       = f'w_{i}'
                y_vals    = np_closes[start_idx : end_idx + 1]
                a_vals    = alpha_matrix[start_idx : end_idx + 1]
                for step_idx in range(256):
                    row_dict = {'unique_id': uid, 'ds': dummy_dates[step_idx],
                                'y': y_vals[step_idx]}
                    for a_idx, a_name in enumerate(nf_exog_cols):
                        row_dict[a_name] = a_vals[step_idx][a_idx]
                    nf_rows.append(row_dict)

            # NF 내부 batch_size=32 기준: 32의 배수가 되도록 더미 패딩
            NF_BATCH_SIZE = 32
            pad_count     = (-chunk_len) % NF_BATCH_SIZE
            if pad_count > 0:
                last_end = chunk_end - 1
                y_pad    = np_closes[last_end - 255 : last_end + 1]
                a_pad    = alpha_matrix[last_end - 255 : last_end + 1]
                for p in range(pad_count):
                    uid = f'_pad_{p}'
                    for step_idx in range(256):
                        row_dict = {'unique_id': uid, 'ds': dummy_dates[step_idx],
                                    'y': y_pad[step_idx]}
                        for a_idx, a_name in enumerate(nf_exog_cols):
                            row_dict[a_name] = a_pad[step_idx][a_idx]
                        nf_rows.append(row_dict)

            batch_df = pd.DataFrame(nf_rows)
            batch_df.fillna(0.0, inplace=True)
            with SuppressOutput():
                out_df = nf_forecaster.predict(df=batch_df)

            for i in range(chunk_len):
                uid     = f'w_{i}'
                uid_out = (out_df.loc[uid] if uid in out_df.index
                           else out_df[out_df['unique_id'] == uid])
                for m_alias in nf_models_list:
                    m_real = {'patchtst': 'PatchTST', 'itransformer': 'iTransformer',
                              'nhits': 'NHITS', 'tide': 'TiDE'}[m_alias]
                    if m_real in uid_out:
                        traj6 = uid_out[m_real].values[:6]
                        chunk_data[i][f'pred_{m_alias}'] = get_direction(traj6)
                        chunk_data[i][f'conf_{m_alias}'] = get_conf(traj6)

        # ── TimesFM / Chronos (fallback) ──────────────────────────────────────
        if fallback_models:
            for i, end_idx in enumerate(range(chunk_start, chunk_end)):
                df_slice = df.iloc[end_idx - 255 : end_idx + 1]
                for name, model in fallback_models.items():
                    if getattr(model, 'available', False):
                        try:
                            with torch.inference_mode(), \
                                 torch.autocast(device_type='cuda', dtype=torch.float16):
                                out = model.predict(df_slice, horizon=6)
                                if out and out.median is not None:
                                    chunk_data[i][f'pred_{name}'] = get_direction(out.median[-1])
                                    chunk_data[i][f'conf_{name}'] = float(out.confidence[-1].mean())
                        except Exception:
                            pass

        new_df = pd.DataFrame(chunk_data, columns=RL_REQUIRED_COLS)
        is_new = not os.path.exists(abs_output)
        new_df.to_csv(abs_output, mode='a', header=is_new, index=False)

        del chunk_data, new_df
        if nf_forecaster is not None:
            del nf_rows, batch_df, out_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"🎉 하이브리드 마이닝 완료! 파일: {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='학습용 앙상블 CSV 생성')
    parser.add_argument('--input',  default='data/training_features_5m.csv',
                        help='원본 피처 CSV 경로')
    parser.add_argument('--output', default='data/ensemble/rl_training_data_full.csv',
                        help='출력 앙상블 CSV 경로')
    args = parser.parse_args()

    generate_training_csv(args.input, args.output)
