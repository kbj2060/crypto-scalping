"""
신규 Elite 시그널 3개를 rl_training_data_full.csv에 추가하는 보조 스크립트.

추가 컬럼:
  sig_volume_confirm  : 방향성 거래량 확인 + 유동성 깊이
  sig_liquidity_trap  : EQH/EQL 스탑 헌팅 반전 감지
  sig_trend_health    : 추세 건강도 종합 점수

CSV 전체를 재생성하지 않고 컬럼 3개만 계산 후 병합합니다.
OHLCV 데이터는 training_features_5m.csv에서 가져옵니다.

사용법:
    cd /home/llewyn/crypto-scalping
    python scripts/add_new_elite_signals.py
    python scripts/add_new_elite_signals.py \\
        --rl-csv data/rl_training_data_full.csv \\
        --features-csv data/training_features_5m.csv
"""
import os, sys, logging, argparse
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, os.path.join(_ROOT_DIR, 'strategies')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features.elite import NewEliteSignalEngine  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NEW_COLS      = NewEliteSignalEngine.COLS                       # ['sig_volume_confirm', ...]
OHLCV_COLS    = ['high', 'low', 'open', 'volume']
OPTIONAL_COLS = ['taker_buy_quote', 'quote_volume']             # sig_trend_health delta factor


def add_new_signals(
    rl_csv:       str = 'data/rl_training_data_full.csv',
    features_csv: str = 'data/training_features_5m.csv',
) -> None:

    # ── 1. RL CSV 로드 ────────────────────────────────────────────────────────
    logger.info(f"RL CSV 로드: {rl_csv}")
    df_rl = pd.read_csv(rl_csv, parse_dates=['timestamp'])
    logger.info(f"  행 수: {len(df_rl):,}  컬럼 수: {len(df_rl.columns)}")

    # 이미 모든 컬럼이 있으면 스킵
    already = [c for c in NEW_COLS if c in df_rl.columns]
    missing = [c for c in NEW_COLS if c not in df_rl.columns]
    if not missing:
        logger.info(f"✅ 이미 존재: {already} — 추가 작업 없음.")
        return
    logger.info(f"  추가할 컬럼: {missing}  (기존 존재: {already or 'none'})")

    # ── 2. OHLCV 로드 (training_features_5m.csv에서 필요한 컬럼만) ───────────
    logger.info(f"OHLCV 로드: {features_csv}")
    avail = pd.read_csv(features_csv, nrows=0).columns.tolist()
    load_cols = (['timestamp'] +
                 [c for c in OHLCV_COLS    if c in avail] +
                 [c for c in OPTIONAL_COLS if c in avail])
    df_feat = pd.read_csv(features_csv, usecols=load_cols, parse_dates=['timestamp'])
    logger.info(f"  로드된 컬럼: {load_cols}")

    # ── 3. 타임스탬프 기준 left join ─────────────────────────────────────────
    df = df_rl.merge(df_feat, on='timestamp', how='left')

    # close 충돌 처리 (rl_csv의 close 우선)
    if 'close_x' in df.columns:
        df.rename(columns={'close_x': 'close'}, inplace=True)
        df.drop(columns=['close_y'], errors='ignore', inplace=True)

    # OHLCV 누락 시 close로 대체 (최소한의 fallback)
    for col in OHLCV_COLS:
        if col not in df.columns:
            logger.warning(f"  ⚠️ '{col}' 없음 → close로 대체")
            df[col] = df['close']
        else:
            df[col] = df[col].fillna(df['close'])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ── 4. 신규 시그널 벡터 계산 ─────────────────────────────────────────────
    logger.info("NewEliteSignalEngine 계산 중...")
    NewEliteSignalEngine().compute(df)

    # ── 5. 결과를 df_rl에 컬럼 추가 후 저장 ──────────────────────────────────
    for col in missing:
        if col in df.columns:
            df_rl[col] = df[col].values
        else:
            logger.warning(f"  ⚠️ '{col}' 계산 실패 — 0으로 채움")
            df_rl[col] = np.float32(0.0)

    df_rl.to_csv(rl_csv, index=False)
    logger.info(f"✅ 저장 완료: {rl_csv}")

    # 통계 요약
    for col in missing:
        s = df_rl[col]
        nz = (s != 0).sum()
        logger.info(
            f"  {col}: μ={s.mean():.4f}  σ={s.std():.4f}  "
            f"nonzero={nz:,}/{len(s):,} ({nz/len(s)*100:.1f}%)"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='신규 Elite 시그널 3개 추가')
    parser.add_argument('--rl-csv',       default='data/rl_training_data_full.csv',
                        help='대상 RL 학습 CSV')
    parser.add_argument('--features-csv', default='data/training_features_5m.csv',
                        help='OHLCV 원본 피처 CSV')
    args = parser.parse_args()
    add_new_signals(args.rl_csv, args.features_csv)
