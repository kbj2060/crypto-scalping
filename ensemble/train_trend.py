"""
Brain B: TrendContextBrain — 다차원 하이브리드 트렌드 맥락 감지기 (JP Morgan Quant Ed.)
================================================================================
아키텍처 변경사항:
    - 데이터 병합: 5m OHLCV/Microstructure + 1h AI/Signal 예측 병합 (4h 리샘플링)
    - 입력 피처 차원: 16차원 (BASE_FEAT_DIM = 16)
    - 모델: Transformer Encoder + Pre-LN + GELU + Regime Aware Pooling
"""

import os, logging, math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터 (High-Frequency 융합형)
# ───────────────────────────────────────────────────────────────────────────
WINDOW        = 48      
D_MODEL       = 64      # (수정) 128 -> 64: 암기력 제한
N_HEADS       = 4       # (수정) 8 -> 4: 복잡도 축소
N_LAYERS      = 2       # (수정) 4 -> 2: 깊이 축소 (오버피팅 방지)
D_FF          = 128     # (수정) 256 -> 128
DROPOUT       = 0.4     # (수정) 0.15 -> 0.4: 뉴런을 무작위로 40%씩 끄면서 학습 (암기력 파괴)
BASE_FEAT_DIM = 16      
LABEL_HORIZON_MIN = 2   
LABEL_HORIZON_MAX = 6
FLAT_THRESH   = 0.008

@dataclass
class TrendSignal:
    trend_dir: int
    strength: float
    rev_prob: float
    probs: Tuple[float, float, float]

    @property
    def is_up(self) -> bool: return self.trend_dir == 2
    @property
    def is_down(self) -> bool: return self.trend_dir == 0
    @property
    def is_flat(self) -> bool: return self.trend_dir == 1

    def to_arbiter_dict(self) -> dict:
        return {'trend_dir': self.trend_dir, 'strength': self.strength,
                'rev_prob': self.rev_prob, 'probs': list(self.probs)}

# ───────────────────────────────────────────────────────────────────────────
# 피처 엔지니어링 (16-Dimension Hybrid Vector)
# ───────────────────────────────────────────────────────────────────────────
class HybridFeatureExtractor:
    """
    [Price Structure]
      0: log_return, 1: body_ratio, 2: upper_shadow, 3: lower_shadow
      4: vol_rel, 5: hl_range, 6: close_pos
    [Microstructure & Flow - 5m 기반]
      7: smart_money_flow_mean, 8: squeeze_power_max, 9: whale_retail_ratio_mean
      10: chop_index_last, 11: rsi_last
    [Quant & AI Signals - 1h 기반]
      12: pred_mdjd_last, 13: conf_mdjd_last, 14: garch_vol_z_last, 15: ou_funding_z_last
    """
    RET_SCALE = 20.0
    EPS = 1e-8

    def __init__(self):
        self.feat_dim = BASE_FEAT_DIM

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        T = len(df)
        feats = np.zeros((T, self.feat_dim), dtype=np.float32)
        
        c = df['close'].values.astype(np.float64)
        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64)
        vol_ma = pd.Series(v).rolling(20, min_periods=1).mean().values
        
        logret = np.zeros(T)
        logret[1:] = np.log(np.maximum(c[1:], self.EPS) / np.maximum(c[:-1], self.EPS))

        for i in range(T):
            hl = max(h[i] - l[i], self.EPS)
            feats[i, 0] = float(np.tanh(logret[i] * self.RET_SCALE))
            feats[i, 1] = float(np.clip((c[i] - o[i]) / max(o[i], self.EPS), -0.15, 0.15) / 0.15)
            feats[i, 2] = float(np.clip((h[i] - max(o[i], c[i])) / max(o[i], self.EPS), 0, 0.05) / 0.05)
            feats[i, 3] = float(np.clip((min(o[i], c[i]) - l[i]) / max(o[i], self.EPS), 0, 0.05) / 0.05)
            feats[i, 4] = float(np.clip((v[i] / max(vol_ma[i], self.EPS) - 1) / 2, -1, 1))
            feats[i, 5] = float(np.clip(hl / max(c[i], self.EPS), 0.0, 0.3) / 0.3)
            feats[i, 6] = float((c[i] - l[i]) / hl) if hl > self.EPS else 0.5
            
        # 5m & 1h Features (Missing values handled via fillna in preprocess)
        feats[:, 7] = np.clip(df['smart_money_flow'].values, -5, 5) / 5.0
        feats[:, 8] = np.clip(df['squeeze_power'].values, 0, 10) / 10.0
        feats[:, 9] = np.clip(df['whale_retail_ratio'].values, 0, 5) / 5.0
        feats[:, 10] = (df['chop_index'].values - 50) / 50.0  # -1 to 1
        feats[:, 11] = (df['rsi'].values - 50) / 50.0         # -1 to 1
        feats[:, 12] = df['pred_mdjd'].values                 # AI Prediction
        feats[:, 13] = df['conf_mdjd'].values                 # AI Confidence
        feats[:, 14] = np.clip(df['garch_vol_z'].values, -3, 3) / 3.0
        feats[:, 15] = np.clip(df['ou_funding_z'].values, -3, 3) / 3.0

        return np.nan_to_num(feats, 0.0)

# ───────────────────────────────────────────────────────────────────────────
# Positional Encoding & Model (유지, 차원 확장됨)
# ───────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.temporal_bias = nn.Parameter(torch.zeros(1, max_len, 1))

    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:, :T] + torch.sigmoid(self.temporal_bias[:, :T]) * 0.1
        return self.dropout(x)

class TrendContextBrain(nn.Module):
    def __init__(self, feat_dim=BASE_FEAT_DIM, d_model=D_MODEL):
        super().__init__()
        self.feat_dim = feat_dim
        self.input_proj = nn.Sequential(nn.Linear(feat_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.pos_enc = PositionalEncoding(d_model, max_len=WINDOW+4, dropout=DROPOUT)
        
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=N_HEADS, dim_feedforward=D_FF, 
                                               dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        
        pool_dim = d_model * 2
        self.direction_head = nn.Sequential(nn.LayerNorm(pool_dim), nn.Linear(pool_dim, d_model), nn.SiLU(), nn.Dropout(DROPOUT), nn.Linear(d_model, 3))
        self.strength_head = nn.Sequential(nn.LayerNorm(pool_dim), nn.Linear(pool_dim, d_model//2), nn.SiLU(), nn.Linear(d_model//2, 1), nn.Sigmoid())
        self.reversal_head = nn.Sequential(nn.LayerNorm(pool_dim), nn.Linear(pool_dim, d_model//2), nn.SiLU(), nn.Linear(d_model//2, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.pos_enc(self.input_proj(x))
        h = self.encoder(h)
        pooled = torch.cat([h[:, -1, :], h.mean(dim=1)], dim=-1)
        return self.direction_head(pooled), self.strength_head(pooled), self.reversal_head(pooled)

    def save(self, path: str, meta: dict = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'state_dict': self.state_dict(), 'feat_dim': self.feat_dim, 'd_model': D_MODEL, 'meta': meta}, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'TrendContextBrain':
        ckpt = torch.load(path, map_location=device, weights_only=False)
        feat_dim = ckpt.get('feat_dim', BASE_FEAT_DIM)
        d_model  = ckpt.get('d_model', D_MODEL)
        model = cls(feat_dim=feat_dim, d_model=d_model).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, candles: np.ndarray) -> 'TrendSignal':
        """(WINDOW, feat_dim) numpy array → TrendSignal"""
        x = torch.tensor(candles[-WINDOW:], dtype=torch.float32).unsqueeze(0)
        d_log, s_pred, r_pred = self(x)
        probs = torch.softmax(d_log, dim=-1).squeeze(0).tolist()
        trend_dir = int(d_log.argmax(dim=-1).item())
        strength  = float(s_pred.squeeze())
        rev_prob  = float(r_pred.squeeze())
        return TrendSignal(trend_dir=trend_dir, strength=strength,
                           rev_prob=rev_prob, probs=tuple(probs))

    def predict_from_df(self, df: pd.DataFrame,
                        timestamp_col: str = 'timestamp',
                        min_candles: int = WINDOW) -> Optional['TrendSignal']:
        """5분봉 DataFrame → 4h 리샘플 → TrendSignal.
        없는 컬럼은 0으로 채워 robust하게 동작."""
        df = df.copy()
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col).sort_index()
        elif not isinstance(df.index, pd.DatetimeIndex):
            return None

        agg = {'open': 'first', 'high': 'max', 'low': 'min',
               'close': 'last', 'volume': 'sum'}
        opt_agg = {
            'smart_money_flow': 'sum', 'squeeze_power': 'max',
            'whale_retail_ratio': 'max', 'chop_index': 'last', 'rsi': 'last',
            'pred_mdjd': 'last', 'conf_mdjd': 'last',
            'garch_vol_z': 'max', 'ou_funding_z': 'last',
        }
        for col, fn in opt_agg.items():
            if col in df.columns:
                agg[col] = fn
        df_4h = df.resample('4h', closed='right', label='right').agg(agg).dropna(subset=['close'])
        if len(df_4h) < min_candles:
            return None

        extractor = HybridFeatureExtractor()
        for col in ['smart_money_flow','squeeze_power','whale_retail_ratio',
                    'chop_index','rsi','pred_mdjd','conf_mdjd','garch_vol_z','ou_funding_z']:
            if col not in df_4h.columns:
                df_4h[col] = 0.0
        candles = extractor.transform(df_4h.tail(min_candles))
        if candles.shape[-1] != self.feat_dim:
            return None
        return self.predict(candles)

# ───────────────────────────────────────────────────────────────────────────
# 데이터 병합 및 4h 리샘플링 유틸리티 (핵심 알파)
# ───────────────────────────────────────────────────────────────────────────
def merge_and_resample(df_5m_path: str, df_1h_path: str) -> pd.DataFrame:
    logger.info("데이터세트 로드 및 4시간 프레임 하이브리드 병합 시작...")
    
    # 1. 5분 데이터 로드 (OHLCV + Microstructure)
    df_5m = pd.read_csv(df_5m_path)
    df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
    df_5m = df_5m.set_index('timestamp').sort_index()
    
    # 2. 1시간 데이터 로드 (Quant Signals)
    df_1h = pd.read_csv(df_1h_path)
    if 'timestamp' in df_1h.columns:
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
        df_1h = df_1h.set_index('timestamp').sort_index()
    else:
        raise ValueError("1h 데이터에 timestamp가 필수입니다.")

    # 3. 5m -> 4h 리샘플링 (가격은 정밀하게, 플로우는 누적/평균)
    agg_5m = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
        # (수정) mean 대신 강력한 신호 한 방(max, min)을 포착
        'smart_money_flow': 'sum',      # 돈의 흐름은 누적해야 의미가 있음
        'squeeze_power': 'max',         # 스퀴즈는 터진 순간의 최대 파워가 중요
        'whale_retail_ratio': 'max',    # 고래 개입의 최대 불균형 포착
        'chop_index': 'last', 'rsi': 'last'
    }
    df_5m_4h = df_5m.resample('4h', closed='right', label='right').agg({
        k: v for k, v in agg_5m.items() if k in df_5m.columns
    })
    
    agg_1h = {
        'pred_mdjd': 'last', 'conf_mdjd': 'last', 
        'garch_vol_z': 'max', 'ou_funding_z': 'last'
    }
    df_1h_4h = df_1h.resample('4h', closed='right', label='right').agg({
        k: v for k, v in agg_1h.items() if k in df_1h.columns
    })
    
    # 5. 병합 (Inner Join으로 유효 구간만 추출)
    df_merged = df_5m_4h.join(df_1h_4h, how='inner').dropna(subset=['close'])
    df_merged.fillna(method='ffill', inplace=True) # 누락된 시그널 ffill
    df_merged.fillna(0, inplace=True)              # 그래도 없으면 0
    
    logger.info(f"병합 완료: 총 {len(df_merged)}봉 (4h)")
    return df_merged.reset_index()

# ───────────────────────────────────────────────────────────────────────────
# Dataset & Trainer (생략 없이 간결하게)
# ───────────────────────────────────────────────────────────────────────────
class TrendDataset(Dataset):
    def __init__(self, feats, closes):
        self.feats, self.closes = feats.astype(np.float32), closes.astype(np.float64)
        self.indices = list(range(WINDOW, len(feats) - LABEL_HORIZON_MAX - 1))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.feats[t - WINDOW : t]
        cur_close = self.closes[t - 1]
        future = self.closes[t + LABEL_HORIZON_MIN - 1 : t + LABEL_HORIZON_MAX]

        mid_ret = (float(future.max()) + float(future.min())) / 2 / float(cur_close + 1e-8) - 1
        label = 2 if mid_ret > FLAT_THRESH else 0 if mid_ret < -FLAT_THRESH else 1
        str_lbl = float(np.tanh(abs(mid_ret) * 20.0))
        
        past_ret = float(self.closes[t - 1] / max(self.closes[t - 6], 1e-8) - 1)
        fut_ret = float(future[-1]) / float(cur_close + 1e-8) - 1
        rev = 1.0 if (past_ret > 0 and fut_ret < -FLAT_THRESH) or (past_ret < 0 and fut_ret > FLAT_THRESH) else 0.0

        return torch.tensor(x), torch.tensor(label), torch.tensor(str_lbl), torch.tensor(rev)

class TrendBrainTrainer:
    def __init__(self, df, device='cuda', batch_size=256, lr=3e-4):
        self.device = device
        self.extractor = HybridFeatureExtractor()
        feats, closes = self.extractor.transform(df), df['close'].values
        split = int(len(feats) * 0.85)
        
        self.train_loader = DataLoader(TrendDataset(feats[:split], closes[:split]), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(TrendDataset(feats[split:], closes[split:]), batch_size=batch_size)
        
        self.model = TrendContextBrain(self.extractor.feat_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2) # 1e-4 -> 1e-2 로 100배 강화

        labels = [self.train_loader.dataset[i][1].item() for i in range(len(self.train_loader.dataset))]
        counts = np.bincount(labels, minlength=3).astype(np.float32)
        self.weights = torch.tensor((1.0 / (counts + 1e-6)) * sum(counts)/3, dtype=torch.float32).to(device)
        self.best_acc = 0.0

    def train(self, epochs: int = 1000, save_path: str = 'data/ensemble/ckpt/trend_brain_hybrid.pth', patience: int = 30):
        """
        과적합 방지 및 조기 종료가 적용된 기관급 훈련 루프.
        
        Args:
            epochs: 최대 훈련 에폭 수
            save_path: 최고 성능 모델 저장 경로
            patience: 검증 손실(Val Loss)이 개선되지 않을 때 기다려주는 최대 에폭 수
        """
        # 1. 1사이클 학습률 스케줄러 초기화 (학습 속도와 안정성 극대화)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=3e-4, 
            steps_per_epoch=len(self.train_loader), 
            epochs=epochs,
            pct_start=0.1
        )

        best_val_loss = float('inf')
        epochs_no_improve = 0
        self.best_acc = 0.0

        logger.info(f"🚀 학습 시작: 최대 {epochs} 에폭 (Early Stopping Patience: {patience})")

        for ep in range(1, epochs + 1):
            # ──────────────────────────────────────────────────────────
            # [Phase 1] 훈련 (Training)
            # ──────────────────────────────────────────────────────────
            self.model.train()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0

            for x, dl, sl, rl in self.train_loader:
                x = x.to(self.device)
                dl, sl, rl = dl.to(self.device), sl.to(self.device), rl.to(self.device)

                self.optimizer.zero_grad()
                d_log, s_pred, r_pred = self.model(x)

                # -----------------------------------------------------------
                # [수정된 Loss] 일반 CE 대신 Focal Loss 적용
                # -----------------------------------------------------------
                ce_loss = F.cross_entropy(d_log, dl, weight=self.weights, reduction='none')
                pt = torch.exp(-ce_loss)  # 모델의 예측 확률
                focal_loss = (((1 - pt) ** 2) * ce_loss).mean() # 감마=2

                str_loss = (F.mse_loss(s_pred, sl.unsqueeze(1), reduction='none') * (dl != 1).float().unsqueeze(1)).mean()
                rev_loss = F.binary_cross_entropy(r_pred, rl.unsqueeze(1))
                
                # Focal Loss에 가장 높은 가중치 부여
                loss = focal_loss + 0.3 * str_loss + 0.5 * rev_loss
                # -----------------------------------------------------------

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                tr_loss += loss.item() * len(x)
                tr_correct += (d_log.argmax(dim=-1) == dl).sum().item()
                tr_total += len(x)

            avg_tr_loss = tr_loss / max(tr_total, 1)
            tr_acc = tr_correct / max(tr_total, 1)

            # ──────────────────────────────────────────────────────────
            # [Phase 2] 검증 (Validation) - 모델의 진짜 실력 테스트
            # ──────────────────────────────────────────────────────────
            # ──────────────────────────────────────────────────────────
            # [Phase 2] 검증 (Validation) - 모델의 진짜 실력 테스트
            # ──────────────────────────────────────────────────────────
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for x_val, dl_val, sl_val, rl_val in self.val_loader:
                    x_val = x_val.to(self.device)
                    dl_val, sl_val, rl_val = dl_val.to(self.device), sl_val.to(self.device), rl_val.to(self.device)

                    d_log, s_pred, r_pred = self.model(x_val)

                    # --- [수정] Validation에도 Focal Loss 동일하게 적용 ---
                    ce_loss = F.cross_entropy(d_log, dl_val, weight=self.weights, reduction='none')
                    pt = torch.exp(-ce_loss)  
                    focal_loss = (((1 - pt) ** 2) * ce_loss).mean() 

                    str_loss = (F.mse_loss(s_pred, sl_val.unsqueeze(1), reduction='none') * (dl_val != 1).float().unsqueeze(1)).mean()
                    rev_loss = F.binary_cross_entropy(r_pred, rl_val.unsqueeze(1))
                    
                    loss = focal_loss + 0.3 * str_loss + 0.5 * rev_loss
                    # ----------------------------------------------------

                    val_loss += loss.item() * len(x_val)
                    val_correct += (d_log.argmax(dim=-1) == dl_val).sum().item()
                    val_total += len(x_val)

            avg_val_loss = val_loss / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            # ──────────────────────────────────────────────────────────
            # [Phase 3] 로깅 및 이상 감지 (Logging & Leakage Detection)
            # ──────────────────────────────────────────────────────────
            logger.info(
                f"Epoch {ep:03d}/{epochs} | "
                f"Train Loss: {avg_tr_loss:.4f} Acc: {tr_acc*100:.1f}% | "
                f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc*100:.1f}%"
            )

            # 데이터 누수(미래 참조) 강력 경고 로직
            if val_acc > 0.85 and ep <= 5:
                logger.error("🚨 [치명적 경고] 학습 초반부터 검증 정확도가 85%를 넘었습니다!")
                logger.error("🚨 입력 피처에 '미래의 정답(Target)'이 섞여 들어간 데이터 누수(Leakage) 상태입니다.")
                logger.error("🚨 당장 학습을 멈추고 pred_ 계열이나 conf_ 계열 피처를 의심하십시오.")

            # ──────────────────────────────────────────────────────────
            # [Phase 4] 조기 종료 및 체크포인트 (Early Stopping)
            # ──────────────────────────────────────────────────────────
            # 정확도(Acc)가 아닌 손실(Loss)을 기준으로 판단해야 모델의 '확신도'가 좋아집니다.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_acc = val_acc
                epochs_no_improve = 0
                
                # 최고 성능일 때만 모델 저장
                self.model.save(save_path, meta={'epoch': ep, 'val_acc': val_acc, 'val_loss': best_val_loss})
                logger.info(f"   🌟 [NEW BEST] 검증 손실 갱신 ({best_val_loss:.4f}) → 모델 저장 완료!")
            else:
                epochs_no_improve += 1
                logger.debug(f"   ⚠️ 성능 개선 없음 ({epochs_no_improve}/{patience})")
                
                if epochs_no_improve >= patience:
                    logger.warning(f"🛑 {patience} 에폭 연속 검증 성능 개선이 없어 과적합 방지를 위해 학습을 강제 종료합니다.")
                    logger.info(f"🏆 최종 최고 검증 정확도: {self.best_acc*100:.1f}%")
                    break

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df_merged = merge_and_resample(
        df_5m_path='data/training_features_5m.csv',
        df_1h_path='data/ensemble/rl_training_data_full.csv'
    )
    trainer = TrendBrainTrainer(df_merged, device='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.train(epochs=100)