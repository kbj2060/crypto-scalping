"""
MacroHFT Signal Model 학습 스크립트 v2.5
================================================================================
모델 아키텍처(macroHFT_model v2.5)와 호환.
학습 루프 자체는 v2.4와 동일.
"""

import sys, os, argparse, logging, json, math
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from macroHFT_model import ForecastingMacroHFT, MacroHFTConfig, DirectionalLoss
from core.feature_selector import auto_select_features
from core.feature_engineering import ULTIMATE_FEATURE_COLS, MUST_INCLUDE_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ✅ 수정 — super().__init__ 전에 base_optimizer를 먼저 만들고,
#           state property 충돌을 피하기 위해 상속 대신 위임(delegation) 방식으로 변경
class SAM:
    """
    SAM: e_w를 별도 dict에 저장해서 base_optimizer.state와 완전히 분리.
    base_optimizer는 한 번도 건드리지 않아서 AdamW 내부 state 초기화를 방해하지 않음.
    """
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.rho            = rho
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self._e_w           = {}   # e_w를 base_optimizer.state와 완전히 분리 저장

    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad.detach() * scale.to(p)
                p.data.add_(e_w)
                self._e_w[p] = e_w
        if zero_grad: self.zero_grad()

    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p not in self._e_w: continue
                p.data.sub_(self._e_w[p])   # w 복원
        self.base_optimizer.step()           # 순수 AdamW step — state 간섭 없음
        self._e_w.clear()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        norms = [
            p.grad.detach().norm(2).to(self.param_groups[0]['params'][0])
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]
        return torch.stack(norms).norm(2)

    def zero_grad(self):          self.base_optimizer.zero_grad()
    def step(self, closure=None): self.base_optimizer.step(closure)
    def state_dict(self):         return self.base_optimizer.state_dict()
    def load_state_dict(self, d): self.base_optimizer.load_state_dict(d)
    @property
    def state(self):              return self.base_optimizer.state

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n])
        self.backup = {}

class MacroHFTDataset(Dataset):
    def __init__(self, config, df, temporal_cols, stride=None):
        self.temporal_data = df[temporal_cols].values.astype(np.float32)
        if config.target_col not in df.columns: df[config.target_col] = 0.0
        self.target_data = df[config.target_col].values.astype(np.float32)
        self.window_size = config.input_window
        self.horizon     = config.forecast_horizon
        
        self.stride = stride if stride is not None else max(config.input_window // 4, 1)
        max_start = len(df) - (self.window_size + self.horizon)
        self.indices = list(range(0, max_start + 1, self.stride))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        t = i + self.window_size
        # [수정] 피처의 마지막 관측치는 t-1입니다. 따라서 타겟도 t-1 인덱스부터 가져와야 
        # 직후의 H 봉을 정확히 예측하게 됩니다. (1-step gap 제거)
        return (torch.tensor(self.temporal_data[i:t]),
                torch.tensor(self.target_data[t-1:t-1 + self.horizon]))


class MacroHFTSignalModel:
    def __init__(self, config=None):
        self.config = config
        self.model  = self.feature_cols = None
        self.scaler_params = {}
        self.target_scaler = {}
        self.ema = None

    def _create_scheduler(self, optimizer, steps_per_epoch):
        cfg = self.config
        warmup    = steps_per_epoch * cfg.warmup_epochs
        min_ratio = cfg.min_lr / cfg.learning_rate

        if cfg.lr_scheduler == 'cosine_restarts':
            T0   = steps_per_epoch * cfg.restart_period
            Tmul = cfg.restart_mult

            def fn(s):
                if s < warmup:
                    return float(s) / max(1, warmup)
                s_post = s - warmup
                T_cur, t_acc = T0, 0
                while t_acc + T_cur <= s_post:
                    t_acc += T_cur
                    T_cur  = int(T_cur * Tmul)
                t_in_cycle = s_post - t_acc
                cos_val = 0.5 * (1 + math.cos(math.pi * t_in_cycle / max(1, T_cur)))
                return max(min_ratio, cos_val)
            return fn

        if cfg.lr_scheduler == 'cosine':
            total = steps_per_epoch * cfg.max_epochs
            def fn(s):
                if s < warmup: return float(s) / max(1, warmup)
                p = (s - warmup) / max(1, total - warmup)
                return max(min_ratio, 0.5 * (1 + math.cos(math.pi * p)))
            return fn

        # plateau는 그대로 반환 (SAM과 별도로 epoch 단위 처리)
        return None

    def _save_checkpoint(self, tag):
        os.makedirs(self.config.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config.model_dir, f'macrohft_{tag}.pt'))
        meta = {
            'feature_cols':  self.feature_cols,
            'scaler_params': {k: v.tolist() for k, v in self.scaler_params.items()},
            'target_scaler': {k: float(v) for k, v in self.target_scaler.items()},
            'config':        self.config.__dict__,
        }
        with open(os.path.join(self.config.model_dir, f'macrohft_{tag}_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def _load_checkpoint(self, tag):
        path = os.path.join(self.config.model_dir, f'macrohft_{tag}.pt')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.config.device, weights_only=True))
            logger.info(f"💾 [Load] '{tag}' 모델 가중치를 성공적으로 불러왔습니다.")
        else:
            logger.warning(f"⚠️ [Load] {path} 파일이 존재하지 않습니다.")

    def _sharpe_weight(self, epoch, max_w=0.3):
        w, r = self.config.sharpe_warmup_epochs, 100
        if epoch < w: return 0.0
        return max_w * min(1.0, (epoch - w) / r)

    def _diagnose(self, train_acc, val_acc, gap):
        if train_acc < 0.54:
            return f"  🔵 [언더피팅] Train Dir {train_acc:.1%} < 54%"
        if train_acc - val_acc > 0.05:  # gap 대신 accuracy gap 기준
            return f"  🔴 [과적합] Train-Val Acc Gap={train_acc - val_acc:+.1%}"
        return f"  ✅ [정상] Train {train_acc:.1%} / Val {val_acc:.1%}"

    def fit(self, cfg, train_df, val_df, feature_cols):
        cfg.num_features  = len(feature_cols)
        self.config       = cfg
        self.feature_cols = feature_cols
        self.model        = ForecastingMacroHFT(cfg).to(cfg.device)

        fm = train_df[feature_cols].mean()
        fs = train_df[feature_cols].std().replace(0, 1.0)
        self.scaler_params = {'mean': fm.values, 'std': fs.values}

        # target은 이미 로컬 변동성으로 정규화됨 — global scaler 불필요
        tm = 0.0
        ts = 1.0
        self.target_scaler = {'mean': tm, 'std': ts}
        logger.info(f"[Target Scaler] disabled (locally normalized target)")

        def norm(df):
            d = df.copy()
            d[feature_cols] = (d[feature_cols] - fm) / fs
            return d

        train_norm, val_norm = norm(train_df), norm(val_df)
        train_loader = DataLoader(
            MacroHFTDataset(cfg, train_norm, feature_cols, stride=8),
            batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(
            MacroHFTDataset(cfg, val_norm, feature_cols, stride=8),
            batch_size=cfg.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )

        lr_fn   = self._create_scheduler(optimizer, max(len(train_loader) // cfg.accumulation_steps, 1))
        lr_step = [0]
        lr_scale = [1.0]  # Emergency Decay 배수 추적

        criterion = DirectionalLoss(
            large_move_weight=cfg.large_move_weight,
            label_smoothing=0.05
        )
        if cfg.use_ema:
            self.ema = EMAModel(self.model, cfg.ema_decay)

        best_val_acc = 0.0
        patience_cnt = 0
        val_rise_cnt = 0
        prev_val     = float('inf')
        history = {'train_loss': [], 'train_direction_acc': [], 'val_loss': [], 'val_direction_acc': []}

        for epoch in range(cfg.max_epochs):
            in_warmup = (epoch < cfg.warmup_epochs)
            self.model.train()
            tl = tdc = tt = 0

            for step, (seq, tgt) in enumerate(train_loader):
                seq, tgt = seq.to(cfg.device), tgt.to(cfg.device)

                optimizer.zero_grad()
                loss, ld = criterion(self.model(seq), tgt)
                loss = loss / cfg.accumulation_steps
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optimizer.step()

                lr_step[0] += 1
                if lr_fn is not None:
                    new_lr = cfg.learning_rate * lr_fn(lr_step[0]) * lr_scale[0]
                    for pg in optimizer.param_groups:
                        pg['lr'] = new_lr

                if self.ema:
                    self.ema.update(self.model)
                tl  += loss.item() * cfg.accumulation_steps
                tdc += ld['direction_accuracy'] * tgt.size(0)
                tt  += tgt.size(0)

            self.model.eval()
            vl = vdc = vt = 0
            if self.ema: self.ema.apply_shadow(self.model)
            with torch.no_grad():
                for seq, tgt in val_loader:
                    seq, tgt = seq.to(cfg.device), tgt.to(cfg.device)
                    vl_, vd_ = criterion(self.model(seq), tgt)
                    vl  += vl_.item() * tgt.size(0)
                    vdc += vd_['direction_accuracy'] * tgt.size(0)
                    vt  += tgt.size(0)
            if self.ema: self.ema.restore(self.model)

            atl = tl / max(len(train_loader), 1)
            ta  = tdc / max(tt, 1)
            avl = vl / max(vt, 1)
            va  = vdc / max(vt, 1)
            gap = ta - va

            history['train_loss'].append(atl)
            history['train_direction_acc'].append(ta)
            history['val_loss'].append(avl)
            history['val_direction_acc'].append(va)

            warmup_tag = " [WARMUP]" if in_warmup else ""
            logger.info(f"Epoch {epoch+1:03d}/{cfg.max_epochs}{warmup_tag} | "
                        f"Train: {atl:.4f} ({ta:.1%}) | Val: {avl:.4f} ({va:.1%}) | "
                        f"AccGap: {gap:+.1%} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            if (epoch + 1) % 5 == 0:
                logger.info(self._diagnose(ta, va, gap))

            # Warmup 중에는 early stopping 스킵
            if in_warmup:
                if epoch == cfg.warmup_epochs - 1:
                    best_val_acc = va
                    prev_val     = avl
                    val_rise_cnt = 0
                    logger.info(
                        f"  ✅ [Warmup 완료] Val Dir 기준점 = {va:.1%} "
                        f"— 이제부터 Early Stopping 카운트 시작"
                    )
                continue

            # Emergency LR Decay — val loss 연속 상승 시
            val_rise_cnt = val_rise_cnt + 1 if avl > prev_val else 0
            prev_val = avl
            if val_rise_cnt >= 7:
                lr_scale[0] *= 0.5  # cosine 기저에 곱하기
                lr_scale[0] = max(lr_scale[0], cfg.min_lr / cfg.learning_rate)
                logger.warning(
                    f"  🚨 [Emergency LR Decay] scale={lr_scale[0]:.3f}"
                )
                val_rise_cnt = 0

            # Early Stopping — val direction accuracy 기준
            if va > best_val_acc:
                best_val_acc = va
                patience_cnt = 0
                self._save_checkpoint('best')
                logger.info(f"  🌟 [New Best] Val Dir: {va:.1%}")
            else:
                patience_cnt += 1
                logger.info(f"  ⚠️ Patience: {patience_cnt}/{cfg.patience}")
                if patience_cnt >= cfg.patience:
                    logger.info("🛑 Early Stopping.")
                    break

        logger.info("✅ 학습 종료. 최고 성능(Best) 모델로 롤백합니다.")
        self._load_checkpoint('best')
        if self.ema: 
            self.ema.apply_shadow(self.model) # EMA 가중치까지 완벽하게 복원
            
        return history

        
def walk_forward_split(df, n_splits=5, train_ratio=0.6, val_ratio=0.15, purge_bars=64):
    """
    시간순 Walk-Forward CV.
    각 fold마다 train→purge→val→purge→test 구조.
    다양한 regime을 골고루 평가.
    """
    n = len(df)
    fold_size = n // n_splits
    splits = []
    
    for i in range(n_splits):
        test_end = n - i * fold_size
        test_start = test_end - fold_size
        if test_start < 0:
            break
        
        val_end = test_start - purge_bars
        val_start = val_end - int(fold_size * val_ratio / (1 - train_ratio - val_ratio))
        
        train_end = val_start - purge_bars
        train_start = max(0, train_end - int(fold_size * train_ratio / (1 - train_ratio - val_ratio)))
        
        if train_end - train_start < fold_size // 2:
            continue
            
        splits.append({
            'train': df.iloc[train_start:train_end].copy(),
            'val':   df.iloc[val_start:val_end].copy(),
            'test':  df.iloc[test_start:test_end].copy(),
            'fold':  i,
        })
        logger.info(f"  Fold {i}: train[{train_start}:{train_end}] "
                    f"val[{val_start}:{val_end}] test[{test_start}:{test_end}]")
    
    return splits

def load_data(path='data/training_features_5m.csv', h=1):
    df = pd.read_csv(path, parse_dates=['timestamp']).replace([np.inf, -np.inf], np.nan)
    df['ema_1h'] = df['close'].ewm(span=12).mean()
    df['ema_4h'] = df['close'].ewm(span=48).mean()
    df['mtf_trend_1h'] = df['close'] / df['ema_1h'] - 1
    df['mtf_trend_4h'] = df['close'] / df['ema_4h'] - 1

    # [혁신 수정] 미래 참조가 없는 '순수 다음 1봉(h) 내의 VWAP'과 현재 종가 비교
    # 롤링을 일절 배제하여 시간적 엇나감(Temporal Mismatch) 완벽 해소
    next_typical_price = (df['high'].shift(-h) + df['low'].shift(-h) + df['close'].shift(-h)) / 3
    # next_vwap은 미래의 다중 캔들 평균이 아니라, 오직 다음 1개 캔들의 체결 중심가입니다.
    next_vwap = next_typical_price  # (단일 캔들이므로 typical price 자체가 그 캔들의 중심가)
    
    # 목표는 다음 1봉의 중심가(VWAP)가 현재 종가 대비 오르는가 내리는가 입니다.
    # 이는 마지막 틱의 Bid-Ask Bounce를 회피하면서도 h=1 제약을 완벽히 준수합니다.
    df['raw_ret'] = (next_vwap / df['close'] - 1) * 100
    
    # 과거 데이터 기반의 변동성으로 정규화 (Target Leakage 없음)
    past_returns = df['close'].pct_change() * 100
    rolling_std = past_returns.rolling(200, min_periods=50).std()
    df[f'target_ret_{h}'] = df['raw_ret'] / rolling_std.clip(lower=0.01)
    
    from core.cvp import add_cvp_features
    if 'cvp_poc_dist' not in df.columns:
        df = add_cvp_features(df, lookback=200, n_clusters=4, drop_strategy=False)

    df.dropna(inplace=True)
    feats = list(dict.fromkeys([c for c in ULTIMATE_FEATURE_COLS if c in df.columns] + ['mtf_trend_1h', 'mtf_trend_4h']))
    return df, feats

def split_data(df, train_ratio=0.7, val_ratio=0.15, purge_bars=64):
    """
    [FIX] Train/Val/Test 사이에 purge gap을 두어
    슬라이딩 윈도우의 정보 누수를 차단.
    """
    n = len(df)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:t].copy()
    # train 끝 ~ val 시작 사이에 input_window만큼 gap
    val   = df.iloc[t + purge_bars : v].copy()
    # val 끝 ~ test 시작 사이에도 gap
    test  = df.iloc[v + purge_bars :].copy()
    
    logger.info(f"[Split] Train: {len(train)}, Val: {len(val)}, Test: {len(test)}, Purge: {purge_bars} bars")
    return train, val, test

def evaluate_model(mw, test_df):
    cfg = mw.config
    if cfg.target_col not in test_df.columns: return {}
    tc = mw.feature_cols
    fm, fs = mw.scaler_params['mean'], mw.scaler_params['std']
    tm, ts = mw.target_scaler['mean'], mw.target_scaler['std']

    tn = test_df.copy()
    tn[tc] = (tn[tc] - fm) / fs
    tn[cfg.target_col] = (tn[cfg.target_col] - tm) / ts

    loader = DataLoader(MacroHFTDataset(cfg, tn, tc, stride=cfg.input_window // 2),
                        batch_size=cfg.batch_size, shuffle=False)
    mw.model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for seq, tgt in loader:
            out = mw.model(seq.to(cfg.device)).cpu()
            all_logits.append(out[:, 0, 0])  # 단일 로짓 직접 사용
            all_targets.append(tgt[:, 0])
    if not all_logits: return {}

    logits = torch.cat(all_logits).numpy()
    actuals_norm = torch.cat(all_targets).numpy()
    actuals = actuals_norm * ts + tm

    pred_up = logits > 0
    actual_up = actuals > 0
    da = np.mean(pred_up == actual_up)

    # 큰 움직임
    thr = np.percentile(np.abs(actuals), 80)
    lm = np.abs(actuals) > thr
    lda = np.mean(pred_up[lm] == actual_up[lm]) if lm.sum() > 0 else 0.0

    # 확신도별 정확도 (핵심 진단)
    confidence = np.abs(logits)
    for pct in [50, 70, 90]:
        thr_c = np.percentile(confidence, pct)
        mask = confidence > thr_c
        if mask.sum() > 0:
            acc_c = np.mean(pred_up[mask] == actual_up[mask])
            logger.info(f"  [확신도 상위 {100-pct}%] 정확도: {acc_c:.1%} ({mask.sum()}건)")

    logger.info(f"\n  [진단] pred pos_ratio={np.mean(pred_up):.1%}, "
                f"actual pos_ratio={np.mean(actual_up):.1%}")
    logger.info(f"  [진단] logit mean={logits.mean():.4f}, std={logits.std():.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("📊 테스트 결과 (MacroHFT v2.6)")
    logger.info("=" * 60)
    logger.info(f"  방향 정확도:           {da:.1%}")
    logger.info(f"  큰 움직임 방향 정확도: {lda:.1%}")
    logger.info("=" * 60)
    return {'direction_accuracy': float(da),
            'large_move_direction_acc': float(lda), 'test_samples': len(logits)}

# main()을 단순 split으로 교체
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/training_features_5m.csv')
    args = parser.parse_args()
    cfg  = MacroHFTConfig()

    start = datetime.now()
    df, feats = load_data(args.data, cfg.forecast_horizon)
    train_df, val_df, test_df = split_data(df)

    selected = auto_select_features(train_df, feats, target_col=cfg.target_col,
                                    max_features=cfg.num_features, corr_threshold=0.85,
                                    must_include=MUST_INCLUDE_FEATURES)

    model = MacroHFTSignalModel(cfg)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    X_train = train_df[selected].values
    y_train = (train_df[cfg.target_col] > 0).astype(int).values
    X_val = val_df[selected].values
    y_val = (val_df[cfg.target_col] > 0).astype(int).values

    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, gb.predict(X_train))
    val_acc = accuracy_score(y_val, gb.predict(X_val))
    logger.info(f"[Baseline GBM] Train: {train_acc:.1%}, Val: {val_acc:.1%}")

    # 피처 중요도 상위 10개
    importances = sorted(zip(selected, gb.feature_importances_), key=lambda x: -x[1])
    for name, imp in importances[:10]:
        logger.info(f"  {name:30s}: {imp:.4f}")
    history = model.fit(cfg, train_df, val_df, selected)
    metrics = evaluate_model(model, test_df)
    print(f"\n🎉 완료! 소요 시간: {datetime.now() - start}")


if __name__ == '__main__':        
    main()