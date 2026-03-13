import os
import sys
import asyncio
import time
import logging
import gc
import numpy as np
import pandas as pd
import torch
import ccxt.async_support as ccxt
from datetime import datetime, timedelta

# 💡 [1. 경로 설정]
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PATHS = [
    _THIS_DIR,
    os.path.join(_THIS_DIR, "models"),
    os.path.join(_THIS_DIR, "timesfm"),
    os.path.join(_THIS_DIR, "uni2ts", "src"),
    os.path.join(_THIS_DIR, "strategies"),
    os.path.join(_THIS_DIR, "ensemble"), 
]
for p in TARGET_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from core.feature_engineering import FeatureEngineer
from ensemble.ensemble_router import (
    TFTForecaster, MacroHFTForecaster, ChronosForecaster, 
    KronosForecaster, TimesFMForecaster, MoiraiForecaster
)

# [NEW] 12-Agent MoE 모듈 및 전략 Import
from ensemble.train_rl_agent import (
    MoEIQNTrader, RobustIQN, STATE_DIM,
    MODEL_PRED, MODEL_CONF, ELITE_COLS, ALPHA_7_COLS, REGIME_COLS
)
# [NEW] LS 2-Agent (롱돌이/숏돌이) Import
from ensemble.train_ls_agent import DualAgentTrader
from strategies.elite_builder import EliteSignals, row_to_market_row

class Colors:
    GREEN, RED, YELLOW, CYAN, RESET, BOLD = '\033[92m', '\033[91m', '\033[93m', '\033[96m', '\033[0m', '\033[1m'

if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("LiveBot")

# ════════════════════════════════════════════════════════════════
# 1. 데이터 수집기 (비동기 BinanceLiveFetcher - 원본 유지)
# ════════════════════════════════════════════════════════════════
class BinanceLiveFetcher:
    def __init__(self, symbol='ETHUSDT', timeframe='5m', limit=2500):
        self.symbol = symbol.replace('/', '') 
        self.timeframe = timeframe
        self.limit = limit
        self.exchange = ccxt.binance({'options': {'defaultType': 'future'}})

    def load_local_data(self):
        try:
            eth_df = pd.read_csv('data/test/eth_test_data.csv')
            btc_df = pd.read_csv('data/test/btc_test_data.csv')
            for df in [eth_df, btc_df]:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cols = df.columns.drop('timestamp')
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            logger.info(f"{Colors.GREEN}📂 로컬 데이터 로드 성공{Colors.RESET}")
            return eth_df, btc_df
        except Exception as e:
            logger.error(f"로컬 로드 실패: {e}")
            return None, None

    async def fetch_klines_raw(self, symbol, target_limit):
        all_klines = []
        last_end_time = None
        while len(all_klines) < target_limit:
            params = {'symbol': symbol, 'interval': self.timeframe, 'limit': 1000}
            if last_end_time: params['endTime'] = last_end_time - 1
            klines = await self.exchange.fapiPublicGetKlines(params)
            if not klines: break
            all_klines = klines + all_klines
            last_end_time = klines[0][0]
            if len(klines) < 1000: break
        return all_klines[-target_limit:]

    async def fetch_ancillary_data(self, limit=500):
        tasks = [
            self.exchange.fapiDataGetOpenInterestHist({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiDataGetTopLongShortAccountRatio({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiDataGetTopLongShortPositionRatio({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiDataGetGlobalLongShortAccountRatio({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiDataGetTakerlongshortRatio({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiPublicGetFundingRate({'symbol': self.symbol, 'limit': limit})
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _process_to_df(self, eth_klines, btc_klines, ancillary_results):
        eth_df = pd.DataFrame(eth_klines).iloc[:, :11]
        eth_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
        eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], unit='ms')
        eth_df[eth_df.columns.drop('timestamp')] = eth_df[eth_df.columns.drop('timestamp')].apply(pd.to_numeric, errors='coerce')

        btc_df = pd.DataFrame(btc_klines).iloc[:, [0, 4, 5, 7]]
        btc_df.columns = ['timestamp', 'close_btc', 'volume_btc', 'quote_volume_btc']
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
        btc_df[btc_df.columns.drop('timestamp')] = btc_df[btc_df.columns.drop('timestamp')].apply(pd.to_numeric, errors='coerce')

        if ancillary_results:
            mappings = [(0, 'sumOpenInterestValue', 'sum_open_interest_value'), (1, 'longShortRatio', 'sum_toptrader_long_short_ratio'), (3, 'longShortRatio', 'count_long_short_ratio'), (5, 'fundingRate', 'last_funding_rate')]
            for idx, key, new_name in mappings:
                res = ancillary_results[idx]
                if isinstance(res, list) and len(res) > 0:
                    try:
                        temp_df = pd.DataFrame(res)
                        t_col = next((c for c in ['timestamp', 'fundingTime', 'time'] if c in temp_df.columns), None)
                        if t_col and key in temp_df.columns:
                            subset = temp_df[[t_col, key]].rename(columns={t_col: 'timestamp', key: new_name})
                            subset['timestamp'] = pd.to_datetime(subset['timestamp'], unit='ms')
                            subset[new_name] = pd.to_numeric(subset[new_name], errors='coerce')
                            eth_df = pd.merge_asof(eth_df.sort_values('timestamp'), subset.sort_values('timestamp'), on='timestamp', direction='nearest')
                    except Exception: pass
        return eth_df.ffill().bfill(), btc_df

    async def fetch_initial_data(self):
        eth_klines = await self.fetch_klines_raw(self.symbol, self.limit)
        btc_klines = await self.fetch_klines_raw('BTCUSDT', self.limit)
        ancillary = await self.fetch_ancillary_data(500)
        return self._process_to_df(eth_klines, btc_klines, ancillary)

    async def fetch_latest_patch(self):
        eth_klines = await self.exchange.fapiPublicGetKlines({'symbol': self.symbol, 'interval': self.timeframe, 'limit': 5})
        btc_klines = await self.exchange.fapiPublicGetKlines({'symbol': 'BTCUSDT', 'interval': self.timeframe, 'limit': 5})
        ancillary = await self.fetch_ancillary_data(5)
        return self._process_to_df(eth_klines, btc_klines, ancillary)

# ════════════════════════════════════════════════════════════════
# 2. 앙상블 인퍼런스 & 레짐 계산
# ════════════════════════════════════════════════════════════════
class EnsemblePredictor:
    def __init__(self):
        self.models = {
            'TFT': TFTForecaster(),
            'MacroHFT': MacroHFTForecaster(),
            'Chronos': ChronosForecaster(),
            'Kronos': KronosForecaster(),
            'TimesFM': TimesFMForecaster(),
            'Moirai': MoiraiForecaster()
        }
        self.model_order = ['TFT', 'MacroHFT', 'Chronos', 'Kronos', 'TimesFM', 'Moirai']

    async def predict_all_async(self, df: pd.DataFrame):
        preds, confs = [], []
        loop = asyncio.get_event_loop()
        
        def _run_inference(m):
            if not getattr(m, 'available', False): return None
            try: return m.predict(df, horizon=6) 
            except Exception: return None

        tasks = [loop.run_in_executor(None, _run_inference, self.models[name]) for name in self.model_order]
        results = await asyncio.gather(*tasks)
        
        for i, res in enumerate(results):
            p_val, c_val = 0.0, 0.5
            if res is not None and getattr(res, 'median', None) is not None:
                trajectory = np.array(res.median[-1], dtype=np.float32) 
                if len(trajectory) > 1:
                    slope = float(np.polyfit(np.arange(len(trajectory)), trajectory, 1)[0])
                    delta = trajectory[-1] - trajectory[0]
                    if slope > 0 and delta > 0: p_val = 1.0  
                    elif slope < 0 and delta < 0: p_val = -1.0 
                else: p_val = float(trajectory.mean())
                c_val = float(res.confidence[-1].mean())
            preds.append(p_val)
            confs.append(c_val)
            
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return np.array(preds), np.array(confs)

def _compute_regime(df, window=24):
    close = df['close']
    net_change  = close - close.shift(window)
    diff_abs    = close.diff().abs().rolling(window).sum()
    er          = net_change.abs() / (diff_abs + 1e-8)
    raw_vol     = close.pct_change().rolling(window).std()
    vol_z       = (raw_vol - raw_vol.rolling(window * 4).mean()) / (raw_vol.rolling(window * 4).std() + 1e-8)
    ema12       = close.ewm(span=12).mean()
    ema26       = close.ewm(span=26).mean()
    mtf         = (ema12 - ema26) / (ema26 + 1e-8) * 100

    er_v    = float(er.iloc[-1])    if er.notna().iloc[-1]       else 0.0
    volz_v  = float(vol_z.iloc[-1]) if vol_z.notna().iloc[-1]    else 0.0
    nc_v    = float(net_change.iloc[-1]) if net_change.notna().iloc[-1] else 0.0
    mtf_v   = float(mtf.iloc[-1])   if mtf.notna().iloc[-1]      else 0.0

    bull  = er_v >= 0.20 and nc_v > 0 and mtf_v > 0
    bear  = er_v >= 0.20 and nc_v < 0 and mtf_v < 0
    chop  = (not bull) and (not bear) and volz_v < -0.5
    whip  = (not bull) and (not bear) and volz_v >  0.5
    norm  = not (bull or bear or chop or whip)
    return {
        'regime_bull': 1.0 if bull else 0.0, 'regime_bear': 1.0 if bear else 0.0,
        'regime_chop': 1.0 if chop else 0.0, 'regime_whipsaw': 1.0 if whip else 0.0,
        'regime_normal': 1.0 if norm else 0.0,
    }

# 💡 [우회 패치] 훈련 중인 파일을 건드리지 않고, 상속을 통해 속마음(Edge)만 빼옵니다.
class LiveMoEIQNTrader(MoEIQNTrader):
    def decide(self, current_idx, features, pos):
        cur_pos = pos.get('type')
        state   = self._state_tensor(features, pos)

        with torch.no_grad():
            q_bull         = self.model_bull(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            q_bear         = self.model_bear(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            q_sup          = self.model_sup(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            q_res          = self.model_res(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            q_normal_long  = self.model_normal_long(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            q_normal_short = self.model_normal_short(state)[0].mean(dim=1).squeeze(0).cpu().numpy()

            if cur_pos is not None and self._active_pair is not None:
                exit_model = {
                    'bull':         self.model_bull_exit,  'bear':         self.model_bear_exit,
                    'sup':          self.model_sup_exit,   'res':          self.model_res_exit,
                    'normal_long':  self.model_normal_long_exit, 'normal_short': self.model_normal_short_exit,
                }[self._active_pair]
                q_exit = exit_model(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            else:
                q_exit = None

        adv_bull         = q_bull[1]         - q_bull[0]
        adv_bear         = q_bear[1]         - q_bear[0]
        adv_sup          = q_sup[1]          - q_sup[0]
        adv_res          = q_res[1]          - q_res[0]
        adv_normal_long  = q_normal_long[1]  - q_normal_long[0]
        adv_normal_short = q_normal_short[1] - q_normal_short[0]

        kelly_bull         = max(0., adv_bull         / (q_bull.std()         + 0.05))
        kelly_bear         = max(0., adv_bear         / (q_bear.std()         + 0.05))
        kelly_sup          = max(0., adv_sup          / (q_sup.std()          + 0.05))
        kelly_res          = max(0., adv_res          / (q_res.std()          + 0.05))
        kelly_normal_long  = max(0., adv_normal_long  / (q_normal_long.std()  + 0.05))
        kelly_normal_short = max(0., adv_normal_short / (q_normal_short.std() + 0.05))

        # 💡 [추가] 0으로 자르지 않은 순수 엣지(Edge) 값 계산
        edge_bull         = adv_bull         / (q_bull.std()         + 0.05)
        edge_bear         = adv_bear         / (q_bear.std()         + 0.05)
        edge_sup          = adv_sup          / (q_sup.std()          + 0.05)
        edge_res          = adv_res          / (q_res.std()          + 0.05)
        edge_normal_long  = adv_normal_long  / (q_normal_long.std()  + 0.05)
        edge_normal_short = adv_normal_short / (q_normal_short.std() + 0.05)

        CLOSE_KELLY_THRESHOLD = 0.3

        is_chop = features.get('regime_chop', 0.) == 1. or features.get('regime_whipsaw', 0.) == 1.
        is_bull = features.get('regime_bull', 0.) == 1.
        is_bear = features.get('regime_bear', 0.) == 1.

        # 💡 [추가] 현재 국면의 순수 엣지 추출
        curr_long_edge, curr_short_edge = 0.0, 0.0
        if is_chop:
            curr_long_edge, curr_short_edge = edge_sup, edge_res
        elif is_bull or is_bear:
            curr_long_edge, curr_short_edge = edge_bull, edge_bear
        else:
            curr_long_edge, curr_short_edge = edge_normal_long, edge_normal_short

        if cur_pos is not None and self._active_pair is not None:
            exit_signal = (q_exit is not None) and (q_exit[1] > q_exit[0])
            if cur_pos == 'LONG':
                opp_signal = (adv_bear > 0 and kelly_bear > CLOSE_KELLY_THRESHOLD) or \
                             (adv_res  > 0 and kelly_res  > CLOSE_KELLY_THRESHOLD) or \
                             (self._active_pair == 'normal_long' and adv_normal_short > 0 and kelly_normal_short > CLOSE_KELLY_THRESHOLD)
            else:  
                opp_signal = (adv_bull > 0 and kelly_bull > CLOSE_KELLY_THRESHOLD) or \
                             (adv_sup  > 0 and kelly_sup  > CLOSE_KELLY_THRESHOLD) or \
                             (self._active_pair == 'normal_short' and adv_normal_long > 0 and kelly_normal_long > CLOSE_KELLY_THRESHOLD)

            if exit_signal or opp_signal:
                active = self._active_pair
                self._active_pair = None
                return 0, 0.0, {'agent': f'{active}_exit+opp' if opp_signal else f'{active}_exit', 'long_edge': float(curr_long_edge), 'short_edge': float(curr_short_edge)}
            else:
                return (1 if cur_pos == 'LONG' else 2), 0.0, {'agent': 'HOLD', 'long_edge': float(curr_long_edge), 'short_edge': float(curr_short_edge)}

        final_action, selected_kelly = 0, 0.0
        if is_chop:
            active_agent = "SUP/RES (대기중)"
            if adv_sup > 0 and adv_sup >= adv_res: final_action, active_agent, selected_kelly, self._active_pair = 1, "SUP_BUY 🚀", kelly_sup, 'sup'
            elif adv_res > 0: final_action, active_agent, selected_kelly, self._active_pair = 2, "RES_SELL 🚀", kelly_res, 'res'
        elif is_bull:
            active_agent = "BULL_SNIPE (대기중)"
            if adv_bull > 0: final_action, active_agent, selected_kelly, self._active_pair = 1, "BULL_SNIPE 🚀", kelly_bull, 'bull'
        elif is_bear:
            active_agent = "BEAR_SNIPE (대기중)"
            if adv_bear > 0: final_action, active_agent, selected_kelly, self._active_pair = 2, "BEAR_SNIPE 🚀", kelly_bear, 'bear'
        else:  
            active_agent = "NORMAL (대기중)"
            if adv_normal_long > 0 and adv_normal_long >= adv_normal_short: final_action, active_agent, selected_kelly, self._active_pair = 1, "NORMAL_LONG 🚀", kelly_normal_long, 'normal_long'
            elif adv_normal_short > 0: final_action, active_agent, selected_kelly, self._active_pair = 2, "NORMAL_SHORT 🚀", kelly_normal_short, 'normal_short'

        if final_action == 0: self._active_pair = None
        leverage_rate = np.clip(selected_kelly * 0.5, 0.1, 1.0) if final_action != 0 else 0.0
        
        return final_action, leverage_rate, {'agent': active_agent, 'kelly': selected_kelly, 'long_edge': float(curr_long_edge), 'short_edge': float(curr_short_edge)}


# ════════════════════════════════════════════════════════════════
# 3-A. LS 2-Agent 라우터 (롱돌이/숏돌이)
# ════════════════════════════════════════════════════════════════
class LSLiveRouter:
    MODEL_ORDER  = ['TFT', 'MacroHFT', 'Chronos', 'Kronos', 'TimesFM', 'Moirai']
    LIVE_TO_TRAIN = {'TimesFM': 'pred_timesfm', 'Chronos': 'pred_chronos'}

    def __init__(self, model_path='data/ensemble/best_ls_agents.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.elite_extractor = EliteSignals()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LS 모델을 찾을 수 없습니다: {model_path}")

        ckpt = torch.load(model_path, map_location=self.device)
        def _load(key):
            m = RobustIQN(STATE_DIM, 2).to(self.device)
            m.load_state_dict(ckpt[key])
            m.eval()
            return m

        self.trader = DualAgentTrader(
            _load('model_long_entry'), _load('model_short_entry'),
            _load('model_long_exit'),  _load('model_short_exit'),
            self.device
        )
        self.pos, self.entry_price, self.hold_count = None, 0.0, 0
        self.peak_equity, self.current_equity = 1.0, 1.0
        logger.info(f"✅ {Colors.GREEN}LS 2-Agent 롱돌이/숏돌이 라우터 탑재 완료{Colors.RESET} (epoch={ckpt.get('epoch','?')})")

    def get_signal(self, processed_df, preds, confs):
        last_row = processed_df.iloc[-1]
        prev_row = processed_df.iloc[-2]
        smf_std  = processed_df['smart_money_flow'].std() if 'smart_money_flow' in processed_df.columns else 1.0

        cur_market  = row_to_market_row(last_row)
        prev_market = row_to_market_row(prev_row)
        elite_sigs  = self.elite_extractor.compute_all(current=cur_market, prev=prev_market, smf_std=smf_std)

        rev_map  = {v: k for k, v in self.LIVE_TO_TRAIN.items()}
        conf_map = {v.replace('pred_', 'conf_'): k for k, v in self.LIVE_TO_TRAIN.items()}
        live_pred = {n: preds[i] for i, n in enumerate(self.MODEL_ORDER)}
        live_conf = {n: confs[i] for i, n in enumerate(self.MODEL_ORDER)}

        features = {}
        for col in MODEL_PRED:
            src = rev_map.get(col)
            features[col] = float(live_pred[src]) if src else 0.0
        for col in MODEL_CONF:
            src = conf_map.get(col)
            features[col] = float(live_conf[src]) if src else 0.5

        features.update(elite_sigs)
        for col in ALPHA_7_COLS: features[col] = float(last_row.get(col, 0.0))
        features.update(_compute_regime(processed_df))
        features['close'] = float(last_row['close'])

        unr = 0.0
        if self.pos is not None:
            cp  = float(last_row['close'])
            unr = (cp - self.entry_price) / self.entry_price if self.pos == 'LONG' else (self.entry_price - cp) / self.entry_price
            self.current_equity = 1.0 + unr
            if self.current_equity > self.peak_equity: self.peak_equity = self.current_equity
        else:
            self.current_equity = self.peak_equity = 1.0

        pos_dict = {
            'type': self.pos, 'entry_price': self.entry_price,
            'unrealized': unr, 'mdd': min((self.current_equity / self.peak_equity) - 1.0, 0.0),
            'hold_norm': min(self.hold_count / 100.0, 1.0)
        }

        final_action, _, info = self.trader.decide(features, pos_dict)

        if final_action == 1 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = 'LONG', float(last_row['close']), 0
        elif final_action == 2 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = 'SHORT', float(last_row['close']), 0
        elif final_action == 0 and self.pos is not None:
            self.pos, self.entry_price, self.hold_count = None, 0.0, 0
        elif self.pos is not None:
            self.hold_count += 1

        pnl_pct = (self.current_equity - 1.0) * 100
        return final_action, info, pnl_pct

# ════════════════════════════════════════════════════════════════
# 3-B. AI 컴포넌트: 12-Agent MoE 라우터
# ════════════════════════════════════════════════════════════════
class MoELiveRouter:
    MODEL_ORDER = ['TFT', 'MacroHFT', 'Chronos', 'Kronos', 'TimesFM', 'Moirai']
    LIVE_TO_TRAIN = {'TimesFM': 'pred_timesfm', 'Chronos': 'pred_chronos'}

    def __init__(self, model_path='data/ensemble/best_moe_agents.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.elite_extractor = EliteSignals()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"학습된 AI 모델을 찾을 수 없습니다: {model_path}")

        ckpt = torch.load(model_path, map_location=self.device)
        def _load(key):
            m = RobustIQN(STATE_DIM, 2).to(self.device)
            m.load_state_dict(ckpt[key])
            m.eval()
            return m

        self.trader = LiveMoEIQNTrader(
            _load('model_bull'), _load('model_bear'), _load('model_sup'), _load('model_res'),
            _load('model_bull_exit'), _load('model_bear_exit'), _load('model_sup_exit'), _load('model_res_exit'),
            _load('model_normal_long'), _load('model_normal_short'),
            _load('model_normal_long_exit'), _load('model_normal_short_exit'),
            df=pd.DataFrame(), device=self.device
        )
        self.pos, self.entry_price, self.hold_count = None, 0.0, 0
        self.peak_equity, self.current_equity = 1.0, 1.0
        logger.info(f"✅ {Colors.GREEN}12-Agent Dueling MoE 라우터 탑재 완료{Colors.RESET} (epoch={ckpt.get('epoch','?')})")

    def get_signal(self, processed_df, preds, confs):
        last_row = processed_df.iloc[-1]
        prev_row = processed_df.iloc[-2]
        smf_std = processed_df['smart_money_flow'].std() if 'smart_money_flow' in processed_df.columns else 1.0
        
        # Elite 11종 시그널 추출
        cur_market = row_to_market_row(last_row)
        prev_market = row_to_market_row(prev_row)
        elite_sigs = self.elite_extractor.compute_all(current=cur_market, prev=prev_market, smf_std=smf_std)

        features = {}
        live_pred = {n: preds[i] for i, n in enumerate(self.MODEL_ORDER)}
        live_conf = {n: confs[i] for i, n in enumerate(self.MODEL_ORDER)}
        rev_map  = {v: k for k, v in self.LIVE_TO_TRAIN.items()}
        conf_map = {v.replace('pred_', 'conf_'): k for k, v in self.LIVE_TO_TRAIN.items()}
        
        for col in MODEL_PRED:
            src = rev_map.get(col)
            features[col] = float(live_pred[src]) if src else 0.0
        for col in MODEL_CONF:
            src = conf_map.get(col)
            features[col] = float(live_conf[src]) if src else 0.5
            
        features.update(elite_sigs)
        for col in ALPHA_7_COLS: features[col] = float(last_row.get(col, 0.0))
        regime = _compute_regime(processed_df)
        features.update(regime)
        features['close'] = float(last_row['close'])

        unr = 0.0
        if self.pos is not None:
            cp = float(last_row['close'])
            unr = (cp - self.entry_price) / self.entry_price if self.pos == 'LONG' else (self.entry_price - cp) / self.entry_price
            self.current_equity = 1.0 + unr
            if self.current_equity > self.peak_equity: self.peak_equity = self.current_equity
        else: self.current_equity = self.peak_equity = 1.0

        pos_dict = {
            'type': self.pos, 'entry_price': self.entry_price,
            'unrealized': unr, 'mdd': min((self.current_equity / self.peak_equity) - 1.0, 0.0),
            'hold_norm': min(self.hold_count / 100.0, 1.0)
        }

        # 12-Agent 결단
        final_action, leverage_rate, info = self.trader.decide(0, features, pos_dict)

        # 내부 포지션 트래킹 업데이트
        if final_action == 1 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = 'LONG', float(last_row['close']), 0
        elif final_action == 2 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = 'SHORT', float(last_row['close']), 0
        elif final_action == 0 and self.pos is not None:
            self.pos, self.entry_price, self.hold_count = None, 0.0, 0
        elif self.pos is not None:
            self.hold_count += 1

        return final_action, info, elite_sigs, regime

    def print_dashboard(self, current_price, preds, confs, final_action, info, regime, elite_sigs, timestamp, ls_action=None, ls_info=None, ls_pnl=None, ls_pos=None):
        pnl_pct = (self.current_equity - 1.0) * 100
        regime_name = next((k.replace('regime_', '').upper() for k, v in regime.items() if v == 1.0), 'UNKNOWN')
        active_agent = info.get('agent', 'NONE')
        kelly = info.get('kelly', 0.0)
        
        # 💡 [수정] 변수명을 long_kelly -> long_edge로 맞추고, 해석용 함수 추가
        long_edge = info.get('long_edge', 0.0)
        short_edge = info.get('short_edge', 0.0)

        def format_edge(edge):
            if edge > 0.01: return f"{Colors.GREEN}{edge:+.3f} (진입희망){Colors.RESET}"
            elif edge < -0.01: return f"{Colors.RED}{edge:+.3f} (진입거부){Colors.RESET}"
            else: return f"{Colors.YELLOW}{edge:+.3f} (완전중립){Colors.RESET}"

        action_str = {0: f'{Colors.YELLOW}🟨 관망 / 청산 (HOLD / CLOSE){Colors.RESET}', 
                      1: f'{Colors.GREEN}🟩 롱 진입 (LONG){Colors.RESET}', 
                      2: f'{Colors.RED}🟥 숏 진입 (SHORT){Colors.RESET}'}.get(final_action, '?')
        
        pos_str = {'LONG': f'{Colors.GREEN}🟩 LONG 보유{Colors.RESET}', 
                   'SHORT': f'{Colors.RED}🟥 SHORT 보유{Colors.RESET}', 
                   None: f'{Colors.YELLOW}🟨 무포지션{Colors.RESET}'}.get(self.pos, '?')

        print("\n" + Colors.BOLD + "╔════════════════════ [ 12-Agent MoE 사령관 대시보드 ] ════════════════════╗" + Colors.RESET)
        print(f" ⏱️ 타임: {timestamp.strftime('%Y-%m-%d %H:%M')} KST | 💰 ETH: ${current_price:,.2f}")
        
        if self.pos is not None:
            pnl_color = Colors.GREEN if pnl_pct > 0 else Colors.RED
            print(f" ⏳ 보유 캔들: {self.hold_count:<4} | 📈 미실현 수익: {pnl_color}{pnl_pct:+.2f}%{Colors.RESET}")
        
        print("-" * 68)
        print(f" {Colors.CYAN}[ 🧠 6대 파운데이션 AI 앙상블 예측 ]{Colors.RESET}")
        for i, model_name in enumerate(self.MODEL_ORDER):
            pred_dir = f"{Colors.GREEN}상승(L){Colors.RESET}" if preds[i] > 0 else f"{Colors.RED}하락(S){Colors.RESET}" if preds[i] < 0 else f"{Colors.YELLOW}중립(-){Colors.RESET}"
            print(f"  • {model_name:<10} : {pred_dir:<15} (신뢰도: {confs[i]:.1%})")

        print("-" * 68)
        print(f" {Colors.YELLOW}[ ⚔️ 13대 엘리트 퀀트 시그널 분석 ]{Colors.RESET}")
        
        sig_interpretations = {
            'whale': ('고래 매수 다이버전스 포착', '고래 매도 다이버전스 포착', '고래 움직임 관망'),
            'liq_squeeze': ('숏 청산 스퀴즈 위협 (롱 반등)', '롱 청산 스퀴즈 위협 (숏 반등)', '청산 스퀴즈 위험 낮음'),
            'net_taker': ('시장가 매수 쏠림 강함', '시장가 매도 쏠림 강함', '시장가 매수/매도 균형'),
            'orderblock': ('지지 매물대(FVG) 강한 반등', '저항 매물대(FVG) 강한 거절', '주요 오더블록 미형성'),
            'hurst_ofi': ('프랙탈 상승 오더플로우 지속', '프랙탈 하락 오더플로우 지속', '랜덤워크/혼조세 (방향성 없음)'),
            'funding_cascade': ('펀딩비 극단적 음수 (숏 청산대기)', '펀딩비 극단적 양수 (롱 청산대기)', '펀딩비/미결제약정 안정적'),
            'multifractal': ('장기 노이즈 캔슬링: 상승', '장기 노이즈 캔슬링: 하락', '추세 노이즈 상쇄됨'),
            'cluster_fib': ('피보나치+CVP 클러스터 지지', '피보나치+CVP 클러스터 저항', '주요 방어선 부재'),
            'oi_divergence': ('하락 중 OI 증가 (숏 스퀴즈 잠재)', '상승 중 OI 증가 (롱 스퀴즈 잠재)', 'OI와 가격 동조화'),
            'top_trader_squeeze': ('탑트레이더 숏 쏠림 (상승폭발 대기)', '탑트레이더 롱 쏠림 (하락폭발 대기)', '탑트레이더 포지션 균형'),
            'btc_corr_breakout': ('BTC 커플링 이탈 (독자 상승)', 'BTC 커플링 이탈 (독자 하락)', 'BTC 방향성 강력 동조'),
            'ai_squeeze': ('변동성 응축 후 상방 폭발', '변동성 응축 후 하방 폭발', '변동성 평이함 (응축 없음)'),
            'vp_gravity': ('POC(매물대) 하단 이탈 후 탄성 반등', 'POC(매물대) 상단 이탈 후 탄성 하락', '최다 매물대(POC) 부근 체류')
        }

        # 💡 [핵심 추가] 신호 강도(절댓값)를 기준으로 내림차순 정렬 (쎈 놈부터 위로, 0은 맨 아래로)
        sorted_elite_sigs = sorted(elite_sigs.items(), key=lambda item: abs(item[1]), reverse=True)

        for k, v in sorted_elite_sigs:
            base_key = k.replace('sig_', '')
            if base_key in sig_interpretations:
                msg_long, msg_short, msg_none = sig_interpretations[base_key]
                if v > 0:
                    interp = f"{Colors.GREEN}{msg_long:<25}{Colors.RESET}"
                    icon = f"{Colors.GREEN}▲{Colors.RESET}"
                elif v < 0:
                    interp = f"{Colors.RED}{msg_short:<25}{Colors.RESET}"
                    icon = f"{Colors.RED}▼{Colors.RESET}"
                else:
                    interp = f"{Colors.YELLOW}{msg_none:<25}{Colors.RESET}"
                    icon = f"{Colors.YELLOW}-{Colors.RESET}"
            else:
                interp = f"{Colors.GREEN}LONG{Colors.RESET}" if v > 0 else f"{Colors.RED}SHORT{Colors.RESET}" if v < 0 else f"{Colors.YELLOW}NONE{Colors.RESET}"
                icon = f"{Colors.GREEN}▲{Colors.RESET}" if v > 0 else f"{Colors.RED}▼{Colors.RESET}" if v < 0 else f"{Colors.YELLOW}-{Colors.RESET}"

            print(f"  {icon} {base_key:<20} : {interp} (강도: {v:+.2f})")

        print("-" * 68)
        print(f" {Colors.CYAN}[ 🤖 LR MoE 12-Agent 독립 판단 ]{Colors.RESET}")
        print(f" 🌍 시장 레짐: {regime_name:<10} | 📊 현재 포지션: {pos_str}")
        print(f" ⚖️ [현재 국면 엣지 비교] {Colors.GREEN}롱돌이(L): {long_edge:.3f}{Colors.RESET} vs {Colors.RED}숏돌이(S): {short_edge:.3f}{Colors.RESET}")
        print(f" 🤖 담당 특수부대: {active_agent:<15} | 🎯 선택된 Kelly: {kelly:.3f}")
        print(f" 🎯 최종 결단: {action_str}")

        # ── LS 2-Agent 섹션 ────────────────────────────────────────────────
        if ls_action is not None:
            print("-" * 68)
            print(f" {Colors.CYAN}[ 🤖 LS 2-Agent 롱돌이/숏돌이 독립 판단 ]{Colors.RESET}")
            ls_action_str = {
                0: f'{Colors.YELLOW}🟨 관망 / 청산{Colors.RESET}',
                1: f'{Colors.GREEN}🟩 롱 진입 (Long){Colors.RESET}',
                2: f'{Colors.RED}🟥 숏 진입 (Short){Colors.RESET}'
            }.get(ls_action, '?')
            ls_agent_name = (ls_info or {}).get('agent', 'N/A')
            ls_adv = (ls_info or {}).get('adv', None)
            ls_pos_str = {'LONG': f'{Colors.GREEN}🟩 LONG{Colors.RESET}', 'SHORT': f'{Colors.RED}🟥 SHORT{Colors.RESET}', None: f'{Colors.YELLOW}무포지션{Colors.RESET}'}.get(ls_pos, f'{Colors.YELLOW}무포지션{Colors.RESET}')
            adv_str = f" | adv: {ls_adv:+.4f}" if ls_adv is not None else ""
            pnl_str = ""
            if ls_pnl is not None:
                pnl_color = Colors.GREEN if ls_pnl > 0 else (Colors.RED if ls_pnl < 0 else Colors.YELLOW)
                pnl_str = f" | 미실현: {pnl_color}{ls_pnl:+.2f}%{Colors.RESET}"
            print(f"  포지션: {ls_pos_str}{pnl_str}")
            print(f"  에이전트: {ls_agent_name}{adv_str}")
            print(f"  결단: {ls_action_str}")

        print(Colors.BOLD + "╚════════════════════════════════════════════════════════════════════════╝\n" + Colors.RESET)

# ════════════════════════════════════════════════════════════════
# 4. 비동기 메인 루프 (즉시 분석 및 실시간 롤링)
# ════════════════════════════════════════════════════════════════
async def main(use_local=False):
    fetcher = BinanceLiveFetcher(limit=2500)
    fe_engine = FeatureEngineer()
    ensemble = EnsemblePredictor()
    
    try:
        # PPO 봇 대신 MoE 라우터 탑재!
        bot = MoELiveRouter('data/ensemble/best_moe_agents.pth')
    except Exception as e:
        logger.error(f"❌ MoE 라우터 초기화 실패: {e}")
        return

    # LS 2-Agent 라우터 (미학습 시 None으로 유지)
    ls_bot = None
    try:
        ls_bot = LSLiveRouter('data/ensemble/best_ls_agents.pth')
    except Exception as e:
        logger.warning(f"⚠️ LS 라우터 초기화 실패 (미학습 상태일 수 있음): {e}")

    try:
        # 💡 [초기화] 실시간 또는 로컬 데이터 로드
        if use_local:
            eth_buffer, btc_buffer = fetcher.load_local_data()
        else:
            logger.info("초기 캔들 데이터 수집 중...")
            eth_buffer, btc_buffer = await fetcher.fetch_initial_data()
        
        if eth_buffer is None: return

        # 💡 [최초 분석] 대기 없이 즉시 1회 실행
        processed_df = fe_engine.process(eth_buffer, btc_buffer)
        preds, confs = await ensemble.predict_all_async(processed_df)

        final_action, info, elite_sigs, regime = bot.get_signal(processed_df, preds, confs)
        current_time_kst = eth_buffer['timestamp'].iloc[-1] + pd.Timedelta(hours=9)
        current_price = float(eth_buffer['close'].iloc[-1])

        ls_action, ls_info, ls_pnl = (None, None, None)
        if ls_bot is not None:
            ls_action, ls_info, ls_pnl = ls_bot.get_signal(processed_df, preds, confs)

        bot.print_dashboard(current_price, preds, confs, final_action, info, regime, elite_sigs, current_time_kst,
                            ls_action=ls_action, ls_info=ls_info, ls_pnl=ls_pnl, ls_pos=ls_bot.pos if ls_bot else None)

        # 💡 [실시간 루프] 로컬 모드가 아닐 때만 롤링 시작
        first_run = True
        while not use_local:
            if not first_run:
                # 5분봉 기준 다음 캔들 오픈 시간까지 대기
                now = time.time()
                wait_sec = int(max(0, (now - (now % 300) + 300 + 2) - now))
                for r in range(wait_sec, 0, -1):
                    sys.stdout.write(f"\r{Colors.CYAN}⏳ 다음 5분봉까지 대기 중... ({r}초 남음)   {Colors.RESET}")
                    sys.stdout.flush()
                    await asyncio.sleep(1)
                
                print()
                logger.info("🔄 최신 캔들 데이터를 갱신합니다.")
                new_eth, new_btc = await fetcher.fetch_latest_patch()
                eth_buffer = pd.concat([eth_buffer, new_eth]).drop_duplicates('timestamp').tail(2500)
                btc_buffer = pd.concat([btc_buffer, new_btc]).drop_duplicates('timestamp').tail(2500)
            else:
                logger.info(f"{Colors.GREEN}🚀 봇 실시간 롤링 가동 시작!{Colors.RESET}")
                first_run = False

            # 새 데이터로 분석
            processed_df = fe_engine.process(eth_buffer, btc_buffer)
            preds, confs = await ensemble.predict_all_async(processed_df)

            final_action, info, elite_sigs, regime = bot.get_signal(processed_df, preds, confs)
            current_time_kst = eth_buffer['timestamp'].iloc[-1] + pd.Timedelta(hours=9)
            current_price = float(eth_buffer['close'].iloc[-1])

            ls_action, ls_info, ls_pnl = (None, None, None)
            if ls_bot is not None:
                ls_action, ls_info, ls_pnl = ls_bot.get_signal(processed_df, preds, confs)

            bot.print_dashboard(current_price, preds, confs, final_action, info, regime, elite_sigs, current_time_kst,
                                ls_action=ls_action, ls_info=ls_info, ls_pnl=ls_pnl, ls_pos=ls_bot.pos if ls_bot else None)

    finally:
        await fetcher.exchange.close()

if __name__ == "__main__":
    # 실시간 롤링 분석을 위해 use_local=False로 실행
    asyncio.run(main(use_local=False))