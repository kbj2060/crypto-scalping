#!/usr/bin/env python3
"""
run_final_meta_ensemble.py — Final Evolution Meta-Ensemble Engine (V18)

[시스템 요약]
- 3개 모델 실시간 추론 (Gaussian/Sigmoid Actor 동기화)
- 마스터-슬레이브 환경 동기화 (Stateful Persistence)
- 슈미트 트리거 기반 비대칭 청산 로직 적용
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 원본 모듈 임포트
from ensemble.train_rl_dsac_agent import DSAC_STATE_DIM, GaussianActor, DSACCompactTradingEnv
from ensemble.train_rl_dsac_long_agent import SigmoidActor as LongActor, LongSpecialistEnv, STATE_DIM as LONG_STATE_DIM
from ensemble.train_rl_dsac_short_agent import SigmoidActor as ShortActor, ShortSpecialistEnv, STATE_DIM as SHORT_STATE_DIM
from ensemble.train_rl_agent import OnlineHMMDetector, MultiTimeframeFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("Final_Ensemble")

# ──────────────────────────────────────────────────────────────────────────────
# Config: 하이퍼 파라미터 (Hysteresis & Veto)
# ──────────────────────────────────────────────────────────────────────────────
ENTER_TH = 0.06    # 진입 임계값
EXIT_TH  = -0.02   # 청산 임계값 (반대 방향 컨빅션 확인 시 청산)
VETO_TH  = 0.35    # 스페셜리스트의 거부권 행사 강도

FEE_RATE = 0.0004
MAX_KELLY = 1.0

_DEFAULT_CSV = str(_ROOT / "data/splits/year_oos/rl_meta_2026.csv")
_CKPT_P = str(_ROOT / "data/ensemble/ckpt/best_dsac_agents.pth")
_CKPT_L = str(_ROOT / "data/ensemble/ckpt/best_dsac_long_agents.pth")
_CKPT_S = str(_ROOT / "data/ensemble/ckpt/best_dsac_short_agents.pth")

# =============================================================================
# 🧠 PAVM Router (Primary-Anchored Veto Matrix)
# =============================================================================
class FinalPAVMRouter:
    def decide(self, p, l, s):
        """
        p: Primary [-1, 1], l: Long [0, 1], s: Short [0, 1]
        """
        p_mag = abs(p)
        p_sign = 1 if p >= ENTER_TH else (-1 if p <= -ENTER_TH else 0)
        
        # 1. 기본 결정은 Primary를 따름
        master_dir = p_sign
        
        # 🚨 Rule 1: Divergence Veto (강력한 의견 충돌 시 회피)
        # Primary가 롱인데 숏 전문가가 강력하게 짖을 때
        if p_sign == 1 and s > VETO_TH:
            master_dir = 0
        # Primary가 숏인데 롱 전문가가 강력하게 짖을 때
        elif p_sign == -1 and l > VETO_TH:
            master_dir = 0
            
        # 🚀 Rule 2: Agreement Boost & Sniper
        # Primary가 데드존일 때 전문가가 확실한 타점을 잡는 경우
        if master_dir == 0:
            if l > 0.40: master_dir = 1
            elif s > 0.40: master_dir = -1

        return master_dir

# =============================================================================
# 🌍 Master Account (실제 수익 및 체결 관리)
# =============================================================================
class MasterAccount:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.t = 0
        self.pos = 0
        self.entry_price = 0.0
        self.trades = []
        self.fee = FEE_RATE
        
    def step_execution(self, master_dir, current_close):
        # 청산 및 스위칭 조건 체크 (Hysteresis 반영)
        # 롱 포지션일 때: 마스터 결정이 0 이하로 죽거나(Lost Conviction), 숏으로 바뀔 때
        if self.pos == 1:
            if master_dir == -1 or master_dir == 0: # 여기서는 단순화를 위해 0일 때도 청산 (또는 EXIT_TH 적용 가능)
                self._close(current_close)
        # 숏 포지션일 때
        elif self.pos == -1:
            if master_dir == 1 or master_dir == 0:
                self._close(current_close)
                
        # 신규 진입
        if self.pos == 0 and master_dir != 0:
            self.pos = master_dir
            self.entry_price = current_close
            
        self.t += 1
        return self.t >= len(self.df) - 1

    def _close(self, price):
        ret = (price - self.entry_price) / self.entry_price
        pnl = self.pos * ret - (self.fee * 2)
        self.trades.append(pnl)
        self.pos = 0
        self.entry_price = 0.0

    def get_metrics(self):
        if not self.trades: return {"pnl": 0.0, "wr": 0.0, "tr": 0}
        pnls = np.array(self.trades)
        return {
            "pnl": pnls.sum() * 100,
            "wr": (pnls > 0).mean() * 100,
            "tr": len(pnls),
            "sortino": pnls.mean() / (pnls[pnls<0].std() + 1e-8) if len(pnls[pnls<0])>0 else 0
        }

# =============================================================================
# 🚀 메인 추론 루프
# =============================================================================
def run_ensemble(csv_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Ensemble Engine Starting on {device}...")

    # 데이터 로드
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # 1. 전문가 환경 및 모델 로드
    hmm = OnlineHMMDetector()
    mtf = MultiTimeframeFeatures(df["close"].values.astype(np.float32))
    
    env_p = DSACCompactTradingEnv(df, hmm_detector=hmm, mtf_features=mtf, phase="val")
    env_l = LongSpecialistEnv(df, hmm_detector=hmm, mtf_features=mtf, phase="val")
    env_s = ShortSpecialistEnv(df, hmm_detector=hmm, mtf_features=mtf, phase="val")

    actor_p = GaussianActor(state_dim=DSAC_STATE_DIM).to(device)
    actor_l = LongActor(state_dim=LONG_STATE_DIM).to(device)
    actor_s = ShortActor(state_dim=SHORT_STATE_DIM).to(device)

    # 가중치 로드 (weights_only=False 필수)
    actor_p.load_state_dict(torch.load(_CKPT_P, map_location=device, weights_only=False)["actor"])
    actor_l.load_state_dict(torch.load(_CKPT_L, map_location=device, weights_only=False)["actor"])
    actor_s.load_state_dict(torch.load(_CKPT_S, map_location=device, weights_only=False)["actor"])
    
    actor_p.eval(); actor_l.eval(); actor_s.eval()

    # 2. 제어 객체
    router = FinalPAVMRouter()
    master = MasterAccount(df)
    
    # 초기 상태
    s_p = env_p.reset(); s_l = env_l.reset(); s_s = env_s.reset()
    done = False

    log.info(f"OOS Backtest Loop Start: {len(df)} candles")

    while not done:
        # (1) 실시간 추론
        with torch.no_grad():
            ts_p = torch.FloatTensor(s_p).unsqueeze(0).to(device)
            ts_l = torch.FloatTensor(s_l).unsqueeze(0).to(device)
            ts_s = torch.FloatTensor(s_s).unsqueeze(0).to(device)
            
            p_val = float(torch.tanh(actor_p.forward(ts_p)[0]).item())
            l_val = float(torch.sigmoid(actor_l.forward_logits(ts_l)[1]).item())
            s_val = float(torch.sigmoid(actor_s.forward_logits(ts_s)[1]).item())

        # (2) 메타 라우팅 결정
        master_dir = router.decide(p_val, l_val, s_val)
        
        # (3) 마스터 실행 및 시간 진행
        curr_close = df.iloc[master.t]["close"]
        done = master.step_execution(master_dir, curr_close)
        
        # (4) 행동 브로드캐스팅 (Stateful Sync)
        # 마스터의 포지션을 하위 환경에 주입하여 관성을 유지시킴
        act_p = float(master_dir)
        act_l = 1.0 if master_dir == 1 else 0.0
        act_s = 1.0 if master_dir == -1 else 0.0
        
        s_p, _, _, _ = env_p.step(act_p)
        s_l, _, _, _ = env_l.step(act_l)
        s_s, _, _, _ = env_s.step(act_s)

        if master.t % 10000 == 0:
            m = master.get_metrics()
            log.info(f"Step {master.t} | PnL: {m['pnl']:.2f}% | TR: {m['tr']}")

    # 결과 요약
    res = master.get_metrics()
    print(f"\n{'='*50}\n🏆 FINAL ENSEMBLE RESULT\n{'='*50}")
    print(f"Total PnL: {res['pnl']:+.2f}%")
    print(f"Win Rate:  {res['wr']:.1f}%")
    print(f"Trades:    {res['tr']}")
    print(f"Sortino:   {res['sortino']:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_ensemble(_DEFAULT_CSV)