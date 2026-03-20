"""
Quant Visualizer: LS(2-Agent) & RL(4-Agent) Backtesting & Visualization Report
================================================================================
--mode ls : best_ls_agents.pth (GatingRouter 기반 2-Agent)
--mode rl : best_rl_agents.pth  (GatingRouter5 기반 4-Agent)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, _SCRIPT_DIR]:
    if _p not in sys.path: sys.path.insert(0, _p)

import torch

# ─── 공통 경로 ───────────────────────────────────────────────────────────────
CSV_PATH   = 'data/ensemble/rl_training_data_full.csv'
REPORT_DIR = 'data/ensemble/reports'


# ─── 시뮬레이션 공통 루프 ────────────────────────────────────────────────────
def run_simulation(router, val_env, df_val):
    obs  = val_env.reset()
    done = False
    logs = []

    while not done:
        idx      = val_env.current_step
        feat     = df_val.iloc[idx].to_dict()
        pos_info = {
            'type':        val_env.pos,
            'entry_price': val_env.entry_price,
            'unrealized':  val_env.unrealized_pnl,
            'mdd':         val_env.max_drawdown,
            'hold_norm':   val_env.hold_count / 144,
        }
        prev_pos = val_env.pos

        action, leverage_rate, info = router.decide(feat, pos_info)
        obs, reward, done, env_info = val_env.step(action, leverage_rate=leverage_rate)

        trade_marker = None
        if   prev_pos is None and val_env.pos == 'LONG':  trade_marker = 'BUY'
        elif prev_pos is None and val_env.pos == 'SHORT': trade_marker = 'SELL'
        elif prev_pos is not None and val_env.pos is None: trade_marker = 'EXIT'

        logs.append({
            'timestamp':      df_val.iloc[idx]['timestamp'],
            'close':          df_val.iloc[idx]['close'],
            'pnl_pct':        env_info['pnl_pct'],
            'trade':          trade_marker,
            'agent':          info.get('agent', 'FLAT'),
            'regime_bull':    feat.get('regime_bull',    0),
            'regime_bear':    feat.get('regime_bear',    0),
            'regime_chop':    feat.get('regime_chop',    0),
            'regime_whipsaw': feat.get('regime_whipsaw', 0),
        })

    return pd.DataFrame(logs)


# ─── 차트 출력 ───────────────────────────────────────────────────────────────
def plot_report(log_df, val_env, mode_label, save_path):
    final_pnl = log_df['pnl_pct'].iloc[-1]
    wr        = val_env.win_rate * 100
    trades    = val_env.total_trades
    running_max = log_df['pnl_pct'].cummax()
    account_dd  = log_df['pnl_pct'] - running_max
    max_dd      = account_dd.min()
    pnl_arr     = log_df['pnl_pct'].values
    sharpe      = float(np.mean(np.diff(pnl_arr)) / (np.std(np.diff(pnl_arr)) + 1e-8) * np.sqrt(288))

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    title = (
        f'[{mode_label}] Return: {final_pnl:.2f}%  |  '
        f'Trades: {trades}  |  WR: {wr:.0f}%  |  '
        f'MDD: {max_dd:.2f}%  |  Sharpe: {sharpe:.2f}'
    )
    ax1.set_title(title, fontsize=13, fontweight='bold', pad=8)

    # ── 가격 + 레짐 배경 ──────────────────────────────────────────────────────
    ax1.plot(log_df['timestamp'], log_df['close'], color='white', alpha=0.65, linewidth=1)
    ax1.set_ylabel('Price (USDT)', fontsize=11)

    ylo, yhi = log_df['close'].min(), log_df['close'].max()
    ax1.fill_between(log_df['timestamp'], ylo, yhi,
                     where=log_df['regime_bull'] == 1,
                     facecolor='green', alpha=0.10, label='Bull')
    ax1.fill_between(log_df['timestamp'], ylo, yhi,
                     where=log_df['regime_bear'] == 1,
                     facecolor='red',   alpha=0.10, label='Bear')
    ax1.fill_between(log_df['timestamp'], ylo, yhi,
                     where=(log_df['regime_chop'] == 1) | (log_df['regime_whipsaw'] == 1),
                     facecolor='purple', alpha=0.10, label='Chop/Whipsaw')

    # ── 타점 마커 ─────────────────────────────────────────────────────────────
    buys  = log_df[log_df['trade'] == 'BUY']
    sells = log_df[log_df['trade'] == 'SELL']
    exits = log_df[log_df['trade'] == 'EXIT']

    if not buys.empty:
        ax1.scatter(buys['timestamp'],  buys['close']  * 0.998, marker='^', color='lime',   s=90, label='Long Entry',  zorder=5)
    if not sells.empty:
        ax1.scatter(sells['timestamp'], sells['close'] * 1.002, marker='v', color='red',    s=90, label='Short Entry', zorder=5)
    if not exits.empty:
        ax1.scatter(exits['timestamp'], exits['close'],          marker='x', color='yellow', s=60, label='Exit',        zorder=5)

    ax1.legend(loc='upper left', ncol=7, fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.2)

    # ── 누적 PnL ──────────────────────────────────────────────────────────────
    ax2.plot(log_df['timestamp'], log_df['pnl_pct'], color='cyan', linewidth=1.8)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(log_df['timestamp'], log_df['pnl_pct'], 0,
                     where=log_df['pnl_pct'] >= 0, facecolor='cyan',  alpha=0.10)
    ax2.fill_between(log_df['timestamp'], log_df['pnl_pct'], 0,
                     where=log_df['pnl_pct'] <  0, facecolor='red',   alpha=0.10)
    ax2.set_ylabel('PnL (%)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.2)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax3.plot(log_df['timestamp'], account_dd, color='magenta', linewidth=1.4)
    ax3.fill_between(log_df['timestamp'], account_dd, 0, facecolor='magenta', alpha=0.18)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Drawdown (%)', fontsize=11)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.2)

    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"📊 차트 저장: {save_path}")


# ─── LS 2-Agent 평가 ─────────────────────────────────────────────────────────
def evaluate_ls():
    from ensemble.train_ls_agent import (
        TradingEnv, RobustIQN, GatingNet, GatingRouter, STATE_DIM
    )

    BEST_PATH  = 'data/ensemble/best_ls_agents.pth'
    CKPT_PATH  = 'data/ensemble/ls_checkpoint.pth'
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(BEST_PATH):
        print(f"❌ 모델 없음: {BEST_PATH}"); return

    df        = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_val    = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)

    # 모델 로드
    ckpt_best = torch.load(BEST_PATH, map_location=device, weights_only=False)
    model_long  = RobustIQN(STATE_DIM, 2).to(device)
    model_short = RobustIQN(STATE_DIM, 2).to(device)
    model_long.load_state_dict(ckpt_best['model_long'],  strict=False)
    model_short.load_state_dict(ckpt_best['model_short'], strict=False)
    model_long.eval(); model_short.eval()

    # GatingNet: ls_checkpoint에 있으면 복원, 없으면 새로 초기화
    gating_net = GatingNet(STATE_DIM).to(device)
    if os.path.exists(CKPT_PATH):
        ckpt_full = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        if 'gating_net' in ckpt_full:
            gating_net.load_state_dict(ckpt_full['gating_net'], strict=False)
            print(f"✅ GatingNet 복원 완료 (ep={ckpt_full.get('epoch','?')})")
        else:
            print("⚠️  ls_checkpoint에 gating_net 없음 → 새로 초기화")
    else:
        print("⚠️  ls_checkpoint 없음 → GatingNet 새로 초기화")
    gating_net.eval()

    ep   = ckpt_best.get('epoch', '?')
    pnl  = ckpt_best.get('best_pnl', 0.0)
    print(f"✅ LS 모델 로드 (ep={ep}, best_val_pnl={pnl:.2f}%)")

    router  = GatingRouter(model_long, model_short, gating_net, device)
    val_env = TradingEnv(df_val, phase='val', agent_role='neutral', fee=0.0005)

    print("🚀 LS 시뮬레이션 시작...")
    log_df = run_simulation(router, val_env, df_val)
    print(f"🏁 완료 | PnL: {log_df['pnl_pct'].iloc[-1]:.2f}% | "
          f"Tr: {val_env.total_trades} | WR: {val_env.win_rate*100:.0f}%")

    plot_report(log_df, val_env,
                mode_label=f'LS 2-Agent | ep={ep}',
                save_path=f'{REPORT_DIR}/backtest_ls.png')


# ─── RL 4-Agent 평가 ─────────────────────────────────────────────────────────
def evaluate_rl():
    from ensemble.train_rl_agent import (
        TradingEnv, RobustIQN, GatingNet5, GatingRouter5, STATE_DIM
    )

    BEST_PATH = 'data/ensemble/ckpt/best_rl_agents.pth'
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(BEST_PATH):
        print(f"❌ 모델 없음: {BEST_PATH}"); return

    df        = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_val    = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)

    agent_configs = {
        'bull':   {'action_dim': 2},
        'bear':   {'action_dim': 2},
        'chop':   {'action_dim': 3},
        'normal': {'action_dim': 3},
    }

    ckpt   = torch.load(BEST_PATH, map_location=device, weights_only=False)
    models = {}
    for name, cfg in agent_configs.items():
        m = RobustIQN(STATE_DIM, cfg['action_dim']).to(device)
        key = f'model_{name}'
        if key in ckpt:
            m.load_state_dict(ckpt[key], strict=False)
        else:
            print(f"⚠️  체크포인트에 '{key}' 없음 → 랜덤 초기화")
        m.eval()
        models[name] = m

    gating_net = GatingNet5(STATE_DIM).to(device)
    if 'gating_net' in ckpt:
        gating_net.load_state_dict(ckpt['gating_net'], strict=False)
        print("✅ GatingNet5 복원 완료")
    else:
        print("⚠️  체크포인트에 gating_net 없음 → 새로 초기화")
    gating_net.eval()

    ep  = ckpt.get('epoch', '?')
    pnl = ckpt.get('best_pnl', 0.0)
    print(f"✅ RL 모델 로드 (ep={ep}, best_val_pnl={pnl:.2f}%)")

    router  = GatingRouter5(models, gating_net, device)
    val_env = TradingEnv(df_val, phase='val', agent_role='neutral', fee=0.0005)

    print("🚀 RL 시뮬레이션 시작...")
    log_df = run_simulation(router, val_env, df_val)
    print(f"🏁 완료 | PnL: {log_df['pnl_pct'].iloc[-1]:.2f}% | "
          f"Tr: {val_env.total_trades} | WR: {val_env.win_rate*100:.0f}%")

    plot_report(log_df, val_env,
                mode_label=f'RL 5-Way GatingRouter | ep={ep}',
                save_path=f'{REPORT_DIR}/backtest_rl.png')


# ─── 진입점 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ls', 'rl', 'both'], default='both',
                        help='ls=2-Agent GatingRouter | rl=4-Agent MoE | both=둘 다')
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        print(f"❌ 데이터 없음: {CSV_PATH}"); sys.exit(1)

    if args.mode in ('ls', 'both'):
        evaluate_ls()
    if args.mode in ('rl', 'both'):
        evaluate_rl()
