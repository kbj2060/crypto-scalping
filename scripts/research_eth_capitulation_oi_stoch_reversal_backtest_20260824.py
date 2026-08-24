#!/usr/bin/env python3
"""'캡츄레이션(대규모 청산)+OI급감+StochRSI 골든/데드크로스' 전략 fresh-forward 백테스트
(2026-08-24, 사용자 제안 전략)

⚠️ 실제 forceOrder 청산 이벤트 피드(data/live/tail_risk.duckdb)는 유효 데이터가
2026-07-18부터뿐(5주)이고, 이 정확한 메커니즘(청산 컨트래리언)이 이미 §12/§13으로
09-15까지 실행 보류가 사전등록돼 있어 여기서 쓰면 사전등록 오염이다. 조건1이 "청산맵이나
청산 데이터 피드"로 OR 표현돼 있어, 대신 가격/거래량 기반 캡츄레이션 프록시(급락+거래량
스파이크+해머형 되돌림, 30분 윈도우)를 쓴다. OI는 실제 감사된 5분 OI(TOTAL_ETHUSDT_metrics,
+5분 종료라벨 보정본)를 그대로 쓴다 — 이건 청산 피드가 아니라 이미 08-23 무결성 대수술로
검증된 별개 데이터라 게이트 대상 아님.

조건:
  1. 캡츄레이션(프록시): 최근 6봉(30분) 구간에서 급락(30분 로그수익 z<=-3)+거래량 스파이크
     (6봉합 거래량 z>=3, 트레일링 7일 기준)+해머형 되돌림(종가가 구간 저가~고가의 상단 50%
     이상 회복). 숏은 대칭(급등+상단꼬리 되돌림).
  2. OI 급감: 같은 6봉 구간의 OI 변화율 z<=-2 (실제 5분 OI, causal 트레일링 7일 z-score).
  3. StochRSI(14,14,3,3) 트리거 — 캡츄레이션 이후 2시간(24봉) 이내에:
     strict: K가 D를 상향돌파하는 바로 그 봉에서 K도 20을 동시에 상향돌파
     loose : K/D 골든크로스 발생 시점에 K>=20이고, 직전 1시간 내 K<20이 있었음(깊은 과매도 확인)
     숏은 데드크로스+80 대칭.
  진입: 트리거 다음 봉 시가. SL=캡츄레이션 구간 저가/고가(+0.1% 버퍼). TP=R배수(1.5/2/3, 그리드
  전체 보고). 24h 시간청산. 단일 포지션. 비용=표준 테이커 왕복 10bp.

Fresh-forward: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
결과: data/research/eth_capitulation_oi_stoch_reversal_backtest_20260824.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "research" / "eth_capitulation_oi_stoch_reversal_backtest_20260824.json"

_spec = importlib.util.spec_from_file_location(
    "rb", REPO / "scripts" / "research_eth_discretionary_rulebook_v1_freshforward_backtest_20260824.py")
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)

CAP_WIN = 6            # 30분 캡츄레이션 판정 윈도우
RET_Z_TH = 3.0
VOL_Z_TH = 3.0
OI_Z_TH = 2.0
WICK_RECOVERY = 0.5
TRIG_WIN_BARS = 24      # 트리거 대기 2시간
Z_WINDOW = 2016         # 7일 트레일링 (5분봉)
SL_BUFFER = 0.001
MAX_HOLD_BARS = 288      # 24h
COST_RT_TAKER = 0.0010


def load_oi() -> pd.Series:
    a = pd.read_csv(REPO / "data" / "TOTAL_ETHUSDT_metrics_2021_2023.csv")
    b = pd.read_csv(REPO / "data" / "TOTAL_ETHUSDT_metrics_2024_2026.csv")
    df = pd.concat([a, b], ignore_index=True)
    df["create_time"] = pd.to_datetime(df["create_time"])
    df = df.drop_duplicates(subset="create_time").sort_values("create_time")
    return df.set_index("create_time")["sum_open_interest"]


def causal_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """NaN을 보존한 채로 rolling mean/std (pandas는 윈도우 내 NaN을 자동 스킵) —
    0으로 채우면 롤링 통계가 오염된다(oi==0 오염치 208개가 실제로 이 버그를 일으켰음)."""
    s = pd.Series(x)
    mu = s.rolling(window, min_periods=window // 4).mean()
    sd = s.rolling(window, min_periods=window // 4).std()
    return ((s - mu) / sd.replace(0, np.nan)).to_numpy()


def build_frame() -> pd.DataFrame:
    df = _rb.load_5m()
    oi = load_oi()
    # end-label 보정: ts(bar open) 봉의 종가 시점 OI = create_time(ts+5min)
    oi_at_close = oi.reindex(df["ts"] + pd.Timedelta(minutes=5))
    df["oi"] = oi_at_close.to_numpy()

    c, o, h, lo, v = (df[k].to_numpy() for k in ("close", "open", "high", "low", "volume"))
    n = len(df)

    win_low = pd.Series(lo).rolling(CAP_WIN).min().to_numpy()
    win_high = pd.Series(h).rolling(CAP_WIN).max().to_numpy()
    win_vol = pd.Series(v).rolling(CAP_WIN).sum().to_numpy()
    ret30 = np.full(n, np.nan)
    ret30[CAP_WIN:] = np.log(c[CAP_WIN:]) - np.log(c[:-CAP_WIN])

    ret_z = causal_zscore(ret30, Z_WINDOW)
    vol_z = causal_zscore(win_vol, Z_WINDOW)
    oi = df["oi"].to_numpy()
    oi_clean = np.where(oi > 0, oi, np.nan)  # oi<=0은 오염치(208개 발견) — NaN 처리
    oi_chg = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        oi_chg[CAP_WIN:] = (oi_clean[CAP_WIN:] - oi_clean[:-CAP_WIN]) / oi_clean[:-CAP_WIN]
    oi_z = causal_zscore(oi_chg, Z_WINDOW)

    rng = np.maximum(win_high - win_low, 1e-9)
    recov_long = (c - win_low) / rng     # 종가가 구간 상단에 가까울수록 1
    recov_short = (win_high - c) / rng

    df["win_low"] = win_low
    df["win_high"] = win_high
    df["ret_z"] = ret_z
    df["vol_z"] = vol_z
    df["oi_z"] = oi_z
    df["recov_long"] = recov_long
    df["recov_short"] = recov_short
    df["cap_long"] = ((ret_z <= -RET_Z_TH) & (vol_z >= VOL_Z_TH)
                      & (oi_z <= -OI_Z_TH) & (recov_long >= WICK_RECOVERY))
    df["cap_short"] = ((ret_z >= RET_Z_TH) & (vol_z >= VOL_Z_TH)
                       & (oi_z <= -OI_Z_TH) & (recov_short >= WICK_RECOVERY))
    return df


def run(df: pd.DataFrame, trigger_mode: str, r_mult: float) -> list:
    c = df["close"].to_numpy(); o = df["open"].to_numpy()
    h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    win_low = df["win_low"].to_numpy(); win_high = df["win_high"].to_numpy()
    cap_long = df["cap_long"].to_numpy(); cap_short = df["cap_short"].to_numpy()
    ts = df["ts"].to_numpy()
    k = _rb.stoch_rsi_k(c)
    d = pd.Series(k).rolling(3).mean().to_numpy()  # %D = %K의 3봉 SMA (표준)

    gold = np.zeros(len(c), dtype=bool)
    dead = np.zeros(len(c), dtype=bool)
    gold[1:] = (k[:-1] < d[:-1]) & (k[1:] >= d[1:])
    dead[1:] = (k[:-1] > d[:-1]) & (k[1:] <= d[1:])

    if trigger_mode == "strict":
        long_trig = gold & np.concatenate([[False], (k[:-1] < 20) & (k[1:] >= 20)])
        short_trig = dead & np.concatenate([[False], (k[:-1] > 80) & (k[1:] <= 80)])
    else:
        was_oversold = pd.Series(k < 20).rolling(12, min_periods=1).max().astype(bool).to_numpy()
        was_overbought = pd.Series(k > 80).rolling(12, min_periods=1).max().astype(bool).to_numpy()
        long_trig = gold & (k >= 20) & np.concatenate([[False], was_oversold[:-1]])
        short_trig = dead & (k <= 80) & np.concatenate([[False], was_overbought[:-1]])

    n = len(c)
    trades = []
    pos = None
    pending = None  # (dir, cap_i, deadline_i, sl_ref)
    for i in range(n - 1):
        if pos is not None:
            if pos["dir"] == 1:
                hit_sl, hit_tp = lo[i] <= pos["sl"], h[i] >= pos["tp"]
            else:
                hit_sl, hit_tp = h[i] >= pos["sl"], lo[i] <= pos["tp"]
            exit_price = exit_kind = None
            if hit_sl:
                exit_price, exit_kind = pos["sl"], "sl"
            elif hit_tp:
                exit_price, exit_kind = pos["tp"], "tp"
            elif i - pos["entry_i"] >= MAX_HOLD_BARS:
                exit_price, exit_kind = c[i], "time"
            if exit_price is not None:
                gross = pos["dir"] * (exit_price / pos["entry"] - 1.0)
                trades.append({"entry_ts": str(ts[pos["entry_i"]]), "exit_ts": str(ts[i]),
                               "dir": "long" if pos["dir"] == 1 else "short",
                               "exit_kind": exit_kind, "hold_bars": int(i - pos["entry_i"]),
                               "gross": gross, "net_taker": gross - COST_RT_TAKER})
                pos = None
            continue

        if pending is not None:
            pdir, cap_i, deadline, sl_ref = pending
            fired = long_trig[i] if pdir == 1 else short_trig[i]
            if fired:
                entry = o[i + 1]
                sl = sl_ref * (1 - SL_BUFFER) if pdir == 1 else sl_ref * (1 + SL_BUFFER)
                risk = abs(entry - sl)
                tp = entry + pdir * r_mult * risk
                pos = {"dir": pdir, "entry": entry, "sl": sl, "tp": tp, "entry_i": i + 1}
                pending = None
            elif i >= deadline:
                pending = None

        if pending is None:
            if cap_long[i]:
                pending = (1, i, i + TRIG_WIN_BARS, win_low[i])
            elif cap_short[i]:
                pending = (-1, i, i + TRIG_WIN_BARS, win_high[i])
    return trades


def bootstrap_unconditional_long(df: pd.DataFrame, n_trades: int, r_mult: float,
                                 n_boot: int = 500, seed: int = 0) -> dict:
    """확증 필수 대조군: 같은 SL/TP/24h청산 기하학을 '캡츄레이션 조건' 없이 무작위 시점에
    적용했을 때도 비슷한 gross가 나오면, 신호가 아니라 이 매매 구조(짧은 SL+긴 보유+강세장
    드리프트)가 만드는 착시다 — always-long 벤치마크 관행과 동일한 목적."""
    rng = np.random.default_rng(seed)
    c = df["close"].to_numpy(); o = df["open"].to_numpy()
    h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    win_low = df["win_low"].to_numpy()
    n = len(c)
    valid_lo = Z_WINDOW
    valid_hi = n - MAX_HOLD_BARS - 2
    boot_means = []
    for _ in range(n_boot):
        idxs = rng.integers(valid_lo, valid_hi, size=n_trades)
        grosses = []
        for i in idxs:
            entry = o[i + 1]
            sl = win_low[i] * (1 - SL_BUFFER)
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + r_mult * risk
            exit_price = c[min(i + MAX_HOLD_BARS, n - 1)]
            for j in range(i + 1, min(i + MAX_HOLD_BARS, n - 1)):
                if lo[j] <= sl:
                    exit_price = sl
                    break
                if h[j] >= tp:
                    exit_price = tp
                    break
            grosses.append(entry / entry * 0 + (exit_price / entry - 1.0))
        boot_means.append(float(np.mean(grosses)) if grosses else np.nan)
    boot_means = np.array(boot_means)
    return {"n_boot": n_boot, "n_trades_per_boot": n_trades,
           "boot_mean_gross_bp_mean": float(np.nanmean(boot_means) * 1e4),
           "boot_mean_gross_bp_std": float(np.nanstd(boot_means) * 1e4),
           "boot_mean_gross_bp_p5_p50_p95": [float(x * 1e4) for x in
                                              np.nanpercentile(boot_means, [5, 50, 95])]}


def main():
    df = build_frame()
    print(f"bars={len(df)}  cap_long_events={int(df['cap_long'].sum())}"
         f"  cap_short_events={int(df['cap_short'].sum())}")
    report = {
        "strategy": "capitulation(price/vol proxy)+OI급감(실제 5분 OI)+StochRSI(14,14,3,3) K/D+20/80 크로스",
        "note_on_liq_feed_substitution": "실제 forceOrder 청산피드는 09-15 사전등록 게이트 보류 대상이라 미사용, 가격/거래량 캡츄레이션 프록시로 대체(사용자 조건1의 OR 허용 문구 근거)",
        "note_on_risk_section": "사용자 메시지에 3.리스크관리(10x레버리지) 본문 미첨부 — R배수 그리드(1.5/2/3)로 대체, price-move 기준 gross/net bp가 1차 판정축(레버리지는 이미 존재하는 엣지를 배율할 뿐 엣지 유무를 안 바꿈)",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "cost_model_rt_taker_bp": 10.0,
        "params": {"cap_win_bars": CAP_WIN, "ret_z_th": RET_Z_TH, "vol_z_th": VOL_Z_TH,
                  "oi_z_th": OI_Z_TH, "wick_recovery": WICK_RECOVERY,
                  "trig_win_bars": TRIG_WIN_BARS, "z_trailing_window_bars": Z_WINDOW,
                  "sl_buffer": SL_BUFFER, "max_hold_bars": MAX_HOLD_BARS},
        "cap_long_events": int(df["cap_long"].sum()), "cap_short_events": int(df["cap_short"].sum()),
        "grid": {},
    }
    for trigger_mode in ("strict", "loose"):
        for r_mult in (1.5, 2.0, 3.0):
            trades = run(df, trigger_mode, r_mult)
            a = _agg(trades)
            key = f"{trigger_mode}_R{r_mult}"
            report["grid"][key] = {
                "all": a,
                "long": _agg([t for t in trades if t["dir"] == "long"]),
                "short": _agg([t for t in trades if t["dir"] == "short"]),
                "yearly": _yearly(trades),
            }
            print(f"{key}: n={a['n']} win={a.get('win_rate_net_taker', 0):.2f} "
                 f"avg_gross={a.get('avg_gross_bp', 0):.1f}bp "
                 f"sum_net={a.get('sum_net_taker_pct', 0):.1f}%")
    # 확증 게이트: loose_R2.0 롱(가장 유망해 보였던 셀, n=50)을 무조건부 랜덤진입 부트스트랩과 대조
    boot = bootstrap_unconditional_long(df, n_trades=50, r_mult=2.0, n_boot=500)
    report["confound_check_unconditional_long_bootstrap_vs_loose_R2.0_long"] = boot
    observed_gross_bp = report["grid"]["loose_R2.0"]["long"]["avg_gross_bp"]
    print(f"\n[대조군] 무조건부 랜덤진입 500회 부트스트랩(같은 SL/TP/24h): "
         f"평균 gross {boot['boot_mean_gross_bp_mean']:.1f}bp "
         f"(p5~p95: {boot['boot_mean_gross_bp_p5_p50_p95'][0]:.1f}~"
         f"{boot['boot_mean_gross_bp_p5_p50_p95'][2]:.1f}bp)")
    print(f"[관측값] loose_R2.0 롱(캡츄레이션 조건부) gross {observed_gross_bp:.1f}bp")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved -> {OUT}")


def _agg(sel: list) -> dict:
    if not sel:
        return {"n": 0}
    g = np.array([t["gross"] for t in sel]); nt = np.array([t["net_taker"] for t in sel])
    cum = np.cumsum(nt)
    mdd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
    kinds = {kd: int(sum(1 for t in sel if t["exit_kind"] == kd)) for kd in ("tp", "sl", "time")}
    return {"n": len(sel), "win_rate_net_taker": float((nt > 0).mean()),
           "avg_gross_bp": float(g.mean() * 1e4), "avg_net_taker_bp": float(nt.mean() * 1e4),
           "sum_net_taker_pct": float(nt.sum() * 100), "mdd_net_taker_pct": float(mdd * 100),
           "exit_kinds": kinds, "avg_hold_bars": float(np.mean([t["hold_bars"] for t in sel]))}


def _yearly(trades: list) -> dict:
    out = {}
    for y in ("2022", "2023", "2024", "2025", "2026"):
        sel = [t for t in trades if t["entry_ts"][:4] == y]
        out[y] = _agg(sel)
    return out


if __name__ == "__main__":
    main()
