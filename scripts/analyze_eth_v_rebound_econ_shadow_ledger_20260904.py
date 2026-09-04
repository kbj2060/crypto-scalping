#!/usr/bin/env python3
"""V자반등 경제라벨 섀도우 원장 분석 (2026-09-04).

무엇을 보나: ①원장 요약(측면·사유·일별·보유봉) ②오픈 포지션 시가평가 ③각 마감 트레이드를 실제 5분봉으로
`sim_exit` 재구성해 러너 회계와 대조(실데이터 L2P) ④진입가 규약(러너=사이클 마크가격 vs 백테스트=다음 봉 시가)의
슬리피지 ⑤"초기 원장은 승자 편향"인가 -- 백테스트 F0 OOS 체결열에서 무작위 시작점 뒤 같은 가동시간 안에 *마감된*
트레이드만 모은 귀무분포와 대조.
입력: data/live/v_rebound_econ_shadow_state.json (서버 pull), tmp/homer_entry_v2_20260904/trades_tabpfn_F0_{OOS,VAL}.csv,
      바이낸스 선물 5m klines (공개 REST).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
STATE = ROOT / "data/live/v_rebound_econ_shadow_state.json"
OUT = ROOT / "tmp/eth_v_rebound_econ_shadow_ledger_20260904"; OUT.mkdir(parents=True, exist_ok=True)
CELL = (5.0, 1.5, 0.1); COST = 10.0; FWD = 200; UPTIME_BARS = 276  # 23h
import importlib.util
_s = importlib.util.spec_from_file_location("hev2", ROOT / "scripts/research_homer_entry_v2_20260904.py"); hev2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(hev2)
sim_exit = hev2.sim_exit

def klines(start_ms, end_ms):
    rows = []; cur = start_ms
    while cur < end_ms:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines", params={"symbol": "ETHUSDT", "interval": "5m", "startTime": cur, "limit": 1500}, timeout=20).json()
        if not r: break
        rows += r; cur = r[-1][0] + 300000
        if len(r) < 1500: break
    k = pd.DataFrame(rows, columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"])[["t","o","h","l","c"]].astype(float)
    k["ts"] = pd.to_datetime(k["t"], unit="ms", utc=True); return k.drop_duplicates("ts").reset_index(drop=True)

def stats(p):
    p = np.asarray(p, float); w = p > 0
    eq = np.cumsum(p); dd = float((eq - np.maximum.accumulate(eq)).min()) if len(p) else 0.0
    return {"n": int(len(p)), "mean_bp": round(float(p.mean()), 2) if len(p) else None, "median_bp": round(float(np.median(p)), 2) if len(p) else None,
            "sum_bp": round(float(p.sum()), 1), "win_rate": round(float(w.mean()), 3) if len(p) else None,
            "payoff": round(float(p[w].mean() / -p[~w].mean()), 3) if w.any() and (~w).any() else None, "max_dd_bp": round(dd, 1)}

s = json.loads(STATE.read_text()); led = pd.DataFrame(s["ledger"]); pos = pd.DataFrame(s["positions"])
for c in ("entry_utc", "exit_utc"): led[c] = pd.to_datetime(led[c], utc=True)
now = pd.Timestamp.now(tz="UTC"); k = klines(int((led["entry_utc"].min() - pd.Timedelta("2h")).timestamp() * 1000), int(now.timestamp() * 1000))
k = k[k["ts"] + pd.Timedelta("5min") <= now].reset_index(drop=True)          # 완결 봉만
last_close = float(k["c"].iloc[-1]); last_ts = k["ts"].iloc[-1]
print(f"원장 마감 {len(led)}건 · 오픈 {len(pos)}건 · 시작 {s['started_utc'][:19]}Z · 최신 완결봉 {last_ts} close {last_close:.2f}")
print("\n=== ① 마감 원장 요약 ===")
rep = {"overall": stats(led["pnl_bp"]), "by_side": {sd: stats(g["pnl_bp"]) for sd, g in led.groupby("side")},
       "by_reason": {r: stats(g["pnl_bp"]) for r, g in led.groupby("reason")}, "by_basis": {b: stats(g["pnl_bp"]) for b, g in led.groupby("exit_basis")}}
print("overall", rep["overall"]); print("by_side", rep["by_side"]); print("by_reason", rep["by_reason"]); print("by_basis", rep["by_basis"])
led["day"] = led["exit_utc"].dt.floor("D"); print("per exit-day:", {str(d.date()): stats(g["pnl_bp"]) for d, g in led.groupby("day")})
print("hold bars: mean %.1f median %.0f max %d | proba mean %.4f min %.4f" % (led["bars_held"].mean(), led["bars_held"].median(), led["bars_held"].max(), led["proba"].mean(), led["proba"].min()))
print(led[["entry_utc", "exit_utc", "side", "entry", "exit", "atr", "proba", "pnl_bp", "bars_held", "reason"]].to_string())

print("\n=== ② 오픈 포지션 시가평가 (최신 완결봉 종가, 비용 10bp 차감) ===")
if len(pos):                                                   # 2026-09-05: 오픈 포지션 0건이면 컬럼이 없어 KeyError -- 건너뜀
    pos["unreal_bp"] = np.where(pos["side"] == "long", 1, -1) * (last_close - pos["entry"]) / pos["entry"] * 1e4 - COST
    pos["to_stop_bp"] = np.where(pos["side"] == "long", 1, -1) * (last_close - pos["stop"]) / last_close * 1e4
    print(pos[["entry_utc", "side", "entry", "atr", "stop", "best", "armed", "bars_held", "proba", "unreal_bp", "to_stop_bp"]].to_string())
else:
    pos["unreal_bp"] = pd.Series(dtype=float); print("(오픈 포지션 없음)")
mtm = np.concatenate([led["pnl_bp"].to_numpy(float), pos["unreal_bp"].to_numpy(float)])
rep["open_positions"] = {"n": int(len(pos)), "unreal_mean_bp": round(float(pos["unreal_bp"].mean()), 2), "unreal_sum_bp": round(float(pos["unreal_bp"].sum()), 1)}
rep["closed_plus_open_mtm"] = stats(mtm); print("마감+오픈 시가평가:", rep["closed_plus_open_mtm"])

print("\n=== ③ 실봉 재구성 대조 (러너 manage() vs sim_exit, 같은 진입가·ATR, 신호봉 다음 봉부터) ===")
kts = k["ts"].dt.tz_localize(None).to_numpy()   # ⚠️tz-aware .to_numpy()는 Timestamp 객체 배열 -- naive datetime64로
o, h, l, c = (k[x].to_numpy(float) for x in ("o", "h", "l", "c"))
rows = []
for r in led.itertuples(index=False):
    i0 = int(np.searchsorted(kts, np.datetime64(r.entry_utc.tz_convert("UTC").tz_localize(None)), side="right"))   # 신호봉 다음 봉
    if i0 >= len(k): continue
    H, L, C = h[i0:i0 + FWD], l[i0:i0 + FWD], c[i0:i0 + FWD]
    if len(C) < 1: continue
    sg = 1.0 if r.side == "long" else -1.0
    pn, ex = sim_exit(np.array([r.entry]), np.array([r.atr]), np.array([sg]), H[None], L[None], C[None], *CELL)
    rec = pn[0] * 1e4 - COST; ex_ts = k["ts"].iloc[i0 + int(ex[0])]
    nxt_open = o[i0]; slip = sg * (r.entry - nxt_open) / nxt_open * 1e4          # +면 러너가 불리한 가격에 진입
    pn2, _ = sim_exit(np.array([nxt_open]), np.array([r.atr]), np.array([sg]), H[None], L[None], C[None], *CELL)
    rows.append({"entry_utc": r.entry_utc, "side": r.side, "runner_bp": r.pnl_bp, "recon_bp": round(rec, 2), "diff_bp": round(r.pnl_bp - rec, 2),
                 "runner_exit": r.exit_utc, "recon_exit": ex_ts, "exit_match": bool(ex_ts == r.exit_utc), "entry_slip_bp": round(slip, 2),
                 "nextopen_bp": round(pn2[0] * 1e4 - COST, 2)})
R = pd.DataFrame(rows); rep["reconstruction"] = {"n": int(len(R)), "max_abs_diff_bp": round(float(R["diff_bp"].abs().max()), 3), "exit_bar_match": int(R["exit_match"].sum()),
                                                "entry_slip_mean_bp": round(float(R["entry_slip_bp"].mean()), 2), "nextopen_mean_bp": round(float(R["nextopen_bp"].mean()), 2), "runner_mean_bp": round(float(R["runner_bp"].mean()), 2)}
print(R.to_string()); print("요약:", rep["reconstruction"])

print("\n=== ④ 초기 원장 귀무분포: 백테스트 F0 체결열에서 무작위 시작 후 23h(276봉) 안에 *마감된* 트레이드만 ===")
def early_null(path, B=3000, seed=0):
    T = pd.read_csv(path); eb, xb, pn = T["entry_bar"].to_numpy(), T["exit_bar"].to_numpy(), T["pnl_bp"].to_numpy(); rng = np.random.default_rng(seed)
    starts = rng.integers(eb.min(), eb.max() - UPTIME_BARS, B); means, ns, nopen = [], [], []
    for s0 in starts:
        m = (eb >= s0) & (eb < s0 + UPTIME_BARS); closed = m & (xb <= s0 + UPTIME_BARS)
        if closed.sum() >= 5: means.append(pn[closed].mean()); ns.append(int(closed.sum())); nopen.append(int((m & ~closed).sum()))
    means = np.array(means); return {"steady_mean_bp": round(float(pn.mean()), 2), "early_closed_mean_bp_median": round(float(np.median(means)), 2),
                                     "early_closed_mean_p05_p95": [round(float(np.percentile(means, 5)), 2), round(float(np.percentile(means, 95)), 2)],
                                     "early_n_closed_median": int(np.median(ns)), "early_n_open_median": int(np.median(nopen)),
                                     "pct_of_starts_with_early_mean_ge_shadow": round(float((means >= led["pnl_bp"].mean()).mean() * 100), 1), "B": int(len(means))}
for w in ("OOS", "VAL"):
    rep[f"early_null_{w}"] = early_null(ROOT / f"tmp/homer_entry_v2_20260904/trades_tabpfn_F0_{w}.csv"); print(w, rep[f"early_null_{w}"])
# 트레이드 단위 부트스트랩 CI (참고)
rng = np.random.default_rng(1); p = led["pnl_bp"].to_numpy(float); bs = np.array([p[rng.integers(0, len(p), len(p))].mean() for _ in range(5000)])
rep["trade_bootstrap_ci95"] = [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]; print("trade-bootstrap CI95", rep["trade_bootstrap_ci95"])
(OUT / "report.json").write_text(json.dumps(rep, indent=2, default=str, ensure_ascii=False)); R.to_csv(OUT / "reconstruction.csv", index=False)
print("wrote", OUT / "report.json")
