#!/usr/bin/env python3
"""①사이징 4열 반사실 — **배포 경로 그대로** 다시 재기 (2026-09-16).

사용자 *"(a)로 배포 경로 그대로 다시 재줘"*.

09-16 1차 재구성은 **모델 상한만** 썼다(자본 1만 고정·보유 4h 고정·total_notional).
그때 MDD 가 09-15 주장(156→84)과 반대로 나왔다(99→211). 서버에서 실제 경로를 읽어보니
배포는 **세 상한의 min** 이었다 — 1차 재구성이 빠뜨린 것이 이것이다:

    dashboard/server.py:2886~2930
      plan_hold   = planning_hold(...)              # recommend_hold 가 고르는 «지평»
      risk        = risk_sizing(sizing, plan_hold)  # 그 지평의 safe_mae_pct
      cap_model   = entry_notional(equity, safe_mae)["total_notional"]
      cap_equity  = equity * SIZING_CAP_EQUITY_X            (=6.0)
      cap_ledger  = SIZING_CAP_MULT * median(최근 30왕복 명목)   (=2.0)
      cap_notional= min(cap_ledger, cap_equity, cap_model)
      rec_qty     = cap_model / price
      → build_entry_plan: qty*price > (cap_notional − existing) 이면 그 여유로 자른다

즉 **실제 명목 = min(모델, 원장, 순자산×6) − 기존포지션**.
정책/계획/모델 파일 3종과 모델 아티팩트는 서버와 **md5 동일**함을 확인하고 임포트한다.

원장에 없는 유일한 입력이 **순자산**이라 E0 를 스윕해 결론이 그 가정에 의존하는지 본다.

출력: tmp/eth_sizing_deployed_path_20260916/
"""
from __future__ import annotations
import argparse, json, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "tmp/eth_sizing_deployed_path_20260916"
RNG = np.random.default_rng(20260916)
EQUITY_X = 6.0        # server.py:1275 SIZING_CAP_EQUITY_X
LEDGER_MULT = 2.0     # server.py:1225 SIZING_CAP_MULT
LEDGER_WIN = 30       # server.py:1227 SIZING_CAP_WINDOW
LEDGER_MIN = 10       # server.py:1226 SIZING_CAP_MIN_TRIPS
HOLD_FIXED_MIN = 240  # server.py HOLD_FIXED_MIN
COST_BP = 5.88


def _mod(rel: str, name: str):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp); argv, sys.argv = sys.argv, ["x"]
    try: sp.loader.exec_module(m)
    finally: sys.argv = argv
    return m


def mdd_and_mean(bp: np.ndarray, w: np.ndarray, day: pd.Series) -> tuple[float, float]:
    """평균 명목을 맞춘 뒤 일별 합 -> bp/일 과 최대낙폭."""
    w = np.asarray(w, float)
    if w.sum() <= 0: return np.nan, np.nan
    w = w / w.mean()
    g = pd.DataFrame({"d": day, "x": bp * w}).groupby("d").x.sum()
    c = g.cumsum()
    return float(g.mean()), float((c.cummax() - c).max())


def selftest() -> None:
    # 평균 명목 정규화: 상수 가중은 ①고정과 동일해야 한다
    bp = np.array([10., -20., 30.]); d = pd.Series(pd.to_datetime(["2026-01-01"]*3))
    assert mdd_and_mean(bp, np.ones(3), d) == mdd_and_mean(bp, np.full(3, 7.0), d)
    # 낙폭: 첫날 +10, 둘째날 -20 이면 MDD 20
    d2 = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]))
    m, mdd = mdd_and_mean(np.array([10., -20.]), np.ones(2), d2)
    assert abs(m + 5) < 1e-9 and abs(mdd - 20) < 1e-9
    # 상한 합성: min(모델, 원장, 순자산x6) - 기존
    def eff(model, ledger, eq, existing):
        cap = min([v for v in (model, ledger, eq * EQUITY_X) if v])
        return max(0.0, min(model, cap - existing))
    assert eff(10_000, 8_000, 1_000, 0) == 6_000        # 순자산이 묶는다
    assert eff(10_000, 4_000, 1_000, 1_000) == 3_000    # 원장이 묶고 기존을 뺀다
    assert eff(2_000, 8_000, 1_000, 0) == 2_000         # 모델이 묶는다
    assert eff(10_000, 8_000, 1_000, 9_999) == 0.0      # 여유 없음
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest: selftest(); return 0
    selftest(); OUT.mkdir(parents=True, exist_ok=True)
    MQ = _mod("scripts/live_eth_mae_quantile_model_20260913.py", "MQ")
    PO = _mod("scripts/live_eth_risk_sizing_policy_20260913.py", "PO")
    TP = _mod("scripts/live_eth_trade_plan_20260913.py", "TP")
    svm = MQ.svm

    print("[1/4] 피쳐·ATR …", flush=True)
    kl = pd.read_csv(REPO / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
                     usecols=["timestamp", "open", "high", "low", "close", "volume",
                              "quote_volume", "trades"])
    kl["timestamp"] = pd.to_datetime(kl.timestamp)
    c, hi, lo = kl.close.to_numpy(float), kl.high.to_numpy(float), kl.low.to_numpy(float)
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
    atr_pct = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy() / np.maximum(c, 1e-12)
    base = svm.build_features(kl.timestamp, c, kl.quote_volume.to_numpy(float),
                              kl.trades.to_numpy(float), hi, lo).replace([np.inf, -np.inf], np.nan)
    bts = kl.timestamp.to_numpy()
    art = MQ.load_model(); models, mult = art["models"], art["mult"]

    print("[2/4] 원장 …", flush=True)
    t = pd.DataFrame([json.loads(l) for l in open(REPO / "data/live/account_round_trips.jsonl")])
    t["entry_ts"] = pd.to_datetime(t.entry_time, unit="ms")
    t["exit_ts"] = pd.to_datetime(t.exit_time, unit="ms")
    t = t.sort_values("entry_ts").reset_index(drop=True)
    t["notional_real"] = t.entry_price * t.max_qty
    mv = np.where(t.side == "SHORT", -1, 1)
    t["bp"] = (t.exit_price - t.entry_price) / t.entry_price * mv * 1e4 - COST_BP
    day = t.entry_ts.dt.floor("D")
    i = np.searchsorted(bts, t.entry_ts.to_numpy(), "right") - 1
    i = np.clip(i, 0, len(base) - 1)
    t["atr_pct"] = atr_pct[i]

    print("[3/4] 지평별 safe_mae (배포와 같은 표) …", flush=True)
    HOLDS = list(TP.HOLD_CHOICES)
    tbl: dict[int, np.ndarray] = {}
    for hm in HOLDS:
        r = base.iloc[i].copy().reset_index(drop=True)
        r["log_h"] = np.log(float(hm))
        r["side"] = np.where(t.side.to_numpy() == "SHORT", -1, 1)
        tbl[hm] = MQ.safe_mae(models, r[MQ.FEATURES], mult)
    print(f"  지평 {HOLDS} · safe_mae 중앙 " +
          " / ".join(f"{hm}m {np.median(tbl[hm]):.2f}%" for hm in HOLDS))

    print("[4/4] 배포 경로 반사실 …", flush=True)
    # 기존 포지션(같은 시각 열려 있던 왕복의 명목 합) — 원장에서 인과 재구성
    ent, ext, nom = t.entry_ts.to_numpy(), t.exit_ts.to_numpy(), t.notional_real.to_numpy()
    existing = np.array([nom[(ent < ent[k]) & (ext > ent[k])].sum() for k in range(len(t))])
    rows, detail = [], {}
    for E0 in (500.0, 719.0, 1000.0, 1500.0, 3000.0):
        eq = E0 + np.concatenate([[0.0], np.cumsum(t.net_pnl.to_numpy())[:-1]])
        w, holds, binds = np.zeros(len(t)), np.zeros(len(t), int), []
        w_nomodel = np.zeros(len(t))        # 모델 상한을 빼면 = 모델의 순기여 분리
        for k in range(len(t)):
            if eq[k] <= 0: binds.append("bust"); continue
            past = nom[:k]
            cap_led = (LEDGER_MULT * float(np.median(past[-LEDGER_WIN:]))
                       if len(past) >= LEDGER_MIN else None)
            cap_eq = eq[k] * EQUITY_X
            pol = [v for v in (cap_led, cap_eq) if v]
            pol_x = min(pol) / eq[k] if pol else EQUITY_X
            rt = {str(hm): {t.side[k]: {"safe_mae_pct": float(tbl[hm][k])}} for hm in HOLDS}
            h = TP.recommend_hold(rt, t.side[k], pol_x, float(t.atr_pct[k]), acc=TP.PRESCRIBE_ACC)
            hm = int(h["recommended_min"]) if h.get("available") else HOLD_FIXED_MIN
            holds[k] = hm
            safe = float(tbl[hm][k])
            cap_mod = PO.entry_notional(eq[k], safe)["total_notional"]
            cands = [(v, n) for v, n in ((cap_led, "ledger"), (cap_eq, "equity"),
                                         (cap_mod, "model")) if v]
            cap, who = min(cands, key=lambda x: x[0])
            binds.append(who)
            w[k] = max(0.0, min(cap_mod, cap - existing[k]))
            cap2, _ = min([(v, n) for v, n in ((cap_led, "ledger"), (cap_eq, "equity")) if v],
                          key=lambda x: x[0])
            w_nomodel[k] = max(0.0, cap2 - existing[k])
        m_fix, d_fix = mdd_and_mean(t.bp.to_numpy(), np.ones(len(t)), day)
        m_dep, d_dep = mdd_and_mean(t.bp.to_numpy(), w, day)
        m_nm, d_nm = mdd_and_mean(t.bp.to_numpy(), w_nomodel, day)
        rows.append(dict(E0=E0, 고정_bp=round(m_fix, 2), 고정_MDD=round(d_fix, 0),
                         배포_bp=round(m_dep, 2), 배포_MDD=round(d_dep, 0),
                         모델뺀_bp=round(m_nm, 2), 모델뺀_MDD=round(d_nm, 0),
                         zero=int((w == 0).sum()),
                         묶은것=", ".join(f"{n} {binds.count(n)}" for n in
                                        ("model", "equity", "ledger", "bust") if binds.count(n))))
        detail[E0] = (w.copy(), holds.copy(), list(binds))
    D = pd.DataFrame(rows); D.to_csv(OUT / "equity_sweep.csv", index=False)
    print(f"\n{'='*118}")
    print("■ 배포 경로 그대로 — 순자산 E0 스윕 (①고정 대비)")
    print(f"{'E0($)':>8}{'①고정':>9}{'MDD':>6}{'②배포경로':>11}{'MDD':>6}"
          f"{'③모델뺀':>10}{'MDD':>6}  무엇이 묶었나")
    for r in D.itertuples():
        print(f"{r.E0:>8.0f}{r.고정_bp:>+9.2f}{r.고정_MDD:>6.0f}{r.배포_bp:>+11.2f}"
              f"{r.배포_MDD:>6.0f}{r.모델뺀_bp:>+10.2f}{r.모델뺀_MDD:>6.0f}  {r.묶은것}")
    print("  ⇒ ②와 ③의 차이가 곧 **MAE 모델의 순기여**다(③은 원장·순자산 상한만).")
    w, holds, binds = detail[719.0]
    print(f"\n■ E0=719 (원장 순손익으로 역산한 기준) 상세")
    print(f"  권고 지평 분포: " + " · ".join(
        f"{h}분 {int((holds == h).sum())}건" for h in sorted(set(holds.tolist())) if h))
    ww = w / w.mean() if w.mean() > 0 else w
    print(f"  가중치: 중앙 {np.median(ww):.2f} · 최대 {ww.max():.2f} · 최소 {ww.min():.2f}")
    from scipy.stats import spearmanr
    print(f"  w ↔ 실현bp 스피어만 {spearmanr(ww, t.bp).statistic:+.3f}")
    print(f"  실제 명목 ↔ 배포권고 명목 스피어만 {spearmanr(t.notional_real, w).statistic:+.3f} "
          f"(사용자가 실제로 배포 권고를 따랐는가)")
    o = np.argsort(-np.abs(t.bp.to_numpy() * ww))[:5]
    print("  최악·최대 기여 5건:")
    for k in o:
        print(f"    {t.entry_ts[k]:%m-%d %H:%M} {t.side[k]:>5} bp {t.bp[k]:>+8.1f} · w {ww[k]:>5.2f}"
              f" · 지평 {holds[k]:>4}분 · 묶은것 {binds[k]:<7} · 기여 {t.bp[k]*ww[k]:>+8.1f}")
    print("=" * 118)
    print(json.dumps({"rows": len(D), "trips": len(t)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
