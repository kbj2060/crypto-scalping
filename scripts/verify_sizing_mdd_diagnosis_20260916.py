"""①사이징 D 의 MDD 불일치 진단 — 내 재구성 가정 탓인가, 모델 탓인가."""
import json, importlib.util, sys, numpy as np, pandas as pd
from pathlib import Path
R = Path("/home/kbj20/crypto-scalping"); W = Path.cwd()
def _mod(rel, name):
    sp = importlib.util.spec_from_file_location(name, W / rel)
    m = importlib.util.module_from_spec(sp); argv, sys.argv = sys.argv, ["x"]
    try: sp.loader.exec_module(m)
    finally: sys.argv = argv
    return m
MQ = _mod("scripts/live_eth_mae_quantile_model_20260913.py", "MQ")
PO = _mod("scripts/live_eth_risk_sizing_policy_20260913.py", "PO")
kl = pd.read_csv(R / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
                 usecols=["timestamp","open","high","low","close","volume","quote_volume","trades"])
kl["timestamp"] = pd.to_datetime(kl.timestamp)
art = MQ.load_model(); models, mult = art["models"], art["mult"]; svm = MQ.svm
base = svm.build_features(kl.timestamp, kl.close.to_numpy(float), kl.quote_volume.to_numpy(float),
                          kl.trades.to_numpy(float), kl.high.to_numpy(float), kl.low.to_numpy(float))
base["_ts"] = kl.timestamp.values; base = base.replace([np.inf,-np.inf], np.nan)
t = pd.DataFrame([json.loads(l) for l in open(R/"data/live/account_round_trips.jsonl")])
t["entry_ts"] = pd.to_datetime(t.entry_time, unit="ms")
mv = np.where(t.side=="SHORT",-1,1)
t["bp"] = (t.exit_price-t.entry_price)/t.entry_price*mv*1e4 - 5.88
i = np.searchsorted(pd.to_datetime(base["_ts"]).to_numpy(), t.entry_ts.to_numpy(), "right")-1
rw = base.iloc[np.clip(i,0,len(base)-1)].copy().reset_index(drop=True)
rw["side"] = np.where(t.side.to_numpy()=="SHORT",-1,1)

def mdd_of(w):
    w = np.asarray(w,float); w = w/w.mean()
    g = pd.DataFrame({"d":t.entry_ts.dt.floor("D"),"x":t.bp*w}).groupby("d").x.sum()
    c = g.cumsum(); return float(g.mean()), float((c.cummax()-c).max())

print("■ 가정을 하나씩 바꿔 MDD 가 어디서 뒤집히는지")
print(f"{'가정':<38}{'bp/일':>9}{'MDD':>8}")
m,d = mdd_of(np.ones(len(t))); print(f"{'①고정':<38}{m:>+9.2f}{d:>8.0f}")
for hb,lab in ((12,"1h"),(48,"4h"),(96,"8h")):
    rw["log_h"] = np.log(hb*5.0)
    sm = MQ.safe_mae(models, rw[MQ.FEATURES], mult)
    for field in ("total_notional",):
        w = [PO.entry_notional(10_000.0, float(s))[field] for s in sm]
        m,d = mdd_of(w); print(f"{'②D 보유'+lab+' · '+field:<38}{m:>+9.2f}{d:>8.0f}")
rw["log_h"] = np.log(48*5.0)
sm = MQ.safe_mae(models, rw[MQ.FEATURES], mult)
print(f"\n안전MAE 분포: 중앙 {np.median(sm):.2f}% · p10 {np.percentile(sm,10):.2f} · p90 {np.percentile(sm,90):.2f}")
w = np.array([PO.entry_notional(10_000.0, float(s))["total_notional"] for s in sm]); w = w/w.mean()
from scipy.stats import spearmanr
print(f"가중치 w_D: 중앙 {np.median(w):.2f} · 최대 {w.max():.2f} · 최소 {w.min():.2f}")
print(f"w_D ↔ 안전MAE 스피어만 {spearmanr(w, sm).statistic:+.3f} (음수여야 정상: 위험 크면 작게)")
print(f"w_D ↔ 실현 bp 스피어만 {spearmanr(w, t.bp).statistic:+.3f}")
o = np.argsort(-np.abs(t.bp.to_numpy()*w))[:5]
print("\n■ MDD 를 만드는 상위 5건 (w_D 가중 손익 기준)")
for k in o:
    print(f"  {t.entry_ts[k]:%m-%d %H:%M} {t.side[k]:>5} bp {t.bp[k]:>+8.1f} · w_D {w[k]:>5.2f} "
          f"· 안전MAE {sm[k]:>5.2f}% · 가중기여 {t.bp[k]*w[k]:>+8.1f}")
print(f"\n상위5건이 ②D 총합의 {100*(t.bp.to_numpy()*w)[o].sum()/(t.bp.to_numpy()*w).sum():.0f}%")
