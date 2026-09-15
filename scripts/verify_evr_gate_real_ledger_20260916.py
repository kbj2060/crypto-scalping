"""E|r| 게이트 실계좌 재검증 v2 — 아티팩트 evr_history 가 08-22 에서 끊겨 09-14 까지 재계산."""
import json, importlib.util, sys, numpy as np, pandas as pd
from pathlib import Path
R = Path("/home/kbj20/crypto-scalping"); W = Path.cwd()
RNG = np.random.default_rng(20260916)

def _mod(rel, name):
    sp = importlib.util.spec_from_file_location(name, W / rel)
    m = importlib.util.module_from_spec(sp); argv, sys.argv = sys.argv, ["x"]
    try: sp.loader.exec_module(m)
    finally: sys.argv = argv
    return m
RX = _mod("scripts/research_direction_event_expansion_20260915.py", "RX")
import joblib
ART = R / "data/models/direction_4h_top5_20260915"
art = joblib.load(ART / "models.joblib"); cols = art["cols"]

raw = pd.read_parquet(R / "data/binance_vision/panel/ETHUSDT.parquet")
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
p = RX._features(raw, "ETH", False, "2000-01-01")
x = p[cols].to_numpy(np.float32)
ok = np.isfinite(x).all(1)
ev = pd.Series(np.nan, index=pd.to_datetime(p["timestamp"]))
ev[ok] = np.exp(art["evr"].predict(x[ok]))
ev = ev.dropna()
print(f"재계산 E|r| {len(ev):,} · {ev.index.min()} ~ {ev.index.max()}")
h = json.load(open(ART / "evr_history.json"))["ETH"]
old = pd.Series(np.exp(h["pred"]), index=pd.to_datetime(h["ts"]))
j = ev.index.intersection(old.index)
print(f"⭐저장 이력과 겹치는 {len(j):,}개 상관 {np.corrcoef(ev[j], old[j])[0,1]:.6f} · "
      f"최대상대오차 {np.max(np.abs(ev[j]-old[j])/old[j]):.2e}  (재계산 경로 검증)")

pct = ev.expanding(8064).rank(pct=True).shift(1)
rows = [json.loads(l) for l in open(R / "data/live/account_round_trips.jsonl")]
t = pd.DataFrame(rows)
t["entry_ts"] = pd.to_datetime(t.entry_time, unit="ms")
t["notional"] = t.entry_price * t.max_qty
mv = np.where(t.side == "SHORT", -1, 1)
t["bp_5_88"] = (t.exit_price - t.entry_price) / t.entry_price * mv * 1e4 - 5.88   # 09-15 정의
t["bp_real"] = t.net_pnl / (t.entry_price * t.qty_in) * 1e4                      # 실제 수수료
i = pct.index.searchsorted(t.entry_ts.to_numpy(), side="right") - 1
t["evr_pct"] = np.where(i >= 0, pct.to_numpy()[np.clip(i, 0, len(pct)-1)], np.nan)
lag = (t.entry_ts - pct.index.to_numpy()[np.clip(i, 0, len(pct)-1)]).dt.total_seconds()/3600
print(f"백분위 매칭 {t.evr_pct.notna().sum()}/{len(t)} · 진입-피쳐 시차 중앙 {lag.median():.2f}h 최대 {lag.max():.2f}h")

t["band"] = pd.cut(t.evr_pct, [-.01, .5, .8, 1.01], labels=["하위50%", "50~80%", "상위20%"])
print(f"\n■ 09-15 주장 재현 (단위당 순bp = 방향조정 가격변동 − 5.88bp 가정비용)")
print(f"{'구간':<9}{'n':>4}{'09-15주장':>10}{'재검증':>9}{'실제수수료':>10}{'실손익$':>10}{'승률':>8}{'평균명목$':>11}")
CLAIM = {"전체": 20.32, "하위50%": 10.34, "50~80%": 14.32, "상위20%": 53.18}
for b in ["전체", "하위50%", "50~80%", "상위20%"]:
    s = t if b == "전체" else t[t.band == b]
    if len(s) == 0: continue
    print(f"{b:<9}{len(s):>4}{CLAIM[b]:>+10.2f}{s.bp_5_88.mean():>+9.2f}{s.bp_real.mean():>+10.2f}"
          f"{s.net_pnl.sum():>+10.2f}{(s.net_pnl>0).mean()*100:>7.1f}%{s.notional.mean():>11,.0f}")
g = t[t.band == "상위20%"]
print(f"\n게이트 증분 {g.bp_5_88.mean()-t.bp_5_88.mean():+.2f}bp/건 (주장 +32.86)")
draws = np.array([RNG.choice(t.bp_5_88.to_numpy(), len(g), replace=False).mean() for _ in range(20000)])
print(f"무작위 부분집합 평균 {draws.mean():+.2f} · 게이트 백분위 {(draws < g.bp_5_88.mean()).mean()*100:.1f}% (주장 99.9%)")
d = t.entry_ts.dt.floor("D"); print(f"\n고유 진입일 {d.nunique()} / 왕복 {len(t)}  ⇒ 독립 관측은 {d.nunique()}")
for lab, s in (("전체", t), ("상위20%", g)):
    dd = s.entry_ts.dt.floor("D"); u = dd.unique(); by = {k: s.bp_5_88[dd == k].to_numpy() for k in u}
    bs = np.array([np.concatenate([by[k] for k in RNG.choice(u, len(u))]).mean() for _ in range(4000)])
    print(f"  {lab:<7} n={len(s):>3} 고유일={len(u):>2}  일군집 CI95 "
          f"[{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]")
from scipy.stats import spearmanr
print(f"\nE|r|백분위 ↔ 명목 스피어만 {spearmanr(t.evr_pct, t.notional).statistic:+.3f} (주장 −0.426)")
