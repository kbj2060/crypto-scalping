"""소진비율을 세 모집단에서 같이 재서 '라벨 문제'인지 '모델 선택 문제'인지 가른다."""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/llewyn/crypto-scalping")
for p in (ROOT, ROOT/"scripts"):
    sys.path.insert(0, str(p))
def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT/r); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
_s1 = _load("s1v", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_vs = _s1._vs
CTX = ROOT/"data/labels/eth_5m_v_rebound_every_bar_20260901/tabpfn_train_context_frozen_every_bar_20260901.csv"
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC"); VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
_s1.VAL_END = OOS_END
sig, feat, eth = _s1.build_sig()
sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", atr_mult=1.5, t_sustain=0.20, full_bars=12)
st = _s1.label_param(sig, False, ambig="drop", anchor="wick", atr_mult=1.5, t_sustain=0.20, full_bars=12)
long = _s1.long_frame_for(sig, feat, sb, st)
long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                 np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
ts_pos = {t: i for i, t in enumerate(sig["timestamp"].dt.tz_localize(None).to_numpy())}
long["pos"] = [ts_pos.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
long = long.loc[long["pos"] >= 0].reset_index(drop=True)
low, high, op, atr = (sig[c].to_numpy() for c in ("low","high","open","atr"))
pre_atr = _vs.shifted_at(atr, -1)
i = long["pos"].to_numpy().astype(int); dn = long["is_downside"].to_numpy() == 1
anc = np.where(dn, low[i], high[i]); ent = op[np.minimum(i+1, len(op)-1)]
long["consumed"] = np.where(dn, ent-anc, anc-ent) / (1.5*pre_atr[i])
from tabpfn import TabPFNClassifier
ctx = pd.read_csv(CTX); FEAT = [c for c in ctx.columns if c not in ("timestamp","label")]
clf = TabPFNClassifier(device="cuda", random_state=20260829, ignore_pretraining_limits=True)
clf.fit(ctx[FEAT], ctx["label"].to_numpy())
print(f"{'모집단':34s} {'n':>8s} {'소진 중앙값':>11s} {'100%↑':>7s}")
for spn in ("VAL","OOS"):
    s = long.loc[long["split"]==spn].copy()
    s["p"] = np.concatenate([clf.predict_proba(s[FEAT].iloc[k:k+20000])[:,1] for k in range(0,len(s),20000)])
    for nm, sub in (("전체 봉", s),
                    ("라벨 양성(label=1)", s.loc[s["label"]==1]),
                    ("라벨 음성(label=0)", s.loc[s["label"]==0]),
                    ("모델 호출(p>=0.60)", s.loc[s["p"]>=0.60]),
                    ("모델 호출 ∩ 라벨양성", s.loc[(s["p"]>=0.60)&(s["label"]==1)])):
        c = sub["consumed"].to_numpy(); c = c[np.isfinite(c)]
        if len(c) < 10: continue
        print(f"{spn} {nm:28s} {len(c):8,d} {np.median(c)*100:10.0f}% {float((c>=1).mean())*100:6.1f}%")
