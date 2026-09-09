#!/usr/bin/env python3
"""TabM 다중과제 — **헤드별 다른 피쳐 경로** (2026-09-07).

사용자: *"각 헤드는 각자 다른 라벨데이터에 잘 맞는 피쳐만 넣어서 진행해줘"*

## 구조 변경
```
x_A(방향 피쳐 40) -> in_proj_A --+
x_B(크기 피쳐 40) -> in_proj_B --+--> **공유 blocks(백본)** --> head_A / head_B / head_C
x_C(시차 피쳐 40) -> in_proj_C --+
```
입력 투영만 과제별로 두고 **백본 몸통(blocks·expert_scale·norms)은 공유**한다. 각 과제의 손실은
자기 경로로 흐르고, 역전파가 공유 백본에서 만난다 -- 이게 "다른 피쳐 + 하나의 백본"이다.
비교군: 부록 P 의 단일경로 3헤드(모든 헤드가 같은 피쳐)와 1헤드.

## 헤드별 피쳐 (TRAIN 전용, 타깃마다 따로)
TRAIN 을 시간순 2등분해 **두 반쪽 모두에서 부호가 같은** 단일피쳐 AUC 상위 40.
`tmp/eth_anchor_direction_feature_ranking_20260907/perhead_train.json`
  A_dir  최강 bb_width 0.461/0.441       -- 약하다
  B_size 최강 parkinson_vol 0.733/0.724  -- 강하다
  C_time 최강 parkinson_vol 0.756/0.759  -- 더 강하다
⭐겹침: A∩B 32% · A∩C 35% · **B∩C 88%** -- 보조 두 과제는 사실상 같은 축(변동성)이고
  주 과제와는 1/3만 겹친다. 전이가 일어날 여지가 있는지가 이 실험의 내용이다.

## 프로토콜
부록 P 와 동일: 8시드 · TRAIN 내부 홀드아웃 조기종료 · **전 에폭 곡선** ·
**세 창(VAL/OOS/HOLDOUT_SPENT) 동시 평가** · 날 블록 셔플 귀무.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_anchor_direction_tabm3head_20260907 as P  # noqa: E402

OUT = ROOT / "tmp/eth_anchor_tabm_perhead_20260907"
PERHEAD = ROOT / "tmp/eth_anchor_direction_feature_ranking_20260907/perhead_train.json"


class PerHeadTabM(nn.Module):
    """헤드별 입력 투영 + 공유 백본. 백본은 ThreeHeadTabM 축자 차용."""

    def __init__(self, n_a: int, n_b: int, n_c: int, three_head: bool = True) -> None:
        super().__init__()
        self.three_head = three_head
        K, H, L = P.K, P.HIDDEN, P.LAYERS
        self.k = K
        self.scale = nn.ParameterDict()
        self.bias = nn.ParameterDict()
        self.proj = nn.ModuleDict()
        for tag, n in (("a", n_a), ("b", n_b), ("c", n_c)):
            self.scale[tag] = nn.Parameter(torch.randn(K, n) * 0.03 + 1.0)
            self.bias[tag] = nn.Parameter(torch.zeros(K, n))
            self.proj[tag] = nn.Linear(n, H)
        self.blocks = nn.ModuleList(nn.Linear(H, H) for _ in range(L - 1))          # 공유
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(K, H) * 0.03 + 1.0) for _ in range(L - 1))     # 공유
        self.norms = nn.ModuleList(nn.LayerNorm(H) for _ in range(L))               # 공유
        self.dropout = nn.Dropout(P.DROPOUT)
        self.head_a = nn.Linear(H, 3)
        if three_head:
            self.head_b = nn.Linear(H, 3); self.head_c = nn.Linear(H, 2)

    def encode(self, x, tag):
        xk = x.unsqueeze(1) * self.scale[tag].unsqueeze(0) + self.bias[tag].unsqueeze(0)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](self.proj[tag](xk))))
        for i, layer in enumerate(self.blocks):
            res = h
            h = layer(h * self.expert_scale[i].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[i + 1](h))) + res
        return h

    def forward_a(self, xa):
        return self.head_a(self.encode(xa, "a"))

    def forward_all(self, xa, xb, xc):
        out = {"a": self.head_a(self.encode(xa, "a"))}
        if self.three_head:
            out["b"] = self.head_b(self.encode(xb, "b"))
            out["c"] = self.head_c(self.encode(xc, "c"))
        return out


def fit_one(Xa, Xb, Xc, ys, w, ho, seed, three_head, curves):
    torch.manual_seed(seed); np.random.seed(seed)
    m = PerHeadTabM(Xa.shape[1], Xb.shape[1], Xc.shape[1], three_head)
    opt = torch.optim.AdamW(m.parameters(), lr=P.LR, weight_decay=P.WD)
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    Ta, Tb, Tc = T(Xa), T(Xb), T(Xc)
    Ya, Yb, Yc = (torch.tensor(y) for y in ys)
    W = T(w); n = len(Ta); K = P.K
    hoa, hoy = T(ho[0]), ho[1]
    best, state, bad, curve = -9, None, 0, []
    for ep in range(P.EPOCHS):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, P.BATCH):
            b = perm[i:i + P.BATCH]; wb = W[b]
            o = m.forward_all(Ta[b], Tb[b], Tc[b])
            def ce(key, Y):
                l = torch.nn.functional.cross_entropy(
                    o[key].reshape(-1, o[key].shape[-1]), Y[b].repeat_interleave(K), reduction="none")
                return (l.view(len(b), K).mean(1) * wb).sum() / wb.sum().clamp(min=1)
            loss = ce("a", Ya)
            if three_head:
                loss = loss + P.W_SIZE * ce("b", Yb) + P.W_TIME * ce("c", Yc)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pa = torch.softmax(m.forward_a(hoa), -1).mean(1).numpy()
        cl = hoy != 1; sc = np.nan
        if cl.sum() > 20 and len(np.unique(hoy[cl] == 2)) > 1:
            den = pa[cl][:, 0] + pa[cl][:, 2]
            sc = roc_auc_score((hoy[cl] == 2).astype(int),
                               np.where(den > 0, pa[cl][:, 2] / np.maximum(den, 1e-12), 0.5))
        curve.append(float(sc) if np.isfinite(sc) else np.nan)
        if np.isfinite(sc) and sc > best:
            best, state, bad = sc, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= P.PATIENCE:
                break
    if state:
        m.load_state_dict(state)
    curves.append({"seed": seed, "three_head": three_head, "epochs": len(curve),
                   "best_ep": int(np.nanargmax(curve)) if np.isfinite(curve).any() else -1, "best": best})
    return m


def run(D, X, ci, feats, three_head, rng, shuffle=False, seeds=P.SEEDS, curves=None):
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy(); bar = D["bar_idx"].to_numpy()
    y3, ysz, ytm = P.make_targets(D)
    w = D["w_uniq"].to_numpy(float); w = np.where(np.isfinite(w) & (w > 0), w, 1e-6)
    tr = sp == "TRAIN"
    ya = y3.copy()
    if shuffle:
        days = np.unique(day[tr]); perm = rng.permutation(days)
        src = {d: np.flatnonzero(tr & (day == d)) for d in days}
        for d, d2 in zip(days, perm):
            if len(src[d]):
                ya[src[d]] = np.resize(y3[src[d2]], len(src[d]))
    def prep(key):
        fc = [ci[c] for c in feats[key] if c in ci]
        Z = X[:, fc]
        mu, sd = np.nanmean(Z[tr], 0), np.nanstd(Z[tr], 0) + 1e-9
        return np.nan_to_num((Z - mu) / sd).astype(np.float32)
    Xa, Xb, Xc = prep("A_dir"), prep("B_size"), prep("C_time")
    ti = np.flatnonzero(tr); o = ti[np.argsort(bar[ti])]; cut = int(len(o) * 0.85)
    fi, hi_ = o[:cut], o[cut + P.EMBARGO:]
    cw = compute_sample_weight("balanced", ya[fi]).astype(np.float32)
    curves = curves if curves is not None else []
    models = [fit_one(Xa[fi], Xb[fi], Xc[fi], (ya[fi], ysz[fi], ytm[fi]), w[fi] * cw,
                      (Xa[hi_], ya[hi_]), s, three_head, curves) for s in seeds]
    out = {}
    for wn in ("VAL", "OOS", "HOLDOUT_SPENT"):
        te = sp == wn
        if te.sum() < 30:
            continue
        with torch.no_grad():
            p = np.mean([torch.softmax(m.forward_a(torch.tensor(Xa[te], dtype=torch.float32)), -1)
                         .mean(1).numpy() for m in models], axis=0)
        cl = y3[te] != 1
        den = p[cl][:, 0] + p[cl][:, 2]
        dp = np.where(den > 0, p[cl][:, 2] / np.maximum(den, 1e-12), 0.5)
        yy = (y3[te][cl] == 2).astype(int)
        out[f"{wn}_n"] = int(cl.sum())
        out[f"{wn}_auc"] = float(roc_auc_score(yy, dp)) if len(np.unique(yy)) > 1 else np.nan
        lo, hi = P.day_auc_ci(yy, dp, day[te][cl], rng)
        out[f"{wn}_lo"], out[f"{wn}_hi"] = lo, hi
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(P.SRCD / "features154.parquet")
    meta = json.loads((P.SRCD / "meta.json").read_text())
    cols = [c for c in meta["feature_cols"] if c in D.columns]
    X = D[cols].to_numpy(np.float64); ci = {c: i for i, c in enumerate(cols)}
    feats = json.loads(PERHEAD.read_text())
    print(f"[입력] {D.shape} · 헤드별 피쳐 " + " · ".join(f"{k} {len(v)}" for k, v in feats.items()))
    ov = lambda a, b: len(set(feats[a]) & set(feats[b]))
    print(f"[겹침] A∩B {ov('A_dir','B_size')} · A∩C {ov('A_dir','C_time')} · B∩C {ov('B_size','C_time')} / 40")
    rows, curves = [], []
    for th in (True, False):
        t0 = time.time()
        r = run(D, X, ci, feats, th, rng, curves=curves)
        r.update({"three_head": th, "sec": round(time.time() - t0, 1)})
        rows.append(r)
        print(f"   {'3head(헤드별피쳐)' if th else '1head(A피쳐만)':<20} "
              f"VAL {r.get('VAL_auc',np.nan):.4f}[{r.get('VAL_lo',np.nan):.3f},{r.get('VAL_hi',np.nan):.3f}] · "
              f"OOS {r.get('OOS_auc',np.nan):.4f}[{r.get('OOS_lo',np.nan):.3f},{r.get('OOS_hi',np.nan):.3f}] · "
              f"3rd {r.get('HOLDOUT_SPENT_auc',np.nan):.4f}[{r.get('HOLDOUT_SPENT_lo',np.nan):.3f},"
              f"{r.get('HOLDOUT_SPENT_hi',np.nan):.3f}] · {r['sec']}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "configs.csv", index=False)
    C = pd.DataFrame(curves)
    print("\n=== 에폭 곡선 ===")
    for th in (True, False):
        s = C[C.three_head == th]
        print(f"   {'3head' if th else '1head'}: best_ep 중앙 {s.best_ep.median():.0f} · "
              f"ep0최고 {int((s.best_ep==0).sum())}/{len(s)} · 총에폭 중앙 {s.epochs.median():.0f} · "
              f"홀드아웃 best {s.best.mean():.4f}±{s.best.std():.4f}")
    best = rows[0] if rows[0].get("OOS_auc", 0) >= rows[1].get("OOS_auc", 0) else rows[1]
    print(f"\n[귀무] {'3head' if best['three_head'] else '1head'} 날 블록 셔플 B=20", flush=True)
    nv = {w: [] for w in ("VAL", "OOS", "HOLDOUT_SPENT")}
    for _ in range(20):
        rr = run(D, X, ci, feats, bool(best["three_head"]), rng, shuffle=True, seeds=P.SEEDS[:2])
        for w in nv:
            nv[w].append(rr.get(f"{w}_auc", np.nan))
    verdict = {}
    for w in nv:
        a = np.array([x for x in nv[w] if np.isfinite(x)])
        obs, lo = best.get(f"{w}_auc", np.nan), best.get(f"{w}_lo", np.nan)
        verdict[w] = {"obs": float(obs), "lo": float(lo), "null_mean": float(a.mean()),
                      "null_p95": float(np.percentile(a, 95)), "p": float(np.mean(a >= obs)),
                      "pass": bool(lo > 0.5 and obs > np.percentile(a, 95))}
        print(f"   {w:<14} 관측 {obs:.4f}[{lo:.3f}] vs 귀무 평균 {a.mean():.4f} p95 {np.percentile(a,95):.4f}"
              f" (p={verdict[w]['p']:.3f}) → {'통과' if verdict[w]['pass'] else '미달'}", flush=True)
    print(f"\n=== 세 창 모두 통과: {'✅ 예' if all(v['pass'] for v in verdict.values()) else '❌ 아니오'} ===")
    (OUT / "summary.json").write_text(json.dumps({"configs": rows, "verdict": verdict, "feats": feats},
                                                 indent=2, ensure_ascii=False, default=float))
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
