#!/usr/bin/env python3
"""앵커 되돌림/지속 — **TabM 3헤드 다중과제** (2026-09-07).

사용자: *"이전에 만든 tabm 모델에서 3헤드를 가진 모델을 차용해서 지속, 되돌림, 혼재를 하나의
백본에서 학습하는 모델을 만들자"* → 설계 승인 후 실행.

## 차용 원본
`scripts/train_eval_omega1_2_tabm_3head_20260603.py::ThreeHeadTabM` 의 백본을 **축자 차용**한다
(input_scale/bias 로 k개 가상 앙상블 + expert_scale + residual). 그 모듈을 import 하면
omega/exit_head/regime3 사슬이 딸려오므로 **클래스만 복제**한다.

## 우리 3헤드 — "지속/되돌림/혼재"는 한 헤드의 3클래스다
사용자가 말한 셋은 `y3` 하나로 들어간다. 나머지 두 헤드는 **이 문제에서 유일하게 배워지는 축**을
보조 과제로 넣어 백본을 돕는 데 쓴다(부록 K/N: 크기는 배워지고 방향은 안 배워진다).
  A 주   y3 3클래스 (지속승 2 / 혼재 1 / 되돌림승 0)
  B 보조 크기 3분위 (`range_pct` TRAIN 삼분위)      <- 배워지는 축, 백본에 실제 신호 공급
  C 보조 시차 2클래스 (승자 배리어 도달이 TRAIN 중앙값보다 빠른가)
  L = L_A + 0.4*L_B + 0.3*L_C
설계 의도: 배워지는 축의 표현이 방향 헤드로 **전이**되는가.

## ⚠️용량 — 저장소 기본값은 못 쓴다
기본(h192·L3·k8)은 파라미터 111,272 = TRAIN 3,237 의 **34.4배**. 09-06 Tier0 는 파라미터 59개로도
첫 에폭부터 과적합했다. 여기서는 **h32·L2·k32**(k 는 TabM 논문 권장 고정값) 로 축소한다.
batch 256(기본 2048 은 우리 TRAIN 에 거의 전배치).

## 프로토콜 (실행 전 고정)
  구조 2종  3head vs 1head(y3 만) -- 다중과제가 실제로 돕는지 직접 비교
  피쳐 2종  dirtop40 / all150     -- 다중검정 최소화
  시드 8개 · 조기종료는 **TRAIN 내부 홀드아웃**(마지막 15%, 엠바고 48봉) -- VAL 오염 방지
  ⭐**전 에폭 곡선 기록 필수**(feedback_modern_dl_training_checklist)
  ⭐**세 번째 창(HOLDOUT_SPENT)을 처음부터 같이 평가** -- 부록 O 에서 VAL/OOS 통과가 세 번째
    창에서 소멸한 전례. 두 창만 보고 판정하지 않는다.
  귀무 = 날 블록 셔플. 판정 = 세 창 모두 CI 하한>0.5 ∧ 귀무 p95 초과.
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
SRCD = ROOT / "tmp/eth_anchor_features154_20260907"
RANK = ROOT / "tmp/eth_anchor_direction_feature_ranking_20260907"
OUT = ROOT / "tmp/eth_anchor_tabm3head_20260907"
SEEDS = [11, 23, 47, 71, 97, 131, 197, 251]
K, HIDDEN, LAYERS, DROPOUT = 32, 32, 2, 0.10
BATCH, LR, WD, EPOCHS, PATIENCE = 256, 2.0e-3, 2.0e-4, 120, 12
W_SIZE, W_TIME = 0.4, 0.3
EMBARGO = 48
BOOT = 1500


class ThreeHeadTabM(nn.Module):
    """`train_eval_omega1_2_tabm_3head_20260603.ThreeHeadTabM` 백본 축자 차용 + 헤드 교체."""

    def __init__(self, n_features: int, n_out_a: int = 3, n_out_b: int = 3, n_out_c: int = 2,
                 three_head: bool = True) -> None:
        super().__init__()
        self.k, self.n_features, self.three_head = K, int(n_features), three_head
        self.input_scale = nn.Parameter(torch.randn(K, self.n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(K, self.n_features))
        self.in_proj = nn.Linear(self.n_features, HIDDEN)
        self.blocks = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(LAYERS - 1))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(K, HIDDEN) * 0.03 + 1.0) for _ in range(LAYERS - 1))
        self.norms = nn.ModuleList(nn.LayerNorm(HIDDEN) for _ in range(LAYERS))
        self.dropout = nn.Dropout(DROPOUT)
        self.head_a = nn.Linear(HIDDEN, n_out_a)
        if three_head:
            self.head_b = nn.Linear(HIDDEN, n_out_b)
            self.head_c = nn.Linear(HIDDEN, n_out_c)

    def encode(self, x):
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](self.in_proj(xk))))
        for i, layer in enumerate(self.blocks):
            res = h
            h = layer(h * self.expert_scale[i].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[i + 1](h))) + res
        return h

    def forward(self, x):
        h = self.encode(x)
        out = {"a": self.head_a(h)}
        if self.three_head:
            out["b"] = self.head_b(h); out["c"] = self.head_c(h)
        return out


def day_auc_ci(y, p, days, rng, B=BOOT):
    uniq = np.unique(days)
    if len(uniq) < 5:
        return (np.nan, np.nan)
    idx = {d: np.flatnonzero(days == d) for d in uniq}
    out = []
    for _ in range(B):
        ii = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) < 2:
            continue
        out.append(roc_auc_score(y[ii], p[ii]))
    if len(out) < B // 3:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def make_targets(D):
    y3 = D["y3"].to_numpy(int)
    tr = D["split"].to_numpy() == "TRAIN"
    q = np.nanquantile(D.loc[tr, "range_pct"], [1 / 3, 2 / 3])
    yb = np.digitize(D["range_pct"].to_numpy(), q).astype(int)          # 크기 3분위
    tw = D["t_win_min"].to_numpy(float)
    med = np.nanmedian(tw[tr & np.isfinite(tw) & (tw < np.inf)])
    yc = (np.where(np.isfinite(tw), tw, 1e9) <= med).astype(int)        # 빠른 도달?
    return y3, yb, yc


def fit_one(Xtr, ytr, wtr, Xho, yho, seed, three_head, curve_out=None):
    torch.manual_seed(seed); np.random.seed(seed)
    m = ThreeHeadTabM(Xtr.shape[1], three_head=three_head)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    Xt = torch.tensor(Xtr, dtype=torch.float32); Xh = torch.tensor(Xho, dtype=torch.float32)
    Ya = torch.tensor(ytr[0]); Yb = torch.tensor(ytr[1]); Yc = torch.tensor(ytr[2])
    W = torch.tensor(wtr, dtype=torch.float32)
    n = len(Xt); best, best_state, bad, curve = -9, None, 0, []
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            o = m(Xt[b]); wb = W[b]
            la = torch.nn.functional.cross_entropy(
                o["a"].reshape(-1, o["a"].shape[-1]), Ya[b].repeat_interleave(K), reduction="none"
            ).view(len(b), K).mean(1)
            loss = (la * wb).sum() / wb.sum().clamp(min=1)
            if three_head:
                for key, Y, w_ in (("b", Yb, W_SIZE), ("c", Yc, W_TIME)):
                    lx = torch.nn.functional.cross_entropy(
                        o[key].reshape(-1, o[key].shape[-1]), Y[b].repeat_interleave(K), reduction="none"
                    ).view(len(b), K).mean(1)
                    loss = loss + w_ * (lx * wb).sum() / wb.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pa = torch.softmax(m(Xh)["a"], -1).mean(1).numpy()
        cl = yho != 1
        sc = np.nan
        if cl.sum() > 20 and len(np.unique(yho[cl] == 2)) > 1:
            den = pa[cl][:, 0] + pa[cl][:, 2]
            sc = roc_auc_score((yho[cl] == 2).astype(int),
                               np.where(den > 0, pa[cl][:, 2] / np.maximum(den, 1e-12), 0.5))
        curve.append(float(sc) if np.isfinite(sc) else np.nan)
        if np.isfinite(sc) and sc > best:
            best, best_state, bad = sc, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        m.load_state_dict(best_state)
    if curve_out is not None:
        curve_out.append({"seed": seed, "three_head": three_head, "epochs": len(curve),
                          "best_ep": int(np.nanargmax(curve)) if np.isfinite(curve).any() else -1,
                          "best": best, "curve": curve})
    return m


def predict(m, X):
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(torch.tensor(X, dtype=torch.float32))["a"], -1).mean(1).numpy()


def run_config(D, X, cols, fcols, three_head, rng, shuffle=False, seeds=SEEDS, curves=None):
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    bar = D["bar_idx"].to_numpy()
    y3, yb, yc = make_targets(D)
    w = D["w_uniq"].to_numpy(float); w = np.where(np.isfinite(w) & (w > 0), w, 1e-6)
    tr = sp == "TRAIN"
    ya = y3.copy()
    if shuffle:                                          # 날 블록 셔플
        days = np.unique(day[tr]); perm = rng.permutation(days)
        src = {d: np.flatnonzero(tr & (day == d)) for d in days}
        for d, d2 in zip(days, perm):
            if len(src[d]):
                ya[src[d]] = np.resize(y3[src[d2]], len(src[d]))
    Xf = X[:, fcols]
    mu, sd = np.nanmean(Xf[tr], 0), np.nanstd(Xf[tr], 0) + 1e-9      # TRAIN 통계로만 표준화
    Xs = np.nan_to_num((Xf - mu) / sd).astype(np.float32)
    # 조기종료용 TRAIN 내부 홀드아웃: 시간순 마지막 15% + 엠바고
    ti = np.flatnonzero(tr); o = ti[np.argsort(bar[ti])]
    cut = int(len(o) * 0.85)
    fit_i, ho_i = o[:cut], o[cut + EMBARGO:]
    cw = compute_sample_weight("balanced", ya[fit_i]).astype(np.float32)
    models = [fit_one(Xs[fit_i], (ya[fit_i], yb[fit_i], yc[fit_i]), w[fit_i] * cw,
                      Xs[ho_i], ya[ho_i], s, three_head, curves) for s in seeds]
    out = {}
    for wname in ("VAL", "OOS", "HOLDOUT_SPENT"):
        te = sp == wname
        if te.sum() < 30:
            continue
        p = np.mean([predict(m, Xs[te]) for m in models], axis=0)
        cl = y3[te] != 1
        den = p[cl][:, 0] + p[cl][:, 2]
        dp = np.where(den > 0, p[cl][:, 2] / np.maximum(den, 1e-12), 0.5)
        yy = (y3[te][cl] == 2).astype(int)
        out[f"{wname}_n"] = int(cl.sum())
        out[f"{wname}_auc"] = float(roc_auc_score(yy, dp)) if len(np.unique(yy)) > 1 else np.nan
        lo, hi = day_auc_ci(yy, dp, day[te][cl], rng)
        out[f"{wname}_lo"], out[f"{wname}_hi"] = lo, hi
        out[f"{wname}_mixed"] = float(roc_auc_score((y3[te] == 1).astype(int), p[:, 1]))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(SRCD / "features154.parquet")
    meta = json.loads((SRCD / "meta.json").read_text())
    cols = [c for c in meta["feature_cols"] if c in D.columns]
    X = D[cols].to_numpy(np.float64); ci = {c: i for i, c in enumerate(cols)}
    sets = {"dirtop40": [ci[c] for c in json.loads((RANK / "dirtop40_train.json").read_text()) if c in ci],
            "all150": list(range(len(cols)))}
    nparam = sum(p.numel() for p in ThreeHeadTabM(len(sets["dirtop40"])).parameters())
    print(f"[입력] {D.shape} · 피쳐풀 {len(cols)} · split {D.split.value_counts().to_dict()}")
    print(f"[모델] k={K} h={HIDDEN} L={LAYERS} · 파라미터 {nparam:,} (TRAIN 3,237 대비 {nparam/3237:.1f}배)", flush=True)
    rows, curves = [], []
    for fs, fc in sets.items():
        for th in (True, False):
            t0 = time.time()
            r = run_config(D, X, cols, fc, th, rng, curves=curves)
            r.update({"featset": fs, "n_feat": len(fc), "three_head": th, "sec": round(time.time() - t0, 1)})
            rows.append(r)
            print(f"   {fs:<9}{'3head' if th else '1head':<7} "
                  f"VAL {r.get('VAL_auc',np.nan):.4f}[{r.get('VAL_lo',np.nan):.3f},{r.get('VAL_hi',np.nan):.3f}] · "
                  f"OOS {r.get('OOS_auc',np.nan):.4f}[{r.get('OOS_lo',np.nan):.3f},{r.get('OOS_hi',np.nan):.3f}] · "
                  f"3rd {r.get('HOLDOUT_SPENT_auc',np.nan):.4f}[{r.get('HOLDOUT_SPENT_lo',np.nan):.3f},"
                  f"{r.get('HOLDOUT_SPENT_hi',np.nan):.3f}] · {r['sec']}s", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "configs.csv", index=False)
    pd.DataFrame(curves).to_json(OUT / "epoch_curves.json", orient="records")
    print("\n=== ⭐에폭 곡선 진단 (09-06 규칙: 첫 에폭이 최고면 과적합) ===")
    C = pd.DataFrame(curves)
    for th in (True, False):
        s = C[C.three_head == th]
        if not len(s):
            continue
        print(f"   {'3head' if th else '1head'}: best_ep 중앙 {s.best_ep.median():.0f} · "
              f"ep0 최고 시드 {int((s.best_ep == 0).sum())}/{len(s)} · 총에폭 중앙 {s.epochs.median():.0f} · "
              f"best 홀드아웃 AUC {s.best.mean():.4f}±{s.best.std():.4f}")
    best = R.loc[R[["VAL_auc", "OOS_auc", "HOLDOUT_SPENT_auc"]].min(axis=1).idxmax()]
    print(f"\n[귀무] 최고 구성 {best.featset}/{'3head' if best.three_head else '1head'} 날 블록 셔플 B=20", flush=True)
    fc = sets[best.featset]
    nv = {w: [] for w in ("VAL", "OOS", "HOLDOUT_SPENT")}
    for _ in range(20):
        rr = run_config(D, X, cols, fc, bool(best.three_head), rng, shuffle=True, seeds=SEEDS[:2])
        for w in nv:
            nv[w].append(rr.get(f"{w}_auc", np.nan))
    verdict = {}
    for w in nv:
        a = np.array([x for x in nv[w] if np.isfinite(x)])
        obs = best.get(f"{w}_auc", np.nan); lo = best.get(f"{w}_lo", np.nan)
        verdict[w] = {"obs": float(obs), "lo": float(lo), "null_mean": float(a.mean()),
                      "null_p95": float(np.percentile(a, 95)), "p": float(np.mean(a >= obs)),
                      "pass": bool(lo > 0.5 and obs > np.percentile(a, 95))}
        print(f"   {w:<14} 관측 {obs:.4f}[{lo:.3f}] vs 귀무 평균 {a.mean():.4f} p95 {np.percentile(a,95):.4f}"
              f" (p={verdict[w]['p']:.3f}) → {'통과' if verdict[w]['pass'] else '미달'}", flush=True)
    allpass = all(v["pass"] for v in verdict.values())
    print(f"\n=== 세 창 모두 통과: {'✅ 예' if allpass else '❌ 아니오'} ===")
    (OUT / "summary.json").write_text(json.dumps({"configs": rows, "verdict": verdict},
                                                 indent=2, ensure_ascii=False, default=float))
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
