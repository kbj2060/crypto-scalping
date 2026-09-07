#!/usr/bin/env python3
"""**단기 지평 방향** 정식 검정 — first-touch · 지평격자 · 경제성 · 대조군 (2026-09-07, 서버 GPU).

사용자: *"1~4번 돌려줘"*

## 배경 (왜 이걸 도는가)
세션 내내 방향을 20번 기각했는데, 전부 **H=48봉(4시간)** 이었다. 매봉 모집단에서 지평을 바꿔 보니:
    종가진입·30분   0.7381   ·  다음봉시가·30분  0.7368
    다음봉시가·4시간 **0.5590**  (앵커 축 측정 0.53~0.56 과 일치)
**지평이 전부였다.** 진입가·모집단은 부차적이었다. 짧은 지평은 한 번도 안 봤다.

## 그러나 그 0.74 는 아직 못 믿는다
그 라벨은 `reb > cont` -- **어느 쪽이 더 멀리 갔나**다. 거래는 **어느 쪽에 먼저 닿나**로 한다.
30분 안에 되돌림이 더 멀리 가지만 지속이 먼저 닿을 수 있고, 그러면 못 쓴다.
이 세션에서 반복된 패턴이다(AUC 0.544 인데 상위30% 리프트 음수 · 이탈폭 AUC 0.75 인데 스톱폭 0/6).

## 네 갈래
1. **first-touch 라벨** -- ±K×ATR 배리어 중 먼저 닿는 쪽. 5분봉 고가/저가로 판정
   (이 저장소 배리어 컨벤션). 같은 봉에 양쪽 다 닿으면 **모호**로 분리해 보고한다.
2. **지평 격자** H ∈ {6, 12, 24, 48}봉 -- 어디서 무너지는지
3. **경제성** -- 진입 = open[t+1], 승 +K×ATR bp · 패 −K×ATR bp · 왕복 7.8bp.
   상위 X% 진입 시 건당 순손익. '항상 되돌림' · 무작위 대조 포함.
4. **대조군** -- 방향 뒤집기(TRAIN 라벨만) · 날 블록 셔플 귀무 · **무작위 direction 할당**

## 판정
first-touch 라벨에서 세 창 모두 CI 하한 > 0.5 ∧ 귀무 초과 ∧ **비용 후 건당 > 0** 이어야 산다.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_labels.csv"
FEA = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_features_tier0.csv"
OUT = ROOT / "tmp/eth_short_horizon_direction_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H_GRID = (6, 12, 24, 48)
K_BAR = 1.5
COST_BP = 7.8
SEED, N_EST, BOOT, MAX_CTX = 20260907, 4, 800, 12000


def day_ci(y, p, d, rng, B=BOOT):
    u = np.unique(d)
    if len(u) < 5: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        if len(np.unique(y[i])) > 1: o.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) > B // 3 else (np.nan, np.nan)


def dayperm(y, d, rng):
    u = np.unique(d); src = {x: np.flatnonzero(d == x) for x in u}
    pm = rng.permutation(u); z = y.copy()
    for a, b in zip(u, pm):
        if len(src[a]): z[src[a]] = np.resize(y[src[b]], len(src[a]))
    return z


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    import importlib.util, torch
    from tabpfn import TabPFNClassifier
    s = importlib.util.spec_from_file_location("ab", ROOT / "scripts/build_eth_anchor_label_dataset_20260907.py")
    B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
    s2 = importlib.util.spec_from_file_location("vr", ROOT / "scripts/live_eth_sweep_v_rebound_signal_20260829.py")
    VR = importlib.util.module_from_spec(s2); s2.loader.exec_module(VR)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    L = pd.read_csv(LAB); F = pd.read_csv(FEA)
    for d in (L, F):
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_localize(None)
    feats = [c for c in VR.FEATURES if c in F.columns]
    D = L.merge(F[["timestamp", "direction"] + feats], on=["timestamp", "direction"]).sort_values("timestamp").reset_index(drop=True)
    kl = B._load_kl(B.ETH_KL)
    pos = pd.Series(np.arange(len(kl)), index=pd.to_datetime(kl["timestamp"]))
    idx = pos.reindex(D["timestamp"]).to_numpy(); keep = ~pd.isna(idx)
    D = D[keep].reset_index(drop=True); idx = idx[keep].astype(int)
    OP = kl["open"].to_numpy(float); HI = kl["high"].to_numpy(float); LO = kl["low"].to_numpy(float)
    atr = D["atr"].to_numpy(float)                       # 가격 단위
    isdn = (D["direction"].to_numpy() == "downside")     # 하락트리거 → 되돌림은 위
    ts = D["timestamp"]
    sp = np.where(ts < "2025-09-01", "TRAIN", np.where(ts < "2026-01-01", "VAL",
         np.where(ts < "2026-04-01", "OOS", "HOLDOUT_SPENT")))
    day = ts.dt.floor("D").to_numpy()
    X = D[feats].to_numpy(np.float64)
    print(f"[입력] {len(D):,}행 · 피쳐 {len(feats)} · {dev}", flush=True)

    def labels(H):
        """first-touch(±K×ATR) 와 max-excursion 을 같은 진입가(open[t+1])에서."""
        ft = np.full(len(D), np.nan); mx = np.full(len(D), np.nan)
        amb = np.zeros(len(D), bool); bp = np.full(len(D), np.nan)
        for j, i in enumerate(idx):
            a, b = i + 1, i + 1 + H
            if b > len(kl) or not np.isfinite(atr[j]) or atr[j] <= 0: continue
            e = OP[a]; up_b = e + K_BAR * atr[j]; dn_b = e - K_BAR * atr[j]
            h = HI[a:b]; l = LO[a:b]
            iu = np.argmax(h >= up_b) if (h >= up_b).any() else 10**9
            il = np.argmax(l <= dn_b) if (l <= dn_b).any() else 10**9
            mx[j] = float(((h.max() - e) if isdn[j] else (e - l.min()))
                          > ((e - l.min()) if isdn[j] else (h.max() - e)))
            if iu == 10**9 and il == 10**9: continue          # 미해소
            if iu == il: amb[j] = True; continue              # 같은 봉 양쪽 -- 모호
            first_up = iu < il
            ft[j] = float(first_up if isdn[j] else (not first_up))
            bp[j] = K_BAR * atr[j] / e * 1e4                  # 배리어 폭(bp)
        return ft, mx, amb, bp

    def run(y, ok, tag, shuffle=False, flip=False, rand_dir=None):
        tr = ok & (sp == "TRAIN")
        if tr.sum() < 300: return None
        yt = y.copy()
        if flip: yt[tr] = 1 - yt[tr]
        if shuffle: yt[tr] = dayperm(y[tr], day[tr], rng)
        if tr.sum() > MAX_CTX:
            sel = rng.choice(np.flatnonzero(tr), MAX_CTX, replace=False)
            m = np.zeros(len(D), bool); m[sel] = True; tr = m
        c = TabPFNClassifier(device=dev, n_estimators=N_EST if not shuffle else 2,
                             random_state=SEED, ignore_pretraining_limits=True, memory_saving_mode=True)
        c.fit(np.nan_to_num(X[tr]).astype(np.float32), yt[tr].astype(int))
        out = {}
        for w in WINS:
            te = ok & (sp == w)
            if te.sum() < 40 or len(np.unique(y[te])) < 2: continue
            p = c.predict_proba(np.nan_to_num(X[te]).astype(np.float32))[:, 1]
            out[w] = {"auc": roc_auc_score(y[te].astype(int), p), "pred": p, "mask": te}
            out[w]["lo"], _ = day_ci(y[te].astype(int), p, day[te], rng)
        return out

    rows, keep_best = [], {}
    print("\n" + "=" * 108, flush=True)
    print(f"{'H':>3}{'라벨':<14}{'n':>8}{'양성':>7}" + "".join(f"{w[:3]:>17}" for w in WINS) + "   mean3", flush=True)
    print("=" * 108, flush=True)
    for H in H_GRID:
        ft, mx, amb, bp = labels(H)
        for nm, y in (("max-excursion", mx), ("⭐first-touch", ft)):
            ok = np.isfinite(y)
            r = run(y, ok, nm)
            if r is None: continue
            a = [r[w]["auc"] for w in WINS if w in r]
            line = f"{H:>3}{nm:<14}{ok.sum():>8,}{np.nanmean(y[ok]):>7.3f}"
            for w in WINS:
                line += f"  {r[w]['auc']:.4f}[{r[w]['lo']:.3f}]" if w in r else f"{'-':>17}"
            m3 = float(np.mean(a))
            ci3 = all(r[w]["lo"] > 0.5 for w in WINS if w in r)
            rows.append({"H": H, "label": nm, "n": int(ok.sum()), "pos": float(np.nanmean(y[ok])),
                         "mean3": m3, "ci3": bool(ci3),
                         **{f"{w}_auc": r[w]["auc"] for w in WINS if w in r}})
            print(line + f"  {m3:.4f}{'  ✅' if ci3 else '  ❌'}", flush=True)
            if nm.endswith("first-touch"):
                keep_best[H] = (y, ok, r, bp, amb)
        print(f"      (H={H} 모호봉 {amb.sum():,} · 미해소 {int((~np.isfinite(ft)).sum() - amb.sum()):,})", flush=True)

    A = pd.DataFrame(rows); A.to_csv(OUT / "grid.csv", index=False)
    ftA = A[A.label.str.endswith("first-touch")]
    if not len(ftA): print("\nfirst-touch 결과 없음"); return 1
    bH = int(ftA.loc[ftA.mean3.idxmax(), "H"])
    y, ok, r, bp, amb = keep_best[bH]
    print("\n" + "=" * 108, flush=True)
    print(f"⭐최고 first-touch 셀: H={bH}봉({bH*5}분) · mean3 {ftA.mean3.max():.4f}", flush=True)
    print("=" * 108, flush=True)

    print("\n[3] 경제성 -- 상위 X% 진입, 비용 후 건당 bp", flush=True)
    for q in (1.00, 0.50, 0.30, 0.10):
        line = f"   상위 {q:.0%} 진입"
        for w in WINS:
            if w not in r: continue
            te = r[w]["mask"]; p = r[w]["pred"]; yy = y[te]; bb = bp[te]
            k = max(10, int(len(p) * q)); sel = np.argsort(-np.abs(p - 0.5))[:k]
            # 예측 방향으로 진입: p>0.5 면 되돌림 쪽, 아니면 지속 쪽
            correct = np.where(p[sel] > 0.5, yy[sel], 1 - yy[sel])
            net = np.where(correct > 0.5, bb[sel], -bb[sel]) - COST_BP
            line += f"  {w[:3]} {net.mean():+.2f}bp(n={k})"
        print(line, flush=True)
    line = "   대조: 항상 되돌림"
    for w in WINS:
        if w not in r: continue
        te = r[w]["mask"]; yy = y[te]; bb = bp[te]
        net = np.where(yy > 0.5, bb, -bb) - COST_BP
        line += f"  {w[:3]} {net.mean():+.2f}bp"
    print(line, flush=True)
    print(f"   배리어 폭 중앙 {np.nanmedian(bp[ok]):.1f}bp · 비용 {COST_BP}bp "
          f"→ 손익분기 정확도 {(np.nanmedian(bp[ok])+COST_BP)/(2*np.nanmedian(bp[ok])):.1%}", flush=True)

    print("\n[4] 대조군", flush=True)
    rf = run(y, ok, "flip", flip=True)
    if rf:
        print("   방향뒤집기(TRAIN 라벨만) " + " ".join(
            f"{w[:3]} {rf[w]['auc']:.4f}(원본 {r[w]['auc']:.4f}, 합 {rf[w]['auc']+r[w]['auc']:.4f})"
            for w in WINS if w in rf and w in r), flush=True)
    nl = {w: [] for w in WINS}
    for _ in range(8):
        rs = run(y, ok, "null", shuffle=True)
        if rs:
            for w in WINS:
                if w in rs: nl[w].append(rs[w]["auc"])
    line = "   날블록 귀무 B=8      "
    for w in WINS:
        if nl[w] and w in r:
            p95 = float(np.percentile(nl[w], 95))
            line += f"  {w[:3]} p95 {p95:.4f}{'✅' if r[w]['auc'] > p95 else '🔴'}"
    print(line, flush=True)
    # 무작위 direction 할당
    rd = rng.random(len(D)) < 0.5
    reb2 = np.where(rd, 1 - y, y)
    rr = run(np.where(ok, reb2, np.nan), ok, "randdir")
    if rr:
        print("   무작위 direction 할당 " + " ".join(
            f"{w[:3]} {rr[w]['auc']:.4f}" for w in WINS if w in rr), flush=True)
    json.dump({"grid": rows, "best_H": bH}, open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
