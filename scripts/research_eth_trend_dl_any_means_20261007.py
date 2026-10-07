"""ETH 30·60분 방향 — 딥러닝·시계열 기초모델·스태킹 «무슨 수를 써서라도» (2026-10-07).

사용자: «정확도 높은 추세 모델을 딥러닝이든 LLM 이든 무슨 수를 써서라도». 이미 닫힌 것(재실행 안 함):
  GRU/Transformer/xLSTM/TCN 5분봉 .51~.53(09-13) · LLM jevk5 .44~.52(10-05) · TabPFN = HGB(09-26) · HGB .535(10-07 같은 날).
여기서 새로 재는 것(결과 보기 전 고정):
  A GRU(1분봉 240개 × 4채널: 수익·폭·거래량z·테이커불균형) + 5분봉 정적 33피쳐 결합, 두 머리(30·60분), 무작위 시드 5개 평균.
  B Chronos-2 제로샷: 1분봉 512 · 1분봉 512+공변량(거래량·불균형) · 5분봉 512. p_up = 예측 분위 CDF 를 현재가에서 보간.
  C 스태킹: VAL 표본에 로지스틱(HGB·GRU·Chronos 3종) → TEST 표본.
  비교: HGB(같은 33피쳐) · 모델 없는 1피쳐(직전 1시간 반대). 판정 = TEST AUC · HGB 대비 일 블록 CI.
  «정확도» 는 확신 상위 1/5/10/20%(문턱은 VAL 에서)만 예측할 때 적중·bp 로 따로 보고한다.
  D 방향 대신 «다음 60분이 추세장인가»(효율비 ≥ TRAIN 중앙) · «크게 움직이나»(|60분| ≥ TRAIN 중앙) HGB AUC.
라벨·분할·경계는 research_eth_trend30_60_hold_policy_20261007.py 와 같다(5분봉 τ 종가 기준, 1분 창 끝 = τ 마지막 1분, 종가 일치 assert).
실행: python scripts/research_eth_trend_dl_any_means_20261007.py [--selftest]   (QUICK=1 로 축소판)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_eth_trend30_60_hold_policy_20261007 import boot_ci, features, load_1m, ms, to_5m  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/trend_dl_any_means_20261007"
QUICK = os.environ.get("QUICK") == "1"
TR0, TR_END, VA_END, T26 = ms("2022-04-01"), ms("2024-07-01"), ms("2025-01-01"), ms("2026-01-01")
L, HZ = 240, (30, 60)
SEED = 20261007
SEEDS = [int(s) for s in np.random.default_rng(SEED).integers(1, 2**31 - 1, 2 if QUICK else 5)]
N_CHR_VA, N_CHR_TE = (200, 400) if QUICK else (2000, 4000)
torch.set_num_threads(os.cpu_count() or 4)


def prep():
    k1 = load_1m(); b = to_5m(k1); X = features(b)
    t1, c1 = k1.t.to_numpy(), k1.c.to_numpy()
    lc = np.log(c1)
    r = np.diff(lc, prepend=lc[0]) * 1e4
    v = k1.v.to_numpy(); lv = np.log1p(v)
    lvz = lv - pd.Series(lv).rolling(1440, min_periods=60).mean().to_numpy()
    imb = np.where(v > 0, 2 * k1.tb.to_numpy() / np.maximum(v, 1e-9) - 1, 0.0)
    C = np.stack([np.clip(r, -100, 100), np.clip(np.log(k1.h / k1.l).to_numpy() * 1e4, 0, 200),
                  np.nan_to_num(lvz), imb], 1).astype(np.float32)
    bt, c = b.t.to_numpy(), b.c.to_numpy()
    end = np.searchsorted(t1, bt + 240_000)                               # τ 의 마지막 1분봉
    end_ok = (end < len(t1)) & (t1[np.minimum(end, len(t1) - 1)] == bt + 240_000) & (end >= 1440)
    Y, FWD = {}, {}
    for h in HZ:
        n = h // 5
        f = np.log(np.roll(c, -n) / c) * 1e4; f[-n:] = np.nan
        FWD[h], Y[h] = f, (f > 0).astype(np.float32)
    e = np.minimum(end, len(lc) - 61)
    er = np.abs(lc[e + 60] - lc[e]) / np.maximum(np.abs(np.diff(lc))[e[:, None] + np.arange(60)].sum(1), 1e-12)
    valid = end_ok & np.isfinite(FWD[60]) & (FWD[30] != 0) & (FWD[60] != 0) & np.isfinite(X.ret288.to_numpy()) & (end + 61 < len(lc))
    assert np.allclose(c1[end[valid]], c[valid]), "1분 창 끝 종가 ≠ 5분봉 종가"
    return dict(t=bt, end=end, C=C, X=X, Y=Y, FWD=FWD, er=er, valid=valid, c1=c1, lv=lv, imb=imb, c=c, lc=lc)


class Net(nn.Module):
    def __init__(self, nc: int, ns: int, hd: int = 48):
        super().__init__()
        self.gru = nn.GRU(nc, hd, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hd + ns, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, len(HZ)))

    def forward(self, seq, st):
        _, h = self.gru(seq)
        return self.head(torch.cat([h[-1], st], 1))


def gru_fit_predict(D, tr, va, seed) -> np.ndarray:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    C, end = D["C"], D["end"]
    mu, sd = C[end[tr]].mean(0), C[end[tr]].std(0) + 1e-6
    Xs = D["X"].to_numpy(np.float32)
    xm = np.nanmedian(Xs[tr], 0); Xs = np.where(np.isnan(Xs), xm, Xs)
    Xs = ((Xs - Xs[tr].mean(0)) / (Xs[tr].std(0) + 1e-6)).astype(np.float32)
    Yt = np.stack([D["Y"][h] for h in HZ], 1)
    idx_tr, idx_va = np.flatnonzero(tr), np.flatnonzero(va)
    if QUICK:
        idx_tr = rng.choice(idx_tr, 20_000, replace=False)
    net = Net(C.shape[1], Xs.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3); lossf = nn.BCEWithLogitsLoss()
    ar = np.arange(-L + 1, 1)

    def batch(ii):
        return torch.from_numpy((C[end[ii][:, None] + ar] - mu) / sd), torch.from_numpy(Xs[ii])

    def predict(ii):
        net.eval(); out = []
        with torch.no_grad():
            for j in range(0, len(ii), 4096):
                out.append(torch.sigmoid(net(*batch(ii[j:j + 4096]))).numpy())
        net.train(); return np.concatenate(out)

    best, best_state = -1.0, None
    for ep in range(2 if QUICK else 4):
        t0 = time.time(); perm = rng.permutation(idx_tr)
        for j in range(0, len(perm), 1024):
            ii = perm[j:j + 1024]
            s, x = batch(ii)
            loss = lossf(net(s, x), torch.from_numpy(Yt[ii]))
            opt.zero_grad(); loss.backward(); opt.step()
        pv = predict(idx_va)
        auc = np.mean([roc_auc_score(Yt[idx_va, k], pv[:, k]) for k in range(len(HZ))])
        print(f"  seed {seed} ep{ep} VAL AUC(30·60 평균) {auc:.4f} · {time.time() - t0:.0f}s", flush=True)
        if auc > best:
            best, best_state = auc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    P = np.full((len(end), len(HZ)), np.nan, np.float32)
    ok = np.flatnonzero(D["valid"]); P[ok] = predict(ok)
    return P


def chronos_p(D, rows) -> dict[str, np.ndarray]:
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    ql = [0.01, 0.05] + [round(x, 2) for x in np.arange(0.1, 0.91, 0.1)] + [0.95, 0.99]
    end, c1, c = D["end"], D["c1"], D["c"]

    def cdf_up(qs, cur, steps):
        out = np.empty((len(qs), len(steps)))
        for i, q in enumerate(qs):
            q = np.sort(q[0].numpy(), axis=-1)                              # (pred_len, nq)
            for j, s in enumerate(steps):
                out[i, j] = 1 - np.interp(cur[i], q[s - 1], ql, left=0.0, right=1.0)
        return out

    res = {}
    e = end[rows]
    specs = {
        "chr1m": ([c1[x - 511:x + 1] for x in e], c1[e], (30, 60), 60),
        "chr1m_cov": ([{"target": c1[x - 511:x + 1], "past_covariates": {"lv": D["lv"][x - 511:x + 1], "imb": D["imb"][x - 511:x + 1]}}
                       for x in e], c1[e], (30, 60), 60),
        "chr5m": ([c[x - 511:x + 1] for x in rows], c[rows], (6, 12), 12),
    }
    for nm, (inp, cur, steps, pl) in specs.items():
        t0 = time.time(); qs = []
        for j in range(0, len(inp), 256):
            q, _ = pipe.predict_quantiles(inp[j:j + 256], prediction_length=pl, quantile_levels=ql)
            qs += q
        res[nm] = cdf_up(qs, cur, steps)
        print(f"  {nm} {len(inp)}건 {time.time() - t0:.0f}s", flush=True)
    return res


def auc_diff_ci(y, a, b, day, rng, n=500) -> list[float]:
    u, inv = np.unique(day, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(u))]
    d = []
    for _ in range(n):
        ii = np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))])
        if y[ii].min() == y[ii].max():
            continue
        d.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii]))
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def selective(p, y, f, day, va_abs, rng) -> dict:
    out = {}
    for cov in (0.01, 0.05, 0.10, 0.20, 1.0):
        thr = np.quantile(va_abs, 1 - cov) if cov < 1 else -1
        m = np.abs(p - 0.5) >= thr
        s = np.sign(p[m] - 0.5)
        out[f"top{int(cov * 100)}%"] = dict(share=float(m.mean()), acc=float(((p[m] > 0.5) == (y[m] == 1)).mean()),
                                         bp=float((s * f[m]).mean()), bp_ci=boot_ci(s * f[m], day[m], rng))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    D = prep(); t, valid = D["t"], D["valid"]
    tr = valid & (t >= TR0) & (t < TR_END - 12 * 300_000)
    va = valid & (t >= TR_END) & (t < VA_END); te = valid & (t >= VA_END)
    print(f"행 TRAIN {tr.sum()} VAL {va.sum()} TEST {te.sum()} · 시드 {SEEDS}", flush=True)
    res: dict = {"seeds": SEEDS, "n": dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()))}
    X = D["X"]
    P = {}
    for h in HZ:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=False, random_state=SEED)
        P[f"hgb{h}"] = m.fit(X[tr], D["Y"][h][tr]).predict_proba(X)[:, 1]
        P[f"rev1h{h}"] = 1 / (1 + np.exp(X.ret12.to_numpy() / 50))         # 모델 없는 1피쳐: 직전 1시간 반대
    G = [gru_fit_predict(D, tr, va, s) for s in SEEDS]
    for k, h in enumerate(HZ):
        P[f"gru{h}"] = np.mean([g[:, k] for g in G], 0)
        res[f"gru{h}_per_seed_test_auc"] = [float(roc_auc_score(D["Y"][h][te], g[te, k])) for g in G]

    for h in HZ:
        y, f, day = D["Y"][h], D["FWD"][h], t // 86_400_000
        r = {}
        for nm in (f"hgb{h}", f"gru{h}", f"rev1h{h}"):
            p = P[nm]
            r[nm] = dict(test_auc=float(roc_auc_score(y[te], p[te])), test26_auc=float(roc_auc_score(y[te & (t >= T26)], p[te & (t >= T26)])),
                         selective=selective(p[te], y[te], f[te], day[te], np.abs(p[va] - 0.5), rng))
        r["gru_minus_hgb_auc_ci"] = auc_diff_ci(y[te], P[f"gru{h}"][te], P[f"hgb{h}"][te], day[te], rng)
        res[f"h{h}"] = r
        print(f"\n[{h}분] TEST AUC hgb {r[f'hgb{h}']['test_auc']:.4f} · gru {r[f'gru{h}']['test_auc']:.4f} "
              f"(시드 {[round(a, 4) for a in res[f'gru{h}_per_seed_test_auc']]}) · 1h반대 {r[f'rev1h{h}']['test_auc']:.4f} · "
              f"gru−hgb CI {r['gru_minus_hgb_auc_ci']}", flush=True)
        for nm in (f"hgb{h}", f"gru{h}"):
            for k, v in r[nm]["selective"].items():
                print(f"   {nm} {k:6s} 비중 {v['share']:.3f} 적중 {v['acc']:.3f} 따라가기 {v['bp']:+.2f}bp [{v['bp_ci'][0]:+.2f},{v['bp_ci'][1]:+.2f}]")

    # B·C Chronos-2 + 스태킹 (표본)
    rows_va = rng.choice(np.flatnonzero(va), N_CHR_VA, replace=False)
    rows_te = np.sort(rng.choice(np.flatnonzero(te), N_CHR_TE, replace=False))
    CH = chronos_p(D, np.concatenate([rows_va, rows_te]))
    nv = len(rows_va)
    for k, h in enumerate(HZ):
        y = D["Y"][h]
        cols = {nm: np.concatenate([P[nm][rows_va], P[nm][rows_te]]) for nm in (f"hgb{h}", f"gru{h}")}
        cols.update({nm: v[:, k] for nm, v in CH.items()})
        Z = np.stack(list(cols.values()), 1); yy = y[np.concatenate([rows_va, rows_te])]
        lr_ = LogisticRegression(C=1.0).fit(Z[:nv], yy[:nv]); stack = lr_.predict_proba(Z[nv:])[:, 1]
        dte = t[rows_te] // 86_400_000
        s = {nm: float(roc_auc_score(yy[nv:], v[nv:])) for nm, v in cols.items()}
        s["stack"] = float(roc_auc_score(yy[nv:], stack))
        s["stack_minus_hgb_ci"] = auc_diff_ci(yy[nv:], stack, cols[f"hgb{h}"][nv:], dte, rng)
        s["stack_coef"] = dict(zip(cols, lr_.coef_[0].round(3).tolist()))
        res[f"chronos_stack_h{h}"] = s
        print(f"\n[{h}분 · TEST 표본 {len(rows_te)}] " + " · ".join(f"{a} {b:.4f}" for a, b in s.items() if isinstance(b, float))
              + f" · 스택−hgb CI {s['stack_minus_hgb_ci']} · 계수 {s['stack_coef']}", flush=True)

    # D 방향 대신 «추세장인가»·«크게 움직이나»
    for nm, lab in (("trendy60", D["er"]), ("bigmove60", np.abs(D["FWD"][60]))):
        thr = np.median(lab[tr]); y = (lab >= thr).astype(int)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=False, random_state=SEED).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        res[nm] = dict(test_auc=float(roc_auc_score(y[te], p)), thr=float(thr),
                       acc_top20=float((y[te][p >= np.quantile(p, 0.8)]).mean()), acc_bot20=float(1 - y[te][p <= np.quantile(p, 0.2)].mean()))
        print(f"[{nm}] TEST AUC {res[nm]['test_auc']:.4f} · 상위20% 적중 {res[nm]['acc_top20']:.3f} · 하위20% 적중 {res[nm]['acc_bot20']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("result_quick.json" if QUICK else "result.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))


def selftest() -> None:
    rng = np.random.default_rng(0)
    p = np.array([0.9, 0.1, 0.6, 0.4]); y = np.array([1, 0, 0, 1]); f = np.array([5.0, -5.0, -1.0, 1.0])
    s = selective(p, y, f, np.arange(4), np.abs(p - 0.5), rng)
    assert s["top100%"]["acc"] == 0.5 and s["top100%"]["bp"] == 2.0
    assert s["top20%"]["share"] == 0.5 and s["top20%"]["acc"] == 1.0          # |p−.5|=.4 두 건이 VAL 상위 20% 문턱 이상
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
