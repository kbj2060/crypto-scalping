#!/usr/bin/env python3
"""5분 카드 «닿는다면 위 먼저 vs 아래 먼저» — 호가가 캔들을 넘는가 (2026-09-27, 사용자 «5분 호가 기반 방향 카드 연구»).

왜: 캔들 26피쳐 HGB 는 30분 방향 AUC .528 · 5분 .522 로 대부분 동전이었다. 5분 이하 방향에서 이 저장소가 통과시킨 건
    호가(깊은 벽 ±50bp 불균형 → 다음 5분: DEV +7.11 · HO1 +7.69 · HO2 +4.82bp) 하나다. 그걸 카드 축으로 옮겨 잰다.
데이터: 서버 1초 패널 tmp/rt_probe_20260927/panel_1s.parquet (09-15 15:51 ~ 09-26 19:40 UTC, 다른 세션이 오늘 빌드)
        + 그 세션의 analyze.py 피쳐 정의(접두만 exec — 같은 정의를 두 번 쓰지 않는다). 둘 다 서버 gitignore 경로.
라벨: 결정 = 매 분 정각 t. 기준가 mid[t], 배리어 ± 0.5 × (t−300..t 초의 mid 고저폭). mid[t+1..t+300] 에서 먼저 닿는 쪽.
      같은 초 양쪽 = 모호(제외) · 미도달 = 방향 과제 제외(닿음 과제에서만).
피쳐(t 까지만): P 가격(dmid10/60/300·rv60·레인지위치·rg5) · F +체결/수급/OI/청산(직전 완결 분) · B +호가(depth imb 5~50bp·OFI·벽·QI·깊이).
분할: DEV < 2026-09-22T10:25Z ≤ TEST(=HO2, 4.4일 하락장). 🔴HO2 는 오늘 다른 세션이 깊은 벽 «스프레드»로 한 번 봤다 —
      이 카드 모델에는 첫 사용이지만 완전 미접촉은 아니다. 최종 판정은 배포 뒤 전진 장부로.
CI: 1시간 블록 부트스트랩(라벨 5분 겹침·초 자기상관 흡수). 연구 점수이지 승격 근거가 아니다.
    python scripts/research_eth_card5_orderbook_direction_20260927.py     # 서버에서(nice 19)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score as auc

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "tmp/rt_probe_20260927/analyze.py"
src = AN.read_text().split("# ── 1. 원천별 단독")[0]
g = {"__file__": str(AN)}
exec(compile(src, "analyze_prefix", "exec"), g)
P = g["P"]
SPLIT = int(pd.Timestamp("2026-09-22T10:25Z").timestamp())
mid = P.bt_mid.ffill(limit=5)
hi24 = mid.rolling(86400, min_periods=3600).max(); lo24 = mid.rolling(86400, min_periods=3600).min()
P["range_pos"] = (mid - lo24) / (hi24 - lo24)
P["dd_imb50_60"] = P.dd_imb50.rolling(60, min_periods=30).mean()
P["dd_imb10_60"] = P.dd_imb10.rolling(60, min_periods=30).mean()
P["dd_imb50_300"] = P.dd_imb50.rolling(300, min_periods=150).mean()
FE = {
    "P": ["dmid10", "dmid60", "dmid300", "rv60", "range_pos"],
    "F": ["tr_imb10", "tr_imb60", "tr_imb300", "wh_net60", "wh_net300", "rt_net60", "rt_net300", "oi_d60", "oi_d300",
          "lq_prev_net", "lq_prev_tot"],
    "B": ["dd_imb5", "dd_imb10", "dd_imb25", "dd_imb50", "dd_imb50_60", "dd_imb10_60", "dd_imb50_300", "dd_ofi10", "dd_ofi60",
          "dd_wall_asym", "bt_qi10", "bt_qi60", "dd_depth10", "bt_spread_bp"],
}

# ── 라벨: 매 분 정각, 1초 mid 로 다음 300초 선착 ─────────────────────────────────────────
idx = P.index.to_numpy()
full = pd.RangeIndex(idx.min(), idx.max() + 1)
m = mid.reindex(full).to_numpy()
W = np.lib.stride_tricks.sliding_window_view
dec = full[(full % 60 == 0) & (full - full[0] >= 86400)].to_numpy()          # 워밍업 하루(레인지 위치)
pos = dec - full[0]
pos = pos[(pos >= 300) & (pos + 300 < len(m))]
past = W(m, 301)[pos - 300]                                                   # t−300..t
fut = W(m, 300)[pos + 1]                                                      # t+1..t+300
c0 = m[pos]
rg = (np.nanmax(past, 1) - np.nanmin(past, 1)) / c0 * 1e4
up, dn = c0 * (1 + 0.5 * rg / 1e4), c0 * (1 - 0.5 * rg / 1e4)
hu, hd = fut >= up[:, None], fut <= dn[:, None]
fu = np.where(hu.any(1), hu.argmax(1), 999); fd = np.where(hd.any(1), hd.argmax(1), 999)
res = np.where((fu == 999) & (fd == 999), 2, np.where(fu == fd, -1, (fu < fd).astype(int)))
bad = ~np.isfinite(c0) | ~(rg > 0) | (np.isnan(fut).mean(1) > 0.05) | (np.isnan(past).mean(1) > 0.05)
res[bad] = -9
ts = pos + full[0]
D = P.reindex(ts)[sum(FE.values(), [])].copy()
D["res"], D["rg5"] = res, rg
D = D[D.res != -9]
print(f"결정 {len(D):,}분 · {pd.to_datetime(D.index.min(), unit='s')} ~ {pd.to_datetime(D.index.max(), unit='s')} UTC · "
      f"결과 {D.res.value_counts(normalize=True).round(3).to_dict()}", flush=True)
FE["P"] = FE["P"] + ["rg5"]


def boot_auc(y, p, blk, B=500, rng=np.random.default_rng(0)):
    u, inv = np.unique(blk, return_inverse=True)
    v = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
        v.append(auc(y, p, sample_weight=w))
    return np.percentile(v, [2.5, 97.5])


def run(task: str) -> None:
    X = D[D.res.isin((0, 1))] if task == "dir" else D[D.res.isin((0, 1, 2, -1))]
    y = ((X.res == 1) if task == "dir" else (X.res != 2)).astype(int).to_numpy()
    tr, te = X.index < SPLIT, X.index >= SPLIT
    blk = X.index[te] // 3600
    print(f"\n[{task}] DEV {tr.sum():,} · TEST {te.sum():,}({len(np.unique(blk))}시간) · 기저 TEST {y[te].mean():.3f}", flush=True)
    for name, cols in (("P 가격", FE["P"]), ("P+F 체결·OI·청산", FE["P"] + FE["F"]), ("P+F+B 호가", FE["P"] + FE["F"] + FE["B"]),
                       ("B 호가만", FE["B"])):
        for mname, mk in (("HGB", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                                                          min_samples_leaf=300, l2_regularization=1.0,
                                                                          early_stopping=True, validation_fraction=0.2,
                                                                          n_iter_no_change=30, random_state=20260927)),
                          ("로지스틱", lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)))):
            mdl = mk().fit(X.loc[tr, cols], y[tr])
            p = mdl.predict_proba(X.loc[te, cols])[:, 1]
            ci = boot_auc(y[te], p, blk)
            s5 = np.abs(p - y[tr].mean()) >= .05
            extra = f" · 기저서 5pp↑ {s5.mean():.3f} 적중 {((p[s5] > y[tr].mean()) == y[te][s5]).mean():.3f}" if task == "dir" and s5.any() else ""
            print(f"  {name:16s} {mname:5s} AUC {auc(y[te], p):.4f} [{ci[0]:.4f},{ci[1]:.4f}]{extra}", flush=True)
    if task == "dir":   # 무모델 규칙: 깊은 벽(dd_imb50, DEV 분위 q80/q20) → 그쪽
        q80, q20 = X.loc[tr, "dd_imb50"].quantile([.8, .2])
        Z = X[te]; yy = y[te]
        for nm, sel, sd in (("매수벽 q80↑", Z.dd_imb50 >= q80, 1), ("매도벽 q20↓", Z.dd_imb50 <= q20, 0)):
            print(f"  규칙 {nm}: n {sel.sum():,} · 그쪽 먼저 {(yy[sel.to_numpy()] == sd).mean():.3f} (기저 {yy.mean() if sd else 1 - yy.mean():.3f})")


run("dir")
run("reach")
