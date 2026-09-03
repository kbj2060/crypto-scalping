#!/usr/bin/env python3
"""진입 모델 라이브 추론 실측: HGB 동결본 vs TabPFN 5멤버 (2026-09-03, 서버 GPU).

목적은 하나다 -- "TabPFN 분류로 바꿔도 되나"에서 **통계가 아닌 운영 쪽 장벽**이
진짜인지 숫자로 가른다. 2026-09-03 대시보드 타임아웃 사고(V자반등 3초→42.95초)의
원인 프로세스가 b15였으므로, 진입 경로에 TabPFN을 하나 더 얹는 비용을 재야 한다.

재는 것:
  · 콜드스타트: 적합 시간 (재시작마다 치르는 값)
  · 상주 GPU 메모리: 5멤버 컨텍스트를 물고 있을 때 (8.2GB 중 얼마)
  · 봉당 채점 지연: n=1/4/16행 (p50/p95, 워밍업 후 30회)
  · CPU 폴백 지연: GPU가 경합으로 막혔을 때의 최악값
"""
from __future__ import annotations

import json, os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
FROZEN = ROOT / "tmp/eth_entry_limit_fade_v1_20260903/model.joblib"
OUT = ROOT / "tmp/eth_entry_live_inference_cost_20260903"
DEPTH, WAIT, LABEL_THR, SUB = 3.0, 6, 0.0040, 18000


def log(m): print(f"[cost] {m}", flush=True)


def gpu_free_gb():
    import torch
    if not torch.cuda.is_available(): return None
    f, t = torch.cuda.mem_get_info()
    return f / 1e9, t / 1e9


def bench(fn, n_warm=3, n_rep=30):
    for _ in range(n_warm): fn()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    a = np.array(ts) * 1000.0
    return float(np.percentile(a, 50)), float(np.percentile(a, 95))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    import torch
    from tabpfn import TabPFNClassifier
    from sklearn.ensemble import HistGradientBoostingRegressor
    from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP

    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "split", "timestamp", "i",
                       "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in D.columns if c.endswith("_r136")] + \
        [c for c in D.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if D[c].dtype.kind in "fiub"]))
    FEATS = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    X = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (D.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    y = D["y"].to_numpy(); lab = (y > LABEL_THR).astype(int)
    itr = np.flatnonzero(tr)
    log(f"TRAIN {len(itr):,} · 피쳐 {len(FEATS)}")
    res = {"n_features": len(FEATS), "n_train": int(len(itr)), "sub": SUB, "seeds": list(SEEDS)}

    # ---------- HGB ----------
    t0 = time.perf_counter()
    hgb = [HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP).fit(X[tr], y[tr])
           for s in SEEDS]
    res["hgb_fit_s"] = round(time.perf_counter() - t0, 2)
    sz = FROZEN.stat().st_size / 1e6 if FROZEN.exists() else None
    res["hgb_artifact_mb"] = round(sz, 2) if sz else None
    log(f"HGB 5시드 적합 {res['hgb_fit_s']}초 · 동결 artifact {res['hgb_artifact_mb']}MB")
    for n in (1, 4, 16):
        Q = X.iloc[:n]
        p50, p95 = bench(lambda: np.mean([m.predict(Q) for m in hgb], axis=0))
        res[f"hgb_ms_n{n}"] = {"p50": round(p50, 3), "p95": round(p95, 3)}
        log(f"  HGB 채점 n={n:<2d} p50 {p50:8.3f}ms  p95 {p95:8.3f}ms")

    # ---------- TabPFN (GPU) ----------
    f0, tot = gpu_free_gb()
    log(f"GPU 적합 전 여유 {f0:.2f}GB / 총 {tot:.2f}GB")
    t0 = time.perf_counter()
    tp = []
    for k in range(5):
        rs = np.random.default_rng(SEEDS[k]).choice(itr, size=min(SUB, len(itr)), replace=False)
        m = TabPFNClassifier(device="cuda", random_state=SEEDS[k])
        m.fit(X.iloc[rs].to_numpy(), lab[rs]); tp.append(m)
    res["tabpfn_fit_s"] = round(time.perf_counter() - t0, 2)
    f1, _ = gpu_free_gb()
    res["tabpfn_resident_gb"] = round(f0 - f1, 3)
    res["gpu_total_gb"] = round(tot, 2)
    res["gpu_free_after_gb"] = round(f1, 2)
    log(f"TabPFN 5멤버 적합 {res['tabpfn_fit_s']}초 · 상주 {res['tabpfn_resident_gb']:.3f}GB "
        f"· 적합 후 여유 {f1:.2f}GB")
    for n in (1, 4, 16):
        Q = X.iloc[:n].to_numpy()
        p50, p95 = bench(lambda: np.mean([m.predict_proba(Q)[:, 1] for m in tp], axis=0), n_rep=15)
        res[f"tabpfn_ms_n{n}"] = {"p50": round(p50, 1), "p95": round(p95, 1)}
        log(f"  TabPFN 채점 n={n:<2d} p50 {p50:8.1f}ms  p95 {p95:8.1f}ms")
    res["tabpfn_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    # ---------- TabPFN (CPU 폴백 = 경합 최악값) ----------
    try:
        rs = np.random.default_rng(SEEDS[0]).choice(itr, size=min(SUB, len(itr)), replace=False)
        mc = TabPFNClassifier(device="cpu", random_state=SEEDS[0])
        t0 = time.perf_counter(); mc.fit(X.iloc[rs].to_numpy(), lab[rs])
        res["tabpfn_cpu_fit_s"] = round(time.perf_counter() - t0, 2)
        Q = X.iloc[:4].to_numpy()
        p50, p95 = bench(lambda: mc.predict_proba(Q)[:, 1], n_warm=1, n_rep=5)
        res["tabpfn_cpu_ms_n4_1member"] = {"p50": round(p50, 1), "p95": round(p95, 1)}
        log(f"  TabPFN CPU 1멤버 n=4 p50 {p50:.1f}ms (5멤버 환산 {p50*5/1000:.1f}초)")
    except Exception as e:
        res["tabpfn_cpu_error"] = str(e)[:200]
        log(f"  CPU 폴백 측정 실패: {e}")

    (OUT / "cost.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    log("")
    log("=== 요약 ===")
    r1 = res["tabpfn_ms_n4"]["p50"] / max(res["hgb_ms_n4"]["p50"], 1e-9)
    log(f"봉당 채점(n=4) HGB {res['hgb_ms_n4']['p50']:.3f}ms vs TabPFN {res['tabpfn_ms_n4']['p50']:.1f}ms "
        f"= {r1:,.0f}배")
    log(f"상주 GPU {res['tabpfn_resident_gb']:.2f}GB / 총 {res['gpu_total_gb']}GB "
        f"(적합 후 남는 여유 {res['gpu_free_after_gb']}GB)")
    log(f"콜드스타트 HGB {res['hgb_fit_s']}초 vs TabPFN {res['tabpfn_fit_s']}초")
    log(f"산출: {OUT}/cost.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
