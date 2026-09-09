"""TabPFN 봉당 단일행 예측 지연 벤치 — 결정층 리플레이 실현 가능성 판정.

계획 5번(결합 → 결정층 PnL)의 선결 조건. `greedy_replay` 는 포지션 보유 중 **매 봉마다**
`_predict_exit_prob_one` 을 한 번 부른다. TabM 은 103,992 파라미터 MLP 라 사실상 공짜지만,
TabPFN 은 in-context 라 호출마다 컨텍스트 전체를 통과한다 — 그 비용이 감당되는지가 관건이다.

VAL 리플레이의 exit 호출 횟수는 대략 (거래 수) x (평균 보유 봉) 이다. 수천 회 규모이므로
호출당 0.5초면 팔 하나에 수십 분, 4팔 x 5시드 x 2구간이면 며칠이 된다.

`fit_mode="fit_with_cache"` 가 컨텍스트를 미리 인코딩해 반복 질의를 싸게 만드는 경로다.
이 벤치가 재는 것: 모드별 **호출당 지연**과 상주 메모리, 그리고 그것을 리플레이 호출 횟수로
환산한 예상 소요.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/decision_layer"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=5500, help="exit 헤드 균형 컨텍스트 천장")
    ap.add_argument("--n-feat", type=int, default=115)
    ap.add_argument("--n-calls", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expected-calls", type=int, default=4000, help="VAL 리플레이 exit 호출 추정")
    args = ap.parse_args()

    import torch
    from tabpfn import TabPFNClassifier
    import inspect
    fm = inspect.signature(TabPFNClassifier.__init__).parameters["fit_mode"]
    print(f"[fit_mode] 기본값 {fm.default} · 주석 {fm.annotation}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(615372041)
    X = rng.standard_normal((args.n_ctx, args.n_feat)).astype(np.float32)
    y = (rng.random(args.n_ctx) < 0.092).astype(np.int64)   # exit 양성 9.2%
    x1 = rng.standard_normal((1, args.n_feat)).astype(np.float32)

    rep = {"n_ctx": args.n_ctx, "n_feat": args.n_feat, "n_calls": args.n_calls,
           "expected_calls_per_replay": args.expected_calls, "runs": {}}
    for mode in ("fit_preprocessors", "fit_with_cache"):
        for nest in (4, 16):
            key = f"{mode}|n_est={nest}"
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                clf = TabPFNClassifier(n_estimators=nest, random_state=615372041,
                                       device=args.device, fit_mode=mode,
                                       memory_saving_mode="auto", balance_probabilities=False)
                t0 = time.time(); clf.fit(X, y); t_fit = time.time() - t0
                clf.predict_proba(x1)                       # 워밍업 1회는 제외
                lat = []
                for _ in range(args.n_calls):
                    t1 = time.time(); clf.predict_proba(x1); lat.append(time.time() - t1)
                med = float(np.median(lat))
                peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")
                est_min = med * args.expected_calls / 60.0
                rep["runs"][key] = {"fit_s": round(t_fit, 1), "per_call_median_s": round(med, 4),
                                    "per_call_p90_s": round(float(np.percentile(lat, 90)), 4),
                                    "peak_gpu_mib": round(peak, 1),
                                    "est_minutes_per_replay": round(est_min, 1)}
                print(f"  {key:34s} fit {t_fit:5.1f}s · 호출당 {med*1000:7.1f}ms "
                      f"(p90 {np.percentile(lat,90)*1000:.0f}ms) · peak {peak:6.0f}MiB "
                      f"→ 리플레이 1회 ≈ {est_min:6.1f}분", flush=True)
            except Exception as exc:
                rep["runs"][key] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  {key:34s} ❌ {type(exc).__name__}: {str(exc)[:120]}", flush=True)

    ok = {k: v for k, v in rep["runs"].items() if "per_call_median_s" in v}
    if ok:
        best = min(ok.items(), key=lambda kv: kv[1]["per_call_median_s"])
        total = best[1]["est_minutes_per_replay"] * 3 * 5 * 2   # 3팔(B/C/D) x 5시드 x 2구간
        rep["verdict"] = {"fastest": best[0], **best[1],
                          "est_minutes_full_grid_3arms_5seeds_2splits": round(total, 1),
                          "feasible": total <= 600}
        print(f"\n[판정] 최속 {best[0]} · 호출당 {best[1]['per_call_median_s']*1000:.1f}ms", flush=True)
        print(f"  A~D 전체 격자(팔3 x 시드5 x 구간2 = 30 리플레이) ≈ {total:.0f}분 "
              f"({'실행 가능' if total <= 600 else '과다 — 컨텍스트 축소나 팔 축소 필요'})", flush=True)
    p = OUT / "tabpfn_perbar_exit_bench.json"
    p.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"산출물: {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
