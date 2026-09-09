"""결정층 PnL — 현직 TabM(A) vs 전면 TabPFN(D) 짝지은 판정. 고정 사이징.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909` · 계획 5번(사용자 승인)
사용자 결정 2026-09-10: "그냥 TABPFN 으로 모두 진행해서 PNL 테스트해줘".

왜 여기가 판정인가
------------------
헤드별 정확도는 이미 다 쟀다(direction +0.55pp 4/5 · exit +9.33pp 5/5 · quality −0.39pp 1/5).
그런데 이 저장소가 반복해서 무너진 지점이 **정확도는 올랐는데 경제 이득이 0** 이었다.
게다가 공유 백본을 버린 대가(quality 가 direction 을 정규화하던 교차 손실)는 헤드별 정확도로는
원리상 보이지 않는다 — 결정층에서만 보인다.

경로는 재구현하지 않는다
------------------------
`greedy_replay` / `_predict_exit_prob_one` 원본을 그대로 import 해서 쓴다. 바꾸는 건 두 곳뿐:
  1) 부모 확률 — `_prediction_output(direction_proba, quality_proba, thr)` 에 넣는 배열의 출처
  2) exit 확률 — `sidecar._predict_exit_prob_one` 을 **모듈 속성 치환**으로 TabPFN 에 연결
     (원본은 base_np[row_i] 의 pos 자리를 pos_values 로 갈아끼우고 표준화 후 TabM 을 태운다.
      패치본은 같은 행을 만들되 **표준화 없이** TabPFN 에 넣는다 — TabPFN 은 자체 전처리를 한다.)

고정 사이징
----------
사이드카를 쓰지 않는다(29피쳐가 전부 부모 출력이라, 부모가 팔마다 다르면 사이드카도 팔마다
재적합해야 한다 — 그건 2단계다). margin 0.225 · leverage 2.0 상수로 두어 BASE_TEMPLATE
notional 0.45 에 맞춘다. `SCALE_MAP`(배포 로직)은 그대로 살려두므로 컴포넌트·측면별로
notional 이 갈리지만, **모든 팔에 동일하게** 적용되어 비교를 편향시키지 않는다.

팔
--
  A    TabM direction · TabM quality 임계 · TabM exit          — 기준선(현행 배포본)
  B    TabM direction · TabM quality 임계 · **TabPFN exit**    — exit 단독 기여(진단)
  D    **TabPFN 전부**                                          — 전면 교체
  Dnq  **TabPFN 전부**, quality 임계 해제                        — 게이트가 밥값을 하는가

⚠️ 비용: `greedy_replay` 는 포지션 보유 중 **매 봉** exit 를 부른다. 벤치에서 TabPFN 단일행
   예측이 1.76초였다(n_est=4, fit_preprocessors). 그래서 **A 를 먼저 돌려 실제 호출 수를 센다** —
   호출 수를 모르면 B/D/Dnq 의 소요를 추정할 수 없다. 카운터를 항상 켠다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_eval_omega4_3head_parent72_regimespine_balnobb_20260909 as spine  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as p72  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import greedy_replay  # noqa: E402
from research_omega461_tabicl_direction_head_20260909 import (  # noqa: E402
    DIR_LBL, QUAL_LBL, SEEDS, _select_ctx, _make_model,
)


def _fit_batch(x_ctx, y_ctx, n_est, seed, device):
    """배치 질의용(부모 direction/quality) — 한 번에 전 봉을 예측하므로 캐시가 필요 없다."""
    clf = _make_model("tabpfn", n_est, seed, device)
    clf.fit(x_ctx.to_numpy(np.float32), y_ctx)
    return clf
from research_omega461_exit_head_model_ab_20260909 import _build_exit, EXIT_KW  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/decision_layer"
PSTEM = "omega4_3head_parent72_loose_entry_quality_20260620_regimespine_balnobb_{comp}_s{seed}_20260909"
WIDE24_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
BALNOBB_DIR = ROOT / "data/ensemble/supervised/omega461_balnobb_cut2509_20260909"
BASE_CSVS = {"2024": "training_features_2024.csv", "2025": "training_features_2025.csv",
             "2026_rebuilt": "training_features_2026_rebuilt.csv"}
SPLITS = {"validation": ("2025-10-01 00:00:00", "2025-12-31 23:55:00"),
          "oos": ("2026-01-01 00:00:00", "2026-02-28 23:55:00")}
COMPS = {"h48qual": 0.50, "zig075": 0.75}          # quality 임계 (Phase 2 스윕과 동일)
FIX_MARGIN, FIX_LEVERAGE = 0.225, 2.0              # 0.225 x 2.0 = BASE_TEMPLATE notional 0.45
CTX_DIR, CTX_EXIT = 32000, 5500                    # 각 헤드 게이트의 최선 칸(균형 천장)

# ── exit 예측기 치환 ─────────────────────────────────────────────────────────
_ORIG_EXIT = sidecar._predict_exit_prob_one
_EXIT_STATE: dict[str, Any] = {"clf": None, "calls": 0, "seconds": 0.0}


def _patched_exit(base_np, runtime, pos_idx, *, row_i, expert, pos_values, device):
    _EXIT_STATE["calls"] += 1
    if _EXIT_STATE["clf"] is None:                  # 팔 A/현직 — 원본 그대로
        return _ORIG_EXIT(base_np, runtime, pos_idx, row_i=row_i, expert=expert,
                          pos_values=pos_values, device=device)
    t0 = time.time()
    row = base_np[int(row_i)].copy()
    row[np.asarray(pos_idx, dtype=np.int64)] = np.asarray(pos_values, dtype=np.float32)
    p = float(_EXIT_STATE["clf"].predict_proba(row.reshape(1, -1).astype(np.float32))[0, 1])
    _EXIT_STATE["seconds"] += time.time() - t0
    return p


sidecar._predict_exit_prob_one = _patched_exit


def load_frame(start: str, end: str) -> pd.DataFrame:
    parts = []
    for tag, fn in BASE_CSVS.items():
        b = pd.read_csv(ROOT / "data/splits/year_oos" / fn, low_memory=False, parse_dates=["timestamp"])
        w = pd.read_csv(WIDE24_DIR / f"training_features_{tag}_regime3_current_sensitive_hmm_wide24.csv",
                        low_memory=False, parse_dates=["timestamp"])
        n = pd.read_csv(BALNOBB_DIR / f"training_features_{tag}_regime3_balnobb_cut2509_sidecar.csv",
                        low_memory=False, parse_dates=["timestamp"])
        parts.append(b.merge(w, on="timestamp", how="inner").merge(n, on="timestamp", how="inner"))
    df = (pd.concat(parts, ignore_index=True).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)


def metrics(ledger: pd.DataFrame) -> dict:
    r = ledger["trade_return"].to_numpy() if len(ledger) else np.array([])
    if not len(r):
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0, "source_component": {}}
    curve = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    dd = curve / np.maximum(np.maximum.accumulate(curve), 1e-12) - 1.0
    return {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0),
            "trades": int(len(r)), "wr": float((r > 0).mean()),
            "source_component": ledger["source_component"].value_counts().to_dict()}


def build_component(frame: pd.DataFrame, comp: str, seed: int, *, dir_p: np.ndarray,
                    qual_p: np.ndarray, thr: float, device) -> dict:
    """`prepare_component` 와 같은 계약의 딕셔너리를 만든다 — 단 사이드카를 쓰지 않는다."""
    import torch
    bp = ROOT / "tmp/causal_regen_20260516" / PSTEM.format(comp=comp, seed=seed) / "true_3head_tabm_bundle.pt"
    bundle = torch.load(bp, map_location="cpu", weights_only=False)
    base_cols = list(bundle["base_cols"])
    src = parent._prediction_output(frame, dir_p, qual_p, threshold=float(thr),
                                    prefix="omega1_regime3_expertdq")
    cfg = retest.COMPONENTS[comp]
    dec_base = parent._to_decisions(src, oof=False)
    dec, _ = atr_eval._apply_atr_safety_sltp(
        dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"],
        min_tp=cfg["min_tp"], min_sl=cfg["min_sl"], max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
    x = parent._base_input(frame, base_cols)
    loaded = parent._load_payloads(bundle["models"], device=device)
    base_np, exit_runtime, pos_idx = sidecar._prepare_exit_runtime(x, loaded)
    n = len(frame)
    return {"dec": dec, "atr": atr_eval._atr_pct(frame, cfg["atr_window"]),
            "margin": np.full(n, FIX_MARGIN), "leverage": np.full(n, FIX_LEVERAGE),
            "base_np": base_np, "exit_runtime": exit_runtime, "pos_idx": pos_idx,
            "route": hard._route_id(frame), "exit_threshold": cfg["exit_threshold"]}


def tabm_probs(frame: pd.DataFrame, comp: str, seed: int, device) -> tuple[np.ndarray, np.ndarray]:
    import torch
    bp = ROOT / "tmp/causal_regen_20260516" / PSTEM.format(comp=comp, seed=seed) / "true_3head_tabm_bundle.pt"
    bundle = torch.load(bp, map_location="cpu", weights_only=False)
    x = parent._base_input(frame, list(bundle["base_cols"]))
    preds = {e: parent._predict_payload(bundle["models"][e], x, device=device) for e in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    return (parent._routed(preds, route, "direction", 3),
            parent._routed(preds, route, "quality", 3))


def tabpfn_probs(frame: pd.DataFrame, comp: str, seed: int, ctx: dict, *,
                 n_est: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """부모 direction/quality 를 TabPFN 으로. **배치 질의**라 봉당 호출과 달리 싸다.

    컨텍스트는 각 헤드 게이트의 최선 칸(균형 32k)을 그대로 쓴다.
    zig075 의 quality 는 `same_as_direction` 이라 direction 확률을 재사용한다 —
    같은 X·같은 y 에 같은 모델을 두 번 적합하는 것은 낭비이고, 현직 TabM 도 같은 라벨을 본다.
    """
    key = f"{comp}|{seed}|{len(frame)}"
    cache = ctx.setdefault("_probs", {})
    if key in cache:
        return cache[key]
    base_cols = ctx["base_cols"]
    xq = parent._base_input(frame, base_cols)
    sel, _ = _select_ctx(ctx["y_dir"], np.arange(len(ctx["y_dir"])), CTX_DIR, True, seed)
    t0 = time.time()
    dclf = _fit_batch(ctx["x_dir"].iloc[sel], ctx["y_dir"][sel], n_est, seed, device)
    dp = dclf.predict_proba(xq.to_numpy(np.float32)).astype(np.float64)
    print(f"    [{comp}] direction TabPFN ctx {len(sel):,} → {len(frame):,}봉 "
          f"{time.time()-t0:.0f}s", flush=True)
    if comp == "zig075":
        qp = dp
    else:
        selq, _ = _select_ctx(ctx["y_qual"], np.arange(len(ctx["y_qual"])), CTX_DIR, True, seed)
        t0 = time.time()
        qclf = _fit_batch(ctx["x_dir"].iloc[selq], ctx["y_qual"][selq], n_est, seed, device)
        qp = qclf.predict_proba(xq.to_numpy(np.float32)).astype(np.float64)
        print(f"    [{comp}] quality TabPFN ctx {len(selq):,} → {len(frame):,}봉 "
              f"{time.time()-t0:.0f}s", flush=True)
    cache[key] = (dp, qp)
    return dp, qp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A")
    ap.add_argument("--seeds", default=",".join(str(x) for x in SEEDS))
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--exit-ctx", type=int, default=CTX_EXIT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", default="validation,oos")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()

    frames = {s: load_frame(*SPLITS[s]) for s in splits}
    for s, f in frames.items():
        print(f"[{s}] {f.timestamp.min()} ~ {f.timestamp.max()}  {len(f):,}봉", flush=True)

    need_tabpfn = any(a in ("B", "D", "Dnq") for a in arms)
    ctx_cache: dict[str, Any] = {}
    if need_tabpfn:
        print("\n[TabPFN 컨텍스트 준비] 부모 학습 프레임 로드", flush=True)
        base_cols = spine._install("zig075")
        fr = p72._prepare_frames(disable_tp_sl=False, direction_label_dir=DIR_LBL,
                                 quality_mode="same_as_direction", quality_label_dir=None,
                                 quality_min_edge=0.0010, quality_max_mae=0.0100,
                                 quality_min_mfe_mae=1.20, quality_max_hold_bars=288)
        ctx_cache["train_raw"] = fr["train_raw"]
        ctx_cache["base_cols"] = base_cols
        ctx_cache["x_dir"] = parent._base_input(fr["train_raw"], base_cols)
        ctx_cache["y_dir"] = fr["train_raw"]["zigzag_action"].to_numpy(np.int64)
        xe, ye, _fe, _dg = _build_exit(fr["train_raw"], base_cols, "exit ctx")
        ctx_cache["exit"] = (xe, ye)
        print(f"  direction 컨텍스트 {len(fr['train_raw']):,}행 · exit 컨텍스트 {len(ye):,}행", flush=True)
        if any(a in ("D", "Dnq") for a in arms):
            # h48qual 의 quality 는 **별도 라벨 계약**이라 프레임을 한 번 더 짓는다.
            # zig075 는 same_as_direction 이라 direction 확률을 그대로 재사용한다(축퇴).
            frq = p72._prepare_frames(disable_tp_sl=False, direction_label_dir=DIR_LBL,
                                      quality_mode="quality_label_action", quality_label_dir=QUAL_LBL,
                                      quality_min_edge=0.0010, quality_max_mae=0.0100,
                                      quality_min_mfe_mae=1.20, quality_max_hold_bars=288)
            ctx_cache["y_qual"] = frq["train_raw"]["omega4_quality_action"].to_numpy(np.int64)
            print(f"  quality(h48qual) 컨텍스트 {len(ctx_cache['y_qual']):,}행 "
                  f"· 클래스 {np.bincount(ctx_cache['y_qual'], minlength=3).tolist()}", flush=True)

    rep: dict[str, Any] = {"arms": arms, "seeds": seeds, "splits": splits,
                           "fixed_sizing": {"margin": FIX_MARGIN, "leverage": FIX_LEVERAGE,
                                            "note": "SCALE_MAP(배포 로직)은 유지 — 모든 팔 동일"},
                           "n_estimators": int(args.n_estimators), "exit_ctx": int(args.exit_ctx),
                           "results": {}}

    for arm in arms:
        for seed in seeds:
            _EXIT_STATE["clf"] = None
            if arm in ("B", "D", "Dnq"):
                xe, ye = ctx_cache["exit"]
                sel, _ = _select_ctx(ye, np.arange(len(ye)), int(args.exit_ctx), True, seed)
                # 봉당 반복 질의 → fit_with_cache 필수(1,758ms → 136ms)
                clf = _make_model("tabpfn", args.n_estimators, seed, args.device,
                                  fit_mode="fit_with_cache")
                t0 = time.time()
                clf.fit(xe.iloc[sel].to_numpy(np.float32), ye[sel])
                print(f"\n[{arm} s{seed}] exit TabPFN 적합 {len(sel):,}행 {time.time()-t0:.1f}s", flush=True)
                _EXIT_STATE["clf"] = clf
            for split in splits:
                frame = frames[split]
                _EXIT_STATE["calls"] = 0; _EXIT_STATE["seconds"] = 0.0
                comps = {}
                t0 = time.time()
                for comp, thr in COMPS.items():
                    if arm in ("A", "B"):
                        dp, qp = tabm_probs(frame, comp, seed, device)
                    else:
                        dp, qp = tabpfn_probs(frame, comp, seed, ctx_cache,
                                              n_est=args.n_estimators, device=args.device)
                    if arm == "Dnq":
                        thr = 0.0
                    comps[comp] = build_component(frame, comp, seed, dir_p=dp, qual_p=qp,
                                                  thr=thr, device=device)
                _s, ledger = greedy_replay(frame, comps, fee=fee, slip=slip,
                                           cost_mult=retest.COST_MULT, device=device)
                m = metrics(ledger)
                m["exit_calls"] = int(_EXIT_STATE["calls"])
                m["exit_seconds"] = round(float(_EXIT_STATE["seconds"]), 1)
                m["wall_s"] = round(time.time() - t0, 1)
                rep["results"][f"{arm}|{seed}|{split}"] = m
                print(f"  {arm} s{seed} [{split}]  PnL {m['pnl']:+8.2f}%  MDD {m['mdd']:+7.2f}%  "
                      f"{m['trades']:3d}건  WR {m['wr']*100:5.1f}%  "
                      f"exit호출 {m['exit_calls']:,} ({m['exit_seconds']:.0f}s)  "
                      f"총 {m['wall_s']:.0f}s  {m['source_component']}", flush=True)
                (OUT / "decision_layer_pnl.json").write_text(
                    json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(f"\n{'='*96}\n[짝지은 요약]", flush=True)
    for split in splits:
        base = {s: rep["results"].get(f"A|{s}|{split}") for s in seeds}
        for arm in arms:
            if arm == "A":
                continue
            d = [rep["results"][f"{arm}|{s}|{split}"]["pnl"] - base[s]["pnl"]
                 for s in seeds if base.get(s) and f"{arm}|{s}|{split}" in rep["results"]]
            if d:
                print(f"  [{split}] {arm} − A  ΔPnL 중앙 {statistics.median(d):+.2f}pp  "
                      f"이김 {sum(1 for x in d if x > 0)}/{len(d)}", flush=True)
    calls = [v["exit_calls"] for v in rep["results"].values()]
    if calls:
        print(f"\n  exit 호출 수: 중앙 {statistics.median(calls):,.0f} "
              f"[{min(calls):,} ~ {max(calls):,}] — TabPFN 1.76s/호출 기준 "
              f"리플레이 1회 ≈ {statistics.median(calls)*1.76/60:.0f}분", flush=True)
    print(f"\n산출물: {OUT}/decision_layer_pnl.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
