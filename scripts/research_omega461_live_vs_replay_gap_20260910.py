"""라이브 ↔ 리플레이 격차 분해 — 같은 창을 같은 배포 아티팩트로 다시 돌린다.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909` · 사용자 결정 2026-09-10.

왜 이게 먼저인가
---------------
2026-07-06 파리티 감사가 **모든 게이트 통과**로 끝나며 "Live-achievable PnL: +145.34% /
MDD -10.13% / 24건 / WR 54.2% (2026-01-01..06-30)" 를 남겼다. 그 직후 실제 라이브
(2026-07-15..09-03) 실거래 7건의 결과는 **복리 +0.50% · MDD -12.72% · 승률 29% ·
중앙 -4.54%** 였다. 모델 교체 실험을 아무리 해도 이 격차를 모르면 VAL PnL 로 내리는
판정이 무엇을 의미하는지 알 수 없다.

두 갈래를 가른다
--------------
  · 같은 창 리플레이도 0 근처  → 버그가 아니라 **엣지 소멸**(기간/레짐). 모델축 자체를 접어야 한다.
  · 같은 창 리플레이가 크게 양수 → **라이브↔리플레이 괴리**. 그 다음 진입/청산 대조로 위치를 좁힌다.

무엇을 그대로 쓰는가 — 재구현 금지
--------------------------------
  · `retest.COMPONENTS`   : 배포된 부모 번들 + 배포된 risk_sidecar.pkl (동결, 재학습 없음)
  · `retest.load_frame_current` : 배포 경로와 같은 프레임(BASE_2026 + wide24 오버레이)
  · `prepare_component` / `greedy_replay` : 라이브와 같은 결정 경로(우선순위 라우터·사이드카 사이징)
부모 예측만 이 스크립트가 프레임 위에서 직접 만든다(`prepare_component` 가 timestamp 일치를 요구).

⚠️ 기지 한계 (retest 문서에 명시, 은폐 금지)
   `ou_halflife`/`kel`/`evt_excess_z`/`btc_corr_60`/`dual_momentum` 이 원래 alpha6/7 계열
   피쳐 파일과 다르다 — `features/elite.py` 공식이 2026-05-29 이후 바뀌었고 git 이력이 부족해
   옛 버전을 복원할 수 없다. **`ou_halflife` 는 duration 게이트 규칙에 직접 들어간다.**
   따라서 이 리플레이는 "라이브가 그때 본 것"의 **근사**다. 격차가 나오면 이 항목이 1순위 용의자다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import greedy_replay, prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"

# ── exit 예측기 치환 (결정층 스크립트와 같은 패턴) ─────────────────────────────
# 원본은 base_np[row_i] 의 pos 자리를 pos_values 로 갈아끼우고 표준화 후 TabM 을 태운다.
# 패치본은 같은 행을 만들되 **표준화 없이** TabPFN 에 넣는다(TabPFN 은 자체 전처리).
_ORIG_EXIT = sidecar._predict_exit_prob_one
_EXIT: dict = {"clf": None, "calls": 0}


def _patched_exit(base_np, runtime, pos_idx, *, row_i, expert, pos_values, device):
    _EXIT["calls"] += 1
    if _EXIT["clf"] is None:
        return _ORIG_EXIT(base_np, runtime, pos_idx, row_i=row_i, expert=expert,
                          pos_values=pos_values, device=device)
    row = base_np[int(row_i)].copy()
    row[np.asarray(pos_idx, dtype=np.int64)] = np.asarray(pos_values, dtype=np.float32)
    return float(_EXIT["clf"].predict_proba(row.reshape(1, -1).astype(np.float32))[0, 1])


sidecar._predict_exit_prob_one = _patched_exit
JOURNAL = ROOT / "data/live/trade_journal.jsonl"


def metrics(ledger: pd.DataFrame) -> dict:
    r = ledger["trade_return"].to_numpy() if len(ledger) else np.array([])
    if not len(r):
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0, "source_component": {}}
    curve = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    dd = curve / np.maximum(np.maximum.accumulate(curve), 1e-12) - 1.0
    return {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0),
            "trades": int(len(r)), "wr": float((r > 0).mean()),
            "median_trade_pct": float(np.median(r) * 100.0),
            "best_pct": float(r.max() * 100.0), "worst_pct": float(r.min() * 100.0),
            "source_component": ledger["source_component"].value_counts().to_dict()}


def live_trades(start: str, end: str) -> pd.DataFrame:
    if not JOURNAL.exists():
        return pd.DataFrame()
    rows = []
    for line in JOURNAL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("pnl_frac") is None:
            continue
        if "_shadow_" in str(r.get("reason", "")):      # 섀도우 제외 — 실거래만
            continue
        rows.append({"ts": r.get("ts"), "opened_at": r.get("opened_at"), "side": r.get("side"),
                     "pnl_frac": float(r["pnl_frac"]),
                     "gross": float(r.get("gross_return_frac") or 0.0),
                     "fee": float(r.get("fee_cost_frac") or 0.0),
                     "reason": r.get("reason"), "hold_bars": r.get("hold_bars"),
                     "notional": float(r.get("notional_exposure") or 0.0),
                     "margin": float(r.get("margin_fraction") or 0.0)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df[(df["ts"] >= start) & (df["ts"] <= end)].sort_values("ts").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-07-01", help="워밍업 포함 시작(라이브 창보다 앞)")
    ap.add_argument("--end", default="2026-08-30")
    ap.add_argument("--live-start", default="2026-07-15", help="라이브 원장 대조 구간")
    ap.add_argument("--live-end", default="2026-08-30")
    ap.add_argument("--parent", default="tabm", choices=["tabm", "tabpfn"],
                    help="부모 direction/quality 출처. exit·사이드카·라우터는 **양 팔 동일**하게 "
                         "배포 아티팩트를 쓴다 — 바뀌는 건 진입 확률 하나뿐이다. "
                         "TabPFN 도 배포 base_cols(wide24 레짐)로 적합해 척추까지 짝을 맞춘다.")
    ap.add_argument("--tabpfn-n-estimators", type=int, default=4)
    ap.add_argument("--tabpfn-ctx", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=615372041, help="TabPFN 적합 시드(Seed-Diversity)")
    ap.add_argument("--exit", dest="exit_src", default="tabm", choices=["tabm", "tabpfn"],
                    help="exit 예측기. tabpfn 이면 봉당 호출이라 fit_with_cache 필수")
    ap.add_argument("--exit-label-mode", default="path_optimal",
                    choices=["terminal_giveback", "path_optimal"])
    ap.add_argument("--exit-ctx", type=int, default=1484,
                    help="path_optimal 균형 천장 = 2 x 742(TRAIN 양성)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()

    frame = retest.load_frame_current(args.start, args.end)
    print(f"[프레임] {frame.timestamp.min()} ~ {frame.timestamp.max()}  {len(frame):,}봉 "
          f"· 부모={args.parent}", flush=True)

    tp_ctx = None
    from research_omega461_tabicl_direction_head_20260909 import DIR_LBL as DIR_LBL0  # noqa: E402
    if args.parent == "tabpfn":
        # 배포와 같은 프레임 계약으로 TRAIN 컨텍스트를 짓는다(spine 패치 없음 = wide24 레짐).
        import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as p72
        from research_omega461_tabicl_direction_head_20260909 import DIR_LBL, QUAL_LBL, _select_ctx
        fr = p72._prepare_frames(disable_tp_sl=False, direction_label_dir=DIR_LBL,
                                 quality_mode="same_as_direction", quality_label_dir=None,
                                 quality_min_edge=0.0010, quality_max_mae=0.0100,
                                 quality_min_mfe_mae=1.20, quality_max_hold_bars=288)
        frq = p72._prepare_frames(disable_tp_sl=False, direction_label_dir=DIR_LBL,
                                  quality_mode="quality_label_action", quality_label_dir=QUAL_LBL,
                                  quality_min_edge=0.0010, quality_max_mae=0.0100,
                                  quality_min_mfe_mae=1.20, quality_max_hold_bars=288)
        tp_ctx = {"train": fr["train_raw"],
                  "y_dir": fr["train_raw"]["zigzag_action"].to_numpy(np.int64),
                  "y_qual": frq["train_raw"]["omega4_quality_action"].to_numpy(np.int64),
                  "select": _select_ctx}
        print(f"  TabPFN 컨텍스트 원본 {len(tp_ctx['y_dir']):,}행 "
              f"({fr['train_raw'].timestamp.min()} ~ {fr['train_raw'].timestamp.max()})", flush=True)

    comps = {}
    for name, cfg in retest.COMPONENTS.items():
        import torch
        bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
        bcols = list(bundle["base_cols"])
        x = parent._base_input(frame, bcols)
        route = hard._route_id(frame)
        if tp_ctx is None:
            preds = {e: parent._predict_payload(bundle["models"][e], x, device=device)
                     for e in hard.EXPERT_NAMES}
            dp = parent._routed(preds, route, "direction", 3)
            qp = parent._routed(preds, route, "quality", 3)
        else:
            from research_omega461_tabicl_direction_head_20260909 import _make_model
            xtr = parent._base_input(tp_ctx["train"], bcols)
            xq = x.to_numpy(np.float32)

            def _fit_pred(y, seed, what):
                sel, _ = tp_ctx["select"](y, np.arange(len(y)), args.tabpfn_ctx, True, seed)
                import time as _t
                t0 = _t.time()
                clf = _make_model("tabpfn", args.tabpfn_n_estimators, seed, args.device)
                clf.fit(xtr.iloc[sel].to_numpy(np.float32), y[sel])
                out = clf.predict_proba(xq).astype(np.float64)
                print(f"    [{name}] {what} TabPFN ctx {len(sel):,} → {len(frame):,}봉 "
                      f"{_t.time()-t0:.0f}s", flush=True)
                return out

            dp = _fit_pred(tp_ctx["y_dir"], args.seed, "direction")
            qp = dp if name == "zig075" else _fit_pred(tp_ctx["y_qual"], args.seed, "quality")
        oof = parent._prediction_output(
            frame, dp, qp,
            threshold=float(cfg["quality_threshold"]), prefix="omega1_regime3_expertdq")
        d = OUT / "preds" / name
        d.mkdir(parents=True, exist_ok=True)
        pcsv = d / f"predictions_{cfg['q_tag']}.csv"
        oof.to_csv(pcsv, index=False)
        comps[name] = prepare_component(frame, pcsv, cfg, device)
        act = int(omega._active(comps[name]["dec"]).sum())
        pos_margin = int((np.asarray(comps[name]["margin"]) > 0).sum())
        print(f"  [{name}] 배포번들 · 활성봉 {act:,} · 사이드카 margin>0 {pos_margin:,} "
              f"({pos_margin/max(len(frame),1)*100:.1f}%)", flush=True)

    if args.exit_src == "tabpfn":
        # exit 컨텍스트는 **배포 base_cols** 로 짓는다(부모 팔과 척추까지 짝을 맞춤).
        import time as _t
        from research_omega461_exit_head_model_ab_20260909 import _build_exit
        from research_omega461_tabicl_direction_head_20260909 import _make_model, _select_ctx
        import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as _p72
        bcols0 = list(torch.load(retest.COMPONENTS["zig075"]["bundle"], map_location="cpu",
                                weights_only=False)["base_cols"])
        fr0 = _p72._prepare_frames(disable_tp_sl=False, direction_label_dir=DIR_LBL0,
                                   quality_mode="same_as_direction", quality_label_dir=None,
                                   quality_min_edge=0.0010, quality_max_mae=0.0100,
                                   quality_min_mfe_mae=1.20, quality_max_hold_bars=288)
        xe, ye, _fe, _dg = _build_exit(fr0["train_raw"], bcols0, "exit ctx", args.exit_label_mode)
        sel, _ = _select_ctx(ye, np.arange(len(ye)), args.exit_ctx, True, args.seed)
        t0 = _t.time()
        clf = _make_model("tabpfn", args.tabpfn_n_estimators, args.seed, args.device,
                          fit_mode="fit_with_cache")
        clf.fit(xe.iloc[sel].to_numpy(np.float32), ye[sel])
        _EXIT["clf"] = clf
        print(f"  exit TabPFN({args.exit_label_mode}) ctx {len(sel):,}행 "
              f"(양성 {int(ye[sel].sum())}) 적합 {_t.time()-t0:.1f}s", flush=True)
    _EXIT["calls"] = 0
    _s, ledger = greedy_replay(frame, comps, fee=fee, slip=slip,
                               cost_mult=retest.COST_MULT, device=device)
    tag = args.tag or args.parent
    ledger.to_csv(OUT / f"replay_ledger_{tag}.csv", index=False)
    m_all = metrics(ledger)
    print(f"  exit 호출 {_EXIT['calls']:,}회", flush=True)
    print(f"\n[리플레이 전체 {args.start}~{args.end}]  PnL {m_all['pnl']:+.2f}%  "
          f"MDD {m_all['mdd']:+.2f}%  {m_all['trades']}건  WR {m_all['wr']*100:.1f}%  "
          f"{m_all['source_component']}", flush=True)

    # 라이브 대조 구간으로 잘라 재계산
    lc = ledger.copy()
    tcol = "exit_time" if "exit_time" in lc.columns else ("timestamp" if "timestamp" in lc.columns else None)
    m_win = m_all
    if tcol:
        lc[tcol] = pd.to_datetime(lc[tcol], errors="coerce")
        lc = lc[(lc[tcol] >= args.live_start) & (lc[tcol] <= args.live_end)].reset_index(drop=True)
        m_win = metrics(lc)
    lt = live_trades(args.live_start, args.live_end)

    print(f"\n{'='*96}\n[대조 {args.live_start} ~ {args.live_end}]", flush=True)
    print(f"  리플레이  PnL {m_win['pnl']:+8.2f}%  MDD {m_win['mdd']:+7.2f}%  {m_win['trades']:3d}건  "
          f"WR {m_win['wr']*100:5.1f}%  중앙 {m_win.get('median_trade_pct', 0):+.3f}%", flush=True)
    if len(lt):
        p = lt["pnl_frac"].to_numpy()
        cur = np.concatenate([[1.0], np.cumprod(1.0 + p)])
        dd = cur / np.maximum.accumulate(cur) - 1
        print(f"  라이브    PnL {(cur[-1]-1)*100:+8.2f}%  MDD {dd.min()*100:+7.2f}%  {len(p):3d}건  "
              f"WR {(p > 0).mean()*100:5.1f}%  중앙 {np.median(p)*100:+.3f}%", flush=True)
        print(f"  라이브 사유 {lt['reason'].value_counts().to_dict()}", flush=True)
        print(f"  라이브 평균 notional {lt['notional'].mean():.3f} · margin {lt['margin'].mean():.3f}",
              flush=True)
        if "notional_exposure" in ledger.columns:
            print(f"  리플레이 평균 notional "
                  f"{pd.to_numeric(lc.get('notional_exposure'), errors='coerce').mean():.3f}", flush=True)
        lt.to_csv(OUT / "live_trades.csv", index=False)
    else:
        print("  라이브 원장 없음(dev 사본은 오래됨 — 서버에서 실행할 것)", flush=True)

    rep = {"window": [args.start, args.end], "live_window": [args.live_start, args.live_end],
           "replay_all": m_all, "replay_in_live_window": m_win,
           "live_n": int(len(lt)),
           "known_limitation": "ou_halflife/kel/evt_excess_z/btc_corr_60/dual_momentum 가 "
                               "alpha6/7 원본과 다름(features/elite.py 변경). ou_halflife 는 "
                               "duration 게이트에 직접 들어간다 — 격차 시 1순위 용의자."}
    (OUT / f"live_vs_replay_{tag}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float),
                                             encoding="utf-8")
    print(f"\n산출물: {OUT}/live_vs_replay_{tag}.json · replay_ledger_{tag}.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
