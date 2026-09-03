#!/usr/bin/env python3
"""XRP·BTC 인과적 첫발동 -- **SL 격자 상단 확장** + **레짐 게이팅**. 기준은 그대로.

## 왜 이 두 축인가

사용자 지시: "진입모델은 건드리지 말고 증거신호 성능을 최대한 끌어올려라."

오늘 인과적 첫발동 + 임계값 조이기가 **0/40**으로 완패했다. 그런데 같은 날 체결봉 감사에서
직접적인 단서가 나왔다:

> 지정가(depth 3~4 ATR) 체결 봉의 유리폭이 **중앙 2.15~4.05 ATR**, ARM 초과 비율 87~98%.

이건 **트리거 발동 후 가격이 2~4 ATR 더 진행한 뒤에 반전한다**는 뜻이다.
그런데 게이트의 `SL_GRID`는 **[1.5 ~ 4.0]에서 끝난다.**
⇒ 인과적 첫발동으로 들어가면 그 진행분을 그대로 맞으므로 **반전 직전에 스톱아웃**된다.
0/96의 원인이 신호가 아니라 **SL 격자가 짧아서**일 수 있다.

⭐이건 오늘 하루 반복 적용한 **격자 경계 규칙**(포팅 프로토콜 §5-A)과 같은 논리다.

두 번째 축: **레짐 게이팅**. 오늘 배포한 XRP `S96_K9`는 Phase 3b에서 13/16 · OOS +0.1437을
냈는데, 그걸 **경제성 게이트에 적용한 적이 없다**(ETH는 `research_eth_regime_gated_costgate_
ensemble_20260902.py`로 이미 했다).

## 설계 -- 평가 기준은 하나도 안 건드린다

  · 왕복 **10bp**, margin 0.30 x leverage 3.0
  · 발동: **인과적 첫발동**(직전 GAP봉 내 같은 쪽 발동 없음) -- 앵커/dedup 없음
  · **SL 격자 확장**: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0] -> **+ [5, 6, 8, 10, 12]**
  · ARM/Trail 격자 동일. **ARM >= 1.0** 유지
  · **방향뒤집기 대조군 전량**
  · 판정: VAL>0 AND OOS>0 AND 정방향>뒤집기(양 구간)
  · 두께 하한: 후보 100건(VAL·OOS 각각)

  · **레짐 게이팅**: 예측 chop 구간의 발동만 채택하는 변형을 따로 돌린다
    (XRP=S96_K9 / BTC=S24_K3, 각 자산의 배포 분류기).

⚠️HOLDOUT 미터치. VAL+OOS만.
⚠️SL을 넓히면 손실이 커질 수 있다 -- 두께와 뒤집기 대조군이 그걸 잡는다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib        # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals   # noqa: E402

_T = importlib.util.spec_from_file_location(
    "tighten", ROOT / "scripts/research_xrp_btc_causal_trigger_threshold_tightening_20260903.py")
_t = importlib.util.module_from_spec(_T)
_T.loader.exec_module(_t)

OUT = ROOT / "data/research/xrp_btc_causal_wide_sl_regime_gate_20260903.json"

MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.001   # ⭐10bp 유지
VAL_START, OOS_START, HOLDOUT_START = _t.VAL_START, _t.OOS_START, _t.HOLDOUT_START
# ⭐확장: 기존 상한 4.0 -> 12.0 (체결봉 유리폭 실측 2.15~4.05 ATR을 덮도록)
SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]
MIN_CANDIDATES = 100

REGIME = {"XRP": ROOT / "tmp/xrp_regime_s96k9_20260903/model.joblib",
          "BTC": ROOT / "tmp/btc_regime_s24k3_20260902/model.joblib"}
CANON = {"XRP": ROOT / "data/splits/year_oos/xrp_features_2024_2026.csv",
         "BTC": ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"}


def log(m): print(f"[wide] {m}", flush=True)


def run_grid(kl, dec, is_long, atr, H):
    ts = kl["timestamp"]
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    masks = {}
    for wn, (s, e) in {"val": (VAL_START, OOS_START), "oos": (OOS_START, HOLDOUT_START)}.items():
        el = set(np.flatnonzero(purged_decision_mask(ts, start=s, end=e, horizon_bars=H)).tolist())
        masks[wn] = np.array([d in el for d in dec])
    tp = np.full(len(dec), 999.0)
    cells = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for tr in TRAIL_GRID:
                row = {"sl": sl, "arm": arm, "trail": tr}
                for wn, m in masks.items():
                    for tag, sgn in (("fwd", 1.0), ("flip", -1.0)):
                        sc = np.where(is_long, 1.0, -1.0) * sgn
                        r = simulate_single_position(
                            timestamps=ts, open_px=o, high=h, low=l, close=c,
                            decision_indices=dec[m], scores=sc[m], tp_moves=tp[m],
                            sl_moves=(sl * atr)[m], upper_threshold=1.0, lower_threshold=-1.0,
                            horizon_bars=H, margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
                            roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
                            arm_moves=(arm * atr)[m], trail_moves=(tr * atr)[m])
                        led = r.ledger
                        row[f"{wn}_{tag}_bp"] = (float(led["trade_return"].mean() * 1e4)
                                                 if len(led) else float("nan"))
                        if tag == "fwd":
                            row[f"{wn}_n"] = int(len(led))
                cells.append(row)
    return cells, {k: int(v.sum()) for k, v in masks.items()}


def chop_mask(asset, ts_frame):
    """배포 레짐 분류기의 예측 chop(=2) 구간. 없으면 None."""
    mp, cp = REGIME[asset], CANON[asset]
    if not (mp.exists() and cp.exists()):
        log(f"  ⚠️레짐 아티팩트 없음({asset}) -- 게이팅 생략"); return None
    pay = joblib.load(mp)
    fc, med = pay["feature_cols"], pay["feature_medians"]
    df = pd.read_csv(cp)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    miss = [c for c in fc if c not in df.columns]
    if miss:
        log(f"  ⚠️피쳐 누락 {len(miss)}개 -- 게이팅 생략"); return None
    x = df[fc].apply(pd.to_numeric, errors="coerce")
    for c in fc:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(med.get(c, 0.0))
    pred = pay["model"].predict(x)
    m = pd.DataFrame({"timestamp": ts_frame}).merge(
        pd.DataFrame({"timestamp": df["timestamp"], "pred": pred}), on="timestamp", how="left")
    return (m["pred"].to_numpy() == 2)


def main() -> int:
    t0 = time.time()
    rep = {"cost_bp": 10.0, "sl_grid": SL_GRID, "sl_grid_prev_max": 4.0,
           "min_candidates": MIN_CANDIDATES, "holdout_touched": False,
           "criteria_unchanged": True,
           "trigger_rule": "인과적 첫발동(dedup/앵커 없음)", "assets": {}}

    for asset, cfg in _t.ASSETS.items():
        log(""); log("#" * 80); log(asset); log("#" * 80)
        raw = pd.read_csv(cfg["klines"], parse_dates=["timestamp"])
        raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        partner = pd.read_csv(cfg["partner"], usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
        funding = _t.load_funding(cfg["funding"])
        for d in (raw, partner):
            d["timestamp"] = d["timestamp"].astype("datetime64[ns]")
        frame = compute_signals(raw, btc_df=partner, funding_df=funding)
        frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
        kl = frame[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
        atr_all = frame["atr_pct"].to_numpy(float)
        cm = chop_mask(asset, frame["timestamp"])
        log(f"프레임 {len(frame):,}봉 | 예측chop 비중 "
            f"{('%.3f' % np.nanmean(cm)) if cm is not None else '게이팅 없음'}")

        res = {}
        for sname, spec in _t.SIGNALS.items():
            col = frame[spec["col"]].to_numpy(float)
            H = spec["H"][asset]
            thr = spec["thr"][0]                      # 현행 배포 임계 고정(조이기는 이미 기각됨)
            if spec["kind"] == "bounded":
                fb, ft = (col <= thr), (col >= 1.0 - thr)
            else:
                fb, ft = (col <= -thr), (col >= thr)
            fb = np.nan_to_num(fb, nan=False); ft = np.nan_to_num(ft, nan=False)
            cb = _t.first_of_cluster(fb, spec["gap"])
            ct = _t.first_of_cluster(ft, spec["gap"])
            log("")
            log(f"=== {sname}  임계 {thr} GAP={spec['gap']} H={H} ===")
            out = {}
            for variant, gate_on in (("게이팅 없음", False), ("레짐 chop 게이팅", True)):
                if gate_on and cm is None:
                    continue
                sel = (cb | ct)
                if gate_on:
                    sel = sel & np.nan_to_num(cm, nan=False)
                idx = np.flatnonzero(sel)
                idx = idx[np.isfinite(atr_all[idx]) & (atr_all[idx] > 0) & (idx < len(kl) - 1)]
                if len(idx) < 50:
                    log(f"  {variant:<16} 발동 부족({len(idx)})"); continue
                cells, ns = run_grid(kl, idx, cb[idx], atr_all[idx], H)
                thin = ns["val"] < MIN_CANDIDATES or ns["oos"] < MIN_CANDIDATES
                gen = [c for c in cells
                       if c["val_fwd_bp"] > 0 and c["oos_fwd_bp"] > 0
                       and c["val_fwd_bp"] > c["val_flip_bp"] and c["oos_fwd_bp"] > c["oos_flip_bp"]
                       and c["arm"] >= 1.0]
                best_any = max(cells, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
                b = max(gen, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"])) if gen else None
                wide = [c for c in gen if c["sl"] > 4.0]
                log(f"  {variant:<16} 후보 V{ns['val']}/O{ns['oos']}{'(얇음)' if thin else '':<6} "
                    f"진짜 {len(gen):>3}셀 (그중 SL>4.0: {len(wide)})  "
                    f"격자최선 SL={best_any['sl']} VAL {best_any['val_fwd_bp']:+.2f}/OOS {best_any['oos_fwd_bp']:+.2f}")
                if b:
                    log(f"     ⭐최선 SL={b['sl']} ARM={b['arm']} Trail={b['trail']}  "
                        f"VAL {b['val_fwd_bp']:+.2f}(뒤{b['val_flip_bp']:+.2f}) n={b['val_n']} | "
                        f"OOS {b['oos_fwd_bp']:+.2f}(뒤{b['oos_flip_bp']:+.2f}) n={b['oos_n']}")
                out[variant] = {"n_candidates": ns, "thin": bool(thin), "n_genuine": len(gen),
                                "n_genuine_wide_sl": len(wide), "best": b,
                                "grid_best_any": best_any}
            res[sname] = {"thr": thr, "gap": spec["gap"], "H": H, "variants": out}
        rep["assets"][asset] = res

    log(""); log("=" * 84)
    log("종합 -- SL 확장 / 레짐 게이팅으로 기존 기준에서 통과하는가 (10bp, 뒤집기, ARM>=1.0)")
    log("=" * 84)
    log(f"{'자산':<5}{'신호':<26}{'게이팅없음':>11}{'레짐게이팅':>11}  최선")
    tot = 0
    for asset, res in rep["assets"].items():
        for sname, v in res.items():
            a = v["variants"].get("게이팅 없음", {})
            g = v["variants"].get("레짐 chop 게이팅", {})
            na = a.get("n_genuine", 0) if not a.get("thin") else 0
            ng = g.get("n_genuine", 0) if not g.get("thin") else 0
            best = None
            for src in (a, g):
                if src.get("best") and not src.get("thin"):
                    if best is None or min(src["best"]["val_fwd_bp"], src["best"]["oos_fwd_bp"]) > \
                       min(best["val_fwd_bp"], best["oos_fwd_bp"]):
                        best = src["best"]
            tot += best is not None
            bs = (f"SL={best['sl']} VAL {best['val_fwd_bp']:+.2f}/OOS {best['oos_fwd_bp']:+.2f}"
                  if best else "-")
            log(f"{asset:<5}{sname:<26}{na:>11}{ng:>11}  {bs}")
    log("")
    log(f"⇒ 두께 유지하며 통과: **{tot}종**")
    rep["n_passed"] = tot
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
