#!/usr/bin/env python3
"""XRP·BTC 증거신호 마지막 두 축 -- **레짐 게이팅**(재실행) + **다중신호 합의**. 기준은 그대로.

## 왜 이 둘인가

사용자 지시: "진입모델은 건드리지 말고 증거신호 성능을 최대한 끌어올려라."
오늘까지 닫힌 축: 격자 최적화 / 시드 / 임계값 조이기(0/40) / 인과적 첫발동(0/96) /
SL 격자 확장(0/8, 가설 반증) / 비용 완화(사실상 무효).

**남은 두 축을 돌린다.**

### ① 레짐 게이팅 -- 앞선 실행은 결과가 아니라 **미실행**이었다

`⚠️피쳐 누락 8개 -- 게이팅 생략`. 캐노니컬 CSV를 **직접 읽어서** 136피쳐 중 8개가 없었다.
Phase 3에서는 같은 모델이 정상 동작하는데, 그 경로는 `load_btc_frame()`이 `_with_raw_state12`
같은 파생을 태우기 때문이다. ⇒ **그 로더를 그대로 쓴다**(재구현 금지).

XRP는 오늘 배포한 `S96_K9`(Phase3b 13/16 · OOS +0.1437), BTC는 `S24_K3`을 쓴다.

### ② 다중신호 합의 -- 미시도 축

같은 봉에서 **같은 방향으로 K개 이상**의 신호가 발동하면 진입한다.
⚠️ETH에서 복합 AND-필터가 **풀링 아티팩트**로 기각된 전례가 있다
(`eth_composite_and_filter_rejected_pooling_artifact_20260902`) -- 풀링 최고 3.72~4.38x가
VAL/OOS 분리에서 0/7이 됐다. ⇒ **처음부터 VAL/OOS를 분리해서 본다**(이 스크립트의 판정 기준이
이미 VAL·OOS 동시 양수이므로 자동으로 걸린다).

## 설계 -- 평가 기준은 하나도 안 건드린다

  · 왕복 **10bp**, margin 0.30 x leverage 3.0
  · **96셀** SL[1.5~4.0] x ARM[0.5~2.0] x Trail[0.1~0.5]  (SL 확장은 어제 반증돼 원복)
  · 발동: **인과적 첫발동**(dedup/앵커 없음)
  · **방향뒤집기 대조군 전량**, **ARM >= 1.0**
  · 판정: VAL>0 AND OOS>0 AND 정방향>뒤집기(양 구간), 후보 >= 100건

⚠️HOLDOUT 미터치.
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

OUT = ROOT / "data/research/xrp_btc_regime_gate_and_consensus_20260903.json"

MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.001    # ⭐10bp
VAL_START, OOS_START, HOLDOUT_START = _t.VAL_START, _t.OOS_START, _t.HOLDOUT_START
SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]      # ⭐원복(확장은 반증됨)
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]
MIN_CANDIDATES = 100

# 레짐: (배포 아티팩트, Phase3 로더 스크립트)
REGIME = {
    "XRP": (ROOT / "tmp/xrp_regime_s96k9_20260903/model.joblib",
            "research_xrp_regime_s48k6_label_train_20260903.py", "S96_K9"),
    "BTC": (ROOT / "tmp/btc_regime_s24k3_20260902/model.joblib",
            "research_btc_regime_s24k3_label_train_20260902.py", "S24_K3"),
}
CONSENSUS_K = [2, 3]
CONSENSUS_GAP = [3, 6, 12]
CONSENSUS_H = [6, 12, 24]


def log(m): print(f"[final2] {m}", flush=True)


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


def genuine(cells):
    return [c for c in cells
            if c["val_fwd_bp"] > 0 and c["oos_fwd_bp"] > 0
            and c["val_fwd_bp"] > c["val_flip_bp"] and c["oos_fwd_bp"] > c["oos_flip_bp"]
            and c["arm"] >= 1.0]


def chop_mask(asset, ts_frame):
    """⭐Phase 3의 `load_btc_frame()`을 그대로 써서 파생 피쳐를 만든다.
    앞선 실행은 캐노니컬 CSV를 직접 읽어 8피쳐가 없었다(= 미실행)."""
    mp, rel, tag = REGIME[asset]
    if not mp.exists():
        log(f"  ⚠️레짐 아티팩트 없음: {mp}"); return None, tag
    sp = importlib.util.spec_from_file_location(f"reg_{asset}", ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    pay = joblib.load(mp)
    fc = pay["feature_cols"]
    df = m.load_btc_frame(fc)                      # 이름만 btc -- 각 자산 캐노니컬을 읽는다
    miss = [c for c in fc if c not in df.columns]
    if miss:
        log(f"  ⚠️로더를 태워도 피쳐 누락 {len(miss)}개: {miss[:5]}"); return None, tag
    x = df[fc].apply(pd.to_numeric, errors="coerce")
    med = pay["feature_medians"]
    for c in fc:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(med.get(c, 0.0))
    pred = pay["model"].predict(x)
    ts = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    mm = pd.DataFrame({"timestamp": ts_frame}).merge(
        pd.DataFrame({"timestamp": ts, "pred": pred}), on="timestamp", how="left")
    cm = (mm["pred"].to_numpy() == 2)
    log(f"  레짐 {tag} 로드 OK -- 예측 chop 비중 {cm.mean():.3f} "
        f"(매핑 결측 {int(mm['pred'].isna().sum()):,}봉)")
    return cm, tag


def main() -> int:
    t0 = time.time()
    rep = {"cost_bp": 10.0, "sl_grid": SL_GRID, "min_candidates": MIN_CANDIDATES,
           "criteria_unchanged": True, "holdout_touched": False, "assets": {}}

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
        cm, rtag = chop_mask(asset, frame["timestamp"])
        res = {"regime_tag": rtag, "gated": {}, "consensus": {}}

        # ---------- ① 레짐 게이팅 ----------
        log("")
        log("--- ① 레짐 chop 게이팅 ---")
        for sname, spec in _t.SIGNALS.items():
            col = frame[spec["col"]].to_numpy(float)
            H, thr = spec["H"][asset], spec["thr"][0]
            if spec["kind"] == "bounded":
                fb, ft = (col <= thr), (col >= 1.0 - thr)
            else:
                fb, ft = (col <= -thr), (col >= thr)
            fb = np.nan_to_num(fb, nan=False); ft = np.nan_to_num(ft, nan=False)
            cb = _t.first_of_cluster(fb, spec["gap"]); ct = _t.first_of_cluster(ft, spec["gap"])
            out = {}
            for vname, sel in (("게이팅없음", cb | ct),
                               ("chop게이팅", (cb | ct) & (cm if cm is not None else False))):
                if vname == "chop게이팅" and cm is None:
                    continue
                idx = np.flatnonzero(sel)
                idx = idx[np.isfinite(atr_all[idx]) & (atr_all[idx] > 0) & (idx < len(kl) - 1)]
                if len(idx) < 50:
                    log(f"  {sname:<24} {vname:<10} 발동 부족({len(idx)})"); continue
                cells, ns = run_grid(kl, idx, cb[idx], atr_all[idx], H)
                thin = ns["val"] < MIN_CANDIDATES or ns["oos"] < MIN_CANDIDATES
                g = genuine(cells)
                ba = max(cells, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
                out[vname] = {"n": ns, "thin": bool(thin), "n_genuine": len(g),
                              "grid_best": ba}
                log(f"  {sname:<24} {vname:<10} 후보 V{ns['val']}/O{ns['oos']}"
                    f"{'(얇음)' if thin else '':<6} 진짜 {len(g):>2}셀  "
                    f"격자최선 VAL {ba['val_fwd_bp']:+.2f}/OOS {ba['oos_fwd_bp']:+.2f}")
            res["gated"][sname] = out

        # ---------- ② 다중신호 합의 ----------
        log("")
        log("--- ② 다중신호 합의 (같은 봉·같은 방향 K개 이상) ---")
        raw_b, raw_t = [], []
        for sname, spec in _t.SIGNALS.items():
            col = frame[spec["col"]].to_numpy(float)
            thr = spec["thr"][0]
            if spec["kind"] == "bounded":
                b, t_ = (col <= thr), (col >= 1.0 - thr)
            else:
                b, t_ = (col <= -thr), (col >= thr)
            raw_b.append(np.nan_to_num(b, nan=False)); raw_t.append(np.nan_to_num(t_, nan=False))
        nb = np.sum(raw_b, axis=0); nt = np.sum(raw_t, axis=0)
        log(f"  동시발동 분포 bottom: " +
            " ".join(f"{k}개={int((nb == k).sum()):,}" for k in range(1, 5)))
        log(f"  동시발동 분포 top   : " +
            " ".join(f"{k}개={int((nt == k).sum()):,}" for k in range(1, 5)))
        for K in CONSENSUS_K:
            for gap in CONSENSUS_GAP:
                cb = _t.first_of_cluster(nb >= K, gap)
                ct = _t.first_of_cluster(nt >= K, gap)
                idx0 = np.flatnonzero(cb | ct)
                if len(idx0) < 50:
                    log(f"  K>={K} GAP={gap:<3} 발동 부족({len(idx0)})"); continue
                for H in CONSENSUS_H:
                    idx = idx0[np.isfinite(atr_all[idx0]) & (atr_all[idx0] > 0) & (idx0 < len(kl) - 1)]
                    cells, ns = run_grid(kl, idx, cb[idx], atr_all[idx], H)
                    thin = ns["val"] < MIN_CANDIDATES or ns["oos"] < MIN_CANDIDATES
                    g = genuine(cells)
                    ba = max(cells, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
                    key = f"K{K}_GAP{gap}_H{H}"
                    res["consensus"][key] = {"n": ns, "thin": bool(thin), "n_genuine": len(g),
                                             "grid_best": ba, "n_fires": int(len(idx))}
                    mark = "⚠️얇음" if thin else ("✅" if g else "")
                    log(f"  K>={K} GAP={gap:<3} H={H:<3} 발동 {len(idx):>5} "
                        f"후보 V{ns['val']}/O{ns['oos']}  진짜 {len(g):>2}셀  "
                        f"격자최선 VAL {ba['val_fwd_bp']:+.2f}/OOS {ba['oos_fwd_bp']:+.2f}  {mark}")
        rep["assets"][asset] = res

    log(""); log("=" * 84)
    log("종합 -- 레짐 게이팅 / 다중신호 합의로 기존 기준에서 통과하는가")
    log("=" * 84)
    tot = 0
    for asset, res in rep["assets"].items():
        for sname, out in res["gated"].items():
            for vname, v in out.items():
                if v["n_genuine"] > 0 and not v["thin"]:
                    tot += 1
                    log(f"  ✅{asset} {sname} [{vname}] 진짜 {v['n_genuine']}셀")
        for key, v in res["consensus"].items():
            if v["n_genuine"] > 0 and not v["thin"]:
                tot += 1
                log(f"  ✅{asset} 합의 {key} 진짜 {v['n_genuine']}셀")
    log(f"  ⇒ 두께 유지하며 통과: **{tot}건**" + ("" if tot else "  (전부 실패)"))
    rep["n_passed"] = tot
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
