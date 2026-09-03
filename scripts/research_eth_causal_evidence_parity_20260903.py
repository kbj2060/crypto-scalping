#!/usr/bin/env python3
"""ETH 증거신호 **인과 경로 파리티** -- XRP·BTC와 같은 검정을 ETH에도 (2026-09-03).

## 왜 -- 감사 방식이 자산마다 달랐다

2026-09-02 ETH 승격감사는 앵커 미래참조를 **top2 포트폴리오(str_z+demarker) 1건**으로 시연했고
(연구앵커 +11.73/+14.08/+7.97 → 지연확정 −5.66/+1.57/−6.44), 나머지 4종
(orthogonal·smt·liquidity_sweep·taker)은 **일괄 무효 판정만** 받고 개별 측정된 적이 없다.

반면 2026-09-03에 XRP·BTC는 **신호별로** 지연확정(10/10) + 인과적 첫발동 + 레짐 게이팅 +
다중신호 합의를 전부 받았다.

⇒ 같은 저장소 안에서 **자산마다 감사 깊이가 다르다.** 오늘 두 번 겪은
"한 자산만 감사하고 다른 자산은 누락" 패턴이 여기선 방향만 반대다.
**ETH에도 같은 인과 검정을 걸어 3자산 파리티를 맞춘다.**

## 설계 -- XRP·BTC와 **완전히 동일**

  · 왕복 **10bp**, margin 0.30 x leverage 3.0, 96셀 SL[1.5~4.0] x ARM[0.5~2.0] x Trail[0.1~0.5]
  · 발동: **인과적 첫발동**(직전 GAP봉 내 같은 쪽 발동 없음) -- dedup/앵커 없음
  · **방향뒤집기 대조군 전량**, **ARM >= 1.0**, 두께 하한 100
  · 변형 3종: (a) 게이팅 없음 (b) 레짐 chop 게이팅(ETH `S12_K3`) (c) 다중신호 합의 K>=2,3

ETH 배포 셀(HIT_TYPE/H/K)은 호메로스 §1 대조표 기준:
  demarker touch/8/0.70(GAP 12) · short_term_return_z touch/12/1.75(GAP 3)
  taker_delta_z_climax touch/24/2.00(GAP 3) · kalman touch/12/2.5(GAP 12)

⚠️HOLDOUT 미터치. VAL+OOS만.
⚠️여기서 통과가 나와도 정식 관문(무작위진입 귀무 + 측면별 갭 + HOLDOUT)이 남는다.
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

from live_evidence_signal_dashboard_20260823 import compute_signals   # noqa: E402

_F = importlib.util.spec_from_file_location(
    "final2", ROOT / "scripts/research_xrp_btc_regime_gate_and_consensus_20260903.py")
_f = importlib.util.module_from_spec(_F)
_F.loader.exec_module(_f)
_t = _f._t

OUT = ROOT / "data/research/eth_causal_evidence_parity_20260903.json"

KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
PARTNER = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"   # ETH의 교차자산 슬롯 = BTC
FUNDING_DIR = ROOT / "data/research/funding_extracted/ETHUSDT"
REGIME_MODEL = ROOT / "tmp/eth_regime_s12k3_20260902/model.joblib"
REGIME_LOADER = "research_eth_regime_s12k3_label_train_20260902.py"

# ETH 배포 셀 (호메로스 §1 자산별 대조표)
ETH_SIGNALS = {
    "demarker_extreme": {"kind": "bounded", "col": "dem", "thr": 0.10, "gap": 12, "H": 8},
    "short_term_return_z": {"kind": "z", "col": "ret3_z", "thr": 2.5, "gap": 3, "H": 12},
    "taker_delta_z_climax": {"kind": "z", "col": "delta_z", "thr": 2.0, "gap": 3, "H": 24},
    "kalman_deviation_meanrev": {"kind": "z", "col": "kalman_dev_z", "thr": 2.0, "gap": 12, "H": 12},
}
CONSENSUS = [(2, 3, 12), (3, 3, 12), (3, 3, 24), (3, 6, 12), (3, 6, 24), (3, 12, 24)]


def log(m): print(f"[eth-par] {m}", flush=True)


def load_eth_funding():
    """BTC/XRP 로더와 동일 형식. ⚠️[ns] 통일까지 로더 안에서 한다(이 저장소 상습 함정)."""
    fs = sorted(FUNDING_DIR.glob("ETHUSDT-fundingRate-*.csv"))
    if not fs:
        log("  ⚠️ETH 펀딩 CSV 없음 -- funding_z는 NaN으로 간다(orthogonal 트리거만 영향)")
        return None
    f = pd.concat([pd.read_csv(p) for p in fs], ignore_index=True)
    f["calc_time"] = pd.to_datetime(f["calc_time"], unit="ms").astype("datetime64[ns]")
    f = f.sort_values("calc_time").drop_duplicates("calc_time").reset_index(drop=True)
    mean = f["last_funding_rate"].rolling(90, min_periods=30).mean()
    std = f["last_funding_rate"].rolling(90, min_periods=30).std()
    f["funding_z"] = (f["last_funding_rate"] - mean) / std.replace(0.0, np.nan)
    out = f[["calc_time", "funding_z"]].copy()
    for c in out.columns:
        if str(out[c].dtype).startswith("datetime64"):
            out[c] = out[c].astype("datetime64[ns]")
    return out


def eth_chop_mask(ts_frame):
    """ETH `S12_K3` 예측 chop. ⭐Phase3 로더를 태워 파생 피쳐를 만든다(직접 CSV 읽으면 8피쳐 누락)."""
    if not REGIME_MODEL.exists():
        log("  ⚠️ETH 레짐 아티팩트 없음"); return None
    sp = importlib.util.spec_from_file_location("ethreg", ROOT / "scripts" / REGIME_LOADER)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    pay = joblib.load(REGIME_MODEL)
    fc = pay["feature_cols"]
    # ⚠️로더 시그니처가 자산마다 다르다 -- ETH `load_frame()`은 **무인자**,
    # BTC/XRP `load_btc_frame(feat_cols)`는 인자를 받는다. 둘 다 받아준다.
    loader = getattr(m, "load_btc_frame", None) or getattr(m, "load_eth_frame", None) \
        or getattr(m, "load_frame", None)
    if loader is None:
        log("  ⚠️ETH Phase3 로더 함수를 못 찾음"); return None
    try:
        df = loader(fc)
    except TypeError:
        df = loader()
    miss = [c for c in fc if c not in df.columns]
    if miss:
        log(f"  ⚠️피쳐 누락 {len(miss)}개: {miss[:5]}"); return None
    x = df[fc].apply(pd.to_numeric, errors="coerce")
    med = pay["feature_medians"]
    for c in fc:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(med.get(c, 0.0))
    pred = pay["model"].predict(x)
    ts = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    mm = pd.DataFrame({"timestamp": ts_frame}).merge(
        pd.DataFrame({"timestamp": ts, "pred": pred}), on="timestamp", how="left")
    cm = (mm["pred"].to_numpy() == 2)
    log(f"  레짐 S12_K3 로드 OK -- 예측 chop 비중 {cm.mean():.3f} "
        f"(매핑 결측 {int(mm['pred'].isna().sum()):,}봉)")
    return cm


def fires_for(frame, spec):
    col = frame[spec["col"]].to_numpy(float); thr = spec["thr"]
    b, t_ = ((col <= thr), (col >= 1.0 - thr)) if spec["kind"] == "bounded" \
        else ((col <= -thr), (col >= thr))
    return (_t.first_of_cluster(np.nan_to_num(b, nan=False), spec["gap"]),
            _t.first_of_cluster(np.nan_to_num(t_, nan=False), spec["gap"]))


def main() -> int:
    t0 = time.time()
    raw = pd.read_csv(KLINES, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    partner = pd.read_csv(PARTNER, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    for d in (raw, partner):
        d["timestamp"] = d["timestamp"].astype("datetime64[ns]")
    frame = compute_signals(raw, btc_df=partner, funding_df=load_eth_funding())
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
    kl = frame[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
    atr = frame["atr_pct"].to_numpy(float)
    log(f"ETH 프레임 {len(frame):,}봉  {frame['timestamp'].min()} ~ {frame['timestamp'].max()}")
    cm = eth_chop_mask(frame["timestamp"])

    rep = {"asset": "ETHUSDT", "cost_bp": 10.0, "criteria_unchanged": True,
           "holdout_touched": False,
           "note": "XRP·BTC와 동일 검정 -- 인과적 첫발동 + 레짐게이팅 + 다중신호 합의",
           "gated": {}, "consensus": {}}

    log(""); log("--- ① 인과적 첫발동 (+ 레짐 chop 게이팅) ---")
    for sname, spec in ETH_SIGNALS.items():
        cb, ct = fires_for(frame, spec)
        H = spec["H"]
        out = {}
        for vname, sel in (("게이팅없음", cb | ct),
                           ("chop게이팅", (cb | ct) & (cm if cm is not None else False))):
            if vname == "chop게이팅" and cm is None:
                continue
            idx = np.flatnonzero(sel)
            idx = idx[np.isfinite(atr[idx]) & (atr[idx] > 0) & (idx < len(kl) - 1)]
            if len(idx) < 50:
                log(f"  {sname:<24} {vname:<10} 발동 부족({len(idx)})"); continue
            cells, ns = _f.run_grid(kl, idx, cb[idx], atr[idx], H)
            thin = ns["val"] < _f.MIN_CANDIDATES or ns["oos"] < _f.MIN_CANDIDATES
            g = _f.genuine(cells)
            best = max(g, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"])) if g else None
            out[vname] = {"n": ns, "thin": bool(thin), "n_genuine": len(g), "best": best,
                          "n_fires": int(len(idx))}
            log(f"  {sname:<24} {vname:<10} 발동 {len(idx):>5} 후보 V{ns['val']}/O{ns['oos']}"
                f"{'(얇음)' if thin else '':<6} 진짜 {len(g):>2}셀"
                + (f"  ⭐SL={best['sl']} VAL {best['val_fwd_bp']:+.2f}(뒤{best['val_flip_bp']:+.2f}) | "
                   f"OOS {best['oos_fwd_bp']:+.2f}(뒤{best['oos_flip_bp']:+.2f})" if best else ""))
        rep["gated"][sname] = {"spec": spec, "variants": out}

    log(""); log("--- ② 다중신호 합의 ---")
    rb, rt = [], []
    for sname, spec in ETH_SIGNALS.items():
        col = frame[spec["col"]].to_numpy(float); thr = spec["thr"]
        b, t_ = ((col <= thr), (col >= 1.0 - thr)) if spec["kind"] == "bounded" \
            else ((col <= -thr), (col >= thr))
        rb.append(np.nan_to_num(b, nan=False)); rt.append(np.nan_to_num(t_, nan=False))
    nb, nt = np.sum(rb, axis=0), np.sum(rt, axis=0)
    log("  동시발동 bottom: " + " ".join(f"{k}개={int((nb == k).sum()):,}" for k in range(1, 5)))
    log("  동시발동 top   : " + " ".join(f"{k}개={int((nt == k).sum()):,}" for k in range(1, 5)))
    for K, gap, H in CONSENSUS:
        cb = _t.first_of_cluster(nb >= K, gap); ct = _t.first_of_cluster(nt >= K, gap)
        idx = np.flatnonzero(cb | ct)
        idx = idx[np.isfinite(atr[idx]) & (atr[idx] > 0) & (idx < len(kl) - 1)]
        if len(idx) < 50:
            log(f"  K>={K} GAP={gap:<3} H={H:<3} 발동 부족({len(idx)})"); continue
        cells, ns = _f.run_grid(kl, idx, cb[idx], atr[idx], H)
        thin = ns["val"] < _f.MIN_CANDIDATES or ns["oos"] < _f.MIN_CANDIDATES
        g = _f.genuine(cells)
        best = max(g, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"])) if g else None
        rep["consensus"][f"K{K}_GAP{gap}_H{H}"] = {
            "n": ns, "thin": bool(thin), "n_genuine": len(g), "best": best,
            "n_fires": int(len(idx))}
        log(f"  K>={K} GAP={gap:<3} H={H:<3} 발동 {len(idx):>5} 후보 V{ns['val']}/O{ns['oos']}"
            f"{'(얇음)' if thin else '':<6} 진짜 {len(g):>2}셀"
            + (f"  ⭐VAL {best['val_fwd_bp']:+.2f} / OOS {best['oos_fwd_bp']:+.2f}" if best else ""))

    log(""); log("=" * 78)
    log("종합 -- ETH 인과 경로 (10bp, 96셀, 뒤집기, ARM>=1.0, 두께 100)")
    log("=" * 78)
    tot = 0
    for sname, v in rep["gated"].items():
        for vname, o in v["variants"].items():
            if o["n_genuine"] > 0 and not o["thin"]:
                tot += 1
                log(f"  ✅{sname} [{vname}] 진짜 {o['n_genuine']}셀")
    for key, o in rep["consensus"].items():
        if o["n_genuine"] > 0 and not o["thin"]:
            tot += 1
            log(f"  ✅합의 {key} 진짜 {o['n_genuine']}셀")
    log(f"  ⇒ 두께 유지하며 통과: **{tot}건**" + ("" if tot else "  (전부 실패)"))
    rep["n_passed"] = tot
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
