"""BTC·SOL 박스 유지 (2026-10-31) -- BTC 1h 굴림 Platt90 적합 = 연구 행 패리티 + payload 공식 · SOL = 보정 없음 · 둘 다 4h 꺼짐.
개발 PC 전용(1분봉 parquet · 연구 행 = python scripts/research_btc_sol_box_hold_20261031.py --run 산출).

  python test/test_box_hold_btc_sol_20261031.py      (≈2분)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dashboard.box_hold as bh  # noqa: E402
import scripts.fit_box_hold_platt90_20261026 as F  # noqa: E402

ROWS = ROOT / "tmp/box_btc_sol_20261031/btc/features.parquet"
T = pd.Timestamp("2026-10-01")

# (i) fit_box_hold_platt90 --asset btc 의 2026-10-01 (행 수, a, b) = 연구 1h 행으로 맞춘 값(1e-9)
F.ASSET = "btc"
ts, p, y = F.fit_rows(T, False, F.EVENTS)
a, b = F.fit_platt(p, y)
cf = bh.BUNDLES["btc"]["scales"]["1h"]
A = pd.read_parquet(ROWS, filters=[("scale", "==", "1h")]).replace([np.inf, -np.inf], np.nan).dropna(subset=cf["cols"])
A = A[(A.ts >= T - pd.Timedelta(days=90)) & (A.ts + pd.Timedelta(minutes=61) <= T) & (np.exp(A.log_rng) > 0)]
aR, bR = F.fit_platt(cf["model"].predict_proba(A[cf["cols"]])[:, 1], A.hold.to_numpy(float))
assert len(A) == len(y) and abs(a - aR) < 1e-9 and abs(b - bR) < 1e-9, (len(A), len(y), a, aR, b, bR)
print(f"(i) BTC 행 {len(y):,} · a {a:.6f} = 연구 {aR:.6f} · b {b:.6f} = {bR:.6f}")


def k1m_before(sym_asset: str) -> pd.DataFrame:
    F.ASSET = sym_asset
    K = F.load_vision(T - pd.Timedelta(days=2), None).drop_duplicates("t").sort_values("t")
    K = K[K.t < T.value // 10 ** 6].tail(F.NEED + 60)
    return pd.DataFrame({"timestamp": pd.to_datetime(K.t, unit="ms", utc=True), "high": K.h, "low": K.l, "close": K.c,
                         "volume": K.v, "taker_buy_base": K.tb}).reset_index(drop=True)


# (ii) payload(asset="btc") 1h p = sig(a·logit p0 + b) · 다른 모델용 파일 무시 · 30분 보정 없음 · 4h 꺼짐 · 모델 이름
k = k1m_before("btc")
bh.PLATT90["btc"] = Path(tempfile.mkdtemp()) / "x.json"
out = bh.payload(k, [], asset="btc")
raw = out["scales"]
assert out["model"] == "hgb-btc-20261031" and "calib" not in raw["1h"] and raw["4h"]["ok"] is False and raw["30m"]["ok"], out
meta = {"a": a, "b": b, "fit_start": str(T - pd.Timedelta(days=90)), "fit_end": str(T), "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
bh.PLATT90["btc"].write_text(json.dumps({**meta, "model_file": bh.HGB["xrp"]}))
assert bh.payload(k, [], asset="btc")["scales"]["1h"]["p"] == raw["1h"]["p"]
bh.PLATT90["btc"].write_text(json.dumps({**meta, "model_file": bh.HGB["btc"]}))
s = bh.payload(k, [], asset="btc")["scales"]
want = 1 / (1 + np.exp(-(a * bh.logit(raw["1h"]["p"]) + b)))
assert abs(s["1h"]["p"] - want) < 1e-4 and s["1h"]["calib"].startswith("platt90 2026-07-03"), (s["1h"], want)
assert s["30m"]["p"] == raw["30m"]["p"] and "calib" not in s["30m"]
print(f"(ii) BTC 1h p {raw['1h']['p']} → {s['1h']['p']} · 30분 그대로 · 4h 꺼짐 -- OK")

# (iii) SOL: 30분·1h 값 · 4h 꺼짐 · 보정 표에 없음(적합 스크립트 --asset sol 도 거절)
assert "sol" not in bh.PLATT90 and bh.platt90("sol") is None and "sol" not in F.PLATT90
o = bh.payload(k1m_before("sol"), [], asset="sol")
assert o["model"] == "hgb-sol-20261031" and o["scales"]["30m"]["ok"] and o["scales"]["1h"]["ok"] and "calib" not in o["scales"]["1h"] \
    and o["scales"]["4h"]["ok"] is False and o["px"] > 10, o
print(f"(iii) SOL 30분 {o['scales']['30m']['p']} · 1h {o['scales']['1h']['p']} · 4h 꺼짐 · 보정 없음 -- OK")
