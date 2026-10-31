"""XRP 1h 굴림 Platt90 (2026-10-31) -- 적합 스크립트 = 연구 행 패리티 + 서빙 payload 공식. 개발 PC 전용(1분봉 parquet · 연구 행 사본).

  python test/test_box_hold_platt90_xrp_20261031.py      (≈2분)

(i) fit_box_hold_platt90 --asset xrp 의 2026-10-01 (행 수, a, b) = 연구 입력 xrp_rows.parquet(10-20 XRP 박스 연구) 1h 행으로 맞춘 값(1e-9).
(ii) payload(asset="xrp") 1h p = sig(a·logit p0 + b) · calib 꼬리표 · 다른 모델용 파일이면 원래 p · 30분 은 보정 없음.
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

ROWS = Path("/home/kbj20/crypto-scalping/.claude/worktrees/agent-ac0360d621a6b463d/tmp/xrp_box_20261031/in/xrp_rows.parquet")
T = pd.Timestamp("2026-10-01")

F.ASSET = "xrp"
ts, p, y = F.fit_rows(T, False, F.EVENTS)
a, b = F.fit_platt(p, y)
cf = bh.BUNDLES["xrp"]["scales"]["1h"]
A = pd.read_parquet(ROWS, filters=[("scale", "==", "1h")]).replace([np.inf, -np.inf], np.nan).dropna(subset=cf["cols"])
A = A[(A.ts >= T - pd.Timedelta(days=90)) & (A.ts + pd.Timedelta(minutes=61) <= T) & (np.exp(A.log_rng) > 0)]
aR, bR = F.fit_platt(cf["model"].predict_proba(A[cf["cols"]])[:, 1], A.hold.to_numpy(float))
assert len(A) == len(y) and abs(a - aR) < 1e-9 and abs(b - bR) < 1e-9, (len(A), len(y), a, aR, b, bR)
print(f"(i) 행 {len(y):,} · a {a:.6f} = 연구 {aR:.6f} · b {b:.6f} = {bR:.6f}")

K = F.load_vision(T - pd.Timedelta(days=2), None).drop_duplicates("t").sort_values("t")
K = K[K.t < T.value // 10 ** 6].tail(F.NEED + 60)
k1m = pd.DataFrame({"timestamp": pd.to_datetime(K.t, unit="ms", utc=True), "high": K.h, "low": K.l, "close": K.c,
                    "volume": K.v, "taker_buy_base": K.tb}).reset_index(drop=True)
bh.PLATT90["xrp"] = Path(tempfile.mkdtemp()) / "x.json"
raw = bh.payload(k1m, [], asset="xrp")["scales"]                       # 파일 없음 = 원래 p
assert "calib" not in raw["1h"]
meta = {"a": a, "b": b, "fit_start": str(T - pd.Timedelta(days=90)), "fit_end": str(T), "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
bh.PLATT90["xrp"].write_text(json.dumps({**meta, "model_file": bh.HGB["eth"]}))   # ETH 모델용 파일 = 안 씀
assert bh.payload(k1m, [], asset="xrp")["scales"]["1h"]["p"] == raw["1h"]["p"]
bh.PLATT90["xrp"].write_text(json.dumps({**meta, "model_file": bh.HGB["xrp"]}))
s = bh.payload(k1m, [], asset="xrp")["scales"]
want = 1 / (1 + np.exp(-(a * bh.logit(raw["1h"]["p"]) + b)))
assert abs(s["1h"]["p"] - want) < 1e-4 and s["1h"]["calib"].startswith("platt90 2026-07-03"), (s["1h"], want)   # raw p 는 4자리 반올림
assert s["30m"]["p"] == raw["30m"]["p"] and "calib" not in s["30m"]
print(f"(ii) 1h p {raw['1h']['p']} → {s['1h']['p']} · 30분 그대로 · 다른 모델용 파일 무시 -- OK")
