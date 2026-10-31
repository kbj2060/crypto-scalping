"""ETH 1h 굴림 Platt90 서빙 = 연구 two_tier_r4()(C0 + layers()[0]["굴림Platt90"]) 패리티 (2026-10-26). 개발 PC 전용(연구 rows·1분봉 parquet).

  python test/test_box_hold_platt90_20261026.py      (≈3분 · 2026-07·08·09 세 달)

(i) 적합 스크립트 행·라벨 = 연구 rolling 이 그 달에 쓴 행·라벨(100%) · (a, b) 차 보고 · 연구 p 와 같은 행만으로는 1e-6 안.
(ii) 서빙 payload 1h p = sig(a·logit p + b)(반올림 4자리) · 연구 굴림Platt90 행 값과 비교 · 파일 없음/40일 넘음/다른 모델 = 원래 p.
🔴 연구 p 와 서빙 p 는 ~1% 행에서 다르다: S_season_now/fut 가 연구(pandas rolling 평균)와 서빙(np.mean)에서 1~2 ulp 차인데
   그 값이 HGB 구간 경계에 정확히 걸려 나무 분기가 뒤집힌다(그 행 |Δp| 최대 ~0.02). 그래서 (a, b) 는 1e-6 이 아니라 ~1e-4 에서 같다.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dashboard.box_hold as bh  # noqa: E402 -- 먼저 올려 둔다(연구 모듈이 자기 워크트리 dashboard 를 집지 않게 · 모델 파일 md5 같음)

RES = Path("/home/kbj20/crypto-scalping/.claude/worktrees/chart-indicators-tuning-ecfa25/scripts/research_eth_box_hold_calibration_drift_20261025.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R = load("R", RES)
F = load("F", ROOT / "scripts/fit_box_hold_platt90_20261026.py")
assert json.loads((R.OUT / "round4.json").read_text())["1h"]["S"] == "C0+굴림Platt90"

# two_tier_r4() 1h 그대로
A = pd.read_parquet(R.OUT / "rows.parquet"); g = A[A.scale == "1h"]; W = 60
te = g[g.ts >= R.V0].dropna(subset=R.LEAN + ["B_rate30"]).reset_index(drop=True); ts, y = te.ts.to_numpy(), te.y.to_numpy().astype(float)
cf = R.BH["scales"]["1h"]
p0 = cf["model"].predict_proba(te[cf["cols"]])[:, 1]; pS = R.layers(ts, p0, y, W)[0]["굴림Platt90"]

ev = {k: np.asarray(v, np.int64) for k, v in json.loads(F.EVENTS.read_text()).items()}
iso = lambda v: pd.Timestamp(int(v), unit="ms", tz="UTC").isoformat()  # noqa: E731
evl = [{"time_utc": iso(v), "category": "fomc", "title_ko": "FOMC 금리결정 발표"} for v in ev["major"]] + \
      [{"time_utc": iso(v), "category": "econ", "detail": "release_id=180"} for v in ev["claims"]]
K = F.load_vision(pd.Timestamp("2026-06-25"), None).drop_duplicates("t").sort_values("t").reset_index(drop=True)
pos = pd.Series(np.arange(len(K)), index=K.t)
tmp = Path(tempfile.mkdtemp()) / "platt90.json"
bh.PLATT90["eth"] = tmp

worst = {"a": 0.0, "b": 0.0, "ab_same_rows": 0.0, "serve_formula": 0.0, "serve_vs_research": 0.0}
for T in ("2026-07-01", "2026-08-01", "2026-09-01"):
    Tn = np.datetime64(T)
    fit = (ts >= Tn - np.timedelta64(90, "D")) & (ts + (W + 1) * R.MIN <= Tn)
    assert Tn - np.timedelta64(90, "D") >= np.datetime64(R.V0)                     # rolling 의 t0 자르기가 안 걸리는 달
    aR, bR = R.fit_platt(p0[fit], y[fit])
    app = (ts >= Tn) & (ts < np.datetime64(pd.Timestamp(T) + pd.offsets.MonthBegin(1)))
    assert np.allclose(pS[app], R.sig(aR * R.logit(p0[app]) + bR), rtol=0, atol=1e-12)   # 이 (aR, bR) 가 rolling 이 그 달에 쓴 값
    # (i)
    ts2, p2, y2 = F.fit_rows(pd.Timestamp(T), False, F.EVENTS)
    assert len(ts2) == fit.sum() and (ts2 == ts[fit]).all() and (y2 == y[fit]).all(), "행·라벨 불일치"
    a, b = F.fit_platt(p2, y2)
    same = np.abs(p2 - p0[fit]) < 1e-9
    a1, b1 = F.fit_platt(p2[same], y2[same]); a2, b2 = R.fit_platt(p0[fit][same], y[fit][same])
    assert abs(a1 - a2) < 1e-6 and abs(b1 - b2) < 1e-6, (a1, a2, b1, b2)
    assert (~same).mean() < 0.02, (~same).mean()
    assert abs(a - aR) < 1e-3 and abs(b - bR) < 1e-3, (a, aR, b, bR)
    for k, v in (("a", abs(a - aR)), ("b", abs(b - bR)), ("ab_same_rows", max(abs(a1 - a2), abs(b1 - b2)))):
        worst[k] = max(worst[k], v)
    print(f"{T}: 행 {fit.sum():,} 같음 · 라벨 같음 · p 다른 행 {(~same).sum()}({(~same).mean():.2%}, 최대 |Δp| {np.abs(p2 - p0[fit]).max():.4f}) · "
          f"a {a:.6f} vs 연구 {aR:.6f} · b {b:.6f} vs {bR:.6f} · p 같은 행만 차 {max(abs(a1 - a2), abs(b1 - b2)):.1e}")
    # (ii) 그 달 행 12개를 서빙 payload 로
    tmp.write_text(json.dumps({"a": a, "b": b, "fit_start": str(pd.Timestamp(T) - pd.Timedelta(days=90)), "fit_end": T,
                               "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "model_file": "box_hold_hgb_20261016.joblib"}))
    for j in np.random.default_rng(int(Tn.astype(int))).choice(np.where(app)[0], 12, replace=False):
        i = int(pos[int(pd.Timestamp(ts[j]).value // 10 ** 6)])
        k = K.iloc[i - F.NEED - 59:i + 1]
        k1m = pd.DataFrame({"timestamp": pd.to_datetime(k.t, unit="ms", utc=True), "high": k.h, "low": k.l, "close": k.c,
                            "volume": k.v, "taker_buy_base": k.tb}).reset_index(drop=True)
        s = bh.payload(k1m, evl)["scales"]["1h"]
        x = bh.features(k1m, evl)["1h"]
        praw = float(cf["model"].predict_proba(pd.DataFrame([[x[c] for c in cf["cols"]]], columns=cf["cols"]))[0, 1])
        want = float(R.sig(a * R.logit(praw) + b))
        assert abs(s["p"] - want) <= 5e-5 + 1e-12 and s["calib"].startswith("platt90 "), (s, want)
        worst["serve_formula"] = max(worst["serve_formula"], abs(s["p"] - want))
        if abs(praw - p0[j]) < 1e-9:                                                # 분기 뒤집힘 행이 아니면 연구 행 값과 같다
            d = abs(s["p"] - pS[j]); worst["serve_vs_research"] = max(worst["serve_vs_research"], d)
            assert d <= 5e-5 + 2e-4, (ts[j], s["p"], pS[j])
        else:
            print(f"  {ts[j]} 분기 뒤집힘 행: 서빙 {s['p']:.4f} · 연구 {pS[j]:.4f}")
        assert all("calib" not in bh.payload(k1m, evl)["scales"][n] for n in ("30m", "4h"))
    # 원래 동작 불변: 파일 없음 · 40일 넘음 · 다른 모델
    for bad in (None, {"created_utc": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=41)).isoformat()}, {"model_file": "other.joblib"}):
        if bad is None:
            tmp.unlink()
        else:
            tmp.write_text(json.dumps({"a": a, "b": b, "fit_start": "x", "fit_end": "y", "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                       "model_file": "box_hold_hgb_20261016.joblib", **bad}))
        s = bh.payload(k1m, evl)["scales"]["1h"]
        assert s["p"] == round(praw, 4) and "calib" not in s, (bad, s)

print("최대 차:", {k: f"{v:.2e}" for k, v in worst.items()})
print("platt90 패리티 OK")
