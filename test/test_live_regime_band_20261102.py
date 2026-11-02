"""학습-서빙 패리티·인과성: 과거 정시 시점으로 패널을 잘라 서빙 경로를 돌리면 학습 데이터셋 행과 같은 피쳐·확률이 나온다.
(잘린 패널은 미래 봉을 못 가지므로 같으면 미래를 안 썼다는 증거이기도 하다.)   python test/test_live_regime_band_20261102.py"""
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_regime_band_20261102 as S  # noqa: E402


def main():
    d = pd.read_parquet(S.acc.OUT / "data.parquet")
    bundle = joblib.load(S.MODEL)
    panels = {c: S.rf.load(c) for c in ("BTC", "XRP", "ETH")}
    rng = np.random.default_rng(0)
    worst_f, worst_p, n = 0.0, 0.0, 0
    for coin in ("XRP", "ETH", "BTC"):
        rows = d[(d.coin == coin) & (d.ts >= "2026-06-10")].dropna(subset=bundle["features"]).sample(8, random_state=1)
        for _, r in rows.iterrows():
            last_open = r.ts - pd.Timedelta(minutes=5)      # 학습 행 ts = 진입 봉(i+1) 시가 → 결정 봉 i 시가 = ts−5분
            cut = {c: p.loc[: last_open] for c, p in panels.items()}
            cut = {c: p.iloc[-3000:] for c, p in cut.items()}
            out = S.predict_latest(cut, coin, bundle)
            x = np.array([out["features"][f] for f in bundle["features"]], float)
            y = r[bundle["features"]].to_numpy(float)
            worst_f = max(worst_f, float(np.nanmax(np.abs(x - y))))
            py = float(bundle["model"].predict_proba(r[bundle["features"]].to_frame().T.astype(float))[:, 1][0])
            worst_p = max(worst_p, abs(out["p_trend"] - py))
            n += 1
    print(f"패리티 {n}건: 피쳐 최대 차이 {worst_f:.2e} · 확률 최대 차이 {worst_p:.2e}")
    assert worst_f < 1e-8 and worst_p < 1e-8, "학습-서빙 불일치"
    # 미래 불변성: 같은 시점에서 뒤에 봉을 더 붙여도(잘라낸 것과) 결과는 같아야 하므로 위 패리티가 곧 증거. 마지막 봉 분(55) 가드:
    try:
        S.predict_latest({c: p.iloc[-3001:-1] for c, p in panels.items()}, "XRP", bundle)      # 마지막 봉 23:50 → 가드가 걸려야 한다
        raise SystemExit("가드 실패: 55분이 아닌 마지막 봉을 받아들임")
    except AssertionError:
        pass
    print("OK")


if __name__ == "__main__":
    main()
