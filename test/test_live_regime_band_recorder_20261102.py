"""기록기 재생 검증: 과거 구간을 «실시간처럼» 매 시 잘라 record → 채점(score)하면 학습 데이터셋의 같은 시각 라벨·확률과 일치해야 한다.
(전진 증거가 아니라 기록기·채점기 정확성 검증. 모델은 이 구간을 학습에 포함했으므로 정확도 수치는 해석 금지.)   python test/test_live_regime_band_recorder_20261102.py"""
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_regime_band_recorder_20261102 as R  # noqa: E402

S = R.S


def main():
    full = {c: S.rf.load(c) for c in R.COINS}
    d = pd.read_parquet(S.acc.OUT / "data.parquet")
    bundle = joblib.load(S.MODEL)
    end = full["XRP"].index[-1]
    hours = pd.date_range(end.floor("h") - pd.Timedelta(hours=60), end.floor("h") - pd.Timedelta(hours=8), freq="h")      # 라벨이 다 찬 최근 구간
    log = Path(tempfile.mkdtemp()) / "fwd.jsonl"
    for h in hours:
        cut = {c: p.loc[: h - pd.Timedelta(minutes=5)].iloc[-3000:] for c, p in full.items()}
        R.record(cut, log)
    R.record(cut, log)                                                                            # 같은 시각 재기록은 중복 없이 무시
    n_lines = sum(1 for _ in log.open())
    assert n_lines == len(hours) * 4, f"중복/누락: {n_lines} != {len(hours) * 4}"
    df, summ = R.score(full, log)
    assert summ["n_pending"] == 0 and summ["n_scored"] == n_lines
    # 학습 데이터셋의 같은 (코인, 결정 시각) 행과 비교: 학습 행 ts = 진입 봉 시가 = 결정 시각 T
    key = d.assign(ts=pd.to_datetime(d.ts)).set_index(["coin", "ts"])
    worst_p, bad_y = 0.0, 0
    for _, r in df.iterrows():
        k = (r.coin, pd.Timestamp(r.ts))
        row = key.loc[k]
        p = float(bundle["model"].predict_proba(row[bundle["features"]].to_frame().T.astype(float))[:, 1][0])
        worst_p = max(worst_p, abs(p - r.p_trend))
        bad_y += int(row["yA48"] != r.y)
    print(f"재생 {len(df)}건: 확률 최대 차이 {worst_p:.2e} · 라벨 불일치 {bad_y} · 채점 정확도(학습 포함 구간이라 참고 불가) {summ['acc']:.3f}")
    assert worst_p < 1e-8 and bad_y == 0
    print("OK")


if __name__ == "__main__":
    main()
