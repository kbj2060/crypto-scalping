"""크론 래퍼 검증(가짜 응답 주입, 네트워크 없음): 형성 중 봉 제거 · 기록/채점 누적 · 중복 없음 · 요약.   python test/test_live_regime_band_cron_20261102.py"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_regime_band_cron_20261102 as C  # noqa: E402

R, S = C.R, C.S


def make_fetch(full, last_open):
    """패널을 klines 행 형식으로 돌려주는 가짜 fetch. last_open 의 다음 봉(= 정시 봉)을 «형성 중»으로 한 개 더 붙인다."""
    def fetch(symbol, end_ms=None, limit=1500):
        df = full[symbol.replace("USDT", "")].loc[: last_open + pd.Timedelta(minutes=5)]
        if end_ms is not None:
            df = df[(df.index.astype("int64") // 10**6) <= end_ms]
        df = df.iloc[-limit:]
        ot = df.index.astype("int64") // 10**6
        return [[int(t), str(r.o), str(r.h), str(r.l), str(r.c), "0", int(t) + 299999, str(r.qv), int(r.n), "0", str(r.tbqv), "0"] for t, r in zip(ot, df.itertuples())]
    return fetch


def main():
    full = {c: S.rf.load(c) for c in R.COINS}
    tmp = Path(tempfile.mkdtemp())
    log, scored, summ = tmp / "f.jsonl", tmp / "s.jsonl", tmp / "sum.json"
    t0 = pd.Timestamp("2026-10-10 06:55")                                        # 결정 봉(시가 55분)
    for k in range(60):                                                          # 60시간을 «실시간처럼»
        last_open = t0 + pd.Timedelta(hours=k)
        now_ms = int((last_open + pd.Timedelta(minutes=5, seconds=120)).timestamp() * 1000)   # 정시+2분: 55분 봉은 닫혔고 00분 봉은 형성 중
        out = C.main(now_ms=now_ms, fetch=make_fetch(full, last_open), log=log, scored=scored, summary=summ)
        assert out["recorded"] == 4, out
    again = C.main(now_ms=now_ms, fetch=make_fetch(full, last_open), log=log, scored=scored, summary=summ)
    assert again["recorded"] == 0, "같은 시각 재실행은 중복 기록하면 안 된다"
    n_log = sum(1 for _ in log.open())
    assert n_log == 60 * 4, n_log
    s = json.loads(summ.read_text())
    # 마지막 결정 시각에서 4h(48봉)가 지난 행만 채점 = 처음 60−4=56시간분(마지막 4시간 결정은 미성숙)
    assert s["n_scored"] == (60 - 4) * 4, s["n_scored"]
    p = C.panel("XRPUSDT", now_ms, make_fetch(full, last_open))
    assert p.index[-1] == last_open and p.index[-1].minute == 55, "형성 중 봉이 제거되지 않음"
    print("cron 래퍼: 기록", n_log, "채점", s["n_scored"], "정확도(학습 포함 구간·참고 불가)", round(s["acc"], 3), "확신도별", {k: v["n"] for k, v in s["by_conf"].items()})
    print("OK")


if __name__ == "__main__":
    main()
