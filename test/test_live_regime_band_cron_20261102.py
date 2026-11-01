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
    # 지연 실행(+8분): 마지막 닫힌 봉이 00분 봉이어도 같은 결정 시각(직전 정시)을 기록해야 한다 -- 이미 기록돼 있으므로 0건
    late_ms = int((last_open + pd.Timedelta(minutes=5 + 8)).timestamp() * 1000)
    late = C.main(now_ms=late_ms, fetch=make_fetch(full, last_open + pd.Timedelta(minutes=5)), log=log, scored=scored, summary=summ)
    assert late["recorded"] == 0 and late["decision_ts"].endswith(str(last_open + pd.Timedelta(minutes=5)).split(" ")[1]), late
    # 새 시각을 지연(+8분)으로 처음 기록하는 경우
    nxt = last_open + pd.Timedelta(hours=1)
    first = C.main(now_ms=int((nxt + pd.Timedelta(minutes=13)).timestamp() * 1000), fetch=make_fetch(full, nxt + pd.Timedelta(minutes=5)), log=log, scored=scored, summary=summ)
    assert first["recorded"] == 4, first
    # 오래된 데이터(+90분)는 거부하고 아무것도 쓰지 않는다
    before = sum(1 for _ in log.open())
    try:
        C.main(now_ms=int((nxt + pd.Timedelta(minutes=5 + 90)).timestamp() * 1000), fetch=make_fetch(full, nxt), log=log, scored=scored, summary=summ)
        raise SystemExit("오래된 데이터를 받아들임")
    except AssertionError:
        pass
    assert sum(1 for _ in log.open()) == before
    p = C.panel("XRPUSDT", now_ms, make_fetch(full, last_open))
    assert p.index[-1] == last_open and p.index[-1].minute == 55, "형성 중 봉이 제거되지 않음"
    print("cron 래퍼: 기록", n_log, "채점", s["n_scored"], "정확도(학습 포함 구간·참고 불가)", round(s["acc"], 3), "확신도별", {k: v["n"] for k, v in s["by_conf"].items()})
    print("OK")


if __name__ == "__main__":
    main()
