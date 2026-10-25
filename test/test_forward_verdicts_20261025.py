"""전진 판정 실행 도구 시험(네트워크 없음) — scripts/forward_verdicts_20261025.py."""
import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import forward_verdicts_20261025 as FV  # noqa: E402

D = dt.date


@pytest.mark.parametrize("cand,day", [("vol", D(2026, 12, 23)), ("box", D(2026, 12, 23)), ("trend90", D(2027, 1, 22)), ("trend180", D(2027, 4, 22)), ("rl", D(2027, 1, 22))])
def test_date_guard(cand, day):
    with pytest.raises(SystemExit):
        FV.window(cand, day - dt.timedelta(days=1))
    for t in (day, day + dt.timedelta(days=30)):   # 판정일 당일·뒤: 창은 늘 10-24 ~ 판정일 전날
        s, e, real = FV.window(cand, t)
        assert real and s == FV.FWD and e.date() == day - dt.timedelta(days=1)
        assert FV.LIMIT_MS == int(FV.pd.Timestamp(day).value // 10**6)   # 가져오기 상한 = 판정일 00:00
    with pytest.raises(SystemExit):
        FV.main([cand], today=day - dt.timedelta(days=1))


def test_dry_run_past_refuses_forward():
    for a, b in (("2026-10-01", "2026-10-24"), ("2026-10-24", "2026-10-30"), ("2026-10-10", "2026-10-05"), ("2026-12-01", "2026-12-22")):
        with pytest.raises(SystemExit):
            FV.window("vol", D(2027, 6, 1), (a, b))
    s, e, real = FV.window("vol", D(2026, 10, 25), ("2026-09-01", "2026-10-23"))
    assert not real and e.date() == D(2026, 10, 23) and FV.LIMIT_MS == FV.FWD_MS


def test_fetch_limit_blocks_forward_before_any_request():
    FV.window("vol", D(2026, 10, 25), ("2026-09-01", "2026-09-10"))
    with pytest.raises(AssertionError):
        FV.klines("ETHUSDT", "5m", FV.FWD_MS - 300_000, FV.FWD_MS + 300_000)
    with pytest.raises(AssertionError):
        FV.funding("ETHUSDT", FV.FWD_MS - 1, FV.FWD_MS + 1)


def test_md5_mismatch_fails_loudly(tmp_path):
    p = tmp_path / "ckpt.zip"; p.write_bytes(b"frozen")
    assert FV.check_md5(p, FV.md5(p)) == FV.md5(p)
    with pytest.raises(SystemExit, match="md5 불일치"):
        FV.check_md5(p, "0" * 32)


def _vol_local_ready():
    need = [FV.QX4_ART] + [ROOT / f"tmp/system_trend_ruleb_20261025/rest_k5_{s}.parquet" for s in ("ETH", "XRP", "BTC", "SOL")] + \
           [ROOT / f"tmp/vol4h_forecast_20261025/rest_vol_{s}.parquet" for s in ("ETH", "XRP", "BTC", "SOL")] + \
           [Path(f"/home/kbj20/crypto-scalping/data/binance_vision/panel/{s}USDT.parquet") for s in ("ETH", "XRP", "BTC", "SOL")]
    return all(p.exists() for p in need)


@pytest.mark.skipif(not _vol_local_ready(), reason="로컬 캐시·아티팩트 없음")
def test_vol_dry_run_past_local():
    """같은 판정 코드를 과거 창(2026-09-20~10-05)에 — klines 는 로컬 연구 자료에서 만든 행(네트워크 없음)."""
    import live_vol4h_qx4_20261025 as L
    start, end, real = FV.window("vol", D(2026, 10, 25), ("2026-09-20", "2026-10-05"))
    rows = {s: L._rows(L.R.load(s)[0].loc[start - FV.pd.Timedelta(days=75):end + FV.pd.Timedelta("23:55:00")]) for s in L.COINS}
    rep = FV.vol_verdict(start, end, rows=rows)
    assert not real and rep["artifact_md5"] == FV.QX4_MD5 and set(rep["gates"]) == {"V1_ETH_QX4_vs_M3_ci_hi_lt0", "V2_ETH_QX4_vs_M0_ci_hi_lt0", "V3_replication_ge2", "V4_ETH_lvl_slope_ci_has_1"}
    for s, r in rep["coins"].items():
        # 결정 = 09-20 00:00(봉 09-19 23:55) ~ 10-05 20:00(라벨 끝 23:55) → 16·24 − 3 행 · 일 묶음은 연구처럼 봉 시작 날짜(17일)
        assert r["days"] == 17 and 16 * 24 - 8 <= r["n"] <= 16 * 24 - 3, (s, r["n"])
        assert r["dq_QX4_vs_M3"][1] <= r["dq_QX4_vs_M3"][0] <= r["dq_QX4_vs_M3"][2]
