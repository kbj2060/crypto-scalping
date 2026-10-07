"""사전등록 전진 검정: 핑퐁 페이드+물타기의 «크기 ∝ 직전 24h 변동성» (동결 2026-10-07 · 판정 데이터 2026-10-08 00:00 UTC~).

설계 원문: docs/experiments/eth_pingpong_vol_sizing_fwd_prereg_20261007.md (이 스크립트와 같은 커밋 = 동결).
  --status  판정 창 거래 수·일수만(손익 안 봄)        --look  점검일(180·365일)에 도달했을 때만 판정
  --parity  후보 생성기가 연구 후보(candidates.parquet)와 같은지 대조(2025-07~2026-08)        --selftest
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numba
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_eth_pingpong_levers_20261007 import D0, HOLD, NADD, TP0, sequential, sim  # noqa: E402
from research_eth_trend30_60_hold_policy_20261007 import PREV, load_1m, ms, to_5m  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAILY = Path("/home/kbj20/crypto-scalping/data/binance_vision/klines1m_daily")
LOOKS = ROOT / "docs/experiments/eth_pingpong_vol_sizing_fwd_looks.json"
FREEZE = ms("2026-10-08")
MED, REF = 157.3941696083474, 0.7657638410664205        # TRAIN(2022-01~2024-06) 후보 σ 중앙 · 2025-01~2026-10-05 봉 배수 중앙
CLIP = (0.25, 3.0)
ZZ, LEG_MIN, NEAR = 0.005, 40.0, 10.0                    # research_eth_pingpong_extract_20261007.py 와 같은 값
LOOK_DAYS, Z_CRIT = (180, 365), (2.80, 1.98)            # O'Brien-Fleming 2회, 단측 α=.025
SEED, N_BOOT = 20261008, 5000


def size_mult(sig_bp):
    """화면·검정 공용: 직전 24h 실현변동성(4h 척도 bp) → 권장 배수(1배 = 2025년 이후 보통 날)."""
    return np.clip(np.asarray(sig_bp, float) / MED, *CLIP) / REF


def fetch_daily(day: pd.Timestamp) -> pd.DataFrame | None:
    p = DAILY / f"ETHUSDT-1m-{day:%Y-%m-%d}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    url = f"https://data.binance.vision/data/futures/um/daily/klines/ETHUSDT/1m/ETHUSDT-1m-{day:%Y-%m-%d}.zip"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read()
    except Exception:  # noqa: BLE001 -- 아직 공개 안 된 날
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        txt = z.read(z.namelist()[0]).decode()
    df = pd.read_csv(io.StringIO(txt), header=None)
    if not str(df.iat[0, 0]).isdigit():
        df = df.iloc[1:]
    df = pd.DataFrame({"t": df[0].astype("int64"), "h": df[2].astype(float), "l": df[3].astype(float), "c": df[4].astype(float),
                       "v": df[5].astype(float), "tb": df[9].astype(float)})
    DAILY.mkdir(parents=True, exist_ok=True); df.to_parquet(p)
    return df


def load_all() -> pd.DataFrame:
    k = load_1m()
    day = pd.Timestamp(int(k.t.iloc[-1]), unit="ms").normalize()
    extra = []
    while day < pd.Timestamp.now(tz=None).normalize():
        d = fetch_daily(day)
        if d is not None:
            extra.append(d)
        day += pd.Timedelta(days=1)
    if extra:
        k = pd.concat([k[["t", "h", "l", "c", "v", "tb"]]] + extra).drop_duplicates("t", keep="last").sort_values("t")
        full = pd.DataFrame({"t": np.arange(k.t.iloc[0], k.t.iloc[-1] + 1, 60_000)}).merge(k, on="t", how="left")
        full["c"] = full.c.ffill(); full["h"] = full.h.fillna(full.c); full["l"] = full.l.fillna(full.c)
        full[["v", "tb"]] = full[["v", "tb"]].fillna(0.0)
        k = full.reset_index(drop=True)
    return k


@numba.njit(cache=False)
def zz_state(c, th):
    """research_eth_pingpong_extract_20261007.zz_state 와 같은 인과 지그재그(다리 방향·시작 피벗·극값)."""
    n = len(c)
    d_ = np.zeros(n, np.int8); p0 = np.full(n, np.nan); i0 = np.zeros(n, np.int64); ext_ = np.full(n, np.nan)
    d, ext, ext_i, piv, piv_i = 1, c[0], 0, c[0], 0
    for i in range(1, n):
        x = c[i]
        if d > 0:
            if x > ext:
                ext, ext_i = x, i
            elif x <= ext * (1 - th):
                piv, piv_i = ext, ext_i; d = -1; ext, ext_i = x, i
        else:
            if x < ext:
                ext, ext_i = x, i
            elif x >= ext * (1 + th):
                piv, piv_i = ext, ext_i; d = 1; ext, ext_i = x, i
        d_[i] = d; p0[i] = piv; i0[i] = piv_i; ext_[i] = ext
    return d_, p0, i0, ext_


def candidates(k: pd.DataFrame) -> pd.DataFrame:
    """5분 마감 분 · 현재 다리 ≥40bp · 극값 10bp 안 · 같은 다리 첫 건. t = 그 1분봉 시작(ms), d = 다리 방향(페이드 = −d)."""
    c = k.c.to_numpy()
    d, p0, i0, ext = zz_state(c, ZZ)
    leg = d * (c / p0 - 1) * 1e4; fe = d * (c / ext - 1) * 1e4
    is5 = (k.t.to_numpy() + 60_000) % 300_000 == 0
    pos = np.flatnonzero(is5 & (leg >= LEG_MIN) & (fe >= -NEAR) & (np.arange(len(k)) < len(k) - HOLD - 3))
    C = pd.DataFrame({"t": k.t.to_numpy()[pos], "d": d[pos].astype(int), "i0": i0[pos]})
    return C[~C.duplicated("i0")].reset_index(drop=True)


def trades(k: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    """V0 순차 거래 + 진입 시 배수. 손익 = 단위-bp(비용 전)."""
    b = to_5m(k); t5 = b.t.to_numpy()
    r2 = pd.Series((np.diff(np.log(b.c.to_numpy()), prepend=np.nan) * 1e4) ** 2)
    sig = np.sqrt(r2.rolling(288).sum().to_numpy() / 6)
    t1 = k.t.to_numpy()
    tdec = C.t.to_numpy("int64") + 60_000; ei = np.searchsorted(t1, tdec)
    bi = np.searchsorted(t5, tdec // 300_000 * 300_000 - 300_000)
    ok = (ei + HOLD < len(t1)) & (bi < len(t5)) & np.isfinite(sig[np.minimum(bi, len(t5) - 1)])
    C, ei, bi, tdec = C[ok].reset_index(drop=True), ei[ok], bi[ok], tdec[ok]
    n = len(C)
    pnl, q, xi, mae, kind = sim(k.h.to_numpy(), k.l.to_numpy(), k.c.to_numpy(), ei, -C.d.to_numpy(np.int64),
                                np.full(n, D0), np.full(n, TP0), NADD, HOLD)
    sel = sequential(ei, xi, np.ones(n, bool))
    return pd.DataFrame({"t": tdec, "day": tdec // 86_400_000, "pnl": pnl, "q": q, "mult": size_mult(sig[bi])})[sel]


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(365)


def verdict(T: pd.DataFrame, look: int) -> dict:
    rng = np.random.default_rng(SEED)
    w = T.mult / T.mult.mean()                           # 판정 창 평균 1(같은 평균 노출)
    a = (T.pnl * w).groupby(T.day).sum(); e = T.pnl.groupby(T.day).sum()
    D = pd.concat([a, e], axis=1).fillna(0.0).to_numpy()
    idx = rng.integers(0, len(D), (N_BOOT, len(D))); S = D[idx]
    bs = (S[:, :, 0].mean(1) / S[:, :, 0].std(1) - S[:, :, 1].mean(1) / S[:, :, 1].std(1)) * np.sqrt(365)
    diff = sharpe(D[:, 0]) - sharpe(D[:, 1]); z = diff / bs.std()
    lo3 = T.mult <= T.mult.quantile(1 / 3); hi3 = T.mult >= T.mult.quantile(2 / 3)
    sec1 = float(T.pnl[hi3].mean() - T.pnl[lo3].mean())
    eq_a, eq_e = np.cumsum(T.pnl * w), np.cumsum(T.pnl)
    mdd = lambda q: float((q - np.maximum.accumulate(q)).min())  # noqa: E731
    pas = bool(z >= Z_CRIT[look] and sec1 > 0)
    return dict(look=look + 1, days=int(T.day.nunique()), n=int(len(T)), sharpe_weighted=float(sharpe(D[:, 0])),
                sharpe_equal=float(sharpe(D[:, 1])), diff=float(diff), z=float(z), z_crit=Z_CRIT[look],
                sec1_hi_minus_lo_bp=sec1, mdd_weighted=mdd(eq_a), mdd_equal=mdd(eq_e),
                result="확정" if pas else ("기각" if look == len(LOOK_DAYS) - 1 or z < 0 else "계속"))


def window() -> pd.DataFrame:
    k = load_all()
    T = trades(k, candidates(k[k.t >= FREEZE - 30 * 86_400_000].reset_index(drop=True)))   # 30일 워밍업(지그재그·24h σ)
    return T[T.t >= FREEZE].reset_index(drop=True)


def main() -> None:
    T = window()
    days = (int(T.day.max()) - FREEZE // 86_400_000 + 1) if len(T) else 0
    if "--status" in sys.argv:
        nxt = [d for d in LOOK_DAYS if d > days]
        print(f"판정 창 거래 {len(T)} · 경과 {days}일 · 다음 점검 {nxt[0] if nxt else '없음'}일")
        return
    looks = json.loads(LOOKS.read_text()) if LOOKS.exists() else []
    if len(looks) >= len(LOOK_DAYS) or any(r["result"] in ("확정", "기각") for r in looks):
        print("판정 완료:", looks[-1]); return
    li = len(looks)
    if days < LOOK_DAYS[li]:
        print(f"점검 {li + 1} 은 {LOOK_DAYS[li]}일째에만 — 지금 {days}일. 결과를 보지 않는다."); return
    T = T[T.day < FREEZE // 86_400_000 + LOOK_DAYS[li]]
    r = verdict(T, li); looks.append(r)
    LOOKS.write_text(json.dumps(looks, ensure_ascii=False, indent=1))
    print(json.dumps(r, ensure_ascii=False, indent=1))


def parity() -> None:
    k = load_1m()
    k = k[(k.t >= ms("2025-06-01")) & (k.t < ms("2026-08-25"))].reset_index(drop=True)
    mine = candidates(k)
    ref = pd.read_parquet(PREV / "tmp/pingpong_extract_20261007/candidates.parquet")
    lo, hi = ms("2025-07-01"), ms("2026-08-20")
    a = set(mine.t[(mine.t >= lo) & (mine.t < hi)]); r = set(ref.t[(ref.t >= lo) & (ref.t < hi)])
    print(f"후보 일치: 내 {len(a)} · 연구 {len(r)} · 교집합 {len(a & r)} ({len(a & r) / max(len(r), 1):.4f})")
    assert len(a & r) / len(r) > 0.99 and len(a - r) / len(a) < 0.01


def selftest() -> None:
    assert np.allclose(size_mult([MED * REF, 0.0, 1e9]), [1.0, 0.25 / REF, 3.0 / REF])
    c = np.array([100, 100.6, 100.2, 99.9, 99.5, 100.1])           # 0.5% 반전마다 다리 전환
    d, p0, i0, ext = zz_state(c, 0.005)
    assert d.tolist() == [0, 1, 1, -1, -1, 1] and p0[3] == 100.6 and p0[5] == 99.5, (d, p0)
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else parity() if "--parity" in sys.argv else main()
