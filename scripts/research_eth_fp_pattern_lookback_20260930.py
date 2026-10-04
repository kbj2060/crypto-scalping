"""풋프린트 패턴 표식(다이버·소진·흡수?)의 봉 수 × 세션 검정 (2026-09-30, 사용자 «봉 갯수 테스트 · 판정과 잘 맞는 봉 수 · 세션을 나눠서도»).

화면(dashboard/live/app.js::fpPatternMarks)은 «보이는 창» 안에서 판정해 1h 창과 2h 창의 표식이 달랐다. 그래서 판정을 창과 떼어
**판정 봉 수 L**(새 고점을 몇 봉과 비교하나)과 **문턱 기준 봉 수 C**(상위 20%·10% 를 몇 봉으로 재나, 뒤로만)를 고정값으로 두고,
어느 값에서 «표식이 말하는 방향»이 가장 잘 맞는지 잰다.

사전 고정(결과 보기 전):
  데이터   ETHUSDT 선물 1분봉 2022-01~2026-09 → 5분봉(고·저·종·거래량·테이커매수). 델타 = 2×테이커매수 − 거래량.
           흡수의 «극값 줄 공격 체결» = 5분 고가(저가)를 **처음 찍은 1분봉**의 테이커 매수(매도)량 -- 화면(가격 줄 칸)의 근사.
  정의     화면과 같다. 새 고가 = 고가 > 직전 L봉 최고가. 다이버↓ = 그 최고가 봉 이후 델타 합 < 0. 소진↓ = 델타 > 0 · 델타 ≥ C봉 |델타| 80분위 ·
           종가가 봉 아래 40% 안. 흡수?↓ = 극값 1분 테이커 매수 ≥ C봉 90분위 · 다음 봉 고가 ≤ 이 고가. 저가 쪽은 거울.
  방향     위 표식 = 하락, 아래 표식 = 상승(교과서 해석). 흡수도 교과서(버팀 = 되돌림)로 부호를 준다 -- 09-29 1초 검정은 반대였다.
  라벨     판정이 알려지는 봉의 종가 → H봉 뒤 종가(bp, 방향 부호). 다이버·소진은 그 봉, 흡수는 다음 봉(다음 봉이 닫혀야 안다).
  대조군   같은 쪽 «새 극값» 봉 중 그 표식이 없는 봉(흡수는 «버텼는데 큰 체결이 없는» 극값). 초과 = 표식 − 대조군. 새 극값 자체의 방향을 뺀다.
  격자     L ∈ {3,4,6,8,12,18,24} · C ∈ {24,48,144,288}(다이버는 C 무관) · H ∈ {3,6,12}봉(15·30·60분).
  분할     TRAIN 2022-01~2024-12 · TEST 2025-01~. 선택 = TRAIN, H=6(30분), 표식별 초과 평균 최대(n ≥ 300). TEST 는 그 한 칸으로 판정.
  세션     UTC 기준 대시보드 LED 와 같은 경계, 겹치지 않게: 아시아 00–08 · 유럽 08–14:30 · 미국 14:30–21 · 그 밖 21–24.
  CI       일 단위 포아송 블록 부트스트랩 B=1000(95%).
  🔴화면 기준은 정보성(IC)이다 -- 비용 판정이 아니다. 격자가 크니 TRAIN 에서 고른 한 칸의 TEST 만 «판정»으로 읽는다.

실행: python scripts/research_eth_fp_pattern_lookback_20260930.py [--selftest] [--data DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

LS = (3, 4, 6, 8, 12, 18, 24)
CS = (24, 48, 144, 288)
HS = (3, 6, 12)
SPLIT = pd.Timestamp("2025-01-01")
START = pd.Timestamp("2000-01-01")    # --start: 그 전 봉은 분위 창 예열에만 쓰고 집계에서 뺀다(09-30 «24년은 거래량 부족» → 2025/2026 판)
SESSIONS = (("아시아", 0.0, 8.0), ("유럽", 8.0, 14.5), ("미국", 14.5, 21.0), ("그밖", 21.0, 24.0))
KINDS = ("div", "exh", "abs")


def default_data_dir() -> Path:
    here = Path(__file__).resolve().parents[1] / "data/binance_vision/klines1m"
    if any(here.glob("ETHUSDT-1m-*.parquet")):
        return here
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent).stdout.strip()
    return Path(common).resolve().parent / "data/binance_vision/klines1m"   # 워크트리면 메인 체크아웃의 데이터


def load_5m(data_dir: Path) -> pd.DataFrame:
    fs = sorted(glob.glob(str(data_dir / "ETHUSDT-1m-*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    d["b"] = d["t"] // 300_000 * 300
    g = d.groupby("b", sort=True)
    out = pd.DataFrame({"high": g["h"].max(), "low": g["l"].min(), "close": g["c"].last(),
                        "vol": g["v"].sum(), "tb": g["tb"].sum()})
    out["top_buy"] = d.loc[g["h"].idxmax(), "tb"].to_numpy()                 # 고가를 처음 찍은 1분의 테이커 매수
    out["bot_sell"] = (d.loc[g["l"].idxmin(), "v"] - d.loc[g["l"].idxmin(), "tb"]).to_numpy()
    full = np.arange(out.index.min(), out.index.max() + 300, 300)             # 빈 봉은 NaN -- 창이 구멍을 건너뛰지 않게
    out = out.reindex(full)
    out["delta"] = 2 * out["tb"] - out["vol"]
    out.index = pd.to_datetime(out.index, unit="s")
    return out


def marks(df: pd.DataFrame, L: int, C: int) -> dict[str, np.ndarray]:
    """봉별 표식 부호 배열(+1 아래 표식 = 상승 예상, −1 위 표식, 0 없음)과 대조군 마스크. 인과적: 봉 t 는 t 까지(흡수는 t+1 까지)."""
    h, lo, c, dl = (df[k].to_numpy(float) for k in ("high", "low", "close", "delta"))
    n = len(h)
    cs = np.nancumsum(np.nan_to_num(dl))
    win = np.lib.stride_tricks.sliding_window_view
    idx = np.arange(L, n)
    Wh, Wl = win(h, L)[:-1], win(lo, L)[:-1]                                   # Wh[k] = high[k:k+L] → 봉 k+L 의 직전 L봉
    ok = np.isfinite(Wh).all(1) & np.isfinite(Wl).all(1) & np.isfinite(h[idx]) & np.isfinite(lo[idx]) & np.isfinite(c[idx])
    jh = idx - L + np.argmax(np.where(np.isfinite(Wh), Wh, -np.inf), 1)       # 첫 최댓값(화면과 같은 «엄격한 >» 갱신)
    jl = idx - L + np.argmin(np.where(np.isfinite(Wl), Wl, np.inf), 1)
    nh = ok & (h[idx] > h[jh])
    nl = ok & (lo[idx] < lo[jl])
    rng = h[idx] - lo[idx]
    pos = np.where(rng > 0, (c[idx] - lo[idx]) / np.where(rng > 0, rng, 1), 0.5)
    q = lambda s, p: s.rolling(C, min_periods=C).quantile(p, interpolation="lower").to_numpy()   # noqa: E731
    bigD = q(df["delta"].abs(), 0.8)[idx]
    tbQ, bsQ = q(df["top_buy"], 0.9)[idx], q(df["bot_sell"], 0.9)[idx]
    d_i = dl[idx]
    nxt_h = np.append(h[idx[:-1] + 1], np.nan) if len(idx) else h[idx]
    nxt_l = np.append(lo[idx[:-1] + 1], np.nan) if len(idx) else lo[idx]
    div_t = nh & (cs[idx] < cs[jh]); div_b = nl & (cs[idx] > cs[jl])
    exh_t = nh & (d_i > 0) & (d_i >= bigD) & (pos <= 0.4); exh_b = nl & (d_i < 0) & (-d_i >= bigD) & (pos >= 0.6)
    held_t = nh & (nxt_h <= h[idx]); held_b = nl & (nxt_l >= lo[idx])
    tb_i, bs_i = df["top_buy"].to_numpy(float)[idx], df["bot_sell"].to_numpy(float)[idx]
    abs_t = held_t & (tb_i >= tbQ); abs_b = held_b & (bs_i >= bsQ)
    def pack(t, b, ct, cb):
        s = np.zeros(n, np.int8); ctl = np.zeros(n, np.int8)
        s[idx[t]] = -1; s[idx[b & ~t]] = 1                                     # 같은 봉에 둘 다면 위를 우선(드묾)
        ctl[idx[ct]] = -1; ctl[idx[cb & ~ct]] = 1
        return s, ctl
    return {"div": pack(div_t, div_b, nh & ~div_t, nl & ~div_b),
            "exh": pack(exh_t, exh_b, nh & ~exh_t, nl & ~exh_b),
            "abs": pack(abs_t, abs_b, held_t & ~abs_t & np.isfinite(tbQ), held_b & ~abs_b & np.isfinite(bsQ))}


def fwd_bp(close: np.ndarray, H: int, lag: int) -> np.ndarray:
    """봉 t 의 판정이 알려지는 종가(t+lag) → H봉 뒤 종가, bp. 끝에 모자라면 NaN."""
    n = len(close)
    out = np.full(n, np.nan)
    s = np.arange(n - lag - H)
    out[s] = (close[s + lag + H] / close[s + lag] - 1) * 1e4
    return out


def block_ci(val: np.ndarray, grp: np.ndarray, day: np.ndarray, B: int = 1000, seed: int = 7) -> tuple[float, float, float]:
    """표식 − 대조군 평균 차의 일 블록 포아송 부트스트랩(95%). grp: 1 표식 · 0 대조군."""
    days, inv = np.unique(day, return_inverse=True)
    D = len(days)
    s1 = np.bincount(inv, weights=val * (grp == 1), minlength=D); n1 = np.bincount(inv, weights=(grp == 1), minlength=D)
    s0 = np.bincount(inv, weights=val * (grp == 0), minlength=D); n0 = np.bincount(inv, weights=(grp == 0), minlength=D)
    point = s1.sum() / max(n1.sum(), 1) - s0.sum() / max(n0.sum(), 1)
    w = np.random.default_rng(seed).poisson(1.0, (B, D))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (w @ s1) / (w @ n1) - (w @ s0) / (w @ n0)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def cell(df: pd.DataFrame, m: dict, kind: str, H: int, sel: np.ndarray) -> dict:
    s, ctl = m[kind]
    ret = fwd_bp(df["close"].to_numpy(float), H, 1 if kind == "abs" else 0)
    ev = sel & (s != 0) & np.isfinite(ret)
    cv = sel & (ctl != 0) & np.isfinite(ret)
    v1, v0 = s[ev] * ret[ev], ctl[cv] * ret[cv]                                 # 표식 방향 부호 수익 · 대조군은 같은 쪽 방향으로
    if len(v1) < 30 or len(v0) < 30:
        return {"n": int(len(v1)), "hit": None, "mean": None, "ctl": None, "ex": None, "lo": None, "hi": None}
    day = df.index.to_numpy()[ev | cv].astype("datetime64[D]")
    allv = np.where(ev, s * ret, ctl * ret)[ev | cv]
    grp = ev[ev | cv].astype(int)
    ex, lo, hi = block_ci(allv, grp, day)
    return {"n": int(len(v1)), "hit": float((v1 > 0).mean()), "mean": float(v1.mean()), "ctl": float(v0.mean()),
            "ex": ex, "lo": lo, "hi": hi}


def run(data_dir: Path, out_dir: Path) -> None:
    df = load_5m(data_dir)
    t = df.index
    train, test = np.asarray((t >= START) & (t < SPLIT)), np.asarray(t >= SPLIT)
    hr = (t.hour + t.minute / 60.0).to_numpy()
    sess = {name: (hr >= a) & (hr < b) for name, a, b in SESSIONS}
    print(f"5분봉 {len(df):,} ({t[0]} ~ {t[-1]}) · 결측 {int(df['close'].isna().sum()):,}")
    rows = []
    for L in LS:
        for C in CS:
            m = marks(df, L, C)
            for kind in KINDS:
                if kind == "div" and C != CS[0]:
                    continue                                                    # 다이버는 C 무관 -- 한 번만
                for H in HS:
                    for split, sel in (("TRAIN", train), ("TEST", test)):
                        r = cell(df, m, kind, H, sel)
                        rows.append({"kind": kind, "L": L, "C": C if kind != "div" else None, "H": H, "split": split, "session": "전체", **r})
                        if H == 6:
                            for name, ss in sess.items():
                                r2 = cell(df, m, kind, H, sel & ss)
                                rows.append({"kind": kind, "L": L, "C": C if kind != "div" else None, "H": H, "split": split, "session": name, **r2})
        print(f"L={L} 끝")
    res = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "grid.csv", index=False)
    picks = {}
    for kind in KINDS:
        tr = res[(res.kind == kind) & (res.split == "TRAIN") & (res.H == 6) & (res.session == "전체") & (res.n >= 300)].dropna(subset=["ex"])
        best = tr.sort_values("ex", ascending=False).iloc[0]
        te = res[(res.kind == kind) & (res.split == "TEST") & (res.H == 6) & (res.session == "전체") & (res.L == best.L)
                 & ((res.C == best.C) if kind != "div" else res.C.isna())].iloc[0]
        picks[kind] = {"L": int(best.L), "C": None if kind == "div" else int(best.C),
                       "train": {k: best[k] for k in ("n", "hit", "mean", "ctl", "ex", "lo", "hi")},
                       "test": {k: te[k] for k in ("n", "hit", "mean", "ctl", "ex", "lo", "hi")}}
    (out_dir / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(picks, ensure_ascii=False, indent=1, default=float))


def selftest() -> None:
    idx = pd.date_range("2025-01-01", periods=20, freq="5min")
    base = dict(high=101.0, low=99.0, close=100.0, vol=10.0, tb=5.5, top_buy=1.0, bot_sell=1.0)
    df = pd.DataFrame([base] * 20, index=idx)
    df["delta"] = 2 * df["tb"] - df["vol"]
    df.iloc[12, df.columns.get_loc("high")] = 103.0                      # 새 고가
    df.iloc[12, df.columns.get_loc("delta")] = -50.0                     # 누적 CVD 가 직전 고점 때보다 낮다 → 다이버↓
    m = marks(df, 6, 12)
    assert m["div"][0][12] == -1 and (m["div"][0][:12] == 0).all(), m["div"][0]
    assert m["div"][1][12] == 0, "표식 봉은 대조군이 아니다"
    df2 = df.copy(); df2.iloc[12, df2.columns.get_loc("delta")] = 90.0; df2.iloc[12, df2.columns.get_loc("close")] = 99.5
    m2 = marks(df2, 6, 12)
    assert m2["exh"][0][12] == -1 and m2["div"][0][12] == 0, (m2["exh"][0][12], m2["div"][0][12])   # 소진↓, CVD 는 올랐으니 다이버 아님
    df3 = df.copy(); df3.iloc[12, df3.columns.get_loc("delta")] = 1.0; df3.iloc[12, df3.columns.get_loc("top_buy")] = 50.0
    m3 = marks(df3, 6, 12)
    assert m3["abs"][0][12] == -1, "새 고가 + 큰 극값 매수 + 다음 봉이 못 넘음 = 흡수?"
    df3.iloc[13, df3.columns.get_loc("high")] = 104.0
    assert marks(df3, 6, 12)["abs"][0][12] == 0, "다음 봉이 넘으면 흡수 아님"
    c = np.arange(10, dtype=float) + 100
    r0, r1 = fwd_bp(c, 2, 0), fwd_bp(c, 2, 1)
    assert abs(r0[0] - (102 / 100 - 1) * 1e4) < 1e-9 and abs(r1[0] - (103 / 101 - 1) * 1e4) < 1e-9, "흡수 라벨은 한 봉 늦게 시작"
    assert np.isnan(r0[-1]) and np.isnan(r1[-3])
    v = np.array([1.0, 1.0, 0.0, 0.0]); g = np.array([1, 1, 0, 0]); dd = np.array(["2025-01-01", "2025-01-02"] * 2, dtype="datetime64[D]")
    p, lo, hi = block_ci(v, g, dd)
    assert p == 1.0 and lo <= 1.0 <= hi
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("tmp/fp_pattern_lookback_20260930"))
    ap.add_argument("--start", default=None)
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    START = pd.Timestamp(a.start) if a.start else START
    SPLIT = pd.Timestamp(a.split) if a.split else SPLIT
    if a.selftest:
        selftest()
    else:
        run(a.data or default_data_dir(), a.out)
