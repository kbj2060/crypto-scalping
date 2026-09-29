#!/usr/bin/env python3
"""옵션 판정 전부를 **2026년 데이터만으로** 다시 돌린다 (2026-09-29, 사용자 지시).

사용자: «옵션 데이터 검증할 때는 26년만 검증해줘. 이전 옵션 데이터는 무의미해.» → «모두 재검증해줘».
옛 판정 중 2025년 이전 데이터에 기댄 것 네 개와, 지난 답에서 «쓸모 있다»고 한 예상 폭 하나를 다시 잰다.
검정 코드는 옛 스크립트를 **그대로** 부르고 입력만 2026년으로 자른다(식을 두 벌 두지 않는다).

사전등록 (실행 전 고정)
- 창: 옵션 입력(DVOL·체인)은 2026-01-01 이후만. 5분봉은 2026-01-01 ~ 2026-09-27 23:55 UTC
  (data.binance.vision 일봉 파일이 09-27 까지 있다. 로컬 REST 는 IP 밴 위험이라 안 쓴다).
- 분할(방향·변동성 검정): TRAIN 01-01~05-31 · OOS 06-01~07-31 · HOLDOUT 08-01~09-27.
- A 핀닝(격자 f): research_eth_option_pinning_lowvol_20260916 그대로. 판정 = 원 스크립트 기준.
- B 행사가 자석 + max pain: research_eth_gex_strike_magnet_20260920 그대로(체인 08-15~09-27),
  통제③ 표본외는 2026-01-01~08-14 로 교체. max pain 은 같은 틀(만기 1h 전 진입→만기, 방향맞춘 수익,
  가까워짐 비율, 날 군집 CI)을 새로 더한다.
- C VRP 방향: research_eth_dvol_vrp_signal_20260910 그대로. 판정 = 세 창 초과분 > 0 이고 OOS·HOLDOUT > 10bp.
- D DVOL 변동성 전망: research_eth_dvol_volexpansion_20260910 그대로. 판정 = 세 창 증분 부호 일치 + DM CI.
- E 예상 폭 보정: DVOL 로 만든 24h 1σ 띠 안에 실제 24h 수익이 들어온 비율(정규면 68.3%)과
  실제/예상 표준편차 비. 대조 = 과거 24h 실현변동성(RV24)으로 만든 띠.
출력: tmp/opt2026_20260929/
"""
from __future__ import annotations
import contextlib, glob, io, os, runpy, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/home/kbj20/crypto-scalping")
W = ROOT / "tmp/opt2026_20260929"
T0, T_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-09-27 23:55")
TRAIN_END, OOS_END = pd.Timestamp("2026-05-31 23:59"), pd.Timestamp("2026-07-31 23:59")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def say(*a) -> None:
    print(*a, flush=True)


# ── 입력: 2026년만 ──────────────────────────────────────────────────────────
def load_px5() -> pd.DataFrame:
    """로컬 5분봉(~09-15) + data.binance.vision 일봉 파일(09-15~09-27). 2026 만."""
    h = pd.read_csv(REPO / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
                    usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"],
                    parse_dates=["timestamp"])
    a = pd.concat([pd.read_csv(f, usecols=["open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"])
                   for f in sorted(glob.glob(str(W / "kl/ETHUSDT-5m-*.csv")))])
    a = a.rename(columns={"taker_buy_volume": "taker_buy_base"})   # 옛 로더(핀닝)가 이 두 열을 요구한다
    a["timestamp"] = pd.to_datetime(a.pop("open_time"), unit="ms")
    d = pd.concat([h, a]).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    d = d[(d.timestamp >= T0) & (d.timestamp <= T_END)].reset_index(drop=True)
    gap = d.timestamp.diff().dt.total_seconds().div(300).gt(1).sum()
    assert gap < 20, f"5분봉 구멍 {gap}"
    return d


def load_dvol() -> pd.DataFrame:
    """Deribit 공개 API 시간봉 DVOL, 2026-01-01 부터. 최신 1000개씩 거꾸로 받는다(continuation)."""
    out, end = [], int(pd.Timestamp.utcnow().timestamp() * 1000)
    start = int(T0.tz_localize("UTC").timestamp() * 1000)
    while end > start:
        r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                         params={"currency": "ETH", "start_timestamp": start, "end_timestamp": end,
                                 "resolution": "3600"}, timeout=30).json()["result"]
        out += r["data"]
        if not r.get("continuation") or not r["data"]:
            break
        end = int(r["continuation"])
    d = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts")
    d["timestamp"] = pd.to_datetime(d.ts, unit="ms")
    d = d[d.timestamp >= T0].sort_values("timestamp").reset_index(drop=True)
    assert d.timestamp.diff().dt.total_seconds().max() <= 3 * 3600, "DVOL 구멍"
    return d[["timestamp", "open", "high", "low", "close"]]


def run_quiet(fn, log: Path):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    log.write_text(buf.getvalue(), encoding="utf-8")
    return buf.getvalue()


# ── A 핀닝 ──────────────────────────────────────────────────────────────────
def run_pinning(px: pd.DataFrame, dv: pd.DataFrame) -> str:
    """옛 스크립트는 REPO 아래 고정 경로를 읽는다 -> 같은 상대경로를 가진 가짜 REPO 를 만든다."""
    fake = W / "fake_repo"
    (fake / "binance_data/klines/ETHUSDT").mkdir(parents=True, exist_ok=True)
    (fake / "data/derivatives/deribit_dvol").mkdir(parents=True, exist_ok=True)
    px.to_csv(fake / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv", index=False)
    dv.to_csv(fake / "data/derivatives/deribit_dvol/ETH_dvol_hourly.csv", index=False)
    import research_eth_option_pinning_lowvol_20260916 as P
    P.REPO, P.OUT = fake, W / "A_pinning"
    P.OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = ["x"]
    return run_quiet(P.main, W / "A_pinning.log")


# ── B 자석 + max pain ───────────────────────────────────────────────────────
def max_pain(g: pd.DataFrame) -> float:
    """보유자 지급 총액이 최소가 되는 행사가(스트라이크 후보 안에서)."""
    k, c, p = g.strike.to_numpy(float), g.oi_call.to_numpy(float), g.oi_put.to_numpy(float)
    pay = [(c * np.clip(K - k, 0, None) + p * np.clip(k - K, 0, None)).sum() for K in k]
    return float(k[int(np.argmin(pay))])


def run_magnet(px: pd.DataFrame) -> str:
    root = W / "magnet_root"
    tmp = root / "tmp/rt_probe_20260920"
    (root / "scripts").mkdir(parents=True, exist_ok=True); tmp.mkdir(parents=True, exist_ok=True)
    src = ROOT / "scripts/research_eth_gex_strike_magnet_20260920.py"
    shutil.copy(src, root / "scripts" / src.name)       # ROOT=parents[1] 이 root 가 되게
    c = pd.read_parquet(W / "chain_near_deribit_options.parquet")
    c = c[c.recorded_at_utc <= T_END.tz_localize("UTC")]
    c.to_parquet(tmp / "chain_near.parquet")
    k = px.copy(); k["open_time"] = k.timestamp.astype("int64") // 10**6
    k[k.timestamp >= pd.Timestamp("2026-08-10")].to_parquet(tmp / "klines_5m.parquet")
    k[k.timestamp < pd.Timestamp("2026-08-15")].to_parquet(tmp / "klines_5m_hist.parquet")
    txt = run_quiet(lambda: runpy.run_path(str(root / "scripts" / src.name), run_name="__main__"),
                    W / "B_magnet.log")
    return txt + "\n" + run_maxpain(c, px)


def run_maxpain(c: pd.DataFrame, px: pd.DataFrame) -> str:
    rng = np.random.default_rng(20260929)
    ks = (px.timestamp.astype("int64") // 10**9).to_numpy(); kc = px.close.to_numpy(float)

    def px_at(sec):  # 그 시각 전에 닫힌 5분봉 종가
        i = np.searchsorted(ks + 300, sec, side="right") - 1
        return float(kc[i]) if 0 <= i < len(kc) else np.nan
    c = c.copy()
    c["sec"] = (c.recorded_at_utc - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    c["exp_sec"] = (c.expiration_ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    rows = []
    for (s, e), g in c.groupby(["sec", "exp_sec"]):
        h = (e - s) / 3600
        if not (0.4 <= h <= 1.6) or len(g) < 3:
            continue
        s0, s1 = px_at(s), px_at(e)
        if not (np.isfinite(s0) and np.isfinite(s1)):
            continue
        kp = max_pain(g)
        kpl = float(g.strike.to_numpy()[rng.integers(len(g))])            # 플라시보: 같은 체인의 아무 행사가
        kgrid = float(g.strike.to_numpy()[np.abs(g.strike.to_numpy() - s0).argmin()])
        rows.append(dict(exp=e, h=h, s0=s0, s1=s1, kp=kp, kpl=kpl, kgrid=kgrid,
                         day=str(pd.to_datetime(e, unit="s").date())))
    E = pd.DataFrame(rows).sort_values("h").groupby("exp", as_index=False).first()
    out = [f"\n## max pain — 만기 1h 전 진입 → 만기 정산(ETHUSDT 08:00 종가), 만기 {len(E)}개 · 고유일 {E.day.nunique()}"]
    for key, lab in (("kp", "max pain"), ("kgrid", "순수격자 최근접"), ("kpl", "└플라시보")):
        d0 = (E[key] - E.s0) / E.s0 * 1e4; d1 = (E[key] - E.s1) / E.s1 * 1e4
        sig = (E.s1 - E.s0) / E.s0 * 1e4 * np.sign(d0)
        m = sig[d0 != 0].to_numpy(); days = E.day[d0 != 0].to_numpy()
        uniq = np.unique(days)
        boot = [np.mean(np.concatenate([m[days == x] for x in rng.choice(uniq, len(uniq))])) for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        half = len(m) // 2
        out.append(f"  {lab:14} 진입거리 중앙 {d0.abs().median():5.0f}bp · 정산거리 {d1.abs().median():5.0f}bp · "
                   f"가까워짐 {(d1.abs() < d0.abs()).mean():.1%} · 방향맞춘 수익 {m.mean():+6.2f}bp "
                   f"[{lo:+.1f},{hi:+.1f}] · 적중 {(m > 0).mean():.1%} · 전반 {m[:half].mean():+.1f}/후반 {m[half:].mean():+.1f} · n={len(m)}")
    out.append("  판정 = CI 0 배제 AND 플라시보·순수격자와 갈림 AND 전·후반 부호 일치 AND 순(−1.4bp) > 0")

    # 🔴통과한 뒤에 붙인 통제(사후) -- «max pain 쪽» 이 «최근에 있던 가격 쪽»(평균회귀)과 같은 말인지.
    #   max pain 은 OI 가 쌓인 곳이고 OI 는 최근 가격대에 쌓이므로, 옵션 정보 없는 «최근 평균가» 앵커가
    #   같은 수익을 내면 옵션 정보가 아니다.
    E["prev1h"] = [(s0 - px_at(e - 7200)) / px_at(e - 7200) * 1e4 for s0, e in zip(E.s0, E.exp)]
    E["prev24"] = [(s0 - px_at(e - 3600 * 25)) / px_at(e - 3600 * 25) * 1e4 for s0, e in zip(E.s0, E.exp)]
    E["vwap24"] = [np.mean(kc[max(0, np.searchsorted(ks, e - 3600 * 25)):np.searchsorted(ks, e - 3600)]) for e in E.exp]
    sp = np.sign(E.kp - E.s0)
    sig = (E.s1 - E.s0) / E.s0 * 1e4 * sp
    out.append("\n  [사후 통제] max pain 방향이 평균회귀의 다른 이름인가")
    out.append(f"    corr(sign(d), 직전 1h 수익) {np.corrcoef(sp, E.prev1h)[0, 1]:+.3f} · "
               f"corr(sign(d), 직전 24h 수익) {np.corrcoef(sp, E.prev24)[0, 1]:+.3f}")
    sv = np.sign(E.vwap24 - E.s0)
    sigv = (E.s1 - E.s0) / E.s0 * 1e4 * sv
    agree = (sv == sp).mean()
    out.append(f"    옵션 없는 앵커 «24h 평균가 쪽» 방향맞춘 수익 {sigv.mean():+.2f}bp · max pain 과 방향 일치 {agree:.0%}")
    out.append(f"    둘이 갈린 만기 {int((sv != sp).sum())}개: max pain 쪽 {sig[sv != sp].mean():+.2f}bp / "
               f"24h 평균가 쪽 {sigv[sv != sp].mean():+.2f}bp")
    q = pd.qcut(E.prev24.rank(method="first"), 4, labels=False)
    out.append("    직전 24h 수익 4분위별 max pain 방향맞춘 수익 " +
               str([round(float(sig[q == i].mean()), 1) for i in range(4)]))
    fri = pd.to_datetime(E.exp, unit="s").dt.dayofweek.to_numpy() == 4
    out.append(f"    금요일 만기 {sig[fri].mean():+.2f}bp (n={fri.sum()}) · 평일 {sig[~fri].mean():+.2f}bp (n={(~fri).sum()})")
    return "\n".join(out)


# ── C VRP 방향 · D DVOL 변동성 전망 ─────────────────────────────────────────
def patch_dvol_modules(px: pd.DataFrame, dv: pd.DataFrame):
    import research_eth_dvol_vrp_signal_20260910 as S
    import research_eth_dvol_volexpansion_20260910 as V
    csv = W / "dvol_2026.csv"; dv.to_csv(csv, index=False)
    S.DVOL_CSV = csv
    S.fetch_dvol_tail = lambda t0: pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"),
                                                 "close": pd.Series(dtype=float)})
    S.load_px5 = lambda: px[["timestamp", "open", "high", "low", "close"]].copy()
    S.TRAIN_END = V.TRAIN_END = TRAIN_END
    S.OOS_END = V.OOS_END = OOS_END
    S.OUT = V.OUT = W / "CD_dvol"
    return S, V


def run_vrp(S) -> str:
    sys.argv = ["x", "--holdout"]
    return run_quiet(S.main, W / "C_vrp.log")


def run_volfc(V) -> str:
    sys.argv = ["x", "--holdout"]
    return run_quiet(V.main, W / "D_volfc.log")


# ── E 예상 폭 보정 ──────────────────────────────────────────────────────────
def band_hits(r24: np.ndarray, sig: np.ndarray) -> tuple[float, float]:
    """(1σ 띠 안 비율, 실제 std / 예상 σ 의 RMS 비). 정규이고 보정됐으면 (0.683, 1.0)."""
    ok = np.isfinite(r24) & np.isfinite(sig) & (sig > 0)
    z = r24[ok] / sig[ok]
    return float((np.abs(z) <= 1).mean()), float(np.sqrt(np.mean(z ** 2)))


def run_band(px: pd.DataFrame, dv: pd.DataFrame) -> str:
    p = px.set_index("timestamp")["close"]
    r5 = np.log(p).diff()
    h = pd.DataFrame({"close": p.resample("1h").last()})
    h["rv24"] = (r5.rolling(288, min_periods=288).std() * np.sqrt(288 * 365)).resample("1h").last()
    h = h.join(dv.set_index("timestamp")["close"].rename("dvol") / 100, how="inner").dropna()
    h["r24"] = np.log(h.close.shift(-24) / h.close)
    out = ["\n## E 예상 폭 보정 — 24h 1σ 띠 (정규·보정이면 띠 안 68.3% · RMS z 1.00)"]
    for lab, lo, hi in (("TRAIN", T0, TRAIN_END), ("OOS", TRAIN_END, OOS_END), ("HOLDOUT", OOS_END, T_END), ("2026 전체", T0, T_END)):
        m = (h.index > lo) & (h.index <= hi)
        cells = []
        for src in ("dvol", "rv24"):
            inside, rms = band_hits(h.r24[m].to_numpy(), (h[src][m] / np.sqrt(365)).to_numpy())
            cells.append(f"{src:5} 띠 안 {inside:5.1%} · RMS z {rms:.2f}")
        out.append(f"  {lab:9} " + "   |   ".join(cells) + f"   (DVOL−RV24 평균 {((h.dvol - h.rv24)[m].mean()) * 100:+.1f}pt)")
    out.append("  읽는 법: 띠 안 > 68% · RMS < 1 이면 그 띠는 실제보다 넓다(과대). 겹치는 24h 창이라 시간마다 독립 아님.")
    return "\n".join(out)


def _selftest() -> None:
    g = pd.DataFrame({"strike": [2600.0, 2700.0, 2800.0], "oi_call": [100.0, 50.0, 0.0], "oi_put": [0.0, 50.0, 100.0]})
    assert max_pain(g) == 2700.0, max_pain(g)            # 손계산: 2600 25k · 2700 20k · 2800 25k
    g2 = pd.DataFrame({"strike": [2600.0, 2700.0, 2800.0], "oi_call": [500.0, 0.0, 0.0], "oi_put": [0.0, 0.0, 0.0]})
    assert max_pain(g2) == 2600.0                          # 콜만 있으면 가장 낮은 행사가
    inside, rms = band_hits(np.random.default_rng(0).normal(0, 0.02, 200_000), np.full(200_000, 0.02))
    assert abs(inside - 0.683) < 0.005 and abs(rms - 1) < 0.01
    print("selftest ok")


def main() -> int:
    _selftest()
    px, dv = load_px5(), load_dvol()
    say(f"5분봉 {px.timestamp.min()} ~ {px.timestamp.max()} ({len(px):,}) · DVOL {dv.timestamp.min()} ~ {dv.timestamp.max()} ({len(dv):,})")
    rep = [f"# 옵션 판정 2026년 재검정 ({pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC)",
           f"5분봉 {px.timestamp.min()} ~ {px.timestamp.max()} · DVOL {dv.timestamp.min()} ~ {dv.timestamp.max()}",
           f"분할 TRAIN ~{TRAIN_END:%m-%d} · OOS ~{OOS_END:%m-%d} · HOLDOUT ~{T_END:%m-%d}"]
    say("A 핀닝"); rep += ["\n# A 핀닝", run_pinning(px, dv)]
    say("B 자석·max pain"); rep += ["\n# B 자석·max pain", run_magnet(px)]
    S, V = patch_dvol_modules(px, dv)
    say("C VRP"); rep += ["\n# C VRP 방향", run_vrp(S)]
    say("D DVOL 변동성"); rep += ["\n# D DVOL 변동성 전망", run_volfc(V)]
    say("E 예상 폭"); rep += ["\n# E", run_band(px, dv)]
    (W / "report.txt").write_text("\n".join(rep), encoding="utf-8")
    say(f"-> {W / 'report.txt'}")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
