"""지지·저항 5분 재계산(B) vs 1시간+쓸림 필터+채우기(A1) -- 다방면 비교 (2026-11-13, 사용자 «5분 업데이트가 실제로 지지/저항을 더 잘 반영하는지
얼마나 오래 머무는지·청산이 얼마나 일어나는지 등 다방면으로»). 11-09 청산 몫 검정(불통과 +0.4%p)의 후속.

사전등록: docs/prereg/sr_refresh_multi_20261113.md (계산 전 커밋). 설계는 그 문서가 정본이다.
팔·화면 재현은 11-09 스크립트(research_liqmap_sr_refresh_5m_vs_1h_20261109)와 같은 함수(compute_spliced_levels · _window_at · 쓸림 · 채우기).
사건 = 결정 시각 τ(매 5분 경계)에 보이는 레벨에 (τ, τ+60분] 1분봉이 처음 닿은 분 m. 라벨 = m+1 분부터 60분, 레벨 L 기준 대칭 ±D(0.5σ1h)
첫 통과: 지지는 위(L+D) 먼저 = 버팀. 위약 = 같은 팔 레벨 묶음을 ±25·±50bp 통째로 옮긴 가짜 레벨 4벌(수명 구조 보존 · 진짜와 겹치면 뺌).
실행: python scripts/research_sr_refresh_multi_20261113.py --selftest | --pilot | --run
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_liquidation_map_20260824 as LM  # noqa: E402
import research_liqmap_sr_refresh_5m_vs_1h_20261109 as P  # noqa: E402  (klines = data.binance.vision 일 파일 · 캐시)

OUT = Path("/home/kbj20/crypto-scalping/tmp/sr_refresh_multi_20261113")
LAKE = OUT / "lake/liquidations"
N_SHOW, LOOKBACK, H_MIN, SAME, LIQ_TOL, DWELL_TOL, N_PLACEBO, BOOT = 3, 24, 60, 15e-4, 15e-4, 10e-4, 4, 2000
END = "2026-11-11"
COINS = {"ETH": "2026-09-28", "BTC": "2026-09-28", "SOL": "2026-09-28", "XRP": "2026-09-28", "SOXL": "2026-11-03"}
PILOT = ("ETH", "2026-09-21", "2026-09-27")   # 위약 양성 대조 확인용 -- 주 판정에서 뺀다


# ---------------------------------------------------------------- 팔(화면 재현)
def show(mp: dict, cur: float, m5: tuple, since: int, tau: int, fill: bool) -> tuple[list[float], list[float]]:
    """app.js srLevelsLive(3) -- (지지, 저항). 안 쓸린 서버 레벨 가까운 순 3개 · fill 이면 heatmap_bins 로 채움."""
    sides = []
    for key, below in (("support_levels", True), ("resistance_levels", False)):
        ok = lambda p: (p < cur if below else p > cur) and not P.swept(m5, since, tau, p)  # noqa: E731
        keep = [lv["price"] for lv in mp.get(key) or [] if lv["price"] > 0 and ok(lv["price"])][:N_SHOW]
        if fill and len(keep) < N_SHOW:
            have = {lv["price"] for lv in mp.get(key) or []}
            more = sorted((b for b in mp.get("heatmap_bins") or [] if b["price"] not in have and b["weight_pct"] >= 0.05
                           and abs(b["price"] - cur) / cur <= 0.05 and ok(b["price"])),
                          key=lambda b: (-b["weight_pct"], abs(b["price"] - cur)))[:N_SHOW - len(keep)]
            keep += [b["price"] for b in more]
        sides.append(sorted(keep, key=lambda p: abs(p - cur)))
    return sides[0], sides[1]


def build(h1: pd.DataFrame, m5df: pd.DataFrame, taus: np.ndarray) -> dict:
    h, m = LM._bar_arrays(h1), LM._bar_arrays(m5df)
    close5 = dict(zip(m[0] + 300, m[1]["close"]))
    hour_maps: dict[int, dict] = {}
    arms = {"A1": [], "B": []}
    for tau in taus:
        tau = int(tau); hour = tau // 3600 * 3600; cur = close5[tau]
        if hour not in hour_maps:
            i1 = int(np.searchsorted(h[0], hour))
            w = h1.iloc[max(0, i1 - LOOKBACK):i1].reset_index(drop=True)
            hour_maps[hour] = LM.compute_spliced_levels(w, float(w.close.iloc[-1]), m5df[m5df.t < hour]) if len(w) >= 20 else {}
        arms["A1"].append(show(hour_maps[hour], cur, m, hour, tau, True))
        mb = LM.compute_spliced_levels(LM._window_at(h, m, tau, LOOKBACK), cur, m5df[m5df.t < tau])
        arms["B"].append(show(mb, cur, m, tau, tau, False))
    return arms


# ---------------------------------------------------------------- 사건·라벨
def first_touch_min(t1: np.ndarray, lo: np.ndarray, hi: np.ndarray, tau: int, L: float, below: bool) -> int:
    """(τ, τ+60분] 에 시작한 1분봉 중 L 에 처음 닿은 봉 번호(없으면 −1). 지지 = 저가 ≤ L · 저항 = 고가 ≥ L."""
    a, b = np.searchsorted(t1, [tau, tau + H_MIN * 60])
    hit = np.nonzero(lo[a:b] <= L)[0] if below else np.nonzero(hi[a:b] >= L)[0]
    return int(a + hit[0]) if len(hit) else -1


def hold_label(lo: np.ndarray, hi: np.ndarray, i: int, L: float, D: float, below: bool) -> int:
    """i+1 분부터 60분, L·(1±D) 첫 통과. 1 = 버팀(지지는 위 먼저) · 0 = 뚫림 · −1 = 둘 다 같은 봉 · −2 = 시간 끝."""
    up, dn = L * (1 + D), L * (1 - D)
    for j in range(i + 1, min(i + 1 + H_MIN, len(lo))):
        u, d = hi[j] >= up, lo[j] <= dn
        if u and d:
            return -1
        if u or d:
            return int(u) if below else int(d)
    return -2


def events(levels: list[tuple[list, list]], taus: np.ndarray, cur: np.ndarray, sig: np.ndarray, k1: pd.DataFrame, liq: pd.DataFrame) -> pd.DataFrame:
    """τ 마다 보이는 레벨의 첫 닿음 → (쪽, 닿은 분, L 0.15% 칸) 으로 중복 제거(같은 닿음을 연달아 τ 가 다시 세지 않게, 가장 이른 τ)."""
    t1, lo, hi, c1 = k1.t.to_numpy(), k1.low.to_numpy(), k1.high.to_numpy(), k1.close.to_numpy()
    lt, lp, lu = liq.ts_ms.to_numpy() // 1000, liq.price.to_numpy(), liq.usd.to_numpy()
    seen, rows = set(), []
    for k, tau in enumerate(taus):
        for below, lvs in ((True, levels[k][0]), (False, levels[k][1])):
            for L in lvs:
                i = first_touch_min(t1, lo, hi, int(tau), L, below)
                if i < 0:
                    continue
                key = (below, int(t1[i]), round(np.log(L) / SAME))
                if key in seen:
                    continue
                seen.add(key)
                a, b = np.searchsorted(lt, [t1[i], t1[i] + 300])
                near = np.abs(lp[a:b] / L - 1) <= LIQ_TOL
                stick = float((np.abs(c1[i + 1:i + 31] / L - 1) <= DWELL_TOL).mean()) if i + 31 <= len(c1) else np.nan   # 닿은 뒤 30분 종가가 ±10bp 안에 머문 몫
                rows.append((int(tau), below, L, abs(L / cur[k] - 1), int(t1[i]), hold_label(lo, hi, i, L, sig[k], below), float(lu[a:b][near].sum()), stick))
    df = pd.DataFrame(rows, columns=["tau", "sup", "L", "dist", "touch_t", "hold", "liq_usd", "stick"])
    df["day"] = pd.to_datetime(df.tau, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return df


SHIFTS = (25e-4, -25e-4, 50e-4, -50e-4)   # 위약 = 진짜 레벨 묶음을 통째로 옮김(수명·개수 구조 보존) -- 파일럿에서 «거리 섞기»는 매 τ 새 레벨이라 닿음이 3배였다


def placebo_levels(levels: list[tuple[list, list]], cur: np.ndarray, seed: int) -> list[tuple[list, list]]:
    """seed 번째 이동(SHIFTS)만큼 같은 쪽 레벨을 옮긴다 -- 지금가의 맞은쪽으로 넘어가거나 진짜 레벨 ±0.15% 안에 떨어지면 뺀다."""
    f = 1 + SHIFTS[seed % len(SHIFTS)]
    out = []
    for k, (sup, res) in enumerate(levels):
        side = []
        for s, lv in ((0, sup), (1, res)):
            real = np.array(lv)
            side.append([L * f for L in lv if ((L * f < cur[k]) if s == 0 else (L * f > cur[k]))
                         and not (len(real) and (np.abs(real / (L * f) - 1) <= SAME).any())])
        out.append(tuple(side))
    return out


# ---------------------------------------------------------------- 서술 지표
def lifetime(levels: list[tuple[list, list]], lag: int) -> float:
    """τ 에 보인 레벨이 lag 경계 뒤에도 (같은 쪽 ±0.15%) 보이는 몫."""
    keep = tot = 0
    for k in range(len(levels) - lag):
        for s in (0, 1):
            nxt = np.array(levels[k + lag][s])
            for L in levels[k][s]:
                tot += 1; keep += bool(len(nxt) and (np.abs(nxt / L - 1) <= SAME).any())
    return keep / tot if tot else float("nan")


def run_lengths(levels: list[tuple[list, list]]) -> np.ndarray:
    """같은 쪽 ±0.15% 로 이어진 연속 표시 길이(경계 수) -- 수명 분포."""
    out, live = [], {0: [], 1: []}
    for k in range(len(levels)):
        for s in (0, 1):
            new = []
            for L in levels[k][s]:
                m = next((x for x in live[s] if abs(x[0] / L - 1) <= SAME), None)
                new.append([L, (m[1] + 1) if m else 1])
            out += [x[1] for x in live[s] if not any(abs(x[0] / y[0] - 1) <= SAME for y in new)]
            live[s] = new
    return np.array(out + [x[1] for s in (0, 1) for x in live[s]])


# ---------------------------------------------------------------- 채점
def rate(df: pd.DataFrame) -> tuple[float, int, int]:
    v = df[df.hold >= 0]
    return (float(v.hold.mean()) if len(v) else float("nan")), int(v.hold.sum()), int(len(v))


def boot(real: dict, plac: dict, seed: int = 11) -> dict:
    """X = 버팀(진짜) − 버팀(위약) 를 팔마다 · ΔX = X_B − X_A1 · 날 블록 부트스트랩(같은 날 뽑기를 모든 팔·위약에 공통)."""
    days = sorted(set(real["B"].day) | set(real["A1"].day))
    def agg(df):
        v = df[df.hold >= 0]
        g = v.groupby("day").hold.agg(["sum", "count"]).reindex(days, fill_value=0)
        return g["sum"].to_numpy(float), g["count"].to_numpy(float)
    S = {k: agg(v) for k, v in real.items()}; Q = {k: agg(v) for k, v in plac.items()}
    X = lambda w, k: (w @ S[k][0]) / (w @ S[k][1]) - (w @ Q[k][0]) / (w @ Q[k][1])  # noqa: E731
    one = np.ones(len(days))
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)) for _ in range(BOOT)]).astype(float)
    out = {}
    for name, f in (("X_A1", lambda w: X(w, "A1")), ("X_B", lambda w: X(w, "B")), ("dX", lambda w: X(w, "B") - X(w, "A1"))):
        bs = np.array([f(w) for w in W])
        out[name] = {"est": round(float(f(one)), 4), "lo90": round(float(np.quantile(bs, .10)), 4),
                     "ci95": [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}
    out["days"] = len(days)
    return out


def liq_lake(coin: str, d0: str, d1: str) -> pd.DataFrame:
    fs = [p for p in sorted(glob.glob(str(LAKE / f"coin={coin}" / "date=*" / "*.parquet"))) if d0 <= p.split("date=")[1][:10] <= d1]
    return pd.concat([pd.read_parquet(p, columns=["ts_ms", "usd", "price"]) for p in fs], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def run_coin(coin: str, d0: str, d1: str) -> dict:
    warm = str(date.fromisoformat(d0) - timedelta(days=2))
    h1, m5, k1 = P.klines(coin, "1h", warm, d1), P.klines(coin, "5m", warm, d1), P.klines(coin, "1m", d0, d1)
    end_s = int(pd.Timestamp(d1, tz="UTC").timestamp()) + 86400 - H_MIN * 60 - 3600
    taus = np.arange(int(pd.Timestamp(d0, tz="UTC").timestamp()), end_s, 300)
    taus = taus[np.isin(taus - 300, m5.t.to_numpy())]
    cl = m5.set_index(m5.t + 300).close
    cur = cl.reindex(taus).to_numpy()
    r = np.log(cl).diff()
    sig = (r.rolling(288, min_periods=200).std() * np.sqrt(12) * 0.5).reindex(taus).to_numpy()   # D = 0.5σ1h (11-10 연구와 같은 폭)
    liq = liq_lake(coin, d0, d1)
    print(f"[{coin}] {d0}~{d1} τ {len(taus)} · 1m {len(k1)} · 청산 {len(liq)}", flush=True)
    arms = build(h1, m5, taus)
    real = {a: events(v, taus, cur, sig, k1, liq) for a, v in arms.items()}
    plac = {a: pd.concat([events(placebo_levels(v, cur, s), taus, cur, sig, k1, liq) for s in range(N_PLACEBO)]) for a, v in arms.items()}
    out = {"coin": coin, "d0": d0, "d1": d1, "n_tau": int(len(taus)), "liq_n": int(len(liq)), "D_med_bp": round(float(np.nanmedian(sig)) * 1e4, 1)}
    for a in arms:
        rr, rp = real[a], plac[a]
        hr, hp = rate(rr), rate(rp)
        rl = run_lengths(arms[a]) * 5
        out[a] = {"hold_real": [round(hr[0], 4), hr[1], hr[2]], "hold_placebo": [round(hp[0], 4), hp[1], hp[2]],
                  "touches_per_day_real": round(len(rr) / out["n_tau"] * 288, 1), "touches_per_day_placebo": round(len(rp) / N_PLACEBO / out["n_tau"] * 288, 1),
                  "ties": int((rr.hold == -1).sum()), "timeouts": int((rr.hold == -2).sum()),
                  "liq_usd_per_touch_real": round(float(rr.liq_usd.mean()), 1), "liq_usd_per_touch_placebo": round(float(rp.liq_usd.mean()), 1),
                  "touch_with_liq_real": round(float((rr.liq_usd > 0).mean()), 4), "touch_with_liq_placebo": round(float((rp.liq_usd > 0).mean()), 4),
                  "keep_15m": round(lifetime(arms[a], 3), 4), "keep_60m": round(lifetime(arms[a], 12), 4),
                  "life_min_median": float(np.median(rl)), "life_ge60_share": round(float((rl >= 60).mean()), 4),
                  "dist_med_bp": round(float(np.nanmedian([abs(L / cur[k] - 1) for k, lv in enumerate(arms[a]) for s in lv for L in s])) * 1e4, 1),
                  "stick30_real": round(float(rr.stick.mean()), 4), "stick30_placebo": round(float(rp.stick.mean()), 4)}
    out["boot"] = boot(real, plac)
    # 닿음 순간 청산 $ 초과(진짜 − 위약, 닿음당) B − A1 · 날 블록
    out["liq_excess_per_touch"] = {a: round(out[a]["liq_usd_per_touch_real"] - out[a]["liq_usd_per_touch_placebo"], 1) for a in arms}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat({f"{a}_real": real[a] for a in arms} | {f"{a}_plac": plac[a] for a in arms}, names=["set"]).to_parquet(OUT / f"events_{coin}_{d0}.parquet")
    return out


def selftest() -> None:
    lo = np.array([100.0, 99.0, 99.5, 99.9, 98.0]); hi = np.array([101.0, 100.0, 101.2, 100.1, 99.0])
    assert hold_label(lo, hi, 1, 100.0, 0.01, True) == 1            # 지지 100 · 위 101 먼저
    assert hold_label(lo, hi, 2, 100.0, 0.01, True) == 0            # 그 뒤 98 = 아래 99 먼저
    lo2 = np.array([100.0, 99.5, 98.9]); hi2 = np.array([100.5, 100.2, 99.5])
    assert hold_label(lo2, hi2, 0, 100.0, 0.01, True) == 0          # 아래 99 먼저 = 뚫림
    assert hold_label(lo2, hi2, 0, 98.0, 0.01, False) == 0          # 저항 98 · 위 98.98 먼저 = 뚫림
    lo3 = np.array([100.0, 98.5]); hi3 = np.array([100.0, 101.5])
    assert hold_label(lo3, hi3, 0, 100.0, 0.01, True) == -1         # 같은 봉 둘 다
    t1 = np.array([0, 60, 120]); assert first_touch_min(t1, np.array([101.0, 99.0, 98.0]), np.array([102.0, 100.0, 99.0]), 0, 99.5, True) == 1
    lv = [([99.0, 98.0], [101.0]), ([97.0], [103.0, 104.0])]; c = np.array([100.0, 100.0])
    pl = placebo_levels([([99.0, 98.75], [101.0])], np.array([100.0]), 0)   # +25bp: 99→99.2475 · 98.75→98.997(진짜 99 ±0.15% 안 → 뺌) · 101→101.2525
    assert [round(x, 2) for x in pl[0][0]] == [99.25] and [round(x, 2) for x in pl[0][1]] == [101.25], pl
    pl = placebo_levels([([99.8], [])], np.array([100.0]), 2)                # +50bp: 99.8→100.299 은 지금가 위 → 뺌
    assert pl[0][0] == [], pl
    assert lifetime([([99.0], []), ([99.05], []), ([90.0], [])], 1) == 0.5
    assert sorted(run_lengths([([99.0], []), ([99.05], []), ([90.0], [])]).tolist()) == [1, 2]
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--pilot" in sys.argv:
        r = run_coin(*PILOT); OUT.mkdir(parents=True, exist_ok=True); (OUT / "pilot.json").write_text(json.dumps(r, ensure_ascii=False, indent=1)); print(json.dumps(r, ensure_ascii=False, indent=1))
    elif "--run" in sys.argv:
        res = {c: run_coin(c, d0, END) for c, d0 in COINS.items()}
        (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(json.dumps(res, ensure_ascii=False, indent=1))
