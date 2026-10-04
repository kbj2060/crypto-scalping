"""LVN 돌파 뒤 «이동거리·체류·가격 효율» (2026-09-30, 사용자 반박 자료 ③ «LVN 은 방향이 아니라 돌파 후 이동거리/체류시간/가격 효율로 봐야»).

앞 검정(research_eth_profile_vwap_levels_20260930.py)은 «닿은 뒤 ±X bp 중 먼저»와 «30분 안 통과율»만 쟀다.
이번엔 «뚫은 뒤»를 잰다. 레벨·닿음·위약은 앞 스크립트 함수를 그대로 쓴다.

사전 고정(결과 보기 전, 2026-09-30):
  레벨   LVN/HVN = PV.hvn_lvn(정시마다 직전 24h 1분 tp 프로파일). 분 j 에 쓰는 레벨 = 직전 종가 위/아래 가장 가까운 것.
         닿음 = PV.touches(재무장 30bp, 해제 레벨 ±10bp 안 닿음 버림).
  돌파   닿은 분 j 부터 15분(j..j+14) 안에 1분 종가가 레벨을 통과 쪽으로 넘은 첫 분 m(저항 닿음이면 c[m] > L). 없으면 «돌파 없음».
         돌파는 분 m 이 닫혀야 안다 → 기준가 c[m], 지표는 분 m+1 부터.
  지표(방향 b = 통과 쪽)
    1순위 mfe60n = 60분 안 최대 순방향 이동 bp(b=+1 이면 max 고가, −1 이면 min 저가) ÷ rv60(분 m 까지 60분 실현변동성 bp). 기대 +.
    참고   mfe15·30·60(bp) · cont30·60 = b·(c[m+H]/c[m]−1) bp(기대 +) · dwell60 = 60분 중 종가가 L ±10bp 안인 분 비율(기대 −) ·
           leff30 = log(|c[m+30]/c[m]−1| bp ÷ 30분 거래대금 $M)(기대 +, 0 이동 제외) · brk = 닿음 중 15분 안 돌파 비율(기대 +).
  대조   ① LVN 위약(7·14·21·28일 전 LVN 을 정시 시가 비율로 옮김 -- 앞 검정과 같음) = 판정. ② HVN 돌파 = 참고(반박의 «HVN vs LVN»).
  기간   2025 · 2026(2024-08~ 예열). CI = 일 블록 포아송 부트스트랩(R.block_ci).
  판정   1순위 지표가 2025·2026 둘 다 기대 부호로 CI 0 배제 → 통과. 한 해만 → «한 해만». 반대 부호로 둘 다 → «반대로 통과».
  누수점검 기준·지표를 한 분 더 늦춰(c[m+1], m+2~) 유지되는지.

실행: python scripts/research_eth_lvn_break_20260930.py [--selftest] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_eth_fp_event_response_20260930 as E      # noqa: E402  load_1m
import research_eth_fp_pattern_lookback_20260930 as R    # noqa: E402  default_data_dir · block_ci
import research_eth_profile_vwap_levels_20260930 as PV   # noqa: E402  hvn_lvn · nearest · touches · swv

BRK_WIN, HS, BAND_BP = 15, (15, 30, 60), 10.0
KEYS = ("mfe60n", "mfe15", "mfe30", "mfe60", "cont30", "cont60", "dwell60", "leff30")
SIGN = {"mfe60n": 1, "mfe15": 1, "mfe30": 1, "mfe60": 1, "cont30": 1, "cont60": 1, "dwell60": -1, "leff30": 1, "brk": 1}


def breaks(J: np.ndarray, LV: np.ndarray, S: np.ndarray, c: np.ndarray) -> np.ndarray:
    """닿음 (j, L, s) → 첫 돌파 분 m(없으면 −1). s = 반전 방향(저항 −1 · 지지 +1), 통과 방향 = −s."""
    n = len(c); m = np.full(len(J), -1, np.int64)
    for q, (j, L, s) in enumerate(zip(J, LV, S)):
        seg = c[j:min(n, j + BRK_WIN)]
        with np.errstate(invalid="ignore"):
            hit = seg > L if s < 0 else seg < L
        if hit.any():
            m[q] = j + int(np.argmax(hit))
    return m


def metrics(m: np.ndarray, b: np.ndarray, L: np.ndarray, h, l, c, qv, rv60, extra: int = 0) -> dict[str, np.ndarray]:
    """돌파 분 m(유효한 것만)의 지표. 기준 c[m+extra], 창 = 분 m+extra+1 부터."""
    e = m + extra; ref = c[e]
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for H in HS:
            hi = np.nanmax(PV.swv(h, H)[e + 1], 1); lo = np.nanmin(PV.swv(l, H)[e + 1], 1)
            out[f"mfe{H}"] = np.where(b > 0, hi / ref - 1, 1 - lo / ref) * 1e4
        out["mfe60n"] = out["mfe60"] / rv60[e]
        for H in (30, 60):
            out[f"cont{H}"] = b * (c[e + H] / ref - 1) * 1e4
        out["dwell60"] = (np.abs(PV.swv(c, 60)[e + 1] / L[:, None] - 1) * 1e4 <= BAND_BP).mean(1)
        mv = np.abs(c[e + 30] / ref - 1) * 1e4; usd = PV.swv(qv, 30)[e + 1].sum(1) / 1e6
        out["leff30"] = np.where((mv > 0) & (usd > 0), np.log(mv / usd), np.nan)
    return out


def run(data_dir: Path, out_dir: Path) -> None:
    m1 = E.load_1m(data_dir)
    m1 = m1[m1.index >= PV.WARM.value // 10**6]
    pad = (-len(m1)) % PV.D1
    if pad:
        m1 = m1.reindex(np.r_[m1.index.to_numpy(), m1.index[-1] + 60_000 * np.arange(1, pad + 1)])
    t = m1.index.to_numpy(np.int64)
    assert t[0] % 86_400_000 == 0 and (np.diff(t) == 60_000).all()
    h, l, c, v = (m1[k].to_numpy(float) for k in ("h", "l", "c", "v"))
    tp, qv, n = (h + l + c) / 3, v * c, len(c)
    tsd = pd.to_datetime(t, unit="ms")
    yr = np.full(n, "", dtype=object)
    for name, a, b in PV.SPLITS:
        yr[(tsd >= a) & (tsd < b)] = name
    day = t // 86_400_000
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = np.r_[np.nan, np.diff(np.log(c))]
    G = np.r_[0.0, np.cumsum(np.nan_to_num(r1) ** 2)]
    rv60 = np.full(n, np.nan); rv60[60:] = np.sqrt(G[61:] - G[1:-60]) * 1e4
    HV, LV = PV.hvn_lvn(tp, v)
    P_hr = PV.open_before(c, np.arange(len(HV)) * 60)
    print(f"1분봉 {n:,} ({tsd[0]} ~ {tsd[-1]}) · 레벨 준비 끝", flush=True)

    def multi(M):
        U, Dn = PV.nearest(M, c)
        return PV.touches(U, Dn, c, h, l, None, PV.HVN_TOL_BP)

    def plac_rows(M, k):
        kk = 24 * k
        out = np.full_like(M, np.nan); out[kk:] = M[:-kk] * (P_hr[kk:] / P_hr[:-kk])[:, None]
        return out

    sets = {"LVN": [multi(LV)], "LVN위약": [multi(plac_rows(LV, k)) for k in PV.LAGS_D], "HVN": [multi(HV)]}
    frames = []
    for g, lst in sets.items():
        J, LVv, S = (np.concatenate(x) for x in zip(*lst))
        keep = yr[J] != ""; J, LVv, S = J[keep], LVv[keep], S[keep]
        m = breaks(J, LVv, S, c)
        frames.append(pd.DataFrame({"grp": g, "yr": yr[J], "day": day[J], "brk": (m >= 0).astype(float), "kind": "touch"}))
        ok = (m >= 0) & (m + 62 < n)
        mm, b, LL = m[ok], -S[ok].astype(int), LVv[ok]
        for extra in (0, 1):
            d = pd.DataFrame(metrics(mm, b, LL, h, l, c, qv, rv60, extra))
            d["grp"], d["yr"], d["day"], d["kind"], d["rv60"] = g, yr[mm], day[mm], f"brk+{extra}", rv60[mm + extra]
            frames.append(d)
        print(f"{g}: 닿음 {len(J):,} · 돌파 {int(ok.sum()):,}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for ref in ("LVN위약", "HVN"):
        for kind, keys in (("touch", ("brk",)), ("brk+0", KEYS), ("brk+1", KEYS)):
            for key in keys:
                row = {"대조": ref, "기준": kind, "지표": key}
                for y in ("2025", "2026"):
                    x = df[(df.kind == kind) & (df.yr == y) & df.grp.isin(["LVN", ref])].dropna(subset=[key])
                    g1 = (x.grp == "LVN").to_numpy(int)
                    ex, lo, hi = R.block_ci(x[key].to_numpy(float), g1, x.day.to_numpy())
                    row.update({f"n{y}": int(g1.sum()), f"LVN{y}": float(x[key][g1 == 1].mean()), f"대조{y}": float(x[key][g1 == 0].mean()),
                                f"차{y}": ex, f"lo{y}": lo, f"hi{y}": hi})
                sg = SIGN[key]
                ok = [(row[f"lo{y}"] * sg > 0) for y in ("2025", "2026")]
                rev = [(row[f"hi{y}"] * sg < 0) for y in ("2025", "2026")]
                row["판정"] = "통과" if all(ok) else "반대로 통과" if all(rev) else "한 해만" if any(ok) or any(rev) else "불합격"
                rows.append(row)
    res = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "grid.csv", index=False)
    df.to_parquet(out_dir / "events.parquet")                              # 사후 진단(변동성 맞춤)용
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))


def selftest() -> None:
    n = 200
    c = np.full(n, 99.9); h = c + 0.05; l = c - 0.05
    c[12] = 100.2                                                          # 저항 100 닿음(분 10) → 분 12 종가 돌파
    m = breaks(np.array([10, 10]), np.array([100.0, 100.0]), np.array([-1, 1]), c)
    assert m[0] == 12 and m[1] == 10, m                                     # 지지 쪽(아래로 뚫음)은 분 10 종가 99.9 < 100 이 곧 돌파
    c2 = np.full(n, 99.9); assert breaks(np.array([10]), np.array([100.0]), np.array([-1]), c2)[0] == -1
    rv = np.full(n, 10.0); qv = np.full(n, 1e6)
    hh = np.full(n, 100.3); ll = np.full(n, 100.1); cc = np.full(n, 100.2)
    hh[12] = 150.0                                                          # 돌파 분 자신의 고가는 지표에 안 들어간다
    hh[13] = 101.2                                                          # 다음 분부터 센다
    o = metrics(np.array([12]), np.array([1]), np.array([100.0]), hh, ll, cc, qv, rv)
    assert abs(o["mfe15"][0] - (101.2 / 100.2 - 1) * 1e4) < 1e-6, o["mfe15"]
    assert abs(o["mfe60n"][0] - o["mfe60"][0] / 10) < 1e-9
    assert o["dwell60"][0] == 0.0                                           # 100.2 는 100 에서 20bp → 띠(10bp) 밖
    o1 = metrics(np.array([12]), np.array([1]), np.array([100.0]), hh, ll, cc, qv, rv, extra=1)
    assert abs(o1["mfe15"][0] - (100.3 / 100.2 - 1) * 1e4) < 1e-6, "한 분 늦추면 분 13 은 창 밖"
    cc2 = cc.copy(); cc2[13:73] = 100.05
    assert metrics(np.array([12]), np.array([1]), np.array([100.0]), hh, ll, cc2, qv, rv)["dwell60"][0] == 1.0
    lo_ = np.full(n, 100.1); lo_[20] = 99.0
    od = metrics(np.array([12]), np.array([-1]), np.array([100.0]), hh, lo_, cc, qv, rv)
    assert abs(od["mfe15"][0] - (1 - 99.0 / 100.2) * 1e4) < 1e-6 and abs(od["cont30"][0] - 0.0) < 1e-9
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("tmp/lvn_break_20260930"))
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.data or R.default_data_dir(), a.out)
