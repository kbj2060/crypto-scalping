"""대시보드 재료(체결·크기별 흐름·OI·청산·베이시스·펀딩·호가·롱숏비·전일 POC) 사이의 상관·선후 관계 — 서술 통계.

🔴탐색이다. 판정(통과/불통과)·승격·매매 근거로 쓰지 않는다. 계산 전에 아래 목록을 고정했고, 계산한 것은 전부 출력한다(고르지 않는다).
범위: ETH · lake 로컬 미러 · 2026-09-29(현물 테이프 시작) ~ 2026-11-14.
      11-15~ 는 체결 정보 연구(docs/prereg/tape_info_abd_20261115.md)의 봉인 시험 구간이라 **읽지 않는다**.
봉: 1분 · 5분 · 60분. 행 τ = 봉 τ 가 닫힌 뒤 아는 값. 선후 = (X_τ , Y_{τ+1}) — 같은 봉을 공유하지 않는다.
불확실성: 날짜(UTC) 블록 부트스트랩 300회 2.5~97.5 분위.
표: T1 동시점 스피어만(부호 있는 것 / 크기) · T2 선후(X→다음 수익, 수익→다음 X, |X|→다음 |수익|) · T3 조건부 상태.

  python scripts/research_dashboard_material_correlation_20261120.py --selfcheck
  python scripts/research_dashboard_material_correlation_20261120.py --out tmp/dash_corr_20261120
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
LAKE = os.environ.get("DASHCORR_LAKE", os.path.expanduser("~/backups/crypto-scalping-server/data/lake"))
D0, D1 = "2026-09-29", "2026-11-14"   # 포함. 11-15~ 봉인 구간 금지
NBOOT = 300
SIGNED = ["ret", "delta", "delta_bn", "delta_okx", "delta_spot", "whale", "mid", "retail", "d_oi", "liq_net",
          "d_basis", "basis", "funding", "obi", "d_ls_global", "d_ls_top", "poc_pull"]
MAG = ["absret", "vol", "n_trades", "avg_size", "abs_delta", "abs_d_oi", "liq_tot", "abs_d_basis", "abs_poc_dist"]


def files(sub: str) -> list[str]:
    out = []
    for f in sorted(glob.glob(f"{LAKE}/{sub}/date=*/part.parquet")):
        d = f.split("date=")[1][:10]
        if D0 <= d <= D1:
            out.append(f)
    assert out, f"no files: {sub}"
    return out


def minute_frame() -> pd.DataFrame:
    import duckdb
    c = duckdb.connect()
    rp = lambda sub: f"read_parquet({files(sub)!r}, union_by_name=true)"   # noqa: E731
    tape = lambda sub, p: c.sql(   # noqa: E731
        f"select ts_sec//60*60 t, sum(buy_qty) {p}b, sum(sell_qty) {p}s, sum(whale_buy_qty-whale_sell_qty) {p}w, "
        f"sum(retail_buy_qty-retail_sell_qty) {p}r, sum(buy_n+sell_n) {p}n from {rp(sub)} group by 1").df().set_index("t")
    bn, ok, sp = tape("binance/tape/coin=ETH", "bn_"), tape("okx/tape/coin=ETH", "ok_"), tape("binance/spot_tape/coin=ETH", "sp_")
    mk = c.sql(f"select ts_ms//60000*60 t, arg_max(mark, ts_ms) px, arg_max((mark-index_px)/index_px*1e4, ts_ms) basis, "
               f"arg_max(funding_rate, ts_ms) funding from {rp('binance/mark_1s/coin=ETH')} group by 1").df().set_index("t")
    oi = c.sql(f"select ts_ms//60000*60 t, arg_max(open_interest, ts_ms) oi_bn from {rp('binance/oi_1s/coin=ETH')} group by 1").df().set_index("t")
    oio = c.sql(f"select ts_ms//60000*60 t, arg_max(oi_base, ts_ms) oi_ok from {rp('okx/oi/coin=ETH')} group by 1").df().set_index("t")
    lq = c.sql(f"select ts_ms//60000*60 t, sum(case when side='long' then usd else 0 end) liq_long, "
               f"sum(case when side='short' then usd else 0 end) liq_short from {rp('binance/liquidations/coin=ETH')} group by 1").df().set_index("t")
    lqo = c.sql(f"select ts_ms//60000*60 t, sum(case when pos_side='long' then sz_base*bk_px else 0 end) liq_long_ok, "
                f"sum(case when pos_side='short' then sz_base*bk_px else 0 end) liq_short_ok from {rp('okx/liquidations/coin=ALL')} "
                f"where inst_id='ETH-USDT-SWAP' group by 1").df().set_index("t")
    ob = c.sql(f"select epoch(ts)::bigint//60*60 t, avg(obi) obi from {rp('binance/micro_1m/coin=ETH')} where not data_stale group by 1").df().set_index("t")
    ls = c.sql(f"select epoch(ts)::bigint t, global_ls_ratio ls_global, top_pos_ls_ratio ls_top from {rp('binance/oi_lsratio_5m/coin=ETH')}").df().drop_duplicates("t").set_index("t")
    prof = c.sql(f"select ts_sec//86400 as d, price_bin as k, sum(buy_qty+sell_qty) as v from {rp('binance/tape/coin=ETH')} group by 1,2").df()
    t0 = int(pd.Timestamp(D0, tz="UTC").timestamp()); t1 = int(pd.Timestamp(D1, tz="UTC").timestamp()) + 86400
    m = pd.DataFrame(index=pd.RangeIndex(t0, t1, 60, name="t"))
    for df in (bn, ok, sp, mk, oi, oio, lq, lqo, ob, ls):
        m = m.join(df)   # 🔴concat(axis=1) 은 이 데이터에서 행을 잃은 적이 있다 -- join
    from market_ctx import value_area
    va = {int(d): value_area(g["k"], g["v"], 0.1) for d, g in prof.groupby("d")}
    prev = pd.Series(m.index // 86400 - 1, index=m.index).map(lambda d: va.get(int(d)))   # 전일(UTC) 프로파일만 = 그 날 내내 아는 값
    for key in ("poc", "val", "vah"):
        m[key] = prev.map(lambda x: x[key] if x else np.nan)
    return m


def bars(m: pd.DataFrame, k: int) -> pd.DataFrame:
    """1분 표 → k분 봉. 흐름은 합, 수준은 마지막. 원천(가격·바이낸스 테이프)이 빈 분이 낀 봉은 통째로 버린다."""
    g = m.index // (60 * k)
    s, la = m.groupby(g).sum(min_count=1), m.groupby(g).last()
    ok = m[["px", "bn_b"]].notna().all(axis=1).groupby(g).sum() == k
    b = pd.DataFrame(index=s.index)
    b["day"] = s.index * 60 * k // 86400
    lp = np.log(la["px"])
    b["ret"] = lp.diff() * 1e4
    fut = lambda c: s["bn_" + c].fillna(0) + s["ok_" + c].fillna(0)   # noqa: E731
    b["delta_bn"], b["delta_okx"], b["delta_spot"] = s["bn_b"] - s["bn_s"], s["ok_b"] - s["ok_s"], s["sp_b"] - s["sp_s"]
    b["delta"] = b["delta_bn"] + b["delta_okx"].fillna(0)
    b["vol"] = fut("b") + fut("s")
    b["whale"], b["retail"] = fut("w"), fut("r")
    b["mid"] = b["delta"] - b["whale"] - b["retail"]
    b["n_trades"] = fut("n"); b["avg_size"] = b["vol"] / b["n_trades"]
    b["d_oi"] = (la["oi_bn"].ffill() + la["oi_ok"].ffill()).diff()
    ll, lsh = s["liq_long"].fillna(0) + s["liq_long_ok"].fillna(0), s["liq_short"].fillna(0) + s["liq_short_ok"].fillna(0)
    b["liq_net"], b["liq_tot"] = lsh - ll, lsh + ll           # + = 숏이 청산(강제 매수)
    b["basis"], b["d_basis"], b["funding"] = la["basis"], la["basis"].diff(), la["funding"]
    b["obi"] = m["obi"].groupby(g).mean()
    b["d_ls_global"], b["d_ls_top"] = la["ls_global"].ffill().diff(), la["ls_top"].ffill().diff()
    dist = (la["px"] - la["poc"]) / la["px"] * 1e4            # 전일 POC 위 = +
    b["poc_pull"], b["abs_poc_dist"] = -dist, dist.abs()      # POC 쪽으로 당긴다면 poc_pull 과 다음 수익이 양의 상관
    b["out_va"] = np.where(la["px"] > la["vah"], 1, np.where(la["px"] < la["val"], -1, 0)).astype(float)
    for c in ("ret", "delta", "d_oi", "d_basis"):
        b["abs" + ("" if c == "ret" else "_") + c] = b[c].abs()
    return b[ok & ok.shift(1, fill_value=False)]   # 첫 봉·빈 분이 낀 봉·그 다음 봉(수익·증분이 빈 봉에 걸친다) 제외


def rk(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank()


def boot_corr(x: np.ndarray, y: np.ndarray, day: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, int]:
    """순위 상관(스피어만) + 날짜 블록 부트스트랩 구간. x·y 는 같은 길이, NaN 쌍은 뺀다."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, day = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy(), day[ok]
    if len(x) < 50 or x.std() == 0 or y.std() == 0:
        return (np.nan, np.nan, np.nan, int(len(x)))
    r = float(np.corrcoef(x, y)[0, 1])
    ud = np.unique(day); idx = {d: np.flatnonzero(day == d) for d in ud}
    bs = []
    for _ in range(NBOOT):
        ii = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])
        bs.append(np.corrcoef(x[ii], y[ii])[0, 1])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return (r, float(lo), float(hi), int(len(x)))


def boot_mean(v: np.ndarray, day: np.ndarray, rng: np.random.Generator) -> dict:
    ok = np.isfinite(v); v, day = v[ok], day[ok]
    if len(v) < 20:
        return {"n": int(len(v))}
    ud = np.unique(day); idx = {d: np.flatnonzero(day == d) for d in ud}
    bs = [v[np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])].mean() for _ in range(NBOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": int(len(v)), "days": int(len(ud)), "mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def leadlag(b: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    day, nxt = b["day"].to_numpy(), b.shift(-1)
    cont = (b.index.to_series().diff().shift(-1) == 1).to_numpy()      # 다음 봉이 바로 이어질 때만
    rows = []
    for c in [x for x in SIGNED if x != "ret"]:
        f = lambda a, z: boot_corr(np.where(cont, a, np.nan), z, day, rng)   # noqa: E731
        rows.append({"x": c, "x_to_ret": f(b[c].to_numpy(), nxt["ret"].to_numpy()), "ret_to_x": f(b["ret"].to_numpy(), nxt[c].to_numpy()),
                     "same": boot_corr(b[c].to_numpy(), b["ret"].to_numpy(), day, rng)})
    for c in [x for x in MAG if x != "absret"] + ["absret"]:
        rows.append({"x": c, "absx_to_absret": boot_corr(np.where(cont, b[c].to_numpy(), np.nan), nxt["absret"].to_numpy(), day, rng),
                     "same": boot_corr(b[c].to_numpy(), b["absret"].to_numpy(), day, rng)})
    return rows


def states(b: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """조건부 상태 → 다음 봉. fwd = 다음 봉 수익(bp), fabs = 다음 봉 |수익| ÷ 전체 평균."""
    day = b["day"].to_numpy()
    cont = (b.index.to_series().diff().shift(-1) == 1)
    fwd = b["ret"].shift(-1).where(cont); fabs = fwd.abs() / fwd.abs().mean()
    sg = np.sign
    out = []

    def add(name: str, mask: pd.Series, direction: pd.Series | None, note: str = "") -> None:
        mk = mask.fillna(False).to_numpy()
        d = {"state": name, "note": note, "share": float(mk.mean()), "size": boot_mean(fabs.to_numpy()[mk], day[mk], rng)}
        if direction is not None:
            d["dir_bp"] = boot_mean((fwd * direction).to_numpy()[mk], day[mk], rng)   # + = 그 방향으로 더 감
        out.append(d)

    big = b["abs_delta"] >= b["abs_delta"].quantile(2 / 3)
    ag = sg(b["ret"]) == sg(b["delta"])
    add("체결 큼 · 가격 같은 쪽", big & ag, sg(b["delta"]), "방향 = 체결 쪽")
    add("체결 큼 · 가격 반대쪽", big & ~ag & (b["ret"] != 0), sg(b["delta"]), "방향 = 체결 쪽(가격 쪽이면 부호 반대)")
    eff = b["absret"] / b["abs_delta"].replace(0, np.nan)
    add("체결 큼 · 가격 거의 제자리(효율 하위 20%)", big & (eff <= eff[big].quantile(.2)), sg(b["delta"]), "«흡수» · 방향 = 체결 쪽")
    small = b["abs_delta"] <= b["abs_delta"].quantile(1 / 3)
    add("체결 작음 · 가격 크게(|수익| 상위 20%)", small & (b["absret"] >= b["absret"].quantile(.8)), sg(b["ret"]), "«얇은 호가» · 방향 = 가격 쪽")
    mvbig = b["absret"] >= b["absret"].quantile(.75)
    for pr, pn in ((1, "상승"), (-1, "하락")):
        for oi, on in ((1, "OI 증가"), (-1, "OI 감소")):
            add(f"가격×OI: {pn} + {on} (이동 상위 25%)", mvbig & (sg(b["ret"]) == pr) & (sg(b["d_oi"]) == oi), pd.Series(1.0, index=b.index), "방향 = 위(+)")
    zw, zr, zm = (b[c] / b[c].rolling(288 if len(b) > 5000 else 24, min_periods=12).std().shift(1) for c in ("whale", "retail", "mid"))
    add("고래 vs 리테일 갈림(|z|≥.5 둘 다)", (zw.abs() >= .5) & (zr.abs() >= .5) & (sg(zw) != sg(zr)), sg(zw), "방향 = 고래 쪽")
    add("고래 vs 중형 갈림(|z|≥.5 둘 다)", (zw.abs() >= .5) & (zm.abs() >= .5) & (sg(zw) != sg(zm)), sg(zw), "방향 = 고래 쪽")
    add("셋 다 같은 쪽(|z|≥.5)", (zw.abs() >= .5) & (zr.abs() >= .5) & (zm.abs() >= .5) & (sg(zw) == sg(zr)) & (sg(zw) == sg(zm)), sg(zw), "방향 = 쏠린 쪽")
    liq = b["liq_tot"] > 0
    hot = b["liq_tot"] >= b.loc[liq, "liq_tot"].quantile(.9) if liq.any() else liq
    add("청산 급증(청산 있는 봉의 상위 10%)", hot, -sg(b["liq_net"]), "방향 = 청산을 낸 가격 방향(롱 청산 = 아래) · + = 연쇄 지속")
    dis = (sg(b["delta_bn"]) != sg(b["delta_okx"])) & (b["delta_okx"].abs() > 0)
    add("바이낸스 vs OKX 체결 부호 갈림", dis, sg(b["delta_okx"]), "방향 = OKX 쪽")
    dsp = (sg(b["delta"]) != sg(b["delta_spot"])) & (b["delta_spot"].abs() > 0)
    add("선물 vs 현물 체결 부호 갈림", dsp, sg(b["delta_spot"]), "방향 = 현물 쪽")
    add("전일 가치영역 밖", b["out_va"] != 0, -b["out_va"], "방향 = 가치영역 안쪽(되돌아옴)")
    add("전일 POC ±10bp 안", b["abs_poc_dist"] <= 10, None)
    bq = b["basis"].rank(pct=True)
    add("베이시스 상위 10%(선물 비쌈)", bq >= .9, pd.Series(1.0, index=b.index), "방향 = 위(+)")
    add("베이시스 하위 10%(선물 쌈)", bq <= .1, pd.Series(1.0, index=b.index), "방향 = 위(+)")
    oq = b["obi"].rank(pct=True)
    add("호가 불균형 상위 10%(매수 쪽 두꺼움)", oq >= .9, pd.Series(1.0, index=b.index), "방향 = 위(+)")
    add("호가 불균형 하위 10%(매도 쪽 두꺼움)", oq <= .1, pd.Series(1.0, index=b.index), "방향 = 위(+)")
    return out


def selfcheck() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n); day = np.arange(n) // 100
    r, lo, hi, _ = boot_corr(x[:-1], x[1:] * 0 + x[:-1] + rng.normal(size=n - 1) * .1, day[:-1], rng)
    assert r > .95 and lo > .9, (r, lo)
    r0, lo0, hi0, _ = boot_corr(x[:-1], rng.normal(size=n - 1), day[:-1], rng)
    assert lo0 < 0 < hi0, (r0, lo0, hi0)
    # bars(): 1분 표 → 5분 봉 합·마지막·수익, 빈 분이 낀 봉과 그 다음 봉은 버린다
    t = pd.RangeIndex(0, 60 * 30, 60, name="t")
    cols = ["bn_b", "bn_s", "bn_w", "bn_r", "bn_n", "ok_b", "ok_s", "ok_w", "ok_r", "ok_n", "sp_b", "sp_s", "sp_w", "sp_r", "sp_n", "px", "basis",
            "funding", "oi_bn", "oi_ok", "liq_long", "liq_short", "liq_long_ok", "liq_short_ok", "obi", "ls_global", "ls_top", "poc", "val", "vah"]
    m = pd.DataFrame(1.0, index=t, columns=cols)
    m["px"] = 100 * np.exp(np.arange(30) * 1e-4); m["bn_b"] = 3.0; m.loc[60 * 12, "bn_b"] = np.nan
    m[["liq_long", "liq_short", "liq_long_ok", "liq_short_ok"]] = 0.0; m.loc[60 * 21, "liq_long"] = 500.0
    b = bars(m, 5)
    assert list(b.index) == [1, 4, 5], list(b.index)                 # 봉 2(빈 분)·봉 3(직전이 빔)·봉 0(첫 수익 없음) 제외
    assert abs(b.loc[1, "ret"] - 5.0) < 1e-6 and b.loc[1, "delta_bn"] == 10.0 and b.loc[4, "liq_net"] == -500.0, b.loc[[1, 4]]
    print("selfcheck OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true"); ap.add_argument("--out", default="tmp/dash_corr_20261120")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = minute_frame(); m.to_parquet(out / "minute.parquet")
    rng = np.random.default_rng(20261120)
    res = {"range": [D0, D1], "nboot": NBOOT, "coverage": {c: float(m[c].notna().mean()) for c in m.columns}, "by_bar": {}}
    for k in (1, 5, 60):
        b = bars(m, k)
        res["by_bar"][str(k)] = {
            "n": int(len(b)), "days": int(b["day"].nunique()),
            "t1_signed": rk(b[SIGNED]).corr().round(3).to_dict(), "t1_mag": rk(b[MAG]).corr().round(3).to_dict(),
            "t2": leadlag(b, rng), "t3": states(b, rng) if k > 1 else []}
        print(k, "min bars", len(b), flush=True)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out / "result.json")


if __name__ == "__main__":
    main()
