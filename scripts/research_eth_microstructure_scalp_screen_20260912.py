#!/usr/bin/env python3
"""ETH 1분봉 미세구조(duckdb)로 스캘핑이 되는가 — 사전등록 스크린.

판정(2026-09-12): **안 된다.** 신호는 실재하나 비용의 4.7배 모자란다.
  obi 십분위 상하위 격차가 H=15분에서 3.32bp(rho 0.89, H=5 는 rho 0.94 로 더 단조).
  편도 환산 1.66bp 가 이 데이터의 **손익분기 왕복비용**이고, 실제는 7.8bp(peg-maker 실측).
  지평을 늘려도 안 된다 -- 격차가 커지지 않고 신호가 감쇠한다(rho H5 0.94 -> H720 0.18
  -> H1440 -0.26). 21피쳐 GBM 은 TRAIN 상관 +0.159 가 VAL 에서 **-0.008** 로 사라지고,
  선별성도 상위 0.2%(68건)까지 좁혀도 총수익 4.01bp < 비용 7.8bp 다.

사전등록(수익률을 보기 **전**에 고정):
  데이터  microstructure_model_ready_v1 (21피쳐) + mark_price.
          ⚠️mark_price 는 2026-07-17 까지 전량 NULL(컬럼이 나중에 추가됨) -- 그대로 쓰면
            39일로 줄어든다. /fapi/v1/klines 로 메운다(이쪽은 startTime 을 정상 존중한다.
            결함은 /futures/data/* 뿐 -- test/test_futures_data_backward_paging_20260912.py).
  분할    시간순 고정, 셔플 없음. TRAIN 05-03~07-14(70일) / VAL 07-15~08-09(26일) /
          OOS 08-10~08-25(16일). 선택은 TRAIN, 확인은 VAL, OOS 는 마지막에 한 번만.
  비용    왕복 7.8bp 헤드라인.
  통과    ①OOS 평균 순익의 일군집 부트 95%CI 가 0 배제 ②같은 측면 무작위 진입 귀무 대비
          p<0.05 ③VAL/OOS 부호 일치. **셋 다** 충족해야 주장한다.
  결과    VAL 에서 ①이 이미 깨져 OOS 를 쓰지 않았다(소진 방지).

⚠️OOS 구간은 H60 평균 +7.37bp 의 상승 드리프트가 있다 -- 롱 편향 규칙은 여기서 무조건
  좋아 보인다. 같은 측면 귀무 없이 이 창의 수익률을 실력으로 읽으면 안 된다.

    python3 scripts/research_eth_microstructure_scalp_screen_20260912.py [--db PATH]
"""
from __future__ import annotations

import argparse, json, sys, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

COST = 7.8e-4
HORIZONS = (5, 15, 30, 60, 120, 240, 720, 1440)
SPLITS = {"TRAIN": ("2026-05-03", "2026-07-15"),
          "VAL": ("2026-07-15", "2026-08-10"),
          "OOS": ("2026-08-10", "2026-08-26")}


def klines_1m(start: str, end: str, symbol: str = "ETHUSDT",
              cache: str = "/tmp/scalp/eth_klines_1m.parquet") -> pd.DataFrame:
    """1분봉. 재실행마다 19만 봉을 다시 받으면 429 를 맞으므로 로컬에 캐시한다."""
    cp = Path(cache)
    if cp.exists():
        return pd.read_parquet(cp)[["ts", "close"]]
    out, s = [], int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    e = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    while s < e:
        u = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1m"
             f"&startTime={s}&endTime={e}&limit=1500")
        d = json.load(urllib.request.urlopen(u, timeout=25))
        if not d:
            break
        out += [(int(k[0]), float(k[4])) for k in d]
        s = int(d[-1][0]) + 60_000
        if len(d) < 1500:
            break
        time.sleep(0.12)
    k = pd.DataFrame(out, columns=["ms", "close"]).drop_duplicates("ms")
    k["ts"] = pd.to_datetime(k.ms, unit="ms", utc=True)
    cp.parent.mkdir(parents=True, exist_ok=True)
    k[["ts", "close"]].to_parquet(cp)
    return k[["ts", "close"]]


def build(db: str) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(db, read_only=True)
    f = con.execute("""select r.* from microstructure_model_ready_v1 r join microstructure_1m m
                       using (ts) where m.data_stale=false and r.valid_taker_flow and r.valid_nif
                       order by r.ts""").df()
    con.close()
    f["ts"] = pd.to_datetime(f.ts, utc=True)
    f = f.drop_duplicates("ts", keep="last")
    d = f.merge(klines_1m("2026-05-01", "2026-09-13"), on="ts", how="inner").sort_values("ts")
    s = d.set_index("ts")
    for H in HORIZONS:                       # 시간 기준 shift -- 결측 분을 건너뛰지 않는다
        s[f"fwd{H}"] = s.close.shift(-H, freq="min").reindex(s.index) / s.close - 1
    d = s.reset_index()
    d["split"] = None
    for n, (a, b) in SPLITS.items():
        m = (d.ts >= pd.Timestamp(a, tz="UTC")) & (d.ts < pd.Timestamp(b, tz="UTC"))
        d.loc[m, "split"] = n
    return d


def feature_cols(d: pd.DataFrame) -> list[str]:
    skip = {"ts", "split", "close", "shadow_regime_tag"}
    return [c for c in d.columns
            if c not in skip and not c.startswith("fwd") and d[c].dtype.kind in "fiub"]


def screen(d: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """십분위 상위 롱 / 하위 숏. `breakeven_bp` 가 그 셀이 허용하는 최대 왕복비용이다."""
    from scipy.stats import spearmanr
    tr, rows = d[d.split == "TRAIN"], []
    for f in feats:
        if tr[f].nunique() < 10:
            continue
        q = pd.qcut(tr[f].rank(method="first"), 10, labels=False)
        for H in HORIZONS:
            y = tr[f"fwd{H}"]
            m = q.notna() & y.notna()
            g = y[m].groupby(q[m]).mean()
            if len(g) < 10:
                continue
            half = (g.iloc[-1] - g.iloc[0]) / 2
            rows.append(dict(feat=f, H=H, rho=spearmanr(g.index, g.values).statistic,
                             breakeven_bp=half * 1e4, net_bp=(half - COST) * 1e4))
    return pd.DataFrame(rows).sort_values("net_bp", ascending=False)


def day_clustered_ci(d: pd.DataFrame, feat: str, H: int, B: int = 2000, seed: int = 20260912):
    """십분위 극단 매매의 순익을 **날짜로 재표집**해 CI 를 낸다. 겹친 행을 독립으로 세면
    CI 가 몇 배 좁아져 우연을 통과로 읽는다(2026-09-12 에 H=1440 두 셀이 그렇게 보였다)."""
    tr = d[d.split == "TRAIN"]
    y, q = tr[f"fwd{H}"], pd.qcut(tr[feat].rank(method="first"), 10, labels=False)
    m = y.notna() & q.notna()
    sub = tr[m].assign(q=q[m], y=y[m])
    lon, sho = sub[sub.q == 9], sub[sub.q == 0]
    ret = np.concatenate([lon.y.values, -sho.y.values])
    day = np.concatenate([lon.ts.dt.date.values, sho.ts.dt.date.values])
    ud, rng, boot = np.unique(day), np.random.default_rng(seed), []
    for _ in range(B):
        v = np.concatenate([ret[day == x] for x in rng.choice(ud, len(ud), replace=True)
                            if (day == x).any()])
        if len(v):
            boot.append(v.mean() - COST)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return lo, hi, len(ud), len(ret)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/live/microstructure.duckdb")
    a = ap.parse_args()
    if not Path(a.db).exists():
        print(f"{a.db} 없음 — 저장소 루트에서 실행하거나 --db 로 지정", file=sys.stderr)
        return 2
    d = build(a.db)
    feats = feature_cols(d)
    print(f"{len(d):,}행 · 독립일수 {d.ts.dt.date.nunique()} · 피쳐 {len(feats)}개 · 비용 {COST*1e4:.1f}bp")
    for n in SPLITS:
        x = d[d.split == n]
        print(f"  {n:6s} {len(x):>7,}행 {x.ts.dt.date.nunique():>3}일  "
              + "  ".join(f"H{H}:{x[f'fwd{H}'].mean()*1e4:+6.2f}bp" for H in (15, 60)))

    r = screen(d, feats)
    print(f"\n=== TRAIN 십분위 스크린 상위 8 ===")
    print(r.head(8).to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    passed = r[r.net_bp > 0]
    print(f"\n비용 {COST*1e4:.1f}bp 를 넘는 셀: {len(passed)} / {len(r)}")
    print(f"최대 손익분기 왕복비용: {r.breakeven_bp.max():.2f}bp "
          f"({r.loc[r.breakeven_bp.idxmax(), 'feat']} H={int(r.loc[r.breakeven_bp.idxmax(), 'H'])})")

    # ⚠️명목 n 은 검정력이 아니다. H 분 전방수익을 1분마다 뽑으면 창이 (H-1)/H 만큼 겹친다 --
    #   H=1440 이면 99.6% 다. 통과처럼 보이는 셀은 반드시 **일 군집** 으로 다시 잰다.
    for _, c in passed.iterrows():
        lo, hi, ndays, ret_n = day_clustered_ci(d, c.feat, int(c.H))
        verdict = "0 배제" if lo > 0 or hi < 0 else "0 포함 -> 통과 아님"
        print(f"  재검정 {c.feat} H={int(c.H)}: 명목 n={ret_n:,} 이지만 **독립 일수 {ndays}** "
              f"-> 일군집 95%CI [{lo*1e4:+.1f}, {hi*1e4:+.1f}]bp  {verdict}")
    if len(passed) == 0:
        print("\n판정: 통과 셀 0 — VAL/OOS 를 쓰지 않는다(소진 방지). 구속조건은 신호가 아니라 비용이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
