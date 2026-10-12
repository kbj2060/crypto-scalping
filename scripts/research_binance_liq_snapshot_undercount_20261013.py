"""바이낸스 청산 스냅샷 누락분 보정 추정기 (2026-10-13, 사용자 «청산 커버리지 1번 진행»).

문제: 바이낸스 `!forceOrder@arr` 는 종목마다 1,000ms 창에서 **마지막 1건만** 보낸다(공식 문서). 청산이 몰리면 나머지가 사라진다.
보정 근거: Bybit `allLiquidation` 은 묶지 않고 전부 준다(실측: 청산 있는 초-종목의 32% 가 2건+, 최대 128건/초).
  ⇒ Bybit 원 기록에 바이낸스 규칙(종목·1초 창마다 마지막 1건)을 그대로 씌우면 «스냅샷이 잃는 몫»을 정확히 잴 수 있다.
추정기(바이낸스에서도 보이는 것만 쓴다): 1·5분 버킷마다 n = 스냅샷이 찍힌 초 수 → n 칸별 배수 두 벌(«중앙» 표시용 · «합비율» 총량용).
  (처음 시도한 초 단위 k 모델은 조용한 단건까지 부풀려 버렸다 -- 아래 fit_m 등은 그 기록으로 남긴다.)
판정(목적 정확도): 1분·5분 합계의 |log(추정/실제)| 중앙값·90분위와 총량 비율 — 보정 없음 · 고정 배수(전체 비율 하나)와 비교.
  검증 ① Bybit 시간 홀드아웃(앞 60% 학습 / 뒤 40%) ② 창 위상 무작위(바이낸스 창은 초 경계와 안 맞는다) ③ OKX(최대 4건/초라 완전하지 않을 수 있어 보조).
한계(ponytail): 바이낸스 청산 엔진이 Bybit 과 같은 «초당 몰림»을 보인다고 가정한다 -- 바이낸스 자체 정답은 공개 원천이 없다(유료 피드뿐).
  업그레이드 경로 = 유료 전체 피드(Coin Metrics 등) 한 달치로 m(k) 를 바이낸스 실측으로 다시 맞춘다.

실행: python scripts/research_binance_liq_snapshot_undercount_20261013.py <all_liq.parquet> [--selftest]
  입력 열: venue, symbol, ts_ms, side, qty, price (서버 hot·lake 를 읽기 전용으로 모은 것)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

K_WIN = 10                      # 버스트 강도 창(초)
K_BINS = [1, 2, 3, 4, 6, 8, 11]  # k 칸 경계: [1],[2],[3],[4-5],[6-7],[8-10]


def snapshot(df: pd.DataFrame, phase_ms: int = 0) -> pd.DataFrame:
    """바이낸스 규칙: 종목마다 (ts+phase)//1000 창의 마지막 1건만 남긴다."""
    d = df.sort_values("ts_ms")
    w = (d.ts_ms + phase_ms) // 1000
    return d.assign(win=w).groupby(["symbol", "win"], as_index=False).last()


def burst_k(snap: pd.DataFrame) -> np.ndarray:
    """스냅샷 초마다 직전 K_WIN 초(포함) 안 스냅샷 초 수 -- 바이낸스에서도 그대로 보이는 값."""
    out = np.empty(len(snap), dtype=int)
    for _, idx in snap.groupby("symbol").indices.items():
        w = snap.win.to_numpy()[idx]
        lo = np.searchsorted(w, w - K_WIN + 1, side="left")
        out[idx] = np.arange(len(w)) - lo + 1
    return out


def kbin(k: np.ndarray) -> np.ndarray:
    return np.digitize(k, K_BINS[1:])            # 0..len-1


def fit_m(truth: pd.DataFrame, phase_ms: int = 0) -> dict:
    """학습: 같은 창의 실제 합 / 스냅샷 수량을 k 칸별 Σ비율로."""
    s = snapshot(truth, phase_ms)
    s["k"] = burst_k(s)
    tot = truth.assign(win=(truth.ts_ms + phase_ms) // 1000).groupby(["symbol", "win"]).qty_n.sum()
    s["true"] = tot.reindex(pd.MultiIndex.from_frame(s[["symbol", "win"]])).to_numpy()
    s["b"] = kbin(s.k.to_numpy())
    g = s.groupby("b")[["true", "qty_n"]].sum()
    return {"m": {int(b): float(r.true / r.qty_n) for b, r in g.iterrows()}, "const": float(s.true.sum() / s.qty_n.sum())}


def estimate(snap: pd.DataFrame, m: dict) -> np.ndarray:
    b = kbin(burst_k(snap))
    return snap.qty_n.to_numpy() * np.array([m["m"].get(int(x), m["const"]) for x in b])


def score(truth: pd.DataFrame, snap: pd.DataFrame, est: np.ndarray, bucket_s: int) -> dict:
    """버킷 합계 비교 -- 실제가 있는 버킷만(스냅샷 0 이면 추정도 0 이라 log 오차 무한 → 따로 «놓친 버킷» 비율)."""
    tb = truth.groupby([truth.symbol, truth.ts_ms // 1000 // bucket_s]).qty_n.sum()
    key = [snap.symbol, snap.win // bucket_s]
    sb = snap.groupby(key).qty_n.sum().reindex(tb.index, fill_value=0.0)
    eb = pd.Series(est, index=snap.index).groupby(key).sum().reindex(tb.index, fill_value=0.0)
    ok = sb > 0
    le = lambda x: np.abs(np.log(x[ok] / tb[ok]))  # noqa: E731
    return {"buckets": int(len(tb)), "missed": round(float((~ok).mean()), 4),
            "raw_med": round(float(le(sb).median()), 3), "raw_p90": round(float(le(sb).quantile(.9)), 3),
            "est_med": round(float(le(eb).median()), 3), "est_p90": round(float(le(eb).quantile(.9)), 3),
            "raw_total": round(float(sb.sum() / tb.sum()), 3), "est_total": round(float(eb.sum() / tb.sum()), 3)}


def norm(df: pd.DataFrame) -> pd.DataFrame:
    """종목별 수량을 그 종목 중앙값으로 나눠 풀링(단위·코인 크기 상쇄 -- 비율만 본다)."""
    return df.assign(qty_n=df.qty / df.groupby("symbol").qty.transform("median"))


# ── 채택 모델(분 단위): 버킷(1·5분) 안 «스냅샷이 찍힌 초 수» n 칸별 배수 두 벌 ─────────────────────────────
# 초 단위 k 모델(위)은 조용한 단건 분까지 ×1.42 로 부풀려 전형 오차를 0 → 0.35 로 망쳤다(10-13 첫 판) -- 누락은 몰린 분에만 있다.
#   «중앙»  = exp(칸별 log(실제/스냅샷) 중앙값) -- 분마다 화면 표시용(조용한 분은 1.0 그대로)
#   «합비율» = 칸별 Σ실제/Σ스냅샷 -- 총량·버스트 크기용(조용한 분도 평균 누락을 싣는다)
N_BINS = [0, 1, 2, 3, 5, 8, 13, 21, 10**9]


def bucket_table(d: pd.DataFrame, B: int, phase_ms: int = 0) -> pd.DataFrame:
    s = snapshot(d, phase_ms).assign(bk=lambda x: x.win // B)
    raw = s.groupby(["symbol", "bk"]).agg(raw=("qty_n", "sum"), n=("qty_n", "size"))
    tru = d.assign(bk=(d.ts_ms + phase_ms) // 1000 // B).groupby(["symbol", "bk"]).qty_n.sum().rename("true")
    out = tru.to_frame().join(raw, how="left").fillna({"raw": 0.0, "n": 0})     # 🔴이름 붙인 색인으로 맞춘다(첫 시도는 어긋났다)
    assert (out.raw <= out.true * (1 + 1e-9)).all(), "스냅샷이 실제보다 클 수 없다"
    return out.assign(b=np.digitize(out.n, N_BINS[1:]))


def fit_bucket(t: pd.DataFrame) -> dict:
    o = t[t.raw > 0]
    lr = np.log(o.true / o.raw)
    return {"median": {int(b): float(np.exp(v)) for b, v in lr.groupby(o.b).median().items()},
            "ratio": {int(b): float(v) for b, v in (o.true.groupby(o.b).sum() / o.raw.groupby(o.b).sum()).items()}}


def apply_bucket(raw: pd.Series, b: pd.Series, table: dict) -> pd.Series:
    last = table[max(table)]
    return raw * b.map(lambda x: table.get(int(x), last))


def score_bucket(t: pd.DataFrame, est: pd.Series) -> dict:
    ok = t.raw > 0
    e = np.abs(np.log(est[ok] / t.true[ok]))
    return {"med": round(float(e.median()), 3), "p90": round(float(e.quantile(.9)), 3),
            "total": round(float(est.sum() / t.true.sum()), 3), "missed_buckets": round(float((~ok).mean()), 4)}


def run(path: str) -> dict:
    a = pd.read_parquet(path)
    by = norm(a[a.venue == "bybit"].reset_index(drop=True))
    cut = by.ts_ms.quantile(0.6)
    tr, te = by[by.ts_ms < cut], by[by.ts_ms >= cut]
    out = {"bybit_rows": len(by), "train_until": str(pd.to_datetime(cut, unit="ms")),
           "capture_by_symbol": {sym: round(float(snapshot(g).qty_n.sum() / g.qty_n.sum()), 3) for sym, g in by.groupby("symbol")}}
    rng = np.random.default_rng(0)
    tables = {}
    for B in (60, 300):
        tables[B] = fit_bucket(bucket_table(tr, B))
        res = {"none": [], "median": [], "ratio": []}
        for ph in [0, *rng.integers(1, 1000, 4)]:                      # 바이낸스 창 위상은 모른다 -- 5가지 평균
            t = bucket_table(te, B, int(ph))
            res["none"].append(score_bucket(t, t.raw))
            for kind in ("median", "ratio"):
                res[kind].append(score_bucket(t, apply_bucket(t.raw, t.b, tables[B][kind])))
        out[f"holdout_{B}s"] = {k: {m: round(float(np.mean([r[m] for r in v])), 3) for m in v[0]} for k, v in res.items()}
    out["tables"] = {f"{B}s": {k: {str(b): round(v, 3) for b, v in t.items()} for k, t in tb.items()} for B, tb in tables.items()}
    out["n_bins"] = N_BINS[:-1]
    ok = norm(a[a.venue == "okx"].reset_index(drop=True))             # OKX 는 정답이 못 된다(최대 4건/초 -- 이미 묶여 온다)
    out["okx_snapshot_capture"] = round(float(snapshot(ok).qty_n.sum() / ok.qty_n.sum()), 3)
    bn = norm(a[a.venue == "binance"].reset_index(drop=True))          # 정답 없음 -- 관측 대비 추정 배율만
    for B in (60, 300):
        s = bn.assign(win=bn.ts_ms // 1000, bk=bn.ts_ms // 1000 // B)
        g = s.groupby(["symbol", "bk"]).agg(raw=("qty_n", "sum"), n=("qty_n", "size"))
        b = pd.Series(np.digitize(g.n, N_BINS[1:]), index=g.index)
        out[f"binance_{B}s_est_over_observed"] = {k: round(float(apply_bucket(g.raw, b, tables[B][k]).sum() / g.raw.sum()), 3)
                                                 for k in ("median", "ratio")}
        out[f"binance_{B}s_n_dist"] = {int(k): int(v) for k, v in b.value_counts().sort_index().items()}
    return out


def selftest() -> None:
    # 한 종목: 0초에 3건(1,2,3) · 1초에 1건(5) → 스냅샷 = [3, 5]
    d = norm(pd.DataFrame({"symbol": "X", "ts_ms": [100, 200, 900, 1500], "qty": [1.0, 2, 3, 5]}))
    s = snapshot(d)
    assert s.qty.tolist() == [3.0, 5.0] and s.win.tolist() == [0, 1]
    assert burst_k(s).tolist() == [1, 2]
    m = fit_m(d)                                   # k=1 칸: 실제 6 / 스냅샷 3 = 2 · k=2 칸: 5/5 = 1
    assert abs(m["m"][0] - 2.0) < 1e-9 and abs(m["m"][1] - 1.0) < 1e-9, m
    est = estimate(s, m)
    assert np.allclose(est, s.qty_n.to_numpy() * [2.0, 1.0])
    sc = score(d, s, est, 60)                      # 한 버킷: 실제 11 = 추정 3·2+5 = 11 → 오차 0
    assert sc["est_med"] == 0.0 and sc["missed"] == 0.0, sc
    s2 = snapshot(d, 500)                          # 위상 500ms: (100,200)→창0, (900,1500)→창1·창2
    assert s2.qty.tolist() == [2.0, 3.0, 5.0], s2
    t = bucket_table(d, 60)                        # 한 분: 실제 11, 스냅샷 3+5=8, n=2
    assert t.true.tolist() == [11 / d.qty.median()] and t.raw.tolist() == [8 / d.qty.median()] and t.n.tolist() == [2]
    tb = fit_bucket(t)
    assert abs(tb["median"][2] - 11 / 8) < 1e-9 and abs(tb["ratio"][2] - 11 / 8) < 1e-9, tb
    assert abs(apply_bucket(t.raw, t.b, tb["ratio"]).iloc[0] - t.true.iloc[0]) < 1e-9
    assert apply_bucket(pd.Series([2.0]), pd.Series([7]), tb["ratio"]).iloc[0] == 2.0 * tb["ratio"][2]   # 없는 칸 = 가장 높은 칸 값
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        r = run(sys.argv[1])
        print(json.dumps(r, ensure_ascii=False, indent=1))
        Path(sys.argv[1]).with_name("undercount_report.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
