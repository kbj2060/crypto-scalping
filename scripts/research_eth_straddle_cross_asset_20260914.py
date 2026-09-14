"""**스트래들 규칙 타자산 이식** — 셀을 고른 다중검정을 자산으로 갚는다 (2026-09-14).

ETH 에서 「익절 +10% / 손절 −3% 양측 동시진입을, rv48 최저 5분위에서만」이 5창 전부 양수였다.
그런데 그 셀은 6셀 × 5분위를 훑어 고른 것이고 블록순열 귀무 통과율이 3.0% 다 -- 훑은 수를 감안하면
p ≈ 0.18 이라 그 자체로는 증거가 못 된다. **아무것도 다시 맞추지 않고 규칙 그대로** BTC·SOL·XRP 에
옮긴다. 이식은 이 저장소가 쓸 수 있는 유일한 진짜 표본외다(HOLDOUT 은 소진됐다).

이식되는 것: 배리어 (U,D) · 게이트 컬럼(rv48) · 게이트 분위(최저 20%) · 비용 3조각 · 슬리피지 ·
같은 다섯 창. 이식되지 **않는** 것: 분위 경계값(자산마다 변동성 수준이 다르므로 그 자산의 TRAIN 에서
다시 자른다 -- 이건 적합이 아니라 단위 환산이다).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_direction_barrier_label_20260914 as B  # noqa: E402
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402

WINS = ("BACK22_23", "TRAIN", "VAL", "OOS", "TEST")


def load(sym: str):
    f = ROOT / f"binance_data/klines/{sym}/{sym}-5m-api.csv"
    d = pd.read_csv(f, usecols=["timestamp", "high", "low", "close", "quote_volume", "trades"],
                    parse_dates=["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    c = d.close.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.nan)
    rv48 = pd.Series(lr).rolling(48, min_periods=40).std().to_numpy()     # 봉 t 까지만 (인과)
    rv12 = pd.Series(lr).rolling(12, min_periods=10).std().to_numpy()
    ratio = np.log(np.clip(rv12, 1e-9, None)) - np.log(np.clip(rv48, 1e-9, None))
    return d, c, d.high.to_numpy(float), d.low.to_numpy(float), rv48, ratio


def volfc_values(sym: str, d, c, hi, lo, artdir: pathlib.Path) -> np.ndarray:
    """그 자산 **자신의** 홀드아웃 변동성 모델로 낸 전방 변동성 예측(로그)."""
    import joblib
    import live_eth_sizing_vol_model_20260912 as svm
    art = joblib.load(artdir / f"vol_model_holdout2223_{sym}.joblib")
    assert art["symbol"] == sym, f"아티팩트 자산 불일치: {art['symbol']} vs {sym}"
    X = svm.build_features(d["timestamp"], c, d["quote_volume"].to_numpy(float),
                           d["trades"].to_numpy(float), hi, lo)
    return np.log(np.clip(svm.predict_vol(art["models"], X), 1e-9, None))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT")
    ap.add_argument("--up", type=float, default=0.10)
    ap.add_argument("--down", type=float, default=0.03)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--gate", default="rv48", choices=("rv48", "rvratio", "volfc"))
    ap.add_argument("--side", default="low", choices=("low", "high"))
    ap.add_argument("--scale-barriers", action="store_true",
                    help="배리어를 자산 변동성으로 환산(ETH 기준). 게이트 문턱만 환산하고 배리어는 "
                         "절대%로 두면 자산마다 «다른 배리어»를 비교하는 셈이다")
    ap.add_argument("--tag", default="cross_asset")
    a = ap.parse_args()
    out = {}
    ref_vol = None
    if a.scale_barriers:          # ETH 의 학습창 중앙 예측변동성을 1.0 으로 잡는다
        dE, cE, hE, lE, _, _ = load("ETHUSDT")
        tsE = dE.timestamp.to_numpy()
        mE = (tsE >= np.datetime64("2024-03-01")) & (tsE <= np.datetime64("2025-08-31T23:59:59"))
        ref_vol = float(np.nanmedian(np.exp(volfc_values(
            "ETHUSDT", dE, cE, hE, lE, ROOT / "data/research/eth_direction_barrier_label_20260914")[mE])))
        print(f"기준 ETH 학습창 중앙 예측변동성 {ref_vol:.5f} — 배리어를 이 비로 환산한다")
    print(f"규칙: 익절 +{a.up*100:g}% / 손절 −{a.down*100:g}% · {a.gate} "
          f"{'최저' if a.side == 'low' else '최고'} 5분위(그 자산 TRAIN 기준)")
    for sym in a.symbols.split(","):
        d, c, hi, lo, rv48, ratio = load(sym)
        rv = (rv48 if a.gate == "rv48" else
              ratio if a.gate == "rvratio" else
              volfc_values(sym, d, c, hi, lo, ROOT / "data/research/eth_direction_barrier_label_20260914"))
        ts = d.timestamp.to_numpy()
        win = {}
        for k, (s0, s1) in P.WINDOWS.items():
            win[k] = (int(np.searchsorted(ts, np.datetime64(s0))),
                      int(np.searchsorted(ts, np.datetime64(s1 + "T23:59:59"))))
        ok = np.isfinite(rv) & np.isfinite(rv48)
        import sys as _s; _s.path.insert(0, str(ROOT / "scripts"))
        tl, th = win["TRAIN"]
        up, down = a.up, a.down
        if ref_vol is not None:
            mtr = (ts >= np.datetime64("2024-03-01")) & (ts <= np.datetime64("2025-08-31T23:59:59"))
            k = float(np.nanmedian(np.exp(volfc_values(
                sym, d, c, hi, lo, ROOT / "data/research/eth_direction_barrier_label_20260914")[mtr]))) / ref_vol
            up, down = a.up * k, a.down * k
            print(f"  변동성비 {k:.3f} ⇒ 배리어 +{up*100:.2f}% / −{down*100:.2f}%")
        pct = 20 if a.side == "low" else 80
        edge = float(np.nanpercentile(rv[tl:th], pct))
        print(f"\n{sym} · {a.gate} {'최저20' if a.side=='low' else '최고80'}% 경계 {edge:.5f}")
        print(f"{'창':>10} {'n(전체)':>8} {'전체bp':>8} | {'n(게이트)':>9} {'게이트bp':>9} {'블록t':>7} "
              f"{'중앙h':>7} {'연쌍수':>7}")
        cell = {}
        for w in WINS:
            L = B.label_window(c, hi, lo, ok, *win[w], a.stride, up, down)
            if len(L["idx"]) < 100:
                continue
            g = (rv[L["idx"]] <= edge) if a.side == "low" else (rv[L["idx"]] >= edge)
            m, bars = L["m"], L["bars"]
            if g.sum() < 30:
                continue
            mm, bb = m[g], float(np.median(bars[g]))
            blk = (L["idx"][g] // max(int(bb), 1)).astype(np.int64)
            bm = np.array([mm[blk == k].mean() for k in np.unique(blk)])
            t = float(bm.mean() / (bm.std(ddof=1) / np.sqrt(len(bm)))) if len(bm) > 2 and bm.std(ddof=1) > 0 else float("nan")
            hrs = bb * 5 / 60
            cell[w] = {"n_all": int(len(m)), "all_bp": float(m.mean()), "n_low": int(g.sum()),
                       "low_bp": float(mm.mean()), "block_t": t, "hours": hrs,
                       "blocks": int(len(bm)), "pairs_per_year": 8760 / hrs if hrs > 0 else None}
            q = cell[w]
            print(f"{w:>10} {q['n_all']:>8,} {q['all_bp']:>+8.1f} | {q['n_low']:>9,} "
                  f"{q['low_bp']:>+9.1f} {t:>+7.2f} {hrs:>7.1f} {q['pairs_per_year']:>7.0f}", flush=True)
        ok5 = all(w in cell and cell[w]["low_bp"] > 0 for w in WINS)
        print(f"  ⇒ 5창 전부 양수: {ok5}")
        out[sym] = {"edge": edge, "up": up, "down": down, "windows": cell, "pass_5win": ok5}
    p = ROOT / "data/research/eth_direction_barrier_label_20260914" / f"{a.tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(f"\n통과 자산 {sum(v['pass_5win'] for v in out.values())}/{len(out)} · 저장: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
