#!/usr/bin/env python3
"""앵커 이후 **되돌림 vs 지속** 방향 라벨 — 재설계판 (2026-09-07).

사용자: *"증거신호는 앵커로서 역할을 잘 하고 있어. 이제 이걸 가지고 y_cont_H* 를 만들어서
예측 모델을 만들어보자."* → *"우선 이 방향의 라벨 생성 로직부터 다시 만들어야 해."*

## 왜 다시 만드는가 -- 기존 `y_cont_H*`(MFE 비교)의 결함 3가지
1. ⭐**순서를 안 본다.** `mfe_cont > mfe_fade` 는 "어느 쪽이 더 멀리 갔나"다. 위로 2% 갔다가
   아래로 2.1% 간 경로와 그 반대가 **같은 라벨**이 된다. 실제 거래는 정반대인데.
2. **미결정을 억지로 이진화한다.** 양쪽 다 거의 안 움직인 봉도 0/1 중 하나로 밀어넣는다.
   부록 E에서 확인된 대로 앵커 이후 편측성 분포는 무작위와 같으므로, 이 억지 이진화가
   라벨 잡음의 큰 몫이다.
3. **5분봉 해상도로는 순서를 못 가린다.** 한 5분봉이 양쪽 배리어를 다 건드리면 어느 쪽이
   먼저인지 알 수 없다 -- 그런데 MFE 비교는 그 사실 자체를 숨긴다.

## 새 설계 -- **터치 시각을 원시로 저장하고 라벨은 유도한다**
진입 `open[t+1]`(앵커 봉 마감에 주문 -> 다음 봉 시가, L4 계약). 진입가 기준 **대칭** 배리어
`entry x (1 ± K·atr_pct)`. **1분봉으로** 두 배리어의 첫 터치 시각을 각각 찾는다.

  hit_cont_min_K{K}  지속 방향 배리어 첫 터치까지 분  (없으면 NaN)
  hit_fade_min_K{K}  페이드 방향 배리어 첫 터치까지 분 (없으면 NaN)
  ambig_K{K}         같은 1분봉에서 양쪽 동시 터치 (순서 판정 불가)

이 셋만 있으면 어떤 (K, H) 조합의 라벨도 재유도된다:
  y_order(K,H) = 1  hit_cont < hit_fade 이고 hit_cont <= H*5분
               = 0  hit_fade < hit_cont 이고 hit_fade <= H*5분
               = NaN  둘 다 미터치(미결정) 또는 ambig
⭐**대칭 배리어가 기본**이다. 09-06 `y_order`는 페이드 2.0ATR vs 지속 1.5ATR 비대칭이라
그 자체로 한쪽에 유리했다 -- 비대칭은 별도 변형으로만 둔다.

## 5분 vs 1분 파리티 (게이트 L2 정신)
같은 라벨을 5분봉 고가/저가로도 만들어 **불일치율**을 보고한다. 5분봉이 양쪽을 다 건드린
비율이 곧 "5분 해상도로는 못 가리는 비율"이고, 그 구간에서 5분 라벨은 신뢰할 수 없다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT = ROOT / "tmp/eth_anchor_direction_labels_20260907"
K_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)          # ATR 배수 배리어 (변동성 정규화)
PCT_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)       # ⭐고정 % 배리어 (사용자 지정: ±1% 가 기본)
H_GRID = (12, 24, 48, 96)          # 5분봉 수
MAX_MIN = max(H_GRID) * 5          # 스캔 상한 (분)
CHUNK = 4000


def load_1m() -> pd.DataFrame:
    df = pd.read_csv(KL1, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def first_touch(hi1: np.ndarray, lo1: np.ndarray, start: np.ndarray,
                up: np.ndarray, dn: np.ndarray, max_min: int = MAX_MIN):
    """1분봉 start 부터 max_min 분 동안 up(고가 돌파)·dn(저가 이탈) 첫 터치 분 인덱스.

    반환 (t_up, t_dn, ambig) -- 미터치는 -1, ambig 는 같은 분봉에서 양쪽 동시 터치."""
    n = len(start)
    t_up = np.full(n, -1, np.int32); t_dn = np.full(n, -1, np.int32)
    amb = np.zeros(n, bool)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(max_min)[None, :]
        H_ = hi1[idx]; L_ = lo1[idx]
        hu = H_ >= up[a:b, None]
        hd = L_ <= dn[a:b, None]
        au = hu.any(axis=1); ad = hd.any(axis=1)
        iu = np.where(au, hu.argmax(axis=1), -1)
        idn = np.where(ad, hd.argmax(axis=1), -1)
        t_up[a:b] = iu; t_dn[a:b] = idn
        amb[a:b] = au & ad & (iu == idn)
    return t_up, t_dn, amb


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/4] 앵커·5분봉 로드 ...", flush=True)
    D = pd.read_parquet(ROOT / "tmp/eth_anchor_label_dataset_20260907/anchors_labels.parquet")
    eth = B._load_kl(B.ETH_KL); btc = B._load_kl(B.BTC_KL); fund = B._load_funding()
    tmax = min(eth["timestamp"].max(), btc["timestamp"].max(), fund["calc_time"].max())
    eth = eth[eth["timestamp"] <= tmax].reset_index(drop=True)
    ts5 = eth["timestamp"].to_numpy()
    O5, H5, L5 = (eth[c].to_numpy(float) for c in ("open", "high", "low"))
    assert (D["timestamp"].to_numpy() == ts5[D["bar_idx"].to_numpy()]).all(), "bar_idx 정합 실패"

    print("[2/4] 1분봉 정렬 ...", flush=True)
    m1 = load_1m()
    ts1 = m1["timestamp"].to_numpy()
    H1, L1 = m1["high"].to_numpy(float), m1["low"].to_numpy(float)
    entry_ts = ts5[np.minimum(D["bar_idx"].to_numpy() + 1, len(ts5) - 1)]
    start = np.searchsorted(ts1, entry_ts)
    ok = (start < len(ts1) - MAX_MIN) & (ts1[np.minimum(start, len(ts1) - 1)] == entry_ts)
    print(f"      앵커 {len(D):,} · 1분봉 정렬 성공 {ok.sum():,} ({ok.mean()*100:.1f}%)", flush=True)

    entry = O5[np.minimum(D["bar_idx"].to_numpy() + 1, len(O5) - 1)]
    atr = D["atr_pct"].to_numpy(float)
    fade_up = (D["side"].to_numpy() == "bottom")        # bottom 페이드=롱 -> 유리=상승
    res = {"ok_1m": ok}
    for K in K_GRID:
        up = entry * (1 + K * atr); dn = entry * (1 - K * atr)
        s = np.where(ok, start, 0)
        tu, td, amb = first_touch(H1, L1, s, up, dn)
        tu = np.where(ok & (tu >= 0), tu, -1); td = np.where(ok & (td >= 0), td, -1)
        # 측면 정렬: 페이드가 상승이면 fade=up 터치, cont=dn 터치
        tf = np.where(fade_up, tu, td); tc = np.where(fade_up, td, tu)
        res[f"hit_fade_min_K{K:g}"] = np.where(tf >= 0, tf, np.nan)
        res[f"hit_cont_min_K{K:g}"] = np.where(tc >= 0, tc, np.nan)
        res[f"ambig_K{K:g}"] = amb & ok
        # 5분봉 해상도 대조 (같은 배리어, 5분 고가/저가)
        s5 = D["bar_idx"].to_numpy() + 1
        o5 = (s5 + MAX_MIN // 5) < len(H5)
        tu5 = np.full(len(D), -1); td5 = np.full(len(D), -1); amb5 = np.zeros(len(D), bool)
        i5 = np.flatnonzero(o5)
        for a in range(0, len(i5), CHUNK):
            b = min(a + CHUNK, len(i5)); j = i5[a:b]
            idx = s5[j][:, None] + np.arange(MAX_MIN // 5)[None, :]
            hu = H5[idx] >= up[j][:, None]; hd = L5[idx] <= dn[j][:, None]
            au = hu.any(axis=1); ad = hd.any(axis=1)
            iu = np.where(au, hu.argmax(axis=1), -1); idn = np.where(ad, hd.argmax(axis=1), -1)
            tu5[j] = iu; td5[j] = idn; amb5[j] = au & ad & (iu == idn)
        tf5 = np.where(fade_up, tu5, td5); tc5 = np.where(fade_up, td5, tu5)
        res[f"hit_fade_bar5_K{K:g}"] = np.where(tf5 >= 0, tf5, np.nan)
        res[f"hit_cont_bar5_K{K:g}"] = np.where(tc5 >= 0, tc5, np.nan)
        res[f"ambig5_K{K:g}"] = amb5 & o5
        print(f"      K={K:g} 완료", flush=True)

    # ⭐고정 % 배리어 (사용자 지정 2026-09-07: 등락폭 ±1%). ATR 정규화가 없으므로 배리어 폭이
    # 항상 PCT*100 bp -- 왕복 비용 10bp 대비 충분히 크다(±1% = 100bp = 비용의 10배).
    # 대신 미결정 비율이 그 구간 변동성과 상관된다 -> atr_pct 는 인과 피쳐로 남겨 모델이 쓰게 한다.
    for PC in PCT_GRID:
        up = entry * (1 + PC / 100); dn = entry * (1 - PC / 100)
        s = np.where(ok, start, 0)
        tu, td, amb = first_touch(H1, L1, s, up, dn)
        tu = np.where(ok & (tu >= 0), tu, -1); td = np.where(ok & (td >= 0), td, -1)
        tf = np.where(fade_up, tu, td); tc = np.where(fade_up, td, tu)
        res[f"hit_fade_min_P{PC:g}"] = np.where(tf >= 0, tf, np.nan)
        res[f"hit_cont_min_P{PC:g}"] = np.where(tc >= 0, tc, np.nan)
        res[f"ambig_P{PC:g}"] = amb & ok
        s5 = D["bar_idx"].to_numpy() + 1
        o5 = (s5 + MAX_MIN // 5) < len(H5)
        tu5 = np.full(len(D), -1); td5 = np.full(len(D), -1); amb5 = np.zeros(len(D), bool)
        i5 = np.flatnonzero(o5)
        for a in range(0, len(i5), CHUNK):
            b = min(a + CHUNK, len(i5)); j = i5[a:b]
            idx = s5[j][:, None] + np.arange(MAX_MIN // 5)[None, :]
            hu = H5[idx] >= up[j][:, None]; hd = L5[idx] <= dn[j][:, None]
            au = hu.any(axis=1); ad = hd.any(axis=1)
            iu = np.where(au, hu.argmax(axis=1), -1); idn = np.where(ad, hd.argmax(axis=1), -1)
            tu5[j] = iu; td5[j] = idn; amb5[j] = au & ad & (iu == idn)
        tf5 = np.where(fade_up, tu5, td5); tc5 = np.where(fade_up, td5, tu5)
        res[f"hit_fade_bar5_P{PC:g}"] = np.where(tf5 >= 0, tf5, np.nan)
        res[f"hit_cont_bar5_P{PC:g}"] = np.where(tc5 >= 0, tc5, np.nan)
        res[f"ambig5_P{PC:g}"] = amb5 & o5
        print(f"      P={PC:g}% 완료", flush=True)

    L = pd.concat([D[["timestamp", "bar_idx", "side", "anchor", "n_signals", "signals", "atr_pct", "split"]]
                   .reset_index(drop=True), pd.DataFrame(res)], axis=1)
    L.to_parquet(OUT / "direction_labels.parquet", index=False)

    print("\n[3/4] ⭐5분 vs 1분 순서 판정 불일치 (게이트 L2 정신)", flush=True)
    rep = []
    for K in list(K_GRID) + [f"P{x:g}" for x in PCT_GRID]:
        tagk = f"K{K:g}" if not isinstance(K, str) else K
        for H in H_GRID:
            lim = H * 5
            def lab(tag):
                f = L[f"hit_fade_{tag}_{tagk}"].to_numpy(float) * (1 if tag == "min" else 5)
                c = L[f"hit_cont_{tag}_{tagk}"].to_numpy(float) * (1 if tag == "min" else 5)
                f = np.where(np.isfinite(f) & (f < lim), f, np.inf)
                c = np.where(np.isfinite(c) & (c < lim), c, np.inf)
                y = np.where(np.isinf(f) & np.isinf(c), np.nan, (c < f).astype(float))
                return y
            y1, y5 = lab("min"), lab("bar5")
            amb5 = L[f"ambig5_{tagk}"].to_numpy(bool)
            m = np.isfinite(y1) & np.isfinite(y5)
            rep.append({"배리어": tagk, "H": H, "결정비율": float(np.isfinite(y1).mean()),
                        "지속비율": float(np.nanmean(y1)), "5분양쪽동시": float(amb5.mean()),
                        "5분vs1분 불일치": float((y1[m] != y5[m]).mean()), "n": int(m.sum())})
    R = pd.DataFrame(rep)
    print(R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    R.to_csv(OUT / "parity_5m_vs_1m.csv", index=False)

    print("\n[4/4] 저장:", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
