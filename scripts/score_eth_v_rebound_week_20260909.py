#!/usr/bin/env python3
"""V자반등(특화 감지기 = 이벤트 트리거) 칩을 **최근 N일 전 봉**에 재구성한다. GPU/TabPFN 필요.

라이브(`live_eth_sweep_v_rebound_signal_20260829.compute_eth_sweep_v_rebound_signal`)는 화면
표시에 필요한 최근 61봉만 채점한다. 차트용으로 같은 함수/같은 동결 문맥/같은 임계(0.60)를
그대로 쓰되 **채점 창만 N일로 넓힌다**. 예측 비용은 문맥 크기가 지배하므로(314행이 1행과
같은 시간이었던 실측) 창을 넓혀도 사실상 공짜다.

⚠️라이브는 1500봉(5.2일) 한 번 호출로 도는데 7일을 채점하려면 페이징이 필요하다. 롤링 창은
전부 864봉 이하라 더 긴 프레임에서 계산해도 같은 값이 나온다.

출력: tmp/eth_signal_map_20260909/v_rebound_scores.csv (봉별 확률/방향/콜/톤 + 콜 지속구간)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import live_eth_sweep_v_rebound_signal_20260829 as VR   # noqa: E402

OUT = ROOT / "tmp/eth_signal_map_20260909"
KL = "https://fapi.binance.com/fapi/v1/klines"


def page(sym: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    out, cur = [], start_ms
    while cur < end_ms:
        r = requests.get(KL, params={"symbol": sym, "interval": "5m", "limit": 1500,
                                     "startTime": cur}, timeout=20)
        r.raise_for_status(); dd = r.json()
        if not dd: break
        out += dd; cur = dd[-1][0] + 1
        if len(dd) < 1500: break
        time.sleep(0.12)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "trades",
            "taker_buy_base", "tq", "ignore"]
    df = pd.DataFrame(out, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[df["close_time"] < int(time.time() * 1000)]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=float, default=7.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    now = int(time.time() * 1000)
    t0 = now - int(a.days * 86400 * 1000)
    warm = 1100 * 300_000
    print(f"[1/3] klines 수집 (최근 {a.days}일 + 웜업)…", flush=True)
    kl = page("ETHUSDT", t0 - warm, now); btc = page("BTCUSDT", t0 - warm, now)
    print(f"  5분봉 {len(kl):,} ({kl.timestamp.iloc[0]} ~ {kl.timestamp.iloc[-1]})", flush=True)

    print("[2/3] 피쳐/트리거…", flush=True)
    frame = VR._build_features(kl)
    sig = VR.compute_signals(kl, btc_df=btc, funding_df=None)   # 라이브와 동일(펀딩 미사용)
    n_tail = int(a.days * 288) + VR.BADGE_HORIZON_BARS + 2
    cand = VR._every_bar_rows(frame, sig, n_tail)
    cand = cand.merge(frame[["timestamp"] + [c for c in VR.FEATURES if c not in
                      ("is_downside", "sweep_penetration_atr", "flow_aligned_delta_z")]],
                      on="timestamp", how="left").dropna(subset=VR.FEATURES)
    print(f"  채점 행 {len(cand):,} (봉 {cand.pos.nunique():,} × 양방향)", flush=True)

    print("[3/3] TabPFN 채점…", flush=True)
    train = VR._load_train_context()
    from tabpfn import TabPFNClassifier
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TabPFNClassifier(device=dev, random_state=20260829, ignore_pretraining_limits=True)
    t = time.time(); clf.fit(train[VR.FEATURES], train["label"].to_numpy())
    proba = clf.predict_proba(cand[VR.FEATURES])[:, 1]
    print(f"  {dev} · 문맥 {len(train):,}행 · {time.time()-t:.1f}s", flush=True)
    cand = cand.assign(proba=proba)

    best: dict[int, dict] = {}
    for row in cand.itertuples():
        p = int(row.pos)
        if p not in best or row.proba > best[p]["proba"]:
            best[p] = {"t0": frame["timestamp"].iloc[p], "proba": float(row.proba),
                       "direction": "down" if int(row.is_downside) == 1 else "up",
                       "triggers": row.triggers or ""}
    last_pos = len(frame) - 1
    spans = VR._call_spans(best, frame, last_pos, VR.PROBA_THRESHOLD)
    span_end = {s: e for s, e, _ in spans}

    rows = []
    for p in sorted(best):
        b = best[p]
        call = "rebound" if b["proba"] >= VR.PROBA_THRESHOLD else "continuation"
        rows.append(dict(bar_idx=p, timestamp=b["t0"], direction=b["direction"],
                         proba=b["proba"], call=call,
                         tone=VR._predicted_tone(b["direction"], call),
                         is_call=p in span_end,
                         span_end_ts=frame["timestamp"].iloc[span_end[p]] if p in span_end else pd.NaT,
                         triggers=b["triggers"]))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v_rebound_scores.csv", index=False)
    nc = int(D.is_call.sum())
    print(f"\n콜(임계 {VR.PROBA_THRESHOLD}) {nc}건 / 채점봉 {len(D):,} "
          f"· 바닥콜(되돌림 롱) {int(((D.is_call)&(D.direction=='down')).sum())} "
          f"· 천장콜(되돌림 숏) {int(((D.is_call)&(D.direction=='up')).sum())}")
    print(json.dumps({"done": True, "bars": len(D), "calls": nc}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
