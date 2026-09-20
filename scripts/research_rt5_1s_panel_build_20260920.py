"""실시간 5원천(체결/수급/풋프린트 · OI · 청산 · bookTicker · depth)을 UTC 1초 격자 패널로.

입력(전부 서버에서 읽기 전용으로 받은 사본):
  tmp/rt_probe_20260920/export/{trade_tape_1s,oi_1s,tail_risk_1m}.parquet
  data/live/orderflow/bookticker/ETHUSDT/*.bt(.gz)   32B 고정폭
  data/live/orderflow/depthdiff/ETHUSDT/*.jsonl(.gz) 시각별 REST 스냅샷 + diff
출력: tmp/rt_probe_20260920/panel_1s.parquet  (ts_sec 인덱스, 원천마다 접두사)
ponytail: depth 리플레이는 시각 파일 단위로 독립(파일 첫 줄 스냅샷) -- 프로세스 풀로 돌린다.
"""
from __future__ import annotations
import glob, gzip, json, os, struct, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "tmp/rt_probe_20260920/export"
BT_DIR = ROOT / "data/live/orderflow/bookticker/ETHUSDT"
DD_DIR = ROOT / "data/live/orderflow/depthdiff/ETHUSDT"
OUT = ROOT / "tmp/rt_probe_20260920/panel_1s.parquet"
ROW = struct.Struct("<qdfdf")
BANDS_BP = (5, 10, 25, 50)
TICK = 100  # $0.01 -> idx = round(px*100)
IDX_MAX = 100_000 * TICK  # 먼 레벨($26k 매도 등)도 들어온다


def _open(path: str):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


# ── 체결 테이프 ────────────────────────────────────────────────────────────
def trades_1s() -> pd.DataFrame:
    t = pd.read_parquet(EXP / "trade_tape_1s.parquet")
    t["px"] = t.price_bin * 0.1
    t["pxq"] = t.px * (t.buy_qty + t.sell_qty)
    agg = {c: "sum" for c in ["buy_qty", "sell_qty", "buy_n", "sell_n", "pxq"]}
    # 분류 칸은 09-19 03:33 KST 이전이 NULL(«안 갈랐다») -- sum 이 0 으로 바꾸지 않게 min_count=1
    agg.update({c: (lambda x: x.sum(min_count=1)) for c in ["retail_buy_qty", "retail_sell_qty",
                              "whale_buy_qty", "whale_sell_qty", "whale_buy_n", "whale_sell_n",
                              "retail_buy_n", "retail_sell_n", "order_buy_n", "order_sell_n"]})
    agg.update(buy_max="max", sell_max="max", px=["min", "max"])
    g = t.groupby("ts_sec").agg(agg)
    g.columns = ["_".join(c) if c[1] in ("min", "max") and c[0] == "px" else c[0] for c in g.columns]
    g["vwap"] = g.pxq / (g.buy_qty + g.sell_qty)
    g = g.drop(columns="pxq").add_prefix("tr_")
    g.index.name = "ts_sec"
    return g


# ── OI ─────────────────────────────────────────────────────────────────────
def oi_1s() -> pd.DataFrame:
    o = pd.read_parquet(EXP / "oi_1s.parquet").sort_values("ts_ms")
    o["d"] = o.open_interest.diff()
    o["sec"] = o.ts_ms // 1000
    g = o.groupby("sec").agg(oi_last=("open_interest", "last"), oi_d=("d", "sum"), oi_n=("d", "size"))
    g.index.name = "ts_sec"
    return g


# ── 청산(1분) ──────────────────────────────────────────────────────────────
def liq_1m() -> pd.DataFrame:
    l = pd.read_parquet(EXP / "tail_risk_1m.parquet")
    l["ts_min"] = ((l.ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=60)) * 60
    g = l.groupby("ts_min").agg(liq_long=("long_usd_1m", "last"), liq_short=("short_usd_1m", "last"),
                                liq_n=("liq_event_count_1m", "last"), liq_valid=("valid_liq_stream", "last"))
    return g


# ── bookTicker ─────────────────────────────────────────────────────────────
def bt_file(path: str) -> pd.DataFrame:
    with _open(path) as f:
        b = f.read()
    n = (len(b) - 32) // 32
    a = np.frombuffer(b, dtype=np.dtype([("ts", "<i8"), ("bp", "<f8"), ("bq", "<f4"), ("ap", "<f8"), ("aq", "<f4")]),
                      count=n, offset=32)
    df = pd.DataFrame({"sec": a["ts"] // 1000, "bp": a["bp"], "bq": a["bq"].astype("f8"),
                       "ap": a["ap"], "aq": a["aq"].astype("f8")})
    df["mid"] = (df.bp + df.ap) / 2
    df["qi"] = (df.bq - df.aq) / (df.bq + df.aq)
    df["spread_bp"] = (df.ap - df.bp) / df.mid * 1e4
    df["micro_bp"] = ((df.bp * df.aq + df.ap * df.bq) / (df.bq + df.aq) - df.mid) / df.mid * 1e4
    g = df.groupby("sec").agg(bt_mid=("mid", "last"), bt_qi_last=("qi", "last"), bt_qi_mean=("qi", "mean"),
                              bt_spread_bp=("spread_bp", "mean"), bt_micro_bp=("micro_bp", "mean"),
                              bt_bq=("bq", "last"), bt_aq=("aq", "last"), bt_n=("qi", "size"),
                              bt_mid_hi=("mid", "max"), bt_mid_lo=("mid", "min"))
    return g


# ── depth diff 리플레이 ─────────────────────────────────────────────────────
def dd_file(path: str) -> pd.DataFrame:
    bid = np.zeros(IDX_MAX, dtype=np.float64)
    ask = np.zeros(IDX_MAX, dtype=np.float64)
    rows = []
    with _open(path) as f:
        first = json.loads(f.readline())
        snap = first.get("_snapshot") if isinstance(first, dict) else None
        if not snap:
            return pd.DataFrame()
        for p, q in snap["bids"]:
            bid[int(round(float(p) * TICK))] = float(q)
        for p, q in snap["asks"]:
            ask[int(round(float(p) * TICK))] = float(q)
        last_u = snap["lastUpdateId"]
        best = [int(np.flatnonzero(bid)[-1]), int(np.flatnonzero(ask)[0])]  # 전체 스캔은 한 번만
        synced = False
        valid = 1
        cur_sec = None
        acc = dict(add_b=0.0, rem_b=0.0, add_a=0.0, rem_a=0.0, n_msg=0, n_lvl=0)
        for line in f:
            try:
                m = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):  # 크래시로 잘린 줄(끝나지 않은 현재 시각 파일 포함)
                valid = 0
                continue
            if m.get("e") != "depthUpdate":
                continue
            if not synced:
                if m["u"] < last_u:
                    continue
                if not (m["U"] <= last_u <= m["u"]):
                    valid = 0
                synced = True
            elif m["pu"] != last_u:
                valid = 0
            last_u = m["u"]
            sec = m["E"] // 1000
            if cur_sec is None:
                cur_sec = sec
            while sec > cur_sec:  # 초 경계: 직전 초의 상태를 적는다(그 초 마지막 메시지 이후 상태)
                rows.append(_snapshot_row(cur_sec, bid, ask, acc, valid, best))
                acc = dict(add_b=0.0, rem_b=0.0, add_a=0.0, rem_a=0.0, n_msg=0, n_lvl=0)
                cur_sec += 1
            acc["n_msg"] += 1
            for p, q in m["b"]:
                i = int(round(float(p) * TICK)); q = float(q)
                if i >= IDX_MAX: continue
                d = q - bid[i]
                if d > 0: acc["add_b"] += d
                else: acc["rem_b"] -= d
                bid[i] = q
            for p, q in m["a"]:
                i = int(round(float(p) * TICK)); q = float(q)
                if i >= IDX_MAX: continue
                d = q - ask[i]
                if d > 0: acc["add_a"] += d
                else: acc["rem_a"] -= d
                ask[i] = q
            acc["n_lvl"] += len(m["b"]) + len(m["a"])
        if cur_sec is not None:
            rows.append(_snapshot_row(cur_sec, bid, ask, acc, valid, best))
    return pd.DataFrame(rows).set_index("sec") if rows else pd.DataFrame()


def _snapshot_row(sec: int, bid: np.ndarray, ask: np.ndarray, acc: dict, valid: int, best: list) -> dict:
    # 최우선: 직전 최우선 ±2% 창만 본다(10M 배열 전체 스캔 회피). 창에 없으면 전체 스캔.
    w = max(int(best[0] * 0.02), 100)
    nb = np.flatnonzero(bid[max(best[0] - w, 0):best[0] + w + 1])
    bb = (max(best[0] - w, 0) + nb[-1]) if len(nb) else (np.flatnonzero(bid)[-1] if bid.any() else -1)
    na = np.flatnonzero(ask[max(best[1] - w, 0):best[1] + w + 1])
    ba = (max(best[1] - w, 0) + na[0]) if len(na) else (np.flatnonzero(ask)[0] if ask.any() else -1)
    if bb < 0 or ba < 0:
        return dict(sec=sec, dd_valid=0)
    best[0], best[1] = int(bb), int(ba)
    mid = (bb + ba) / 2 / TICK
    r = dict(sec=sec, dd_valid=valid, dd_bb=bb / TICK, dd_ba=ba / TICK,
             dd_add_b=acc["add_b"], dd_rem_b=acc["rem_b"], dd_add_a=acc["add_a"], dd_rem_a=acc["rem_a"],
             dd_n_msg=acc["n_msg"], dd_n_lvl=acc["n_lvl"])
    for bp in BANDS_BP:
        w = int(mid * bp / 1e4 * TICK)
        r[f"dd_bid{bp}"] = bid[max(bb - w, 0):bb + 1].sum()
        r[f"dd_ask{bp}"] = ask[ba:ba + w + 1].sum()
    # 벽: ±50bp 안 최대 단일 레벨
    w = int(mid * 50 / 1e4 * TICK)
    sb = bid[max(bb - w, 0):bb + 1]; sa = ask[ba:ba + w + 1]
    r["dd_wall_b"] = sb.max(); r["dd_wall_b_bp"] = (len(sb) - 1 - sb.argmax()) / TICK / mid * 1e4
    r["dd_wall_a"] = sa.max(); r["dd_wall_a_bp"] = sa.argmax() / TICK / mid * 1e4
    return r


def main() -> None:
    print("trades…", flush=True); tr = trades_1s()
    print("oi…", flush=True); oi = oi_1s()
    print("liq…", flush=True); lq = liq_1m()
    bt_files = sorted(glob.glob(str(BT_DIR / "*.bt*")))
    dd_files = sorted(glob.glob(str(DD_DIR / "*.jsonl*")))
    with ProcessPoolExecutor(max_workers=8) as ex:
        print(f"bookticker {len(bt_files)} files…", flush=True)
        bt = pd.concat(list(ex.map(bt_file, bt_files)))
        bt = bt[~bt.index.duplicated(keep="last")]
        print(f"depth {len(dd_files)} files…", flush=True)
        dd = pd.concat([d for d in ex.map(dd_file, dd_files) if len(d)])
        dd = dd[~dd.index.duplicated(keep="last")]
    lo = int(max(tr.index.min(), bt.index.min())); hi = int(min(tr.index.max(), bt.index.max()))
    idx = pd.RangeIndex(lo, hi + 1, name="ts_sec")
    panel = pd.DataFrame(index=idx)
    panel = panel.join(tr).join(bt).join(dd).join(oi)
    panel["ts_min"] = (panel.index // 60) * 60
    panel = panel.join(lq, on="ts_min").drop(columns="ts_min")
    for c in ["tr_buy_qty", "tr_sell_qty", "tr_buy_n", "tr_sell_n"]:
        panel[c] = panel[c].fillna(0.0)
    panel["oi_last"] = panel.oi_last.ffill(limit=30)
    panel.to_parquet(OUT)
    print("rows", len(panel), "cols", panel.shape[1], "->", OUT, flush=True)
    print(panel.notna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
