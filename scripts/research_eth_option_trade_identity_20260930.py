"""ETH 옵션 체결의 «누가»와 «의도(거래 구조)» -- 공개 데이터로 어디까지 되나 (2026-09-30 연구).

질문: 대시보드 옵션 카드의 방향(테이커)·블록 한 줄이 «누가 산 건지, 어떤 의도인지» 모호하다는 사용자 지적.
  ① Deribit 공개 API 가 실제로 돌려주는 식별 필드 전수(REST·WS·블록 RFQ·combo) -- 문서가 아니라 응답으로
  ② 우리 수집기(option_trades)가 버리는 필드
  ③ 블록·combo 다리 → 구조 분류(규칙) + Deribit 자체 combo 유형 코드로 검증 + 순델타/순베가
  ④ «누가»의 대리 지표(블록/화면 · 크기 · mark 대비 불리함 · liquidation · 같은 ms 쓸기) 분포

데이터: 서버 option_trades(09-27~, 서버에서 read_only COPY → tmp/option_trade_identity_20260930/eth_option_trades.parquet)
      + history.deribit.com 30일 백필(일반 API 는 ~24h 만 보존). 🔴바이낸스 호출 없음.

  python scripts/research_eth_option_trade_identity_20260930.py --probe       # API 필드 전수(네트워크, ~1분)
  python scripts/research_eth_option_trade_identity_20260930.py --history 30  # history 백필(캐시)
  python scripts/research_eth_option_trade_identity_20260930.py               # 분석 → report.json
  python scripts/research_eth_option_trade_identity_20260930.py --selftest    # 구조 분류 규칙 assert(네트워크 없음)
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/option_trade_identity_20260930"
API = "https://www.deribit.com/api/v2/public/"
HIST = "https://history.deribit.com/api/v2/public/"
YEAR_MS = 365.0 * 86400e3


def get(base: str, ep: str) -> dict:
    with urllib.request.urlopen(base + ep, timeout=30) as r:
        return json.load(r)


# ── 구조 분류 ─────────────────────────────────────────────────────────────────
def parse_inst(name: str) -> dict:
    """'ETH-2OCT26-2740-C' → 옵션, 'ETH-PERPETUAL'/'ETH-16OCT26' → 선물."""
    p = name.split("-")
    exp = None if p[1] == "PERPETUAL" else datetime.strptime(p[1], "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
    if len(p) == 4:
        return {"kind": p[3], "exp": exp, "strike": float(p[2].replace("d", "."))}
    return {"kind": "F", "exp": exp, "strike": None}


def classify(legs: list[tuple[str, float]]) -> str:
    """legs = [(instrument, 부호 있는 수량 +매수/−매도), ...] (테이커 관점). 선물 다리는 «+hedge» 꼬리표."""
    net: dict = collections.defaultdict(float)
    for inst, q in legs:
        net[inst] += q
    o = [dict(parse_inst(i), q=q) for i, q in net.items() if abs(q) > 1e-12]
    fut = [x for x in o if x["kind"] == "F"]
    o = sorted((x for x in o if x["kind"] != "F"), key=lambda x: (x["exp"], x["kind"], x["strike"]))
    tag = "+hedge" if fut else ""
    if not o:
        return "future_only"
    return _opt(o) + tag


def _opt(o: list[dict]) -> str:
    n, exps, kinds = len(o), {x["exp"] for x in o}, {x["kind"] for x in o}
    s = [1 if x["q"] > 0 else -1 for x in o]
    q = [abs(x["q"]) for x in o]
    eq = lambda a, b: abs(a - b) <= 1e-9 * max(a, b)
    if n == 1:
        return "single"
    if n == 2 and len(exps) == 1:
        if kinds == {"C", "P"}:
            c, p = (o[0], o[1]) if o[0]["kind"] == "C" else (o[1], o[0])
            if (c["q"] > 0) == (p["q"] > 0):
                return "straddle" if c["strike"] == p["strike"] else "strangle"
            return "synthetic" if c["strike"] == p["strike"] else "risk_reversal"
        if s[0] == s[1]:
            return "other"
        return "vertical" if eq(q[0], q[1]) else "ratio_spread"
    if n == 2:  # 만기 둘
        if len(kinds) == 1 and s[0] != s[1] and eq(q[0], q[1]):
            return "calendar" if o[0]["strike"] == o[1]["strike"] else "diagonal"
        return "other"
    if len(exps) != 1:
        return "other"
    if n == 3 and len(kinds) == 1:
        if s[0] == s[2] != s[1] and eq(q[0], q[2]) and eq(q[1], q[0] + q[2]):
            return "butterfly"
        if eq(q[0], q[1]) and eq(q[1], q[2]) and (s[0] != s[1] == s[2] or s[0] == s[1] != s[2]):
            return "ladder"
        return "other"
    if n == 4 and all(eq(x, q[0]) for x in q):
        if len(kinds) == 1:
            return "condor" if s[0] == s[3] != s[1] == s[2] else "other"
        puts = [x for x in o if x["kind"] == "P"]
        calls = [x for x in o if x["kind"] == "C"]
        if len(puts) != 2:
            return "other"
        ks = {x["strike"] for x in o}
        if len(ks) == 2 and puts[0]["strike"] == calls[0]["strike"]:
            lo_c, lo_p = calls[0]["q"] > 0, puts[0]["q"] > 0
            return "box" if lo_c != lo_p and (calls[1]["q"] > 0) != lo_c else "other"
        if (puts[0]["q"] > 0) != (puts[1]["q"] > 0) and (calls[0]["q"] > 0) != (calls[1]["q"] > 0) \
                and (puts[1]["q"] > 0) == (calls[0]["q"] > 0):
            if puts[1]["strike"] == calls[0]["strike"]:
                return "iron_butterfly"
            return "iron_condor" if puts[1]["strike"] < calls[0]["strike"] else "other"
    return "other"


# Deribit 자체 combo 유형 코드(get_combo_details 로 확인한 다리 정의) → 우리 라벨. 검증용 정답지.
DERIBIT_CODE = {"CS": "vertical", "PS": "vertical", "STRD": "straddle", "STRG": "strangle", "GUTS": "strangle",
                "RR": "risk_reversal", "RRITM": "risk_reversal", "REV": "synthetic", "CCAL": "calendar",
                "PCAL": "calendar", "CDIAG": "diagonal", "PDIAG": "diagonal", "CBUT": "butterfly",
                "PBUT": "butterfly", "IBUT": "iron_butterfly", "ICOND": "iron_condor", "CCOND": "condor",
                "PCOND": "condor", "CLAD": "ladder", "PLAD": "ladder", "BOX": "box", "FS": "future_only",
                "CSR12": "ratio_spread", "PSR12": "ratio_spread", "CSR13": "ratio_spread",
                "PSR13": "ratio_spread", "CSR23": "ratio_spread", "PSR23": "ratio_spread"}


# ── 그릭스(r=0, 지수를 선도 대용) ──────────────────────────────────────────────
def bs_delta_vega(spot: float, k: float, iv_pct: float, yrs: float, call: bool) -> tuple[float, float]:
    """(델타, 베가 USD/1 vol pt/1 ETH). 🔴역옵션(ETH 표시)의 코인 델타가 아니라 USD 델타 -- 방향 부호용."""
    sig = iv_pct / 100.0
    if spot <= 0 or k <= 0 or sig <= 0 or yrs <= 0:
        return 0.0, 0.0
    d1 = (math.log(spot / k) + 0.5 * sig * sig * yrs) / (sig * math.sqrt(yrs))
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return (nd1 if call else nd1 - 1.0), spot * pdf * math.sqrt(yrs) / 100.0


def greeks(legs: list[dict]) -> dict:
    """legs: instrument, q(부호), iv, index_price, ts_ms, price(선물이면 USD 가격). 선물 amount 는 USD → ETH 환산."""
    nd = gd = nv = gv = 0.0
    ivs = []
    for l in legs:
        p = parse_inst(l["instrument"])
        if p["kind"] == "F":
            d = l["q"] / float(l["price"])       # 선물 amount = USD 명목
            nd += d
            gd += abs(d)
            continue
        yrs = (p["exp"].timestamp() * 1e3 - l["ts_ms"]) / YEAR_MS
        de, ve = bs_delta_vega(l["index_price"], p["strike"], l["iv"], yrs, p["kind"] == "C")
        nd, gd, nv, gv = nd + l["q"] * de, gd + abs(l["q"] * de), nv + l["q"] * ve, gv + abs(l["q"] * ve)
        ivs.append(l["iv"])
    spot = legs[0]["index_price"]
    iv = sum(ivs) / len(ivs) if ivs else 0.0
    # ponytail: 하루 1σ 가격 손익 vs 하루 vol 변동(3 vol pt 가정) 손익 비 -- 척도 가정 하나에 의존하는 휴리스틱
    d_usd = abs(nd) * spot * iv / 100 / math.sqrt(365)
    v_usd = abs(nv) * 3.0
    r = d_usd / v_usd if v_usd > 0 else float("inf")
    intent = "방향" if r > 2 else ("변동성" if r < 0.5 else "혼합")
    return {"net_delta_eth": nd, "gross_delta_eth": gd, "net_vega_usd": nv, "gross_vega_usd": gv,
            "delta_share": abs(nd) / gd if gd else 0.0, "vega_share": abs(nv) / gv if gv else 0.0,
            "dir_vs_vol_ratio": r, "intent": intent,
            "vol_side": "long_vol" if nv > 0 else "short_vol", "delta_side": "long" if nd > 0 else "short"}


# ── ① API 필드 전수 ───────────────────────────────────────────────────────────
def probe(ws_sec: float) -> dict:
    out: dict = {"probed_at_utc": datetime.now(timezone.utc).isoformat()}
    tr = get(API, "get_last_trades_by_currency?currency=ETH&kind=option&count=1000")["result"]["trades"]
    keys = lambda rows: dict(collections.Counter(k for t in rows for k in t))
    out["rest_by_currency"] = {"n": len(tr), "fields": keys(tr),
                               "span_h": (tr[0]["timestamp"] - tr[-1]["timestamp"]) / 3.6e6,
                               "example_block": next((t for t in tr if "block_trade_id" in t), None),
                               "example_screen_combo": next((t for t in tr if "combo_id" in t and "block_trade_id" not in t), None)}
    top = collections.Counter(t["instrument_name"] for t in tr).most_common(1)[0][0]
    ti = get(API, f"get_last_trades_by_instrument?instrument_name={top}&count=1000")["result"]["trades"]
    out["rest_by_instrument"] = {"instrument": top, "n": len(ti), "fields": keys(ti)}
    hi = get(HIST, "get_last_trades_by_currency_and_time?currency=ETH&kind=option&count=1000&sorting=asc"
                   f"&start_timestamp={tr[-1]['timestamp'] - 5 * 86400000}&end_timestamp={tr[-1]['timestamp']}")["result"]["trades"]
    out["history_by_currency"] = {"n": len(hi), "fields": keys(hi)}
    out["ws_trades_option_ETH_100ms"] = asyncio.run(_ws(ws_sec))
    rfq, cont = [], None
    for _ in range(400):
        r = get(API, "get_block_rfq_trades?currency=ETH&count=50" + (f"&continuation={cont}" if cont else ""))["result"]
        rfq += r["block_rfqs"]
        cont = r.get("continuation")
        if not cont or not r["block_rfqs"]:
            break
        time.sleep(0.12)
    (OUT / "rfq_eth.json").write_text(json.dumps(rfq))
    out["block_rfq_trades"] = {"n": len(rfq), "fields": keys(rfq),
                               "leg_fields": keys([l for r in rfq for l in r["legs"]]),
                               "trade_fields": keys([t for r in rfq for t in r["trades"]]),
                               "from_utc": _utc(rfq[-1]["timestamp"]) if rfq else None,
                               "to_utc": _utc(rfq[0]["timestamp"]) if rfq else None,
                               "example_with_hedge": next((r for r in rfq if r.get("hedge")), None)}
    ids = get(API, "get_combo_ids?currency=ETH")["result"]
    types: dict = {}
    for i in ids:
        code = i.split("-")[1]
        if code not in types:
            d = get(API, f"get_combo_details?combo_id={i}")["result"]
            types[code] = {"example": i, "legs": [(l["amount"], l["instrument_name"]) for l in d["legs"]]}
    out["combos"] = {"n_ids": len(ids), "type_counts": dict(collections.Counter(i.split("-")[1] for i in ids)),
                     "types": types}
    missing = {}
    for ep in ("get_block_rfqs?currency=ETH", "get_last_block_trades_by_currency?currency=ETH",
               "get_block_trade?id=BLOCK-1"):
        try:
            missing[ep] = get(API, ep).get("error")
        except Exception as e:  # noqa: BLE001 -- 없는 메서드는 HTTP 400 으로 온다
            missing[ep] = str(e)
    out["nonexistent_endpoints"] = missing
    (OUT / "probe.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return out


async def _ws(sec: float) -> dict:
    import websockets
    got, sub = [], {}
    async with websockets.connect("wss://www.deribit.com/ws/api/v2") as w:
        for i, ch in enumerate(("trades.option.ETH.100ms", "block_rfq.trades.ETH", "block_trade_confirmations")):
            await w.send(json.dumps({"jsonrpc": "2.0", "id": i, "method": "public/subscribe", "params": {"channels": [ch]}}))
        end = time.time() + sec
        while time.time() < end:
            try:
                m = json.loads(await asyncio.wait_for(w.recv(), timeout=max(0.1, end - time.time())))
            except asyncio.TimeoutError:
                break
            if "id" in m:
                sub[m["id"]] = m.get("result", m.get("error"))
            elif m.get("method") == "subscription":
                d = m["params"]["data"]
                got += [dict(t, _ch=m["params"]["channel"]) for t in (d if isinstance(d, list) else [d])]
    tr = [t for t in got if t["_ch"].startswith("trades.")]
    return {"seconds": sec, "subscribe_results": sub, "n_trades": len(tr),
            "fields": dict(collections.Counter(k for t in tr for k in t if k != "_ch")),
            "other_channel_msgs": [t for t in got if not t["_ch"].startswith("trades.")][:3]}


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1e3, timezone.utc).strftime("%Y-%m-%d %H:%M")


# ── history 백필 ──────────────────────────────────────────────────────────────
def history(days: int) -> Path:
    import pandas as pd
    end = int(time.time() * 1e3)
    cur, rows, seen = end - days * 86400000, [], set()
    while cur < end:
        r = get(HIST, f"get_last_trades_by_currency_and_time?currency=ETH&kind=option&count=1000&sorting=asc"
                      f"&start_timestamp={cur}&end_timestamp={end}")["result"]
        tr = r["trades"]
        new = [t for t in tr if t["trade_id"] not in seen]
        seen.update(t["trade_id"] for t in new)
        rows += new
        if not r.get("has_more") or not tr:
            break
        cur = tr[-1]["timestamp"] + (1 if tr[0]["timestamp"] == tr[-1]["timestamp"] else 0)
        time.sleep(0.08)
    df = pd.DataFrame(rows).rename(columns={"timestamp": "ts_ms"})
    p = OUT / f"hist_eth_options_{days}d.parquet"
    df.to_parquet(p)
    print("history", len(df), _utc(df.ts_ms.min()), "~", _utc(df.ts_ms.max()))
    return p


def fut_legs_at(ts_ms: int, block_id: str) -> list[dict]:
    """블록의 선물 헤지 다리 -- 같은 ms 의 ETH 선물 체결 중 같은 block_trade_id."""
    base = API if time.time() * 1e3 - ts_ms < 20 * 3600e3 else HIST
    r = get(base, f"get_last_trades_by_currency_and_time?currency=ETH&kind=future&count=100"
                  f"&start_timestamp={ts_ms}&end_timestamp={ts_ms}")["result"]["trades"]
    return [t for t in r if t.get("block_trade_id") == block_id]


# ── ③④ 분석 ───────────────────────────────────────────────────────────────────
def q(s, ps=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) -> dict:
    return {f"p{int(p * 100)}": round(float(s.quantile(p)), 4) for p in ps} | {"n": int(s.count())}


def analyze(hist_days: int) -> dict:
    import pandas as pd
    srv = pd.read_parquet(OUT / "eth_option_trades.parquet")
    hp = OUT / f"hist_eth_options_{hist_days}d.parquet"
    hst = pd.read_parquet(hp) if hp.exists() else None
    rep: dict = {"server": {"n": len(srv), "from": _utc(srv.ts_ms.min()), "to": _utc(srv.ts_ms.max()),
                            "liquidation_nonnull": int(srv.liquidation.notna().sum())}}
    if hst is not None:
        rep["history"] = {"n": len(hst), "from": _utc(hst.ts_ms.min()), "to": _utc(hst.ts_ms.max()),
                          "liquidation_nonnull": int(hst["liquidation"].notna().sum()) if "liquidation" in hst else "필드 없음"}
    for name, df in (("server", srv), ("history", hst)):
        if df is None:
            continue
        df = df.copy()
        df["is_block"] = df["block_trade_id"].notna()
        df["usd"] = df.amount * df.index_price
        rep[name] |= _blocks(df, name) | _screen(df)
    rep |= _rfq()
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    return rep


def _blocks(df, name: str) -> dict:
    import pandas as pd
    b = df[df.is_block]
    out, rows = {}, []
    fut_cache = OUT / f"fut_legs_{name}.json"
    fut = json.loads(fut_cache.read_text()) if fut_cache.exists() else {}
    for bid, g in b.groupby("block_trade_id"):
        lc = int(g.block_trade_leg_count.iloc[0]) if pd.notna(g.block_trade_leg_count.iloc[0]) else len(g)
        seen = len(g)
        ts = int(g.ts_ms.iloc[0])
        if lc > seen and bid not in fut:
            try:
                fut[bid] = fut_legs_at(ts, bid)
            except Exception as e:  # noqa: BLE001
                fut[bid] = [{"error": str(e)}]
            time.sleep(0.08)
        fl = [t for t in fut.get(bid, []) if "error" not in t]
        legs = [{"instrument": r.instrument_name, "q": r.amount * (1 if r.direction == "buy" else -1), "iv": r.iv,
                 "index_price": r.index_price, "ts_ms": r.ts_ms, "price": r.price} for r in g.itertuples()]
        legs += [{"instrument": t["instrument_name"], "q": t["amount"] * (1 if t["direction"] == "buy" else -1),
                  "iv": None, "index_price": t["index_price"], "ts_ms": t["timestamp"], "price": t["price"]} for t in fl]
        lab = classify([(l["instrument"], l["q"]) for l in legs])
        combo = g.combo_id.dropna()
        code = combo.iloc[0].split("-")[1] if len(combo) else None
        rows.append({"block_trade_id": bid, "ts": _utc(ts), "leg_count": lc, "opt_legs_seen": seen,
                     "fut_legs_found": len(fl), "complete": seen + len(fl) == lc, "label": lab,
                     "deribit_code": code, "rfq": g.block_rfq_id.notna().any() if "block_rfq_id" in g else None,
                     "usd": float(g.usd.sum()),
                     "legs": " | ".join(f"{'+' if l['q'] > 0 else ''}{l['q']:g} {l['instrument']}" for l in legs)}
                    | greeks(legs))
    fut_cache.write_text(json.dumps(fut))
    bl = pd.DataFrame(rows)
    if bl.empty:
        return {"blocks": 0}
    bl.to_csv(OUT / f"blocks_classified_{name}.csv", index=False)
    base = bl.label.str.replace("+hedge", "", regex=False)
    v = bl[bl.deribit_code.notna()]
    agree = (v.label.str.replace("+hedge", "", regex=False) == v.deribit_code.map(DERIBIT_CODE))
    multi = bl[bl.opt_legs_seen > 1]
    out["blocks"] = {
        "n_blocks": len(bl), "n_legs": int(bl.opt_legs_seen.sum()), "usd_total": round(bl.usd.sum()),
        "leg_count_eq_opt_legs": int((bl.leg_count == bl.opt_legs_seen).sum()),
        "leg_count_gt_opt_legs": int((bl.leg_count > bl.opt_legs_seen).sum()),
        "fut_hedge_found": int((bl.fut_legs_found > 0).sum()), "complete": int(bl.complete.sum()),
        "rfq_share": float(bl.rfq.mean()) if bl.rfq.notna().any() else None,
        "label_counts": bl.label.value_counts().to_dict(),
        "label_share_by_count": (base.value_counts(normalize=True).round(3)).to_dict(),
        "label_share_by_usd": (bl.groupby(base).usd.sum() / bl.usd.sum()).round(3).to_dict(),
        "unclassified_share": float((base == "other").mean()),
        "multi_leg_unclassified_share": float((multi.label.str.startswith("other")).mean()) if len(multi) else None,
        "with_deribit_combo_code": len(v), "agree_with_deribit_code": int(agree.sum()),
        "disagree_examples": v[~agree][["deribit_code", "label", "legs"]].head(8).to_dict("records"),
        "intent_counts": bl.intent.value_counts().to_dict(),
        "intent_by_label": bl.groupby(base).intent.value_counts().unstack(fill_value=0).to_dict("index"),
        "vol_side_by_label": bl[base != "single"].groupby(base[base != "single"]).vol_side.value_counts()
                             .unstack(fill_value=0).to_dict("index"),
        "delta_share": q(multi.delta_share) if len(multi) else None,
        "vega_share": q(multi.vega_share) if len(multi) else None}
    return out


def _screen(df) -> dict:
    s = df[~df.is_block].copy()
    sign = s.direction.map({"buy": 1, "sell": -1})
    out = {"screen": {"n": len(s), "usd": round(s.usd.sum()), "block_usd_share": round(df[df.is_block].usd.sum() / df.usd.sum(), 3),
                      "combo_share_trades": round(s.combo_id.notna().mean(), 4),
                      "combo_share_usd": round(s[s.combo_id.notna()].usd.sum() / s.usd.sum(), 4),
                      "combo_codes": s.dropna(subset=["combo_id"]).drop_duplicates("combo_trade_id").combo_id
                      .str.split("-").str[1].value_counts().to_dict()}}
    # (b)(e) 테이커 주문 = 같은 ms·종목·방향 묶음(한 공격 주문이 여러 메이커를 먹은 흔적)
    s["sign"] = sign
    s["edge_bp_underlying"] = sign * (s.price - s.mark_price) * 1e4          # 가격이 ETH 단위 = 기초자산 대비 비율
    s["edge_pct_premium"] = sign * (s.price - s.mark_price) / s.mark_price * 100
    nc = s[s.combo_id.isna()]
    orders = nc.groupby(["ts_ms", "instrument_name", "direction"]).agg(
        fills=("amount", "size"), eth=("amount", "sum"), usd=("usd", "sum"),
        edge_bp=("edge_bp_underlying", "mean"), edge_pct=("edge_pct_premium", "mean")).reset_index()
    ms = nc.groupby(["ts_ms", "direction"]).instrument_name.nunique()
    import pandas as pd
    bins = [0, 1, 10, 100, 1e9]
    orders["size_bucket"] = pd.cut(orders.eth, bins, labels=["<1", "1-10", "10-100", ">=100"], right=False)
    blk = df[df.is_block]
    bsign = blk.direction.map({"buy": 1, "sell": -1})
    out["proxies"] = {
        "trade_eth": q(s.amount), "trade_usd": q(s.usd), "block_leg_eth": q(blk.amount),
        "order_eth": q(orders.eth), "order_size_share_count": orders.size_bucket.value_counts(normalize=True).round(3).to_dict(),
        "order_size_share_usd": (orders.groupby("size_bucket", observed=False).usd.sum() / orders.usd.sum()).round(3).to_dict(),
        "multi_fill_orders_share": round((orders.fills > 1).mean(), 4),
        "multi_fill_orders_usd_share": round(orders[orders.fills > 1].usd.sum() / orders.usd.sum(), 4),
        "same_ms_multi_strike_same_dir_share_of_groups": round((ms > 1).mean(), 4),
        "edge_bp_underlying_screen": q(s.edge_bp_underlying.dropna()),
        "edge_pct_premium_screen": q(s.edge_pct_premium.replace([float("inf"), -float("inf")], float("nan")).dropna()),
        "edge_bp_underlying_block": q((bsign * (blk.price - blk.mark_price) * 1e4).dropna()),
        "edge_bp_by_order_size": orders.groupby("size_bucket", observed=False).edge_bp.median().round(3).to_dict(),
        "edge_pct_by_order_size": orders.groupby("size_bucket", observed=False).edge_pct.median().round(2).to_dict(),
        "edge_bp_multi_vs_single_fill": {"multi": round(float(orders[orders.fills > 1].edge_bp.median()), 3),
                                         "single": round(float(orders[orders.fills == 1].edge_bp.median()), 3)},
        "worse_than_mark_share": round(float((s.edge_bp_underlying > 0).mean()), 3),
        "liquidation_values": df["liquidation"].value_counts().to_dict() if "liquidation" in df else "필드 없음"}
    return out


def _rfq() -> dict:
    p = OUT / "rfq_eth.json"
    if not p.exists():
        return {}
    rfq = json.loads(p.read_text())
    rows = []
    for r in rfq:
        legs = [(l["instrument_name"], r["amount"] * l["ratio"] * (1 if l["direction"] == "buy" else -1)
                 * (1 if r["direction"] == "buy" else -1)) for l in r["legs"]]
        # 🔴RFQ 의 legs.direction 은 «콤보를 buy 할 때» 다리 방향 -- 체결 방향(r.direction)이 sell 이면 뒤집는다
        #   (rfq_direction_check: 공개 테이프 다리 방향과 67/67 일치, 뒤집지 않으면 45/67)
        h = r.get("hedge")
        if h:
            legs.append((h["instrument_name"], h["amount"] / h["price"] * (1 if h["direction"] == "buy" else -1)))
        rows.append({"id": r["id"], "label": classify(legs), "combo": r["combo_id"], "hedge": bool(h),
                     "n_legs": len(r["legs"])})
    lab = collections.Counter(x["label"] for x in rows)
    v = [x for x in rows if x["combo"] and x["combo"].split("-")[1] in DERIBIT_CODE]   # 단일 다리는 combo_id=종목명
    agree = sum(x["label"].replace("+hedge", "") == DERIBIT_CODE.get(x["combo"].split("-")[1]) for x in v)
    return {"rfq": {"n": len(rows), "from": _utc(rfq[-1]["timestamp"]), "to": _utc(rfq[0]["timestamp"]),
                    "labels": dict(lab), "hedge_n": sum(x["hedge"] for x in rows),
                    "with_combo_id": len(v), "agree_with_combo_code": agree,
                    "unclassified_share": round(sum(x["label"].startswith("other") for x in rows) / len(rows), 3)}}


def rfq_direction_check() -> dict:
    """RFQ legs.direction 이 체결 방향을 이미 반영했는지 -- 공개 테이프(같은 ms 블록 다리)와 대조."""
    import pandas as pd
    rfq = json.loads((OUT / "rfq_eth.json").read_text())
    srv = pd.read_parquet(OUT / "eth_option_trades.parquet")
    srv = srv[srv.block_rfq_id.notna()]
    same = flip = n = 0
    for r in rfq:
        t = srv[srv.block_rfq_id.astype(str) == str(r["id"])]
        for l in r["legs"]:
            m = t[t.instrument_name == l["instrument_name"]]
            if len(m):
                n += 1
                same += int(m.direction.iloc[0] == l["direction"])
                want = l["direction"] if r["direction"] == "buy" else {"buy": "sell", "sell": "buy"}[l["direction"]]
                flip += int(m.direction.iloc[0] == want)
    return {"legs_matched": n, "tape_eq_leg_dir": same, "tape_eq_leg_dir_x_rfq_dir": flip}


# ── 자체점검 ─────────────────────────────────────────────────────────────────
def _selftest() -> None:
    C = lambda k, e="2OCT26": f"ETH-{e}-{k}-C"
    P = lambda k, e="2OCT26": f"ETH-{e}-{k}-P"
    cases = [
        ([(C(2700), 10)], "single"),
        ([(C(2700), 10), ("ETH-PERPETUAL", -5000)], "single+hedge"),
        ([(C(2700), 5), (P(2700), 5)], "straddle"),
        ([(C(2700), -5), (P(2700), -5)], "straddle"),
        ([(C(2900), 5), (P(2500), 5)], "strangle"),
        ([(C(2500), 5), (P(2900), 5)], "strangle"),                   # GUTS
        ([(C(2900), -5), (P(2500), 5)], "risk_reversal"),
        ([(C(2700), 5), (P(2700), -5)], "synthetic"),
        ([(C(2700), 5), (C(2800), -5)], "vertical"),
        ([(P(2700), -5), (P(2600), 5)], "vertical"),
        ([(C(2700), 5), (C(2800), -10)], "ratio_spread"),
        ([(C(2700), 5), (C(2800), 5)], "other"),
        ([(C(2700, "2OCT26"), -5), (C(2700, "16OCT26"), 5)], "calendar"),
        ([(C(2700, "2OCT26"), -5), (C(2800, "16OCT26"), 5)], "diagonal"),
        ([(C(2600), 1), (C(2700), -2), (C(2800), 1)], "butterfly"),
        ([(P(2600), -1), (P(2700), 2), (P(2800), -1)], "butterfly"),
        ([(C(1700), 1), (C(1800), -1), (C(2400), -1)], "ladder"),
        ([(P(1600), -1), (P(2100), -1), (P(2400), 1)], "ladder"),
        ([(C(2500), 1), (C(2600), -1), (C(2700), -1), (C(2800), 1)], "condor"),
        ([(P(2300), 1), (P(2650), -1), (C(2750), -1), (C(3100), 1)], "iron_condor"),
        ([(P(2300), 1), (P(2700), -1), (C(2700), -1), (C(3100), 1)], "iron_butterfly"),
        ([(C(1000), 1), (P(1000), -1), (C(3000), -1), (P(3000), 1)], "box"),
        ([(C(2700), 5), (C(2700), -5)], "future_only"),               # 같은 종목 상쇄 → 옵션 0
        ([("ETH-1OCT26", -1), ("ETH-16OCT26", 1)], "future_only"),
        ([(C(2700), 1), (P(2600), 1), (C(2900), -1)], "other"),
    ]
    for legs, want in cases:
        got = classify(legs)
        assert got == want, (legs, got, want)
    # Deribit combo 정의 자체(get_combo_details 실측 다리)를 분류하면 코드와 일치해야 한다
    for code, legs in {"CBUT": [(1, C(2760, "1OCT26")), (-2, C(2850, "1OCT26")), (1, C(2860, "1OCT26"))],
                       "PSR23": [(-3, P(2000, "25DEC26")), (2, P(2200, "25DEC26"))],
                       "RRITM": [(1, C(1400, "25DEC26")), (-1, P(2200, "25DEC26"))],
                       "PLAD": [(-1, P(1600, "25DEC26")), (-1, P(2100, "25DEC26")), (1, P(2400, "25DEC26"))]}.items():
        assert classify([(i, a) for a, i in legs]) == DERIBIT_CODE[code], code
    # 그릭스: 롱 스트래들 ≈ 델타 0·롱 베가, 롱 콜 = 롱 델타
    ts = int(datetime(2026, 9, 25, 8, tzinfo=timezone.utc).timestamp() * 1e3)
    L = lambda i, q: {"instrument": i, "q": q, "iv": 50.0, "index_price": 2700.0, "ts_ms": ts, "price": 0.01}
    g = greeks([L(C(2700, "16OCT26"), 10), L(P(2700, "16OCT26"), 10)])
    assert g["vol_side"] == "long_vol" and g["delta_share"] < 0.15 and g["intent"] == "변동성", g
    g = greeks([L(C(2700, "16OCT26"), 10), {"instrument": "ETH-PERPETUAL", "q": -2700 * 10 * 0.52, "iv": None,
                                             "index_price": 2700.0, "ts_ms": ts, "price": 2700.0}])
    assert abs(g["net_delta_eth"]) < 0.5 and g["intent"] == "변동성", g   # 델타 헤지된 콜 = 변동성 거래
    assert greeks([L(C(2700, "16OCT26"), 10)])["delta_side"] == "long"
    print("selftest OK", len(cases) + 4, "구조 케이스")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--ws-sec", type=float, default=60)
    ap.add_argument("--history", type=int, default=0)
    ap.add_argument("--hist-days", type=int, default=30)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.selftest:
        _selftest()
    elif a.probe:
        print(json.dumps(probe(a.ws_sec), indent=1, ensure_ascii=False, default=str)[:6000])
    elif a.history:
        history(a.history)
    else:
        r = analyze(a.hist_days)
        r["rfq_direction_check"] = rfq_direction_check()
        (OUT / "report.json").write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str))
        print(json.dumps(r, indent=1, ensure_ascii=False, default=str))
