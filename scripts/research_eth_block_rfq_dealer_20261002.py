#!/usr/bin/env python3
"""블록 방향 규약 재점검 + «RFQ 요청자 = 고객» 가정(T_rfq) 검정 (2026-10-02, 사용자 «진행해줘»).

이미 알던 것(09-30 research_eth_option_trade_identity_20260930, report.json rfq_direction_check):
  RFQ 패키지 46건(09-24~30)의 옵션 다리 67개 -- 공개 테이프 다리 방향 = legs.direction × rfq.direction 67/67
  (legs.direction 그대로면 45/67). rfq.direction 은 문서상 «taker(= 요청자) 방향». ⇒ RFQ 블록의 공개 direction = 요청자.
  직접 블록(block_rfq_id 없음)은 점검한 적 없다(대시보드 문구 «직접 거래는 수락자»는 근거 없는 서술).
여기서 새로 하는 것:
  ① (a) 확장: 서버 표 사본(09-23~10-01) ∪ 지금 API 로 받은 RFQ 전부 × 테이프(2026 parquet + history 꼬리) -- 다리별 방향·수량,
     반대 다리(legs.direction=sell)·rfq sell 따로.
  ② (b) 직접 블록의 direction 이 누구 쪽인가: 패키지 단위 «direction 쪽이 mark 대비 낸 비용»(bp) 부호.
     RFQ 블록(요청자 = direction 확정)과 화면 체결(테이커 확정)이 양성 대조군. 문서(Block Trading API)의 «maker 관점» 표기가
     공개 테이프에도 적용된다면 직접 블록의 direction 쪽은 비용을 «받는» 쪽이어야 한다(share < 0.5).
  ③ (c) T 에 남는 부호 오류의 크기: 직접 블록 거래량 몫 · |w| 몫 · 직접 블록만 뒤집은 T(T_dflip)와 T 의 GEX 순위 상관 · 판정 변화.
  ④ T_rfq = −Σ(RFQ 다리 체결의 테이커=요청자 부호 수량), 종목 첫 체결부터(2025 체결은 포지션 재료로만, 검정 표본은 2026).
     공개 테이프의 block_rfq_id 행 = RFQ 다리 체결(옵션만; 선물 hedge 다리는 옵션 포지션에 안 들어간다).
     🔴get_block_rfq_trades 는 count 10~50 · 보존 약 1주 · history.deribit.com 미지원 ⇒ 과거 전량은 테이프에서만 나온다.
     검정 = compare(수요압력·만기 소멸·물리 제약) + matrix(변동성·재헤지·헤지 흔적)의 함수를 마스크만 바꿔 같은 사양으로.
     T_direct(직접 블록만) · T_dflip(T 에서 직접 블록 부호 반전)은 (c) 보조 -- 판정 집계에 안 넣는다.
시점 경계: 포지션 = ts < t 체결만(R.positions_at) · 라벨 = t 뒤 첫 완결 봉부터(각 원 스크립트 규약 그대로).
🔴바이낸스 REST 호출 없음(matrix 의 data.binance.vision 캐시를 심볼릭 링크로 재사용). Deribit 요청 사이 0.15초.
출력 tmp/block_rfq_dealer_20261002/ : results.json · summary.md · (compare|pass_*)/

  python scripts/research_eth_block_rfq_dealer_20261002.py             # 전체(~수 분)
  python scripts/research_eth_block_rfq_dealer_20261002.py --selftest  # 네트워크 없음
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_dealer_gex_reconstruct_2026_20261001 as R      # noqa: E402
import research_eth_dealer_assumption_compare_20261002 as A        # noqa: E402
import research_eth_dealer_assumption_matrix_20261002 as M         # noqa: E402

OUT = ROOT / "tmp/block_rfq_dealer_20261002"
VAL = ROOT / "tmp/opt_validate_20261002"
API = "https://www.deribit.com/api/v2/public/"
HIST = "https://history.deribit.com/api/v2/public/"
D_MS = 86_400_000
SGN = {"buy": 1.0, "sell": -1.0}

# 결과 보기 전 고정(2026-10-02). 결과를 보고 바꾸지 않는다.
CRITERIA = {
    "a_rfq_tape_direction": "다리 단위 일치율(테이프 행 direction == 요청자 다리 부호) ≥ 0.99 이면 «공개 테이프 RFQ 블록 direction = 요청자» 확정, "
                            "≤ 0.01 이면 «= 응답자», 그 사이 = 혼재. 수량은 |Σ테이프 − amount×ratio| ≤ 1e-6 상대오차를 일치로 센다",
    "b_direct_perspective": {
        "stat": "패키지(block_trade_id) 단위 edge_bp = Σ s·amt·(price − mark) / Σ amt × 1e4 (s = 공개 direction 부호; 양수 = direction 쪽이 mark 보다 불리)",
        "main": "share(edge>0)(0 정확히는 제외) 일 블록 부트스트랩 95% CI",
        "controls": "RFQ 블록(direction = 요청자 확정)과 화면 체결(테이커 확정)이 CI lo > 0.5 여야 방법이 산다 -- 아니면 (b) 전체 «판별 불가»",
        "verdict": "직접 블록 lo > 0.5 → «direction 쪽 = 비용 낸 쪽(요청자형)» · hi < 0.5 → «direction 쪽 = 비용 받은 쪽(maker 관점 표기)» · 그 외 판별 불가",
        "note": "비용 낸 쪽 = 유동성을 요청한 쪽이라는 해석은 가정이다. 신원(고객/딜러)은 이 검정이 못 가른다",
    },
    "dealer_tests": "matrix CRITERIA 그대로(rule · E · 이론 부호) -- 새 가정 T_rfq 만 추가, 같은 함수",
    "aggregation": "47일 공통 구간 6칸 · 2026 전체 T 계열 칸(변동성·재헤지·만기 -- 헤지 흔적 분 설계는 HF 스크립트 전용이라 T_rfq 미계산)",
    "better_rule": "«T_rfq 가 낫다» = 47일 주 칸에서 반대 0 이고 지지 수가 T·T_block·C·R 각각보다 많음. 거울(Tflip·R)은 독립 증거 아님. "
                   "그 외 = «낫다고 말할 근거 없음»",
    "aux_not_judged": ["T_direct", "T_dflip", "RFQ 다리 수량 일치", "RFQ id 집합 대조"],
}


def get(base: str, ep: str, **kw) -> dict:
    with urllib.request.urlopen(base + ep + "?" + urllib.parse.urlencode(kw), timeout=30) as r:
        return json.load(r)["result"]


# ───────────────────────── RFQ 다리 해석 ─────────────────────────
def req_legs(r: dict) -> dict:
    """RFQ 한 건 → {종목: 요청자 쪽 부호 수량}. legs.direction 은 «콤보를 buy 할 때» 다리 방향이라
    rfq.direction(요청자)이 sell 이면 뒤집는다(09-30 67/67). 수량 = amount × ratio. hedge(선물)는 옵션 포지션 밖이라 뺀다."""
    s = SGN[r["direction"]]; out: dict = {}
    for l in r["legs"]:
        out[l["instrument_name"]] = out.get(l["instrument_name"], 0.0) + s * SGN[l["direction"]] * float(r["amount"]) * float(l["ratio"])
    return out


def fetch_rfq_api() -> list:
    p = OUT / "rfq_api_eth.json"
    if p.exists():
        return json.loads(p.read_text())
    rows, cont = [], None
    for _ in range(2000):
        r = get(API, "get_block_rfq_trades", currency="ETH", count=50, **({"continuation": cont} if cont else {}))
        rows += r["block_rfqs"]; cont = r.get("continuation")
        if not cont or not r["block_rfqs"]:
            break
        time.sleep(0.15)
    p.write_text(json.dumps(rows))
    return rows


def tape() -> pd.DataFrame:
    """2026 체결 parquet + 그 끝 이후 history 꼬리(RFQ 대조용)."""
    t = pd.read_parquet(VAL / "eth_opt_trades_2026.parquet")
    p = OUT / "tape_tail.parquet"
    if not p.exists():
        rows, t0, now = [], int(t["ts_ms"].max()) + 1, int(time.time() * 1000)
        while t0 < now:
            r = get(HIST, "get_last_trades_by_currency_and_time", currency="ETH", kind="option", count=1000, sorting="asc",
                    start_timestamp=t0, end_timestamp=now)["trades"]
            rows += r
            if len(r) < 1000:
                break
            t0 = r[-1]["timestamp"] + (r[0]["timestamp"] == r[-1]["timestamp"])      # 경계 ms 는 다시 받고 trade_id 로 중복 제거
            time.sleep(0.15)
        d = pd.DataFrame(rows).drop_duplicates("trade_id").rename(columns={"timestamp": "ts_ms"})
        d.reindex(columns=t.columns).to_parquet(p)
    tail = pd.read_parquet(p)
    return pd.concat([t, tail]).drop_duplicates("trade_id").reset_index(drop=True)


def direction_check(rfqs: list, tp: pd.DataFrame) -> dict:
    """(a) RFQ 요청자 다리 부호 vs 공개 테이프. 반대 다리(legs.direction=sell) · rfq sell 을 따로 센다."""
    blk = tp[tp["block_rfq_id"].notna()].copy(); blk["rid"] = blk["block_rfq_id"].astype("int64")
    lo_t, hi_t = int(tp["ts_ms"].min()), int(tp["ts_ms"].max())
    by = {k: g for k, g in blk.groupby("rid")}
    c = {"rfq_total": len(rfqs), "rfq_in_tape_window": 0, "rfq_found_in_tape": 0, "legs": 0, "legs_found": 0,
         "rows": 0, "rows_agree_requester": 0, "legs_qty_agree": 0,
         "by_leg_dir": {"buy": [0, 0], "sell": [0, 0]}, "by_rfq_dir": {"buy": [0, 0], "sell": [0, 0]}, "tape_legs_not_in_rfq": 0}
    for r in rfqs:
        if not (lo_t <= r["timestamp"] <= hi_t):
            continue
        c["rfq_in_tape_window"] += 1
        exp = req_legs(r); g = by.get(int(r["id"]))
        c["legs"] += len(exp)
        if g is None:
            continue
        c["rfq_found_in_tape"] += 1
        c["tape_legs_not_in_rfq"] += int((~g["instrument_name"].isin(exp)).sum())
        ldir = {l["instrument_name"]: l["direction"] for l in r["legs"]}
        for inst, q in exp.items():
            m = g[g["instrument_name"] == inst]
            if not len(m):
                continue
            c["legs_found"] += 1
            agree = int((m["direction"].map(SGN) == np.sign(q)).sum())
            c["rows"] += len(m); c["rows_agree_requester"] += agree
            for key, d in (("by_leg_dir", ldir[inst]), ("by_rfq_dir", r["direction"])):
                c[key][d][0] += len(m); c[key][d][1] += agree
            tq = float((m["direction"].map(SGN) * m["amount"]).sum())
            c["legs_qty_agree"] += int(abs(tq - q) <= 1e-6 * max(abs(q), 1.0))
    c["rate_rows_agree"] = round(c["rows_agree_requester"] / max(c["rows"], 1), 4)
    c["verdict"] = ("공개 direction = 요청자" if c["rate_rows_agree"] >= 0.99 else
                    "공개 direction = 응답자" if c["rate_rows_agree"] <= 0.01 else "혼재")
    return c


def rfq_id_overlap(srv: pd.DataFrame, rfqs_api: list, tp: pd.DataFrame) -> dict:
    """서버 표 사본·API·테이프의 ETH RFQ id 집합 대조(겹치는 기간)."""
    s = srv[srv["api"] == "ETH"]
    lo = max(int(s["ts_ms"].min()), int(tp["ts_ms"].min())); hi = min(int(s["ts_ms"].max()), int(tp["ts_ms"].max()))
    a = set(s.loc[s["ts_ms"].between(lo, hi), "rfq_id"].astype("int64"))
    b = set(tp.loc[tp["ts_ms"].between(lo, hi) & tp["block_rfq_id"].notna(), "block_rfq_id"].astype("int64"))
    api = {int(r["id"]) for r in rfqs_api if lo <= r["timestamp"] <= hi}
    return {"window": [str(pd.Timestamp(lo, unit="ms")), str(pd.Timestamp(hi, unit="ms"))], "server": len(a), "tape": len(b), "api": len(api),
            "server_and_tape": len(a & b), "server_only": len(a - b), "tape_only": len(b - a), "api_and_server": len(api & a)}


# ───────────────────────── (b) 직접 블록 관점 ─────────────────────────
def edge_bp(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """key 단위 edge_bp(양수 = direction 쪽이 mark 대비 불리) · day."""
    d = df.assign(s=df["direction"].map(SGN))
    d = d.assign(num=d["s"] * d["amount"] * (d["price"] - d["mark_price"]))
    g = d.groupby(key).agg(num=("num", "sum"), amt=("amount", "sum"), ts=("ts_ms", "min"))
    return pd.DataFrame({"edge_bp": g["num"] / g["amt"] * 1e4, "day": g["ts"] // D_MS})


def share_pos(e: pd.DataFrame) -> dict:
    e = e[np.isfinite(e["edge_bp"]) & (e["edge_bp"] != 0)]
    g = pd.DataFrame({"day": e["day"], "p": (e["edge_bp"] > 0).astype(float), "n": 1.0, "e": e["edge_bp"]}).groupby("day").sum()
    est, lo, hi = A.boot_ratio(g["p"].to_numpy(), g["n"].to_numpy())
    me, ml, mh = A.boot_ratio(g["e"].to_numpy(), g["n"].to_numpy())
    return {"n": len(e), "days": len(g), "share_pos": [round(est, 4), round(lo, 4), round(hi, 4)],
            "mean_edge_bp": [round(me, 3), round(ml, 3), round(mh, 3)], "median_edge_bp": round(float(e["edge_bp"].median()), 3)}


def direct_perspective(tp: pd.DataFrame) -> dict:
    t = tp[(tp["ts_ms"] < pd.Timestamp("2026-10-01", tz="UTC").value // 10**6) & (tp["mark_price"] > 0)]
    b = t[t["block_trade_id"].notna()]
    rfq = b["block_rfq_id"].notna()
    out = {"rfq_blocks": share_pos(edge_bp(b[rfq], "block_trade_id")),
           "direct_blocks": share_pos(edge_bp(b[~rfq], "block_trade_id")),
           "screen_trades": share_pos(edge_bp(t[t["block_trade_id"].isna()], "trade_id"))}
    ctrl_ok = out["rfq_blocks"]["share_pos"][1] > 0.5 and out["screen_trades"]["share_pos"][1] > 0.5
    lo, hi = out["direct_blocks"]["share_pos"][1:]
    out["controls_pass"] = bool(ctrl_ok)
    out["verdict"] = ("판별 불가(대조군 실패)" if not ctrl_ok else "direction 쪽 = 비용 낸 쪽(요청자형)" if lo > 0.5 else
                      "direction 쪽 = 비용 받은 쪽(maker 관점 표기)" if hi < 0.5 else "판별 불가")
    # 사후 서술: 크기별(직접 블록 패키지 명목 ETH 3분위)
    e = edge_bp(b[~rfq], "block_trade_id").join(b[~rfq].groupby("block_trade_id")["amount"].sum().rename("amt"))
    e["q"] = pd.qcut(e["amt"], 3, labels=["small", "mid", "large"])
    out["direct_by_size_posthoc"] = {str(k): share_pos(g) for k, g in e.groupby("q", observed=True)}
    return out


# ───────────────────────── 딜러 검정(재사용) ─────────────────────────
def rfq_trade_ids() -> set:
    s = set()
    for p in (R.CACHE / "opt_trades_hist.parquet", R.OUT / "tail_trades.parquet", R.OUT / "prelisted_trades.parquet"):
        b = pd.read_parquet(p, columns=["trade_id", "block_rfq_id"]); s |= set(b.loc[b["block_rfq_id"].notna(), "trade_id"])
    return s


def run_compare(rfq_tids: set) -> dict:
    """compare main 을 그대로 -- 가정 목록에 T_rfq · T_direct · T_dflip 만 더한다(기존 칸은 양성 대조로 재현)."""
    orig_masks, orig_pos = A.subset_masks, A.positions

    def masks(tr):
        m = orig_masks(tr); rfq = tr["trade_id"].isin(rfq_tids).to_numpy()
        m["T_rfq"] = rfq; m["T_direct"] = m["T_block"] & ~rfq
        return m

    def positions(tr, t_eval, ms_, q_override=None):
        P = orig_pos(tr, t_eval, ms_, q_override)
        if "pos_T_direct" in P:
            P["pos_T_dflip"] = P["pos_T"] - 2 * P["pos_T_direct"]          # 직접 블록만 부호 반전(선형)
        return P

    tfam = ("T", "T_block", "T_screen", "T_small", "T_rfq", "T_direct", "T_dflip")
    d = OUT / "compare"; d.mkdir(parents=True, exist_ok=True)
    with mock.patch.multiple(A, OUT=d, subset_masks=masks, positions=positions, TFAM=tfam, ASM=tfam + ("C", "R")):
        A.main()
    return json.loads((d / "results.json").read_text())


def run_matrix_pass(name: str, tr: pd.DataFrame, block_mask: np.ndarray, B) -> dict:
    """matrix 의 vol · rehedge · hedge 를 «T_block» 자리에 다른 마스크를 넣어 그대로 돈다."""
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    if not (d / "binance").exists():
        (d / "binance").symlink_to(M.OUT / "binance")                       # data.binance.vision 캐시 재사용
    p = d / "results.json"
    if p.exists():
        return json.loads(p.read_text())
    mk = lambda t: {"T": np.ones(len(t), bool), "T_block": t["_bm"].to_numpy()}   # noqa: E731  hedge 는 tr 을 잘라 넘긴다 → 열로 들고 다닌다
    tr = tr.assign(_bm=block_mask)
    with mock.patch.multiple(M, OUT=d, masks_of=mk):
        vol, tb = M.vol_tests(tr, B)
        reh = M.rehedge_tests(tb)
        hed = M.hedge_tests(tr[tr["timestamp"] < M.END].reset_index(drop=True))
    res = {"vol": vol, "rehedge": reh, "hedge": hed}
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return res


def cells(name: str, ps: dict, cmp_: dict, key: str = "T_block", with_reh: bool = True) -> dict:
    """pass 결과(키 key) + compare 결과(name) → matrix 와 같은 모양의 칸."""
    t1, t2, t3 = cmp_["test1"][name], cmp_["test2"][name], cmp_["test3"][name]
    nc = {"verdict": "미계산", "note": ""}
    nr = {**nc, "note": "재헤지의 T 열은 원 T 시간 파일 고정 → T_dflip 미계산"}
    c47 = {"vol": M.cell(ps["vol"][f"{key}|47d"]["judge"]),
           "rehedge": M.cell(ps["rehedge"][f"{key}|47d"]["judge"]) if with_reh else nr,
           "hedge": M.cell(ps["hedge"][key]["judge"]), "demand": M.cell(t1["cross_section"]),
           "expiry": M.cell(t2["aux_same_days_as_CR"]),
           "constraint": {"verdict": {"일관": "지지", "불일치": "반대"}.get(t3["verdict"], "불가"), "est": t3["viol_rate"], "ci": t3["ci"],
                          "days": cmp_["test3"]["n_days"], "note": f"위반율 · 무작위 부호 {t3['null_random_sign_viol_rate']}"}}
    c26 = {"vol": M.cell(ps["vol"][f"{key}|2026"]["judge"]),
           "rehedge": M.cell(ps["rehedge"][f"{key}|2026"]["judge"]) if with_reh else nr,
           "hedge": {**nc, "note": "분 단위 원 설계는 HF 스크립트 전용"}, "expiry": M.cell(t2["main"]),
           "demand": {**nc, "note": "체인 스냅샷(08-15~) 필요"}, "constraint": {**nc, "note": "OI 필요"}}
    return {"47d": c47, "2026": c26}


def agg(c: dict, w: str) -> dict:
    cols = ("vol", "rehedge", "hedge", "demand", "expiry", "constraint") if w == "47d" else ("vol", "rehedge", "hedge", "expiry")
    vs = [c[x]["verdict"] for x in cols if c[x]["verdict"] not in ("제외", "미계산")]
    return {"지지": vs.count("지지"), "반대": vs.count("반대"), "불가": len(vs) - vs.count("지지") - vs.count("반대")}


def fmt(c: dict) -> str:
    if "est" not in c:
        return f"{c['verdict']}" + (f" ({c['note']})" if c.get("note") else "")
    nr = f" · 필요 {c['n_req']}일" if "n_req" in c else ""
    return f"**{c['verdict']}** {c['est']:+.4g} [{c['ci'][0]:+.4g}, {c['ci'][1]:+.4g}] {c['days']}일{nr}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"criteria": CRITERIA}
    # ── 1. 방향 규약
    rfq_api = fetch_rfq_api()
    srv = pd.read_parquet(VAL / "block_rfq.parquet")
    srv_rfqs = [{"id": int(r.rfq_id), "timestamp": int(r.ts_ms), "direction": r.direction, "amount": r.amount, "legs": json.loads(r.legs)}
                for r in srv[srv["api"] == "ETH"].itertuples()]
    allr = {int(r["id"]): r for r in srv_rfqs}; allr.update({int(r["id"]): r for r in rfq_api})
    tp = tape()
    res["rfq_api"] = {"n": len(rfq_api), "from": str(pd.Timestamp(min(r["timestamp"] for r in rfq_api), unit="ms")),
                      "to": str(pd.Timestamp(max(r["timestamp"] for r in rfq_api), unit="ms")),
                      "note": "count 10~50 · continuation 끝까지 · history.deribit.com 은 이 메서드 미지원(400)"}
    res["a_direction"] = direction_check(list(allr.values()), tp)
    res["rfq_id_overlap"] = rfq_id_overlap(srv, rfq_api, tp)
    res["b_direct"] = direct_perspective(tp)
    print("a", json.dumps(res["a_direction"], ensure_ascii=False), "\nb", json.dumps(res["b_direct"], ensure_ascii=False), flush=True)

    # ── 2. T_rfq 데이터 범위
    rt = rfq_trade_ids()
    pre = pd.read_parquet(R.OUT / "prelisted_trades.parquet", columns=["timestamp", "block_rfq_id"])
    t26 = tp[tp["ts_ms"] < pd.Timestamp("2026-10-01", tz="UTC").value // 10**6]
    res["rfq_range"] = {
        "earliest_rfq_leg_in_our_trades": str(pd.Timestamp(int(pre.loc[pre["block_rfq_id"].notna(), "timestamp"].min()), unit="ms")),
        "n_rfq_legs_prelisted_2025": int(pre["block_rfq_id"].notna().sum()),
        "n_rfq_legs_2026": int(t26["block_rfq_id"].notna().sum()), "n_rfq_2026": int(t26["block_rfq_id"].nunique()),
        "amount_share_2026_calendar": {"rfq": round(float(t26.loc[t26["block_rfq_id"].notna(), "amount"].sum() / t26["amount"].sum()), 4),
                                       "direct_block": round(float(t26.loc[t26["block_trade_id"].notna() & t26["block_rfq_id"].isna(), "amount"].sum()
                                                                   / t26["amount"].sum()), 4)}}
    print("range", res["rfq_range"], flush=True)

    # ── 3. 검정
    cmp_ = run_compare(rt)
    res["compare_volume_share_2026_expiries"] = cmp_["volume_share_2026_expiries"]
    res["explain_sum_abs_over_oi"] = {a: cmp_["test3"][a]["explain_sum_abs_over_oi"] for a in ("T", "T_block", "T_rfq", "T_direct", "T_dflip", "C")}
    tr = M.load_trades()
    rfq_m = tr["trade_id"].isin(rt).to_numpy(); direct_m = tr["is_block"].to_numpy() & ~rfq_m
    B = R.Bars(pd.read_parquet(R.OUT / "perp_5m.parquet"), pd.read_parquet(R.OUT / "dvol_1h.parquet"))
    p_rfq = run_matrix_pass("pass_rfq", tr, rfq_m, B)
    p_dir = run_matrix_pass("pass_direct", tr, direct_m, B)
    tr_df = tr.assign(q=np.where(direct_m, -tr["q"].to_numpy(), tr["q"].to_numpy()))
    p_dfl = run_matrix_pass("pass_dflip", tr_df, rfq_m, B)                # 여기서 «T» = T_dflip

    # 양성 대조: pass 의 T·C·R = matrix 원 결과
    old = json.loads((M.OUT / "results.json").read_text())
    pc = {}
    for nm, ps in (("pass_rfq", p_rfq), ("pass_direct", p_dir)):
        pc[nm] = {"vol_T47": [ps["vol"]["T|47d"]["full"]["beta"], old["vol"]["T|47d"]["full"]["beta"]],
                  "vol_T26": [ps["vol"]["T|2026"]["full"]["beta"], old["vol"]["T|2026"]["full"]["beta"]],
                  "hedge_T": [ps["hedge"]["T"]["judge"]["est"], old["hedge"]["T"]["judge"]["est"]],
                  "rehedge_C47": [ps["rehedge"]["C|47d"]["judge"]["est"], old["rehedge"]["C|47d"]["judge"]["est"]]}
    pc["compare_T_demand"] = [cmp_["test1"]["T"]["cross_section"]["est"], json.loads(
        (ROOT / "tmp/dealer_assumption_compare_20261002/results.json").read_text())["test1"]["T"]["cross_section"]["est"]]
    res["positive_control"] = pc

    # ── 매트릭스
    MC = {k: v for k, v in old["matrix"]["cells"].items()}
    for nm, ps, key, reh in (("T_rfq", p_rfq, "T_block", True), ("T_direct", p_dir, "T_block", True), ("T_dflip", p_dfl, "T", False)):
        c = cells(nm, ps, cmp_, key, reh)
        MC[f"{nm}|47d"], MC[f"{nm}|2026"] = c["47d"], c["2026"]
    AG = {k: agg(v, k.split("|")[1]) for k, v in MC.items()}
    res["matrix"] = {"cells": MC, "aggregate": AG}
    g = AG["T_rfq|47d"]
    others = [AG[f"{a}|47d"]["지지"] for a in ("T", "T_block", "C", "R")]
    res["better_verdict"] = ("T_rfq 가 낫다" if g["반대"] == 0 and all(g["지지"] > o for o in others) else "낫다고 말할 근거 없음")
    # (c) 크기: 시간별 week GEX 순위 상관(2026, 같은 사양: GEX 는 w 에 선형)
    g = pd.read_parquet(R.OUT / "hourly_dealer_gex.parquet")[["ts", "gex_week"]].rename(columns={"gex_week": "T"})
    for k, nm in (("D", "pass_direct"), ("R", "pass_rfq")):
        g = g.merge(pd.read_parquet(OUT / nm / "tblock_gex_hourly.parquet")[["ts", "gex_week"]].rename(columns={"gex_week": k}), on="ts")
    gT, gD, gR = g["T"].to_numpy(), g["D"].to_numpy(), g["R"].to_numpy(); n = len(g)
    sp = lambda a, b: round(float(pd.Series(a).corr(pd.Series(b), method="spearman")), 3)   # noqa: E731
    res["c_sign_error"] = {"spearman_week_2026": {"T~T_dflip": sp(gT, gT - 2 * gD), "T~T_direct": sp(gT, gD), "T~T_rfq": sp(gT, gR),
                                                  "T_rfq~T_direct": sp(gR, gD)},
                           "n_hours": n, "note": "T = 재구성 원본 시간 파일, 직접·RFQ = pass 시간 파일(같은 사양), ts 로 맞춤. T_dflip = T − 2·직접(GEX 는 w 에 선형)"}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    write_summary(res)


def write_summary(res: dict):
    MC, AG = res["matrix"]["cells"], res["matrix"]["aggregate"]
    cols = [("vol", "변동성"), ("rehedge", "재헤지"), ("hedge", "헤지 흔적"), ("demand", "수요압력"), ("expiry", "만기 소멸"), ("constraint", "물리 제약")]
    L = []
    for w, title in (("47d", "47일 공통 구간(2026-08-15~09-30)"), ("2026", "2026 전체(T 계열)")):
        L += [f"## {title}", "", "| 가정 | " + " | ".join(n for _, n in cols) + " | 지지/반대/불가 |", "|---" * (len(cols) + 2) + "|"]
        for a in ("T", "T_block", "Tflip", "C", "R", "T_rfq", "T_direct", "T_dflip"):
            c = MC[f"{a}|{w}"]; g = AG[f"{a}|{w}"]
            tag = " (보조)" if a in ("T_direct", "T_dflip") else ""
            L.append(f"| {a}{tag} | " + " | ".join(fmt(c[k]) for k, _ in cols) + f" | {g['지지']}/{g['반대']}/{g['불가']} |")
        L.append("")
    L += [f"T_rfq 판정: **{res['better_verdict']}**", "",
          "설명 비율 Σ|w|/ΣOI: " + " · ".join(f"{a} {v[0]}" for a, v in res["explain_sum_abs_over_oi"].items()), ""]
    (OUT / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


def selftest():
    # 1) RFQ 다리 부호: 콤보 «buy» 정의 = 콜 buy 1 · 풋 sell 2. 요청자가 sell 100 → 콜 −100 · 풋 +200
    r = {"direction": "sell", "amount": 100.0, "legs": [{"instrument_name": "C", "direction": "buy", "ratio": 1},
                                                        {"instrument_name": "P", "direction": "sell", "ratio": 2}]}
    assert req_legs(r) == {"C": -100.0, "P": 200.0}
    assert req_legs({**r, "direction": "buy"}) == {"C": 100.0, "P": -200.0}
    tp = pd.DataFrame({"ts_ms": [5, 5], "block_rfq_id": [7.0, 7.0], "instrument_name": ["C", "P"], "direction": ["sell", "buy"],
                       "amount": [100.0, 200.0]})
    c = direction_check([{**r, "id": 7, "timestamp": 5}], tp)
    assert c["rate_rows_agree"] == 1.0 and c["legs_qty_agree"] == 2 and c["by_leg_dir"]["sell"] == [1, 1]
    # 2) 시점 경계: t 와 같은 시각 체결은 포지션에 안 들어간다(ts < t), 1ms 앞은 들어간다 · 마스크 밖 체결은 0
    h = 3_600_000
    tr = pd.DataFrame({"instrument_name": ["X"] * 3, "timestamp": [h, 2 * h - 1, 2 * h], "trade_seq": [1, 2, 3], "q": [10.0, -3.0, 30.0],
                       "amount": [10.0, 3.0, 30.0], "iv": [50.0] * 3, "K": [100.0] * 3, "sg": [1.0] * 3, "exp_ms": [100 * h] * 3})
    P = A.positions(tr, np.array([2 * h]), {"T": np.ones(3, bool), "T_rfq": np.array([True, False, True])}).iloc[0]
    assert P["pos_T"] == -7.0 and P["pos_T_rfq"] == -10.0
    # 3) edge: 매수가 mark 위 = 양수(불리), 매도가 mark 위 = 음수(유리)
    e = edge_bp(pd.DataFrame({"k": [1, 2], "direction": ["buy", "sell"], "amount": [1.0, 1.0], "price": [0.011, 0.011],
                              "mark_price": [0.010, 0.010], "ts_ms": [0, 0]}), "k")["edge_bp"].to_numpy()
    assert np.allclose(e, [10.0, -10.0])
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else main()
