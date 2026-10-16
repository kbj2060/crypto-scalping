#!/usr/bin/env python3
"""**jevk5:4b 가 «베이시스 할인 해석»을 할 수 있나** — 읽기 · 지식 · 해석 세 층 객관식 시험(2026-10-16).

왜: 사용자 «선물 할인이 깊으면 누르는 세력?» 질문의 답(OI·청산·현물 CVD·헤지로 분해)을 로컬 AI 가 화면 보고 해 줄 수 있나,
  «소규모 모델은 불가능할 것 같다» → 먼저 분석.
사실: ollaya 0.9.0 는 판정 전용(capabilities choice·score·noul, /api/generate·/api/chat 404, 영어만). 문장은 못 쓴다 →
  «해석 = 상황 고르기» 로만 잴 수 있다.
층: ① 읽기(noul 4개: OI↑? 롱청산>숏청산? 현물 순매수? 선물<현물?) ② 지식 객관식 12문항 × 보기 순서 3가지
  ③ 해석 5종(신규 숏 · 롱 강제청산 · 현물 견인 · 구조적 헤지 · 할인 아님) × 25 합성 상황 × {bare: 짧은 이름만, defs: 판별 규칙 포함}.
  정답은 생성기가 정한다(정의상 분해라 예측 아님). 보기 순서는 매 질의 섞는다(위치 편향 차단).
🔴 서버 GPU — 질의 사이 --pace 초(기본 2), 오류는 재시도 없이 죽는다(10-05 CUDA 고장 사고). 터널: ssh -f -N -L 11435:127.0.0.1:11435 -p 2222 llewyn@192.168.1.89
  python scripts/research_llm_jevk5_basis_read_20261016.py run
  python scripts/research_llm_jevk5_basis_read_20261016.py eval
  python scripts/research_llm_jevk5_basis_read_20261016.py --selfcheck
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/llm_jevk5_basis_read_20261016"
URL = os.environ.get("OLLAYA_URL", "http://127.0.0.1:11435") + "/api/decide"
MODEL = "jevk5:4b"

CLASSES = {  # 이름(bare) · 판별 규칙(defs)
    "new_shorts": ("Traders are opening new short positions in perpetual futures",
                   "New shorts: perp discount while open interest RISES, few liquidations, perp takers sell while spot flow is flat"),
    "long_liq": ("Leveraged longs are being forcibly liquidated",
                 "Forced long liquidation: perp discount while open interest FALLS sharply, long liquidations are large, price drops hard"),
    "spot_led": ("Spot buying is pulling spot above futures",
                 "Spot-led: open interest flat, spot takers buy heavily, Coinbase premium positive, price rises while perp lags"),
    "hedge": ("Structural hedging: delta-neutral funds keep a persistent futures short",
              "Structural hedge/carry: discount has persisted for days with persistently negative funding, flat open interest, small flows and liquidations"),
    "none": ("No unusual futures-side pressure; the basis is normal",
             "Normal: basis is near its usual level (not in the low percentiles of the past 7 days)"),
}

KNOWLEDGE = [  # (질문, 정답, 오답들)
    ("When a perpetual future trades below spot for a long time, the funding rate is typically:",
     "Negative (shorts pay longs)", ["Positive (longs pay shorts)", "Exactly zero"]),
    ("Why can a perpetual-futures discount persist longer than a premium?",
     "Arbitraging a discount requires shorting spot, which is costly or constrained",
     ["Exchanges forbid buying futures at a discount", "Discounts are always closed within one minute"]),
    ("Price falls sharply, open interest drops 5%, and long liquidations spike. The selling is mainly:",
     "Forced closing of leveraged longs", ["New short positions being opened", "Spot buyers absorbing supply"]),
    ("Price falls while open interest rises 3%. This most likely means:",
     "New short positions are being opened", ["Longs are closing positions", "Shorts are covering"]),
    ("Price rises while open interest falls. This most likely means:",
     "Shorts are covering (closing positions)", ["New longs are opening", "New shorts are opening"]),
    ("Price rises while open interest rises. This most likely means:",
     "New long positions are being opened", ["Shorts are covering", "Longs are being liquidated"]),
    ("Funds that hold staked ETH and short ETH perpetuals to stay delta-neutral push the perp basis:",
     "Lower (toward a discount, lower funding)", ["Higher (toward a premium)", "They have no effect on the basis"]),
    ("In perpetual futures, every open short position is matched by:",
     "An open long position held by someone else", ["The exchange's own inventory", "Nothing; shorts need no counterparty"]),
    ("Cash-and-carry arbitrage when the perp trades ABOVE spot is to:",
     "Buy spot and short the perp", ["Short spot and buy the perp", "Buy both spot and the perp"]),
    ("A positive Coinbase premium means:",
     "US spot buyers are paying more than offshore markets", ["Futures trade above spot on Binance", "Coinbase charges higher fees"]),
    ("The basis (perp minus spot) falls from 0 to -8 bp while the perp price is unchanged and spot rises. The discount is:",
     "Spot-led (spot moved up)", ["Caused by aggressive perp selling", "Caused by long liquidations"]),
    ("Does a deep futures discount by itself prove that someone intends to push the price down?",
     "No, hedging, liquidations or spot-led moves can create it too", ["Yes, a discount always means intentional selling"]),
]


def u(r: random.Random, a: float, b: float) -> float:
    return r.uniform(a, b)


def scenario(cls: str, r: random.Random) -> dict:
    """정답 cls 인 합성 상황(지난 1시간). 값 범위가 곧 판별 규칙이다."""
    s = dict(basis_bp=u(r, -9, -3), basis_pct=u(r, 1, 8), disc_h=u(r, 0.3, 3), oi_pct=u(r, -0.5, 0.5),
             long_liq=u(r, 0.2, 1.5), short_liq=u(r, 0.2, 1.5), perp_cvd=u(r, -8, 8), spot_cvd=u(r, -5, 5),
             cb_bp=u(r, -3, 3), fund=u(r, 0.0, 0.01), px_bp=u(r, -15, 15))
    if cls == "new_shorts":
        s.update(oi_pct=u(r, 1.5, 5), perp_cvd=u(r, -80, -20), px_bp=u(r, -60, -10))
    elif cls == "long_liq":
        s.update(oi_pct=u(r, -6, -2), long_liq=u(r, 15, 80), perp_cvd=u(r, -150, -40), spot_cvd=u(r, -40, -10), px_bp=u(r, -250, -80))
    elif cls == "spot_led":
        s.update(spot_cvd=u(r, 20, 80), cb_bp=u(r, 5, 20), px_bp=u(r, 20, 80), perp_cvd=u(r, -5, 10))
    elif cls == "hedge":
        s.update(disc_h=u(r, 24, 120), basis_pct=u(r, 3, 15), fund=u(r, -0.02, -0.005))
    elif cls == "none":
        s.update(basis_bp=u(r, -1, 2), basis_pct=u(r, 40, 70), disc_h=0.0)
    return s


def facts(s: dict) -> dict:
    return {"oi_up": s["oi_pct"] > 0, "longliq_more": s["long_liq"] > s["short_liq"], "spot_buy": s["spot_cvd"] > 0, "perp_below": s["basis_bp"] < 0}


def text(s: dict) -> str:
    disc = f"; the discount has persisted for {s['disc_h']:.1f} hours" if s["disc_h"] > 0 else ""
    return "\n".join([
        "ETH market snapshot (last 1 hour):",
        f"- Perp minus spot basis: {s['basis_bp']:+.1f} bp (percentile rank {s['basis_pct']:.0f} of 100 over the past 7 days){disc}",
        f"- Open interest change: {s['oi_pct']:+.1f}%",
        f"- Long liquidations: ${s['long_liq']:.1f}M; short liquidations: ${s['short_liq']:.1f}M",
        f"- Perp taker net buy (CVD): {s['perp_cvd']:+.0f}M USD; spot taker net buy (CVD): {s['spot_cvd']:+.0f}M USD",
        f"- Coinbase premium: {s['cb_bp']:+.1f} bp",
        f"- Funding rate (8h): {s['fund']:+.4f}%",
        f"- Price change: {s['px_bp']:+.0f} bp",
    ])


FACT_Q = {
    "oi_up": "Open interest increased over the last hour.",
    "longliq_more": "Long liquidations were larger than short liquidations.",
    "spot_buy": "Spot takers were net buyers.",
    "perp_below": "The perpetual future is trading below spot.",
}


def questions(cond: str, r: random.Random) -> dict:
    keys = list(CLASSES)
    r.shuffle(keys)
    q = {"situation": {"type": "choice", "instructions": "Which best explains the current state of the futures basis?",
                       "criteria": {k: CLASSES[k][0 if cond == "bare" else 1] for k in keys}}}
    for k, t in FACT_Q.items():
        q[k] = {"type": "noul", "instructions": t, "criteria": {"true": "True", "false": "False"}}
    return q


def decide(state: str, qs: dict) -> dict:
    body = json.dumps({"model": MODEL, "state": state, "questions": qs, "keep_alive": "30m"}).encode()
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=120) as resp:
        return json.load(resp)["answers"]


def jobs(n: int = 25) -> list[dict]:
    r = random.Random(20261016)
    out = []
    for i, (q, a, wrong) in enumerate(KNOWLEDGE):
        for rep in range(3):
            opts = [a] + wrong
            r.shuffle(opts)
            out.append({"id": f"k{i}_{rep}", "layer": "knowledge", "gold": "a", "state": q,
                        "qs": {"answer": {"type": "choice", "instructions": "Choose the correct answer.",
                                          "criteria": {("a" if o == a else f"w{wrong.index(o)}"): o for o in opts}}}})
    for cls in CLASSES:
        for j in range(n):
            s = scenario(cls, r)
            for cond in ("bare", "defs"):
                out.append({"id": f"{cls}_{j}_{cond}", "layer": cond, "gold": cls, "facts": facts(s), "state": text(s), "qs": questions(cond, r)})
    return out


def run(pace: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "answers.jsonl"
    done = {json.loads(x)["id"] for x in cache.open()} if cache.exists() else set()
    todo = [j for j in jobs() if j["id"] not in done]
    with cache.open("a") as fh:
        for k, j in enumerate(todo, 1):
            ans = decide(j["state"], j["qs"])
            fh.write(json.dumps({**{x: j[x] for x in ("id", "layer", "gold")}, "facts": j.get("facts"), "ans": ans}) + "\n")
            fh.flush()
            time.sleep(pace)
            if k % 50 == 0:
                print(f"{k}/{len(todo)}", flush=True)
    print(f"끝 {len(todo)} 건", flush=True)


def top(a: dict) -> str:
    if a["type"] == "noul":  # noul = P(true)
        return "true" if a["noul"] > 0.5 else "false"
    p = a["probabilities"]
    return max(p, key=p.get)


def evaluate() -> None:
    rows = [json.loads(x) for x in (OUT / "answers.jsonl").open()]
    kn = [r for r in rows if r["layer"] == "knowledge"]
    by_q: dict[str, list[bool]] = {}
    for r in kn:
        by_q.setdefault(r["id"].split("_")[0], []).append(top(r["ans"]["answer"]) == "a")
    print(f"① 지식: {sum(map(sum, by_q.values()))}/{len(kn)} 정답 · 문항별(3순서 다 맞음) {sum(all(v) for v in by_q.values())}/{len(by_q)}")
    for q, v in by_q.items():
        print(f"   {q} {''.join('O' if x else 'x' for x in v)}  {KNOWLEDGE[int(q[1:])][0][:90]}")
    sc = [r for r in rows if r["layer"] in ("bare", "defs")]
    for f in FACT_Q:
        ok = [top(r["ans"][f]) == str(r["facts"][f]).lower() for r in sc]
        print(f"② 읽기 {f:13s} {sum(ok)}/{len(ok)} = {sum(ok) / len(ok):.0%}")
    for cond in ("bare", "defs"):
        rs = [r for r in sc if r["layer"] == cond]
        ok = sum(top(r["ans"]["situation"]) == r["gold"] for r in rs)
        print(f"③ 해석 {cond}: {ok}/{len(rs)} = {ok / len(rs):.0%} (무작위 20%)")
        print("   정답\\답 " + " ".join(f"{c[:8]:>9s}" for c in CLASSES))
        for g in CLASSES:
            row = [sum(top(r["ans"]["situation"]) == c for r in rs if r["gold"] == g) for c in CLASSES]
            print(f"   {g:9s} " + " ".join(f"{x:9d}" for x in row))


def _selfcheck() -> None:
    r = random.Random(0)
    for _ in range(200):  # 생성기가 정의대로 판별되는지
        assert scenario("new_shorts", r)["oi_pct"] >= 1.5
        s = scenario("long_liq", r)
        assert s["oi_pct"] <= -2 and s["long_liq"] > s["short_liq"] and s["px_bp"] < -50
        s = scenario("spot_led", r)
        assert s["spot_cvd"] >= 20 and s["cb_bp"] >= 5 and abs(s["oi_pct"]) <= 0.5
        s = scenario("hedge", r)
        assert s["disc_h"] >= 24 and s["fund"] < 0
        s = scenario("none", r)
        assert s["basis_pct"] >= 40 and s["disc_h"] == 0
    js = jobs()
    assert len(js) == 12 * 3 + 5 * 25 * 2 and len({j["id"] for j in js}) == len(js)
    assert all("a" in j["qs"]["answer"]["criteria"] for j in js if j["layer"] == "knowledge")
    assert len({tuple(j["qs"]["situation"]["criteria"]) for j in js if j["layer"] == "bare"}) > 10  # 순서가 섞였다
    print("selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["run", "eval"])
    ap.add_argument("--pace", type=float, default=2.0)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    elif a.cmd == "run":
        run(a.pace)
    elif a.cmd == "eval":
        evaluate()
