#!/usr/bin/env python3
"""**뉴스 판정 워커** — 수집된 헤드라인을 jevk5:4b(ollaya, GPU)로 판정(2026-10-05, 사용자 «수집된 뉴스를 jevk5로 판정하는 것도 연결해줘»).

왜: 10-05 영어 헤드라인 16건 벤치에서 jevk5:4b 가 감성 15/16 · 자산 16/16 · 해킹·장애 완전 분리, GPU 4질문 189ms.
  작은 모델(laya·nli·0.8B)은 어려운 기사를 «중립»으로 도망갔다.
무엇: data/hot/news_rss.sqlite 의 items 중 판정 없는 것을 최신부터 읽어 judgments 에 쓴다(같은 DB, 수집기와 별도 프로세스).
  backfill 기사는 게시 24시간 안의 것만 판정(화면 첫 채움용). ollaya 가 죽어 있으면 30초 뒤 다시.
  질문 5개: 감성(bullish/bearish/neutral) · 자산(BTC/ETH/SOL/XRP/HYPE/macro/other) · 영향 0~3 · 해킹·장애 · 시장 관련성.
  🔴 판정은 아직 검증 전이다 — 매매·푸시 근거로 쓰기 전에 판정 뒤 가격 반응으로 따로 검증한다.
의존: ollaya serve(scripts/ops/supervisor_ollaya.sh, 127.0.0.1:11435). 주문·바이낸스 호출 없음.
  python scripts/live_news_judge_20261005.py
  python scripts/live_news_judge_20261005.py --selftest
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("NEWS_RSS_DB", ROOT / "data/hot/news_rss.sqlite"))
URL = os.environ.get("OLLAYA_URL", "http://127.0.0.1:11435") + "/api/decide"
MODEL = os.environ.get("NEWS_JUDGE_MODEL", "jevk5:4b")
BACKFILL_MAX_AGE_MS = 24 * 3600 * 1000
QUESTIONS = {
    "sentiment": {"type": "choice", "instructions": "What is the likely short-term price impact of this news on crypto markets?",
                  "criteria": {"bullish": "Positive for prices", "bearish": "Negative for prices", "neutral": "Little or no price impact"}},
    "asset": {"type": "choice", "instructions": "Which asset is this news mainly about?",
              "criteria": {"BTC": "Bitcoin", "ETH": "Ethereum / Ether", "SOL": "Solana", "XRP": "Ripple / XRP", "HYPE": "Hyperliquid / HYPE",
                           "macro": "Macroeconomy, central banks, rates, inflation, geopolitics", "other": "Other coins, companies or topics"}},
    "impact": {"type": "score", "instructions": "How large is the market impact?",
               "criteria": ["None", "Minor", "Moderate", "Major market-moving event"]},
    "security_incident": {"type": "noul", "instructions": "The news reports a hack, exploit or outage.",
                          "criteria": {"true": "Hack, exploit or outage", "false": "No hack, exploit or outage"}},
    "market_relevant": {"type": "noul", "instructions": "The news is relevant to crypto or financial markets.",
                        "criteria": {"true": "Relevant to markets", "false": "Not relevant to markets"}},
}


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS judgments(source TEXT, link TEXT, model TEXT, sentiment TEXT, p_bull REAL, p_bear REAL,
        p_neu REAL, asset TEXT, asset_p REAL, impact REAL, incident REAL, relevant REAL, judged_ms INTEGER, ms INTEGER,
        PRIMARY KEY(source, link))""")
    return con


def pending(con: sqlite3.Connection, now_ms: int, limit: int = 20) -> list[tuple]:
    """판정 없는 기사(source, link, 입력 글) 최신부터. backfill 은 게시 24시간 안만."""
    rows = con.execute("""SELECT i.source, i.link, i.title, i.summary FROM items i
        LEFT JOIN judgments j ON j.source = i.source AND j.link = i.link
        WHERE j.link IS NULL AND (i.backfill = 0 OR coalesce(i.pub_ms, i.first_seen_ms) >= ?)
        ORDER BY coalesce(i.pub_ms, i.first_seen_ms) DESC LIMIT ?""", (now_ms - BACKFILL_MAX_AGE_MS, limit)).fetchall()
    return [(s, l, text_of(t, m)) for s, l, t, m in rows]


def text_of(title: str, summary: str) -> str:
    """제목 + 요약(태그 떼고, 제목과 같으면 생략). 2,000자로 자른다."""
    clean = lambda x: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", x or "")).strip()
    t, m = clean(title), clean(summary)
    return (t if not m or m.startswith(t[:60]) else f"{t}\n{m}")[:2000]


def row_of(source: str, link: str, ans: dict, ms: int, now_ms: int) -> tuple:
    p = ans["sentiment"]["probabilities"]
    return (source, link, MODEL, ans["sentiment"]["choice"], p.get("bullish"), p.get("bearish"), p.get("neutral"),
            ans["asset"]["choice"], ans["asset"]["confidence"], ans["impact"]["score"], ans["security_incident"]["noul"],
            ans["market_relevant"]["noul"], now_ms, ms)


def decide(text: str) -> dict:
    body = json.dumps({"model": MODEL, "state": text, "questions": QUESTIONS, "keep_alive": "30m"}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["answers"]


def run() -> None:
    con = connect(DB)
    print(f"판정 {MODEL} via {URL} → {DB}", flush=True)
    while True:
        batch = pending(con, int(time.time() * 1000))
        for source, link, text in batch:
            t0 = time.time()
            try:
                ans = decide(text)
            except Exception as exc:  # noqa: BLE001 -- ollaya 재기동·로드 중이면 기다렸다 다시(판정 안 된 행은 남는다)
                print(f"판정 실패 {exc!r}"[:200], flush=True)
                time.sleep(30)
                break
            con.execute("INSERT OR REPLACE INTO judgments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        row_of(source, link, ans, int((time.time() - t0) * 1000), int(time.time() * 1000)))
        time.sleep(1 if len(batch) == 20 else 5)


def selftest() -> None:
    import tempfile
    assert text_of("A <b>x</b>", "A x more") == "A x" and text_of("T", "<p>body</p>") == "T\nbody"
    ans = {"sentiment": {"choice": "bearish", "probabilities": {"bullish": .05, "bearish": .9, "neutral": .05}},
           "asset": {"choice": "ETH", "confidence": .93}, "impact": {"score": 2.8},
           "security_incident": {"noul": .95}, "market_relevant": {"noul": .97}}
    with tempfile.TemporaryDirectory() as d:
        con = connect(Path(d) / "t.sqlite")
        con.execute("CREATE TABLE items(source, link, title, summary, categories, pub_ms, first_seen_ms, backfill)")
        now = 10 * BACKFILL_MAX_AGE_MS
        con.executemany("INSERT INTO items VALUES(?,?,?,?,?,?,?,?)", [
            ("s", "new", "N", "", "", now - 1000, now, 0),
            ("s", "oldbf", "O", "", "", now - 2 * BACKFILL_MAX_AGE_MS, now, 1),     # 오래된 backfill → 판정 안 함
            ("s", "recentbf", "R", "", "", now - 3600_000, now, 1),
            ("s", "nopub", "P", "", "", None, now - 5, 0)])
        assert [r[1] for r in pending(con, now)] == ["nopub", "new", "recentbf"], pending(con, now)
        con.execute("INSERT INTO judgments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row_of("s", "new", ans, 190, now))
        assert [r[1] for r in pending(con, now)] == ["nopub", "recentbf"]
        assert con.execute("SELECT sentiment, p_bear, asset, impact FROM judgments").fetchone() == ("bearish", .9, "ETH", 2.8)
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run()
