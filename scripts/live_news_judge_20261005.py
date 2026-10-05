#!/usr/bin/env python3
"""**뉴스 판정 워커** — 수집된 헤드라인을 jevk5:4b(ollaya, GPU)로 판정(2026-10-05, 사용자 «수집된 뉴스를 jevk5로 판정하는 것도 연결해줘»).

왜: 10-05 영어 헤드라인 16건 벤치에서 jevk5:4b 가 감성 15/16 · 자산 16/16 · 해킹·장애 완전 분리, GPU 4질문 189ms.
  작은 모델(laya·nli·0.8B)은 어려운 기사를 «중립»으로 도망갔다.
무엇: data/hot/news_rss.sqlite 의 items 중 판정 없는 것을 최신부터 읽어 judgments 에 쓴다(같은 DB, 수집기와 별도 프로세스).
  backfill 기사는 게시 24시간 안의 것만 판정(화면 첫 채움용). ollaya 가 죽어 있으면 30초 뒤 다시.
  본문 없는 글(이미지 게시물 «[No Title] - Post from …», 링크뿐인 트윗)은 모델에 안 보내고 model='skip:empty'(판정값 NULL)로 남긴다 --
  10-05 첫 판정에서 이런 글 21건을 모델이 «HYPE» 로 찍었다(억지 선택).
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
import urllib.error
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


def has_content(text: str) -> bool:
    """URL·«[No Title] - Post from …»·«이름 (@핸들):» 머리를 뺀 글에 단어가 2개 이상(또는 비ASCII 글자 4자 이상)인가.
    «Fed cuts 50»·«BUY BITCOIN» 같은 짧은 속보는 남기고, 링크·이미지뿐인 글만 뺀다."""
    t = re.sub(r"https?://\S+", " ", text)
    t = re.sub(r"\[No Title\] - Post from [^\n]*", " ", t)
    t = re.sub(r"^[^:\n]{1,60}\(@\w+\):", " ", t)
    return len(re.findall(r"[A-Za-z0-9$%.]+", t)) >= 2 or len(re.findall(r"[^\x00-\x7f\s]", t)) >= 4


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


def skip(con: sqlite3.Connection, source: str, link: str, why: str) -> None:
    con.execute("INSERT OR REPLACE INTO judgments(source, link, model, judged_ms, ms) VALUES(?,?,?,?,0)",
                (source, link, f"skip:{why}", int(time.time() * 1000)))


SKIP_CODES = {413, 422}           # «이 입력은 안 된다» -- 404·400 등은 서버·설정 문제라 기다린다(전부 skip 되면 판정 소실)
CANARY = "Bitcoin price rises after ETF inflows"


def judge_batch(con: sqlite3.Connection, batch: list[tuple], ask=None) -> bool:
    """한 묶음 판정. False = ollaya 가 응답 못 함(잠깐 쉬고 다시, 아무것도 넘기지 않는다). 넘기는 경우:
    본문 없음(skip:empty) · 입력 거부 413/422(skip:httpNNN) · 응답 모양 이상(skip:badresp) ·
    실패했는데 카나리 문장은 판정됨 = 이 글만 문제(skip:error). 안 넘기면 큐가 그 한 건에서 영원히 멈추고,
    반대로 서버가 죽었을 때 넘기면 그동안의 글이 전부 판정 없이 사라진다 -- 카나리로 둘을 가른다."""
    ask = ask or decide
    for source, link, text in batch:
        if not has_content(text):
            skip(con, source, link, "empty")
            continue
        t0 = time.time()
        try:
            ans = ask(text)
        except Exception as exc:  # noqa: BLE001 -- 원인은 카나리로 가른다
            code = getattr(exc, "code", None)
            if isinstance(exc, TimeoutError) or "timed out" in str(exc):   # 긴 글 시간 초과는 부하 탓일 수 있다 -- 넘기지 않고 다음 바퀴로
                print(f"판정 시간 초과 {source} {link}"[:200], flush=True)
                return False
            if code in SKIP_CODES:
                print(f"판정 거부 {code} {source} {link}"[:200], flush=True)
                skip(con, source, link, f"http{code}")
                continue
            try:
                ask(CANARY)
            except Exception as exc2:  # noqa: BLE001 -- 서버 쪽 문제: 남겨 두고 기다린다
                print(f"판정 실패(ollaya) {exc2!r}"[:200], flush=True)
                return False
            print(f"판정 실패(이 글만) {exc!r} {source} {link}"[:200], flush=True)
            skip(con, source, link, "error")
            continue
        try:
            row = row_of(source, link, ans, int((time.time() - t0) * 1000), int(time.time() * 1000))
        except (KeyError, TypeError) as exc:                    # ollaya 응답 모양이 바뀜 -- 죽지 않고 남긴다
            print(f"응답 모양 이상 {exc!r} {source} {link}"[:200], flush=True)
            skip(con, source, link, "badresp")
            continue
        con.execute("INSERT OR REPLACE INTO judgments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
    return True


def run() -> None:
    con = connect(DB)
    print(f"판정 {MODEL} via {URL} → {DB}", flush=True)
    while True:
        batch = pending(con, int(time.time() * 1000))
        ok = judge_batch(con, batch)
        time.sleep(30 if not ok else 1 if len(batch) == 20 else 5)


def selftest() -> None:
    import tempfile
    assert has_content("Fed cuts 50") and has_content("Trump (@realDonaldTrump): BUY BITCOIN") and has_content("비트코인 급락")
    assert not has_content("[No Title] - Post from October 4, 2026")
    assert not has_content("Donald J. Trump (@realDonaldTrump): https://x.co/u/abc")
    assert has_content("Donald J. Trump (@realDonaldTrump): The Kennedy Center is crumbling")
    assert has_content("Fed holds rates steady")
    assert text_of("A <b>x</b>", "A x more") == "A x" and text_of("T", "<p>body</p>") == "T\nbody"
    ans = {"sentiment": {"choice": "bearish", "probabilities": {"bullish": .05, "bearish": .9, "neutral": .05}},
           "asset": {"choice": "ETH", "confidence": .93}, "impact": {"score": 2.8},
           "security_incident": {"noul": .95}, "market_relevant": {"noul": .97}}
    with tempfile.TemporaryDirectory() as d:
        con = connect(Path(d) / "t.sqlite")
        con.execute("CREATE TABLE items(source, link, title, summary, categories, pub_ms, first_seen_ms, backfill)")
        now = 10 * BACKFILL_MAX_AGE_MS
        con.executemany("INSERT INTO items VALUES(?,?,?,?,?,?,?,?)", [
            ("s", "new", "New headline about ETH", "", "", now - 1000, now, 0),
            ("s", "oldbf", "Old backfill headline", "", "", now - 2 * BACKFILL_MAX_AGE_MS, now, 1),     # 오래된 backfill → 판정 안 함
            ("s", "recentbf", "Recent backfill headline", "", "", now - 3600_000, now, 1),
            ("s", "nopub", "No pubDate headline here", "", "", None, now - 5, 0)])
        assert [r[1] for r in pending(con, now)] == ["nopub", "new", "recentbf"], pending(con, now)
        con.execute("INSERT INTO judgments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row_of("s", "new", ans, 190, now))
        assert [r[1] for r in pending(con, now)] == ["nopub", "recentbf"]
        assert con.execute("SELECT sentiment, p_bear, asset, impact FROM judgments").fetchone() == ("bearish", .9, "ETH", 2.8)
        def fake(text):                                     # 한 건은 422(거부) · 한 건은 모양 이상 · 나머지는 정상
            if text.startswith("BAD"):
                raise urllib.error.HTTPError(URL, 422, "bad", None, None)
            return {"sentiment": {}} if text.startswith("ODD") else ans
        con.executemany("INSERT INTO items VALUES(?,?,?,?,?,?,?,?)", [
            ("s", "bad", "BAD input that is long", "", "", now, now, 0), ("s", "img", "[No Title] - Post from October 4, 2026", "", "", now, now, 0),
            ("s", "odd", "ODD response shape here", "", "", now, now, 0)])
        assert judge_batch(con, pending(con, now), fake)
        got = dict(con.execute("SELECT link, model FROM judgments").fetchall())
        assert got["bad"] == "skip:http422" and got["img"] == "skip:empty" and got["odd"] == "skip:badresp" and got["nopub"] == MODEL and got["recentbf"] == MODEL, got
        assert pending(con, now) == []                      # 큐가 비었다 -- 거부된 한 건에서 멈추지 않는다
        def down(text):
            raise urllib.error.URLError("refused")
        con.execute("INSERT INTO items VALUES('s','late','Late news here','', '', ?, ?, 0)", (now, now))
        assert not judge_batch(con, pending(con, now), down) and [r[1] for r in pending(con, now)] == ["late"]   # ollaya 죽음 = 남겨 둔다
        def nf(text):
            raise urllib.error.HTTPError(URL, 404, "model not found", None, None)
        for _ in range(5):                                   # 404·접속 불가가 몇 번이 와도 skip 하지 않는다(판정 소실 방지)
            assert not judge_batch(con, pending(con, now), nf) and not judge_batch(con, pending(con, now), down)
        assert [r[1] for r in pending(con, now)] == ["late"]
        def poison(text):                                    # 서버는 정상(카나리 성공)인데 이 글만 500
            if text == CANARY:
                return ans
            raise urllib.error.HTTPError(URL, 500, "tokenizer", None, None)
        con.execute("INSERT INTO items VALUES('s','slow','Slow long article here','', '', ?, ?, 0)", (now, now))
        def slow(text):
            if text == CANARY:
                return ans
            raise TimeoutError("timed out")
        assert not judge_batch(con, [r for r in pending(con, now) if r[1] == "slow"], slow)   # 시간 초과 = 남긴다(카나리가 돼도)
        assert judge_batch(con, [r for r in pending(con, now) if r[1] == "late"], poison)
        assert [r[1] for r in pending(con, now)] == ["slow"]
        assert con.execute("SELECT model FROM judgments WHERE link='late'").fetchone()[0] == "skip:error"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run()
