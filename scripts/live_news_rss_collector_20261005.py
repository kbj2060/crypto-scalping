#!/usr/bin/env python3
"""**뉴스 RSS 수집기** — The Block·Cointelegraph·CoinDesk 헤드라인(2026-10-05, 사용자 «시장 뉴스 및 속보를 읽고 싶다»).

왜: 세 사이트 모두 공식 RSS 가 웹 목록과 같다(10-05 실측 상위 15건 15/15). The Block 웹은 Cloudflare 확인 페이지라 RSS 뿐.
  먼저 «게시 시각(pubDate) → 우리가 처음 본 시각» 지연을 잰다 -- 속보로 쓸 만한지는 이 숫자로 판단한다.
  응답 `age` 헤더(CDN 캐시 나이)도 남긴다: The Block 은 max-age 60 인데 age 1974 가 찍혔다(10-05).
무엇: items = 기사 한 건(source, link, 제목, 요약, 카테고리, pub_ms, first_seen_ms, backfill) -- 첫 폴링에 이미 있던 기사는
  backfill=1(지연 통계에서 뺀다). polls = 폴링 한 번(시각, source, HTTP 상태, 기사 수, 소요 ms, age).
저장: data/hot/news_rss.sqlite(WAL). 피드마다 주기(초)대로 조건부 GET(ETag/Last-Modified). 주문·바이낸스 호출 없음.
10-05 속보 피드 추가(사용자 «거래소 공지와 하이퍼리퀴드 빼고»): Tree News(JSON, 트윗·Truth·뉴스 사이트 미러) · BWEnews ·
  FinancialJuice(매크로) · Truth Social 아카이브(trumpstruth) · Fed · SEC. BWEnews 에는 «UPBIT LISTING» 같은 상장 속보가 섞여 온다
  (원천 그대로 저장, 거르는 건 읽는 쪽에서). Tree 는 link 가 원문 url(트윗 등)이다.
  python scripts/live_news_rss_collector_20261005.py
  python scripts/live_news_rss_collector_20261005.py --selftest
"""
from __future__ import annotations

import email.utils
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("NEWS_RSS_DB", ROOT / "data/hot/news_rss.sqlite"))
FEEDS = {                                                     # 이름: (url, 주기 초)
    "theblock": ("https://www.theblock.co/rss.xml", 60),
    "cointelegraph": ("https://cointelegraph.com/rss", 60),
    "coindesk": ("https://www.coindesk.com/arc/outboundfeeds/rss", 60),
    "tree": ("https://news.treeofalpha.com/api/news?limit=50", 15),
    "bwenews": ("https://rss-public.bwe-ws.com/", 20),            # 10건뿐이라 짧게
    "financialjuice": ("https://www.financialjuice.com/feed.ashx?xy=rss", 30),
    "trumpstruth": ("https://www.trumpstruth.org/feed", 60),
    "fed": ("https://www.federalreserve.gov/feeds/press_all.xml", 300),
    "sec": ("https://www.sec.gov/news/pressreleases.rss", 300),
}
TICK_S = float(os.environ.get("NEWS_RSS_TICK_S", "5"))
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"


def parse(xml: bytes) -> list[tuple]:
    """RSS → (link, title, summary, categories, pub_ms). link 은 쿼리 떼고 끝 / 뗀 값(키)."""
    out = []
    for it in ET.fromstring(xml).iter("item"):
        link = (it.findtext("link") or it.findtext("guid") or "").strip().split("?")[0].rstrip("/")
        if not link:
            continue
        pub = it.findtext("pubDate")
        pub_ms = int(email.utils.parsedate_to_datetime(pub).timestamp() * 1000) if pub else None
        summary = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", it.findtext("description") or "")).strip()
        cats = ",".join(c.text.strip() for c in it.findall("category") if c.text)
        out.append((link, (it.findtext("title") or "").strip(), summary, cats, pub_ms))
    return out


def parse_tree(body: bytes) -> list[tuple]:
    """Tree News /api/news JSON → parse() 와 같은 행. 카테고리 = 출처 종류(Twitter·Blogs·usGov)."""
    return [(x.get("url") or f"tree:{x['_id']}", (x.get("title") or "").strip(), "", x.get("source") or "",
             int(x["time"]) if x.get("time") else None) for x in json.loads(body)]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS items(source TEXT, link TEXT, title TEXT, summary TEXT, categories TEXT,
        pub_ms INTEGER, first_seen_ms INTEGER, backfill INTEGER, PRIMARY KEY(source, link))""")
    con.execute("CREATE TABLE IF NOT EXISTS polls(ts_ms INTEGER, source TEXT, status INTEGER, n_items INTEGER, ms INTEGER, age_s INTEGER)")
    return con


def store(con: sqlite3.Connection, source: str, items: list[tuple], seen_ms: int, backfill: bool) -> int:
    """새 기사만 넣는다(이미 본 link 는 first_seen 을 바꾸지 않는다). 새로 들어간 수."""
    before = con.total_changes
    con.executemany("INSERT OR IGNORE INTO items VALUES(?,?,?,?,?,?,?,?)",
                    [(source, *it, seen_ms, int(backfill)) for it in items])
    return con.total_changes - before


def run() -> None:
    con = connect(DB)
    cond: dict[str, dict] = {s: {} for s in FEEDS}            # 조건부 요청 헤더
    first = {s: con.execute("SELECT count(*) FROM items WHERE source=?", (s,)).fetchone()[0] == 0 for s in FEEDS}
    due = {s: 0.0 for s in FEEDS}                               # 다음 폴링 시각
    print(f"수집 {', '.join(f'{s}({e}s)' for s, (_, e) in FEEDS.items())} → {DB}", flush=True)
    while True:
        for src, (url, every) in FEEDS.items():
            t0 = time.time()
            if t0 < due[src]:
                continue
            due[src] = t0 + every
            status, n, age = 0, 0, None
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA, **cond[src]})
                with urllib.request.urlopen(req, timeout=20) as r:
                    status, body = r.status, r.read()
                    age = int(r.headers["Age"]) if (r.headers.get("Age") or "").isdigit() else None
                    cond[src] = {k: v for k, v in (("If-None-Match", r.headers.get("ETag")),
                                                     ("If-Modified-Since", r.headers.get("Last-Modified"))) if v}
                items = parse_tree(body) if src == "tree" else parse(body)
                n = len(items)
                new = store(con, src, items, int(time.time() * 1000), first[src])
                if new and not first[src]:
                    print(f"{time.strftime('%H:%M:%S')} {src} 새 기사 {new}", flush=True)
                first[src] = False
            except urllib.error.HTTPError as exc:
                status = exc.code                                  # 304 = 바뀐 것 없음
                if exc.code != 304:
                    print(f"{src} HTTP {exc.code}", flush=True)
            except Exception as exc:  # noqa: BLE001 -- 한 사이트 실패가 나머지를 막지 않게 -- polls 에 status 0 으로 남는다
                print(f"{src} 실패 {exc!r}"[:200], flush=True)
            con.execute("INSERT INTO polls VALUES(?,?,?,?,?,?)",
                        (int(t0 * 1000), src, status, n, int((time.time() - t0) * 1000), age))
        time.sleep(TICK_S)


def selftest() -> None:
    import tempfile
    xml = b"""<rss><channel>
      <item><title> A </title><link>https://x.co/a/?utm=1</link><pubDate>Sun, 04 Oct 2026 13:48:48 +0000</pubDate>
        <description>&lt;p&gt;Hello  &lt;b&gt;world&lt;/b&gt;&lt;/p&gt;</description><category>DeFi</category><category>Markets</category></item>
      <item><title>B</title><guid>https://x.co/b</guid></item>
      <item><title>no link</title></item></channel></rss>"""
    rows = parse(xml)
    assert rows == [("https://x.co/a", "A", "Hello world", "DeFi,Markets", 1791121728000),
                    ("https://x.co/b", "B", "", "", None)], rows
    with tempfile.TemporaryDirectory() as d:
        con = connect(Path(d) / "t.sqlite")
        assert store(con, "s", rows, 1, True) == 2
        assert store(con, "s", rows + [("https://x.co/c", "C", "", "", 5)], 9, False) == 1   # a·b 는 그대로
        assert con.execute("SELECT first_seen_ms, backfill FROM items WHERE link='https://x.co/a'").fetchone() == (1, 1)
        assert con.execute("SELECT first_seen_ms, backfill FROM items WHERE link='https://x.co/c'").fetchone() == (9, 0)
    tree = json.dumps([{"_id": "1", "title": "Trump (@realDonaldTrump): hi", "source": "Twitter", "url": "https://x.com/a/1", "time": 7},
                       {"_id": "2", "title": "B", "source": "Blogs"}]).encode()
    assert parse_tree(tree) == [("https://x.com/a/1", "Trump (@realDonaldTrump): hi", "", "Twitter", 7), ("tree:2", "B", "", "Blogs", None)]
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run()
