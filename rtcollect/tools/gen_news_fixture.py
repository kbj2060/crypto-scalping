"""뉴스 RSS 정답지: 원본 live_news_rss_collector_20261005.run() 을 가짜 urlopen·가짜 시계(5초 바퀴 90번)로.

피드 8개 각각 차례대로 정한 응답: 200(ETag·Last-Modified·Age) · 304 · 429(주기 4배) · 500 · 접속 실패 · 깨진 XML · 모르는 개체(&nbsp;) ·
  이름공간(atom:link·dc·기본 xmlns) · CDATA·HTML 설명 · 날짜 여러 꼴 · 링크 쿼리·끝 / · guid · 카테고리 · 이중 이스케이프 · CRLF · BWE 거르기 · Tree JSON.
미리 있는 DB: fed 기사(= backfill 아님) · 8일 전·1일 전 polls(앞 것만 지워진다).
출력 = tests/fixtures/news_{input,golden}.json. 바깥 접속 없음.   python3 rtcollect/tools/gen_news_fixture.py
"""
from __future__ import annotations

import email.message
import io
import json
import sqlite3
import sys
import tempfile
import time
import types
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_news_rss_collector_20261005 as S  # noqa: E402

FIX = ROOT / "rtcollect/tests/fixtures"
T0 = 1_795_000_000.0
TICKS = 90


def rss(items: str, head: str = "<rss><channel>", tail: str = "</channel></rss>") -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{head}{items}{tail}'


def item(title, link=None, guid=None, pub=None, desc=None, cats=(), extra=""):
    s = "<item>"
    if title is not None:
        s += f"<title>{title}</title>"
    if link is not None:
        s += f"<link>{link}</link>"
    if guid is not None:
        s += f"<guid>{guid}</guid>"
    if pub is not None:
        s += f"<pubDate>{pub}</pubDate>"
    if desc is not None:
        s += f"<description>{desc}</description>"
    s += "".join(f"<category>{c}</category>" for c in cats) + extra + "</item>"
    return s


DATES = ["Sun, 04 Oct 2026 13:48:48 +0000", "Mon, 3 Nov 2025 14:00:00 EST", "3 Nov 25 14:00 GMT", "Monday, 03-Nov-25 14:00:00 GMT",
         "Tue, 4 Nov 2025 09:30:00 -0000", "Tue, 04 Nov 2025 25:00:00 +0000", "Nov 4 2025 10:00:00 +0530", "2025-11-03", "Wed, 05 Nov 2025 01:02",
         "Thu, 06 Nov 2025 10.20.30 PDT", "Fri,07 Nov 2025 08:00:00 +0100", "Sat, 08 Nov 1970 00:00:00 +2500", "Sun, 9 November 2025 07:07:07 UT",
         "Sun, 09 Nov 2025 07:07:60 +0000", "", "   ", "Tue, 31 Feb 2026 00:00:00 +0000", "1 Jan 69 00:00:00 +0000", "1 Jan 68 00:00:00 +0000"]


def plan() -> dict:
    """피드마다 호출 차례별 응답(모자라면 마지막 것을 되풀이)."""
    tb = [rss("".join(item(f"Block {i} &amp; co", link=f"https://www.theblock.co/post/{i}/?utm=rss", pub=DATES[i % len(DATES)],
                           desc=f"<![CDATA[<p>Hello  <b>world</b> {i}</p>\n<img src='x'/>]]>", cats=("DeFi", "  Markets ", "")) for i in range(6)))]
    ok = lambda body, **h: {"kind": "ok", "status": 200, "body": body, "headers": h}  # noqa: E731
    http = lambda code: {"kind": "http", "status": code}  # noqa: E731
    P = {
        "theblock": [ok(tb[0], ETag='W/"abc"', Age="12"), http(304), ok(rss(item("Block 9", link="https://www.theblock.co/post/9")) + "\n", ETag='"d2"', Age="x7"),
                     http(429), ok(tb[0], **{"Last-Modified": "Wed, 05 Nov 2025 01:02:03 GMT"})],
        "cointelegraph": [ok(rss(item("CT &amp;amp; Friends", link="https://cointelegraph.com/news/a", pub=DATES[1],
                                      extra='<atom:link href="https://x/y" rel="self"/><dc:creator>Anon</dc:creator>')
                                 + item("&#169; numeric &#x2014; ref", guid="https://cointelegraph.com/news/b/", pub=DATES[2])
                                 + '<atom:item><title>ns item</title><link>https://ns/1</link></atom:item>'
                                 + item("outer", link="https://cointelegraph.com/outer", extra=item("inner", link="https://cointelegraph.com/inner")),
                                 head='<rss xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>'),
                             **{"Last-Modified": "Thu, 06 Nov 2025 00:00:00 GMT"}), http(304)],
        "coindesk": [ok(rss(item("CD 1", link="https://www.coindesk.com/a/", pub=DATES[3], desc="&lt;p&gt;x&lt;/p&gt; &amp;nbsp; y"))),
                     ok("<rss><channel><item><title>broken</title></channel></rss>"), ok(rss(item("bad &nbsp; entity", link="https://www.coindesk.com/b"))),
                     ok(rss(item("CD 2", link="https://www.coindesk.com/c", pub=DATES[4]), head='<feed xmlns="http://www.w3.org/2005/Atom"><rss><channel>', tail="</channel></rss></feed>")),
                     ok(rss(item("CD 3", link="https://www.coindesk.com/d", pub=DATES[5])))],
        "tree": [ok(json.dumps([{"_id": "1", "title": "Trump (@realDonaldTrump): hi &amp; bye", "source": "Twitter", "url": "https://x.com/a/1", "time": 1795000000123},
                                {"_id": 2, "title": "B", "source": "Blogs", "time": 1795000000999.7}, {"_id": "3", "title": None, "url": "", "time": 0},
                                {"_id": "4", "title": "  spaced　", "source": None, "time": "1795000001000"}])),
                 ok("not json"), http(429), ok(json.dumps([{"_id": "5", "title": "T5", "url": "https://x.com/a/5", "time": 1795000002000}]))],
        "bwenews": [ok(rss("".join(item(t, link=f"https://bwe/{i}", pub=DATES[(i + 6) % len(DATES)]) for i, t in enumerate(
                        ["UPBIT LISTING：돌핀", "Upbit 上新：关于 XYZ", "Binance EN: Binance Futures Will Delist", "OKX: OKX to list X",
                         "Tree News: *Citi Partners With Coinbase", "AggrNews: METAMASK RESPONDING", "BWENEWS: patched", "Tree News: Binance lists nothing: denial",
                         "  bithumb Listing: [마켓 추가", "GATEIO: not gate"])))),
                    ok(rss(item("BWENEWS: new one", link="https://bwe/new")))],
        "financialjuice": [{"kind": "urlerror"}, http(500), ok(rss(item("FJ 1", link="https://fj/1?x=1&amp;y=2", pub=DATES[7])))],
        "fed": [ok(rss(item("Fed\r\n statement", link="https://www.federalreserve.gov/a\r\n", pub=DATES[8], desc="line1\r\nline2\rline3") +
                       item("Fed old", link="https://www.federalreserve.gov/old", pub=DATES[9])).replace("\n", "\r\n")),
                ok(rss("".join(item(f"Fed d{i}", link=f"https://www.federalreserve.gov/d{i}", pub=d) for i, d in enumerate(DATES))))],
        "sec": [ok(rss(item("SEC 1", link="", guid=" https://www.sec.gov/news/1/ ", pub=DATES[10], desc="pre <b>bold</b> tail",
                            extra="<!-- c --><category><![CDATA[Enf]]>orcement</category><category>A<x/>B</category>") +
                       item("no link at all") + item(None, link="https://www.sec.gov/news/notitle"))),
                http(304)],
    }
    return P


class Resp:
    def __init__(self, sp):
        self.status = sp["status"]
        self._b = sp["body"].encode()
        m = email.message.Message()
        for k, v in sp.get("headers", {}).items():
            m[k] = v
        self.headers = m

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main() -> None:
    P = plan()
    calls = {s: 0 for s in S.FEEDS}
    reqs = []
    clock = [T0]

    def urlopen(req, timeout=None):
        src = next(s for s, (u, _) in S.FEEDS.items() if u == req.full_url)
        k = calls[src]
        calls[src] += 1
        sp = P[src][min(k, len(P[src]) - 1)]
        reqs.append({"url": req.full_url, "headers": sorted(req.header_items()), "timeout": timeout, "resp": sp})
        if sp["kind"] == "urlerror":
            raise urllib.error.URLError("refused")
        if sp["kind"] == "http":
            raise urllib.error.HTTPError(req.full_url, sp["status"], "x", email.message.Message(), None)
        return Resp(sp)

    class Stop(Exception):
        pass
    n = [0]

    def sleep(s):
        n[0] += 1
        if n[0] >= TICKS:
            raise Stop
        clock[0] += s

    S.urllib.request.urlopen = urlopen
    S.time = types.SimpleNamespace(time=lambda: clock[0], sleep=sleep, strftime=time.strftime)
    with tempfile.TemporaryDirectory() as td:
        S.DB = Path(td) / "news.sqlite"
        con = S.connect(S.DB)
        pre_items = [("fed", "https://www.federalreserve.gov/x", "old fed", "", "", 1, 2, 1)]
        pre_polls = [(int((T0 - 8 * 86400) * 1000), "fed", 200, 1, 3, None), (int((T0 - 86400) * 1000), "sec", 304, 0, 4, 5)]
        con.executemany("INSERT INTO items VALUES(?,?,?,?,?,?,?,?)", pre_items)
        con.executemany("INSERT INTO polls VALUES(?,?,?,?,?,?)", pre_polls)
        con.close()
        try:
            S.run()
        except Stop:
            pass
        con = sqlite3.connect(S.DB)
        items = [list(r) for r in con.execute("SELECT * FROM items ORDER BY rowid")]
        polls = [list(r) for r in con.execute("SELECT * FROM polls ORDER BY rowid")]
        schema = [r[0] for r in con.execute("SELECT sql FROM sqlite_master ORDER BY name")]
    (FIX / "news_input.json").write_text(json.dumps({"t0": T0, "ticks": TICKS, "pre_items": pre_items, "pre_polls": pre_polls, "reqs": reqs}, ensure_ascii=False))
    (FIX / "news_golden.json").write_text(json.dumps({"items": items, "polls": polls, "schema": schema}, ensure_ascii=False))
    print(f"정답지: 요청 {len(reqs)} · 기사 {len(items)} · polls {len(polls)} · 피드별 호출 {calls}")


if __name__ == "__main__":
    main()
