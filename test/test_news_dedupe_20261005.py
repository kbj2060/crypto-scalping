"""뉴스 카드 중복 합치기(2026-10-05, 사용자 «중복은 하나로 합치고») -- 같은 글이 Tree·Truth 로 두 번 들어온다.
  python3 test/test_news_dedupe_20261005.py"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dash_srv_news", ROOT / "dashboard" / "server.py")
srv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = srv
spec.loader.exec_module(srv)

J = dict(p_bull=.1, p_bear=.1, p_neu=.8, asset="macro", impact=1.5, incident=.0, relevant=.6, judged_ms=9, model="jevk5:4b")
NOJ = {k: None for k in J} | {"sentiment": None}
items = [
    {"source": "trumpstruth", "link": "t/1", "title": "Mainstream media ignores positive Trump indicators", "ts_ms": 200, "sentiment": "neutral", **J},
    {"source": "tree", "link": "x/1", "title": "Donald J. Trump (@realDonaldTrump): Mainstream media ignores positive Trump indicators",
     "ts_ms": 100, **NOJ},                                                  # 먼저 왔지만 판정 없음 → 대표 시각·출처, 판정은 Truth 것
    {"source": "financialjuice", "link": "f/1", "title": "FinancialJuice: Iran oil minister steps down", "ts_ms": 150, "sentiment": "bearish", **J},
    {"source": "tree", "link": "x/2", "title": "Donald J. Trump (@realDonaldTrump): https://x.co/u/img", "ts_ms": 120, **NOJ},
    {"source": "trumpstruth", "link": "t/2", "title": "[No Title] - Post from October 4, 2026", "ts_ms": 121, **NOJ},
    {"source": "trumpstruth", "link": "t/3", "title": "[No Title] - Post from October 4, 2026", "ts_ms": 119, **NOJ},   # 다른 이미지 글
]
out = srv.news_dedupe(items)
assert [o["link"] for o in out] == ["f/1", "t/2", "x/2", "t/3", "x/1"], [o["link"] for o in out]   # 최신부터 · 본문 없는 글은 안 합친다
m = out[-1]
assert m["sources"] == ["tree", "trumpstruth"] and m["ts_ms"] == 100 and m["sentiment"] == "neutral" and m["asset"] == "macro", m
assert srv.news_key("DECRYPT: Trump Taps Jay Clayton") == srv.news_key("Trump taps Jay Clayton!")
assert srv.news_key("[No Title] - Post from October 4, 2026") == ""   # 본문 없는 글은 합치지 않는다
assert srv.news_key("Tree News: Upbit 상장 ABC(ABC) KRW 마켓") != srv.news_key("Tree News: Upbit 상장폐지 ABC(ABC) KRW 마켓")
assert srv.news_key("COINDESK: There's an election next month: State of Crypto") == srv.news_key("There’s an election next month: State of Crypto")
assert srv.news_key("THE STREET: Top Bitcoin Billionaires") == srv.news_key("Top Bitcoin Billionaires")
assert srv.news_key("FinancialJuice: Iran oil minister steps down") == srv.news_key("Iran oil minister steps down")
print("ok")
