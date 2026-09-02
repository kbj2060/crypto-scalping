#!/usr/bin/env python3
"""XRP 대시보드 **자산 인식 감사** — XRP 페이지에 ETH 데이터가 새는 곳을 찾는다.

## 왜

2026-09-02 사용자 신고 "비트코인 페이지에 이더리움 증거신호가 나온다"가 있었고,
2026-09-03에 **XRP 페이지에서도 같은 일**이 벌어지고 있던 걸 발견해 고쳤다(증거신호).
그 버그 계열이 다른 패널에도 있는지 전수로 본다.

## 판정 규칙

각 API 상수마다:
  · `?asset=` 파라미터를 붙여 호출하면 → **자산 인식** (서버가 코인별 데이터를 준다)
  · 자산별 전용 URL이 있으면(`/api/xrp-*`) → **자산 인식**
  · 둘 다 아니면 → ⚠️**ETH 고정**. XRP 탭에서 이 패널이 보이면 ETH 데이터를 보는 것이다.

⚠️"ETH 고정"이 전부 버그는 아니다 — 매크로 캘린더처럼 자산 무관한 것도 있고,
`renderEvidenceSignalsProvisional`처럼 **ETH가 아니면 DOM을 안 건드리는 방어**가 이미 걸린 것도 있다.
그래서 이 스크립트는 **후보 목록**을 만들고, 각 항목의 실제 처리 방식을 함께 표시한다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard/live/app.js"
SRV = ROOT / "dashboard/server.py"
OUT = ROOT / "data/research/xrp_dashboard_asset_audit_20260903.json"

# 자산 무관(코인과 상관없는 정보) — ETH 고정이어도 문제 아님
ASSET_AGNOSTIC = {"API_EVENTS_URL", "API_OPS_STATUS_URL", "API_MACRO_CALENDAR_URL",
                  "API_SESSION_ALERTS_URL"}


def log(m): print(f"[dash-audit] {m}", flush=True)


def main() -> int:
    app, srv = APP.read_text(), SRV.read_text()
    consts = dict(re.findall(r'const (API_[A-Z_0-9]+_URL) = "([^"]+)"', app))
    # 서버에서 ?asset= 을 파싱하는 라우트
    multi = set()
    for m in re.finditer(r'load_(\w+)\(_query_coin_asset\(request\)\)', srv):
        multi.add(m.group(1))
    rows = []
    for name, url in sorted(consts.items()):
        per_asset_url = bool(re.match(r"/api/(xrp|btc)-", url))
        # 이 상수를 asset 쿼리와 함께 쓰는가
        uses_asset_q = bool(re.search(re.escape(name) + r"[^\n;]{0,200}?asset=", app)) or \
                       bool(re.search(r"asset=[^\n;]{0,200}?" + re.escape(name), app))
        # 자산별 분기(삼항)에 등장하는가
        in_branch = bool(re.search(r"activeSnapshotAsset[^\n]{0,200}?" + re.escape(name), app))
        agnostic = name in ASSET_AGNOSTIC
        # ETH가 아니면 조기 반환하는 방어가 걸린 렌더 함수와 연결되는가(대략)
        aware = per_asset_url or uses_asset_q or in_branch or agnostic
        rows.append({"const": name, "url": url, "per_asset_url": per_asset_url,
                     "uses_asset_query": uses_asset_q, "in_asset_branch": in_branch,
                     "asset_agnostic": agnostic, "asset_aware": aware})
    log(f"{'API 상수':<40}{'URL':<36}{'판정'}")
    for r in rows:
        why = ("전용URL" if r["per_asset_url"] else
               "?asset=" if r["uses_asset_query"] else
               "자산분기" if r["in_asset_branch"] else
               "자산무관" if r["asset_agnostic"] else "—")
        mark = "✅" if r["asset_aware"] else "⚠️"
        log(f"{mark} {r['const']:<38}{r['url']:<36}{why}")
    bad = [r for r in rows if not r["asset_aware"]]
    log("")
    log(f"⚠️자산 인식 안 되는 상수 {len(bad)}개 -- 각각 실제 렌더 경로를 확인해야 한다:")
    for r in bad:
        log(f"   {r['const']}  ({r['url']})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "not_asset_aware": [r["const"] for r in bad]},
                              ensure_ascii=False, indent=2))
    log(f"report -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
