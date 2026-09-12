#!/usr/bin/env python3
"""왕복 원장을 지정한 백업으로 되돌리고, 지정한 왕복만 뺀다 (2026-09-12, 일회성 복구).

왜 필요했나 — `backfill_account_trip_ledger_20260912.py --rebuild` 는 **멱등이 아니다**.
버릴 줄을 «거래소 재계산본에 키가 없는 줄» 로 정하는데, 그 재계산본이 **조회 시작점에 따라
달라진다**(체결 스트림을 어디서 자르느냐가 폴딩 경계를 바꾼다). 두 번째 실행이 첫 실행과
다른 폴딩을 얻어 멀쩡한 10줄을 «키 없음» 으로 버렸다(68 → 59줄).
⚠️그래서 `--rebuild` 는 **한 번만** 쓰고, 이후에는 이 복구 경로로 되돌린다.

하는 일: 백업을 읽어 JSON 전건 파싱 → `--drop` 으로 준 entry_time 을 빼고 → 원자 교체.
기본 dry-run. 현재 파일은 `.bad_<타임스탬프>` 로 남긴다.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/live/account_round_trips.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise SystemExit(f"{path} {i}행 파싱 실패: {exc}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-backup", required=True, help="되돌릴 백업 파일명(data/live 안)")
    ap.add_argument("--drop", type=int, nargs="*", default=[], help="뺄 entry_time(ms)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    backup = LEDGER.parent / a.from_backup
    rows = read_jsonl(backup)
    cur = read_jsonl(LEDGER) if LEDGER.exists() else []
    drop = set(a.drop)
    keep = [r for r in rows if int(r.get("entry_time", 0)) not in drop]
    gone = [r for r in rows if int(r.get("entry_time", 0)) in drop]

    print(f"현재 {len(cur)}줄 · 백업 {backup.name} {len(rows)}줄 → 복구본 {len(keep)}줄")
    for r in gone:
        print(f"   뺌: entry_time={r['entry_time']} {r.get('side')} 순익 {r.get('net_pnl')} "
              f"check={r.get('pnl_check_bp')}")
    missing = drop - {int(r.get("entry_time", 0)) for r in rows}
    if missing:
        print(f"⚠️백업에 없는 entry_time {sorted(missing)} -- 오타일 수 있다")

    # 백업에는 없는데 현재 파일에만 있는 줄(복구 중 대시보드가 붙였을 수 있다)은 살린다
    have = {int(r.get("entry_time", 0)) for r in keep}
    extra = [r for r in cur if int(r.get("entry_time", 0)) not in have
             and int(r.get("entry_time", 0)) not in drop]
    if extra:
        print(f"현재 파일에만 있는 {len(extra)}줄은 살린다:")
        for r in extra:
            print(f"   유지: entry_time={r['entry_time']} {r.get('side')} 순익 {r.get('net_pnl')}")
    final = sorted(keep + extra, key=lambda r: int(r["entry_time"]))

    def viol(rs):
        out = []
        for i, x in enumerate(rs):
            for y in rs[i + 1:]:
                if x.get("symbol") == y.get("symbol") and x.get("side") == y.get("side"):
                    x0 = int(x["entry_time"]); x1 = int(x.get("exit_time") or x0)
                    y0 = int(y["entry_time"]); y1 = int(y.get("exit_time") or y0)
                    if x0 <= y1 and y0 <= x1:
                        out.append((x, y))
        return out

    v = viol(final)
    print(f"최종 {len(final)}줄 · 불변식 위반 {len(v)}건")
    if not a.apply:
        print("미리보기만 했다. 반영하려면 --apply")
        return 0
    if v:
        print("⛔불변식 위반이 있어 반영하지 않는다.")
        return 5
    if LEDGER.exists():
        shutil.copy2(LEDGER, LEDGER.with_suffix(f".jsonl.bad_{time.strftime('%Y%m%d_%H%M%S')}"))
    tmp = LEDGER.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final), encoding="utf-8")
    os.replace(tmp, LEDGER)
    print(f"복구 완료 — {LEDGER} {len(final)}줄")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
