"""다른 호스트(Pi 수집 노드)의 다코인 수집기 신선도 프로브 (2026-10-12).

서버 ops_watchdog.check_pi_collectors 가 `ssh <호스트> python3 - <저장소>` 의 stdin 으로 이 파일을 보낸다.
Pi 에는 dotenv 가 없고 저장소도 옛 판이라 ops_watchdog 를 import 할 수 없다 -- 그래서 표준 라이브러리만 쓴다.
매니페스트 해석은 ops_watchdog.check_multicoin_collectors 와 같다(유예·fresh 항목). 다른 점 하나:
duckdb 항목도 DB 를 열지 않고 본체·.wal 의 mtime 으로 본다(쓰는 수집기와 락 경합을 만들지 않는다).
출력 = JSON 한 줄: [{name, age_minutes(없으면 null), warn, critical, grace, path}, ...]
"""
import json
import sys
import time
from pathlib import Path


def probe(root: Path, now: float) -> list[dict]:
    manifest = json.loads((root / "data" / "live" / "multicoin_collectors.json").read_text(encoding="utf-8"))
    out = []
    for c in (manifest.get("collectors") or {}).values():
        grace = now - float(c.get("started_at", 0)) < 60.0 * float(c.get("grace_minutes", 15))
        for f in c.get("fresh") or []:
            p = root / f["path"]
            if f.get("kind") == "dir":
                files = [x for x in p.glob(f.get("glob", "*")) if x.is_file()] if p.is_dir() else []
            else:
                files = [x for x in (p, p.with_name(p.name + ".wal")) if x.is_file()]
            newest = max((x.stat().st_mtime for x in files), default=None)
            out.append({"name": f["name"], "age_minutes": None if newest is None else round(max(0.0, now - newest) / 60.0, 1),
                        "warn": f["warn"], "critical": f["critical"], "grace": grace, "path": f["path"]})
    return out


if __name__ == "__main__":
    print(json.dumps(probe(Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser(), time.time())))
