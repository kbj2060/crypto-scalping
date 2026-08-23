"""GEX dev 고아 duckdb → 서버 라이브 DB 일회성 병합 (2026-08-23).

배경: 2026-08-17 서버 이관 커밋이 CREATE TABLE IF NOT EXISTS만 하고 기존 dev duckdb를
복사하지 않아, dev 시절 이력(2026-08-15 06:03 ~ 08-17 04:00 UTC, BTC NEG-감마 15행 포함 —
이관 후 6일간 서버 NEG-감마 0건이므로 현존 유일한 NEG-감마 증거)이 dev 고아파일에만
남아 있었다(eth_gex_status_and_next_direction_candidates_20260820 메모리의 미결 항목).
Tier1 판정에서 이 2일이 조용히 빠지는 것을 막기 위해 물리 병합한다.

실행 위치: 서버(라이브 DB 소재지). 컬렉터 크론(매시 정각) 사이 안전창에서 실행.
사용법: python3 merge_deribit_gex_dev_orphan_20260823.py <orphan_db_path>
겹침 없음 확인됨(dev max 04:00:02 UTC < server min 04:35:53 UTC)이나 WHERE 가드 유지.
"""
import shutil
import sys

import duckdb

LIVE = "data/live/deribit_gex.duckdb"
ORPHAN = sys.argv[1]
BAK = LIVE + ".bak_pre_dev_orphan_merge_20260823"

shutil.copy2(LIVE, BAK)
print(f"backup: {BAK}")

con = duckdb.connect(LIVE)
con.execute(f"ATTACH '{ORPHAN}' AS dev (READ_ONLY)")

for tbl in ["gex_summary", "option_chain_snapshot"]:
    live_min = con.execute(f"SELECT MIN(recorded_at_utc) FROM {tbl}").fetchone()[0]
    before = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    dev_total = con.execute(f"SELECT COUNT(*) FROM dev.{tbl}").fetchone()[0]
    dev_eligible = con.execute(
        f"SELECT COUNT(*) FROM dev.{tbl} WHERE recorded_at_utc < ?", [live_min]
    ).fetchone()[0]
    con.execute(
        f"INSERT INTO {tbl} BY NAME SELECT * FROM dev.{tbl} WHERE recorded_at_utc < ?",
        [live_min],
    )
    after = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    ok = after == before + dev_eligible
    print(
        f"{tbl}: live_before={before} dev_eligible={dev_eligible} "
        f"(dev_total={dev_total}) after={after} ok={ok}"
    )
    if not ok:
        raise SystemExit(f"count mismatch on {tbl} — inspect before trusting merge")

dups = con.execute(
    "SELECT COUNT(*) FROM (SELECT recorded_at_utc, currency FROM gex_summary "
    "GROUP BY 1,2 HAVING COUNT(*) > 1)"
).fetchone()[0]
print(f"duplicate (ts,currency) groups in gex_summary: {dups}")

print(
    con.execute(
        "SELECT currency, COUNT(*) n, MIN(recorded_at_utc) mn, MAX(recorded_at_utc) mx, "
        "SUM(CASE WHEN front_month_gex_usd < 0 THEN 1 ELSE 0 END) n_neg "
        "FROM gex_summary GROUP BY 1 ORDER BY 1"
    ).fetchdf().to_string()
)
con.close()
print("merge complete")
