"""OI 1초 스냅샷 저장/5분 집계 자체점검 (2026-09-19).

확인하는 것 셋:
  1) 같은 초를 두 번 써도 행이 안 늘어난다(재기동 직후 겹치는 구간을 다시 써도 안전).
  2) Δ 는 «직전 봉 끝 -> 이 봉 끝»이다(봉 사이 3~7초에 일어난 변화를 잃지 않는다).
  3) 앞 봉이 비면 그 공백을 이 봉에 몰아주지 않고 gap=1 로 알린다.
실행: python test/test_oi_1s_duckdb.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard.server as srv


def main(tmp: Path) -> None:
    srv.OI_1S_DB_PATH = tmp
    bar = srv.OI_5M_BAR_SECONDS
    base = (int(time.time()) // bar) * bar - 3 * bar   # 세 봉 전부터

    rows = [(base + 10, 100.0), (base + 290, 110.0),                 # 봉0: 100 -> 110
            (base + bar + 10, 125.0), (base + bar + 290, 130.0),     # 봉1: 봉0 끝 대비 +20
            (base + 3 * bar + 10, 200.0), (base + 3 * bar + 290, 205.0)]  # 봉3: 봉2가 비었다
    srv.oi_1s_persist(rows)
    srv.oi_1s_persist(rows)          # 같은 것을 다시
    with srv.duckdb_path_lock(tmp):
        con = srv.duckdb.connect(str(tmp))
        n = con.execute(f"select count(*) from {srv.OI_1S_TABLE}").fetchone()[0]
        con.close()
    assert n == len(rows), f"중복이 쌓였다: {n} != {len(rows)}"

    buckets = {b[0]: b for b in srv.oi_5m_buckets(6)}
    b0, b1, b3 = buckets[base], buckets[base + bar], buckets[base + 3 * bar]
    assert b0[4] == 1 and abs(b0[1] - 10.0) < 1e-9, b0      # 첫 봉: 봉 안에서만
    assert b1[4] == 0 and abs(b1[1] - 20.0) < 1e-9, b1      # 130 - 110, 봉 사이 +15 포함
    assert b3[4] == 1 and abs(b3[1] - 5.0) < 1e-9, b3       # 공백 뒤: 205-200 (195 가 아니다)
    print("ok")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        main(Path(d) / "oi_1s.duckdb")
