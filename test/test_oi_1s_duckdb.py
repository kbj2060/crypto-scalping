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

    ms = lambda sec, milli=0: (base + sec) * 1000 + milli
    rows = [(ms(10), 100.0), (ms(290), 110.0),                       # 봉0: 100 -> 110
            (ms(bar + 10), 125.0), (ms(bar + 290), 130.0),           # 봉1: 봉0 끝 대비 +20
            (ms(3 * bar + 10), 200.0), (ms(3 * bar + 290), 205.0)]   # 봉3: 봉2가 비었다
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

    # 4) 순서가 뒤바뀌어 도착한 스냅샷도 제 봉으로 간다. 바이낸스가 stamp 를 도착 순서대로
    #    주지 않아(2026-09-19 실측 3.5%) 수집기가 최고수위 대신 «본 stamp 집합»으로 거른다 --
    #    그 늦게 온 스냅샷이 봉 끝을 실제로 바꿔야 그 변경이 값어치가 있다.
    srv.oi_1s_persist([(ms(295), 118.0)])                  # 봉0 끝(290초)보다 뒤, 뒤늦게 도착
    b0b = {b[0]: b for b in srv.oi_5m_buckets(6)}[base]
    assert abs(b0b[1] - 18.0) < 1e-9, b0b                  # 118 - 100, 늦게 온 값이 새 종가
    assert b0b[3] == 3, b0b                                # 스냅샷 수도 늘었다
    # 5) 같은 «초»의 두 스냅샷이 둘 다 남는다(PK 가 밀리초라서). 초 PK 였을 땐 10분간
    #    갱신의 4.91%가 여기서 사라졌다. 늦은 쪽이 종가가 되어야 한다.
    srv.oi_1s_persist([(ms(297, 120), 130.0), (ms(297, 830), 140.0)])
    b0c = {b[0]: b for b in srv.oi_5m_buckets(6)}[base]
    assert b0c[3] == 5, b0c                                # 3 -> 5, 같은 초 둘 다 남았다
    assert abs(b0c[1] - 40.0) < 1e-9, b0c                  # 140 - 100, 830ms 쪽이 종가
    print("ok")


def migration(tmp: Path) -> None:
    """옛 스키마(ts_sec PK)가 있는 파일을 그대로 받아 밀리초로 옮긴다.

    이 경로는 가동 중인 실제 DB 위에서 딱 한 번 돌기 때문에 미리 지나가 봐야 한다:
    옛 행이 남아 있는가 · 초 단위 봉 집계가 그대로인가 · 이후 ms 행이 정상으로 붙는가."""
    srv.OI_1S_DB_PATH = tmp
    bar = srv.OI_5M_BAR_SECONDS
    base = (int(time.time()) // bar) * bar - bar
    with srv.duckdb_path_lock(tmp):
        con = srv.duckdb.connect(str(tmp))
        con.execute(f"""CREATE TABLE {srv.OI_1S_TABLE} (ts_sec BIGINT, symbol VARCHAR,
                        open_interest DOUBLE, PRIMARY KEY (ts_sec, symbol))""")
        con.executemany(f"INSERT INTO {srv.OI_1S_TABLE} VALUES (?, ?, ?)",
                        [(base + 10, "ethusdt", 50.0), (base + 290, "ethusdt", 60.0)])
        con.close()

    srv.oi_1s_persist([((base + 295) * 1000 + 400, 70.0)])     # 이관 + 새 ms 행 한 번에
    with srv.duckdb_path_lock(tmp):
        con = srv.duckdb.connect(str(tmp))
        cols = {r[0] for r in con.execute(
            "select column_name from information_schema.columns where table_name = ?",
            [srv.OI_1S_TABLE]).fetchall()}
        got = con.execute(f"select ts_ms, open_interest from {srv.OI_1S_TABLE} order by 1").fetchall()
        tabs = {r[0] for r in con.execute(
            "select table_name from information_schema.tables").fetchall()}
        con.close()
    assert cols == {"ts_ms", "symbol", "open_interest"}, cols
    assert f"{srv.OI_1S_TABLE}_sec_legacy" not in tabs, tabs        # 임시 테이블을 안 남긴다
    assert got == [((base + 10) * 1000, 50.0), ((base + 290) * 1000, 60.0),
                   ((base + 295) * 1000 + 400, 70.0)], got          # 옛 행이 .000ms 로 살아있다
    b = {x[0]: x for x in srv.oi_5m_buckets(3)}[base]
    assert b[3] == 3 and abs(b[1] - 20.0) < 1e-9, b                 # 70 - 50, 봉 집계 그대로

    srv.oi_1s_persist([((base + 296) * 1000, 80.0)])                # 두 번째 호출은 이관을 또 하면 안 된다
    b2 = {x[0]: x for x in srv.oi_5m_buckets(3)}[base]
    assert b2[3] == 4, b2
    print("ok (migration)")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        main(Path(d) / "oi_1s.duckdb")
        migration(Path(d) / "legacy.duckdb")
