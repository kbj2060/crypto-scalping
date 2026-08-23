"""GEX 만기리버설 프로토콜 — Tier1 조건일 카운트 (판독 아님, 주 1회 실행용).

docs/experiments/eth_candidate_gex_expiry_reversal_protocol_20260817.md의 조작적 정의를
그대로 구현해 "NEG-감마 & 고조-ATM-OI 조건 만족일"을 센다. 프로토콜이 명시적으로 허용한
카운트-only 작업이며 리버설/수익률 판독은 하지 않는다 (Tier1 도달 전 판독은 사전등록 위반).

- 감마 부호: 만기(매일 08:00 UTC) 직전 gex_summary 최신 관측치의 front_month_gex_usd 부호.
- ATM OI: 만기 직전 최신 option_chain_snapshot에서 days_to_expiry<=1.0 AND
  |strike/underlying_price - 1|<=0.02 인 계약의 open_interest 합.
- 고조: 누적(expanding) 표본의 상위 tercile(>= 66.67 percentile, 자기 자신 포함).
- Tier1 = 조건 만족일 >= 20일 누적, Tier2 = >= 60일.

실행 위치: 서버(라이브 DB 소재지). 만기 미도래일(오늘 08:00 UTC 이전)은 제외.
"""
import datetime as dt

import duckdb
import numpy as np
import pandas as pd

DB = "data/live/deribit_gex.duckdb"
TIER1, TIER2 = 20, 60

con = duckdb.connect(DB, read_only=True)

gamma = con.execute("""
    SELECT currency,
           CAST(recorded_at_utc AT TIME ZONE 'UTC' AS DATE) AS d,
           arg_max(front_month_gex_usd, recorded_at_utc) AS gex_pre,
           MAX(recorded_at_utc) AS pre_ts
    FROM gex_summary
    WHERE EXTRACT(hour FROM recorded_at_utc AT TIME ZONE 'UTC') < 8
      AND front_month_gex_usd IS NOT NULL
    GROUP BY 1, 2
""").fetchdf()

atm = con.execute("""
    WITH pre_snap AS (
        SELECT currency,
               CAST(recorded_at_utc AT TIME ZONE 'UTC' AS DATE) AS d,
               MAX(recorded_at_utc) AS snap_ts
        FROM option_chain_snapshot
        WHERE EXTRACT(hour FROM recorded_at_utc AT TIME ZONE 'UTC') < 8
        GROUP BY 1, 2
    )
    SELECT c.currency,
           p.d,
           SUM(c.open_interest) AS atm_oi_frontexp
    FROM option_chain_snapshot c
    JOIN pre_snap p
      ON c.currency = p.currency AND c.recorded_at_utc = p.snap_ts
    WHERE c.days_to_expiry <= 1.0
      AND ABS(c.strike / c.underlying_price - 1) <= 0.02
    GROUP BY 1, 2
""").fetchdf()
con.close()

df = gamma.merge(atm, on=["currency", "d"], how="left")

# 만기 미도래일 제외 (오늘인데 아직 08:00 UTC 이전이면 pre-window가 미완성)
now = dt.datetime.now(dt.timezone.utc)
cutoff = pd.Timestamp(now.date() if now.hour >= 8 else now.date() - dt.timedelta(days=1))
df = df[df["d"] <= cutoff].sort_values(["currency", "d"]).reset_index(drop=True)

print(f"기준시각: {now:%Y-%m-%d %H:%M} UTC / 만기 완료일 컷오프: {cutoff}")
for cur, g in df.groupby("currency"):
    g = g.reset_index(drop=True)
    oi = g["atm_oi_frontexp"].to_numpy(dtype=float)
    elevated = np.zeros(len(g), dtype=bool)
    for i in range(len(g)):
        sample = oi[: i + 1]
        sample = sample[~np.isnan(sample)]
        if len(sample) == 0 or np.isnan(oi[i]):
            continue
        elevated[i] = oi[i] >= np.percentile(sample, 100 * 2 / 3)
    neg = g["gex_pre"].to_numpy() < 0
    cond = neg & elevated
    print(f"\n=== {cur} ===")
    for i, row in g.iterrows():
        oi_s = f"{row['atm_oi_frontexp']:,.0f}" if not np.isnan(oi[i]) else "NA"
        print(
            f"  {row['d']}  gex_pre={row['gex_pre']:+.3e}  "
            f"{'NEG' if neg[i] else 'POS'}  atm_oi={oi_s}  "
            f"elevated={'Y' if elevated[i] else 'N'}  cond={'** Y **' if cond[i] else 'N'}"
        )
    n_days, n_neg, n_cond = len(g), int(neg.sum()), int(cond.sum())
    print(
        f"  관측일 {n_days} / NEG-감마일 {n_neg} / 조건만족일(NEG&고조OI) {n_cond}"
        f"  → Tier1({TIER1}일)까지 {max(0, TIER1 - n_cond)}일, Tier2({TIER2}일)까지 {max(0, TIER2 - n_cond)}일"
    )
