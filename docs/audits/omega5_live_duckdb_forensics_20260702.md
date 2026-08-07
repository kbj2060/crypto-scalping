# Omega5 Live DuckDB Forensics - 2026-07-02

- Status: `fail`
- Verdict: `OMEGA5_LIVE_DUCKDB_FORENSICS_BLOCKED`
- Omega5 events: `0` open `0` close `0`
- Decision snapshot matches: `0`
- DuckDB matches: micro `0`, tail `0`
- Missing risk-contract opens: `0`
- Missing trace opens: `0`
- Quarantine events: `0`
- Reconcile close events: `0`

## Blocking
- `omega5_trade_journal_events_missing`

## Warnings
- none

## Data Sources
- Journal: `/home/llewyn/crypto-scalping/data/live/trade_journal.jsonl`
- Decision snapshots: `/home/llewyn/crypto-scalping/data/live/decision_feature_snapshot.jsonl`
- Micro DuckDB: `direct_read_only` rows `71995`
- Tail DuckDB: `direct_read_only` rows `72000`

## Recent Omega5 Events

| kind | decision_ts | source | side | notional | lev | pnl_pct | snapshot | micro | tail | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
