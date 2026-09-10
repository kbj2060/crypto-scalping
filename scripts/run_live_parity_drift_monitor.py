"""라이브 패리티/드리프트/섀도우/수집 건강 모니터 (운영용, 크론 대상).

WS-D 설계 문서(docs/duckdb_live_data_utilization_design_20260719.md) 기반.
D2(드리프트) / D3(섀도우 성과 이탈) / D4(수집 건강)를 실행하고
data/live/monitoring/에 타임스탬프 리포트 + latest 스냅샷을 남긴다.

D1(라이브 vs 오프라인 피처 재계산 패리티)은 전체 피처 파이프라인 재현이 필요해
아직 미구현 -- 별도 작업으로 남겨둠 (2026-07-19 세션 노트 참고).

읽기전용 DuckDB 연결만 사용. 트레이딩 로직/봇 프로세스에 관여하지 않음.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

OUT_DIR = Path("data/live/monitoring")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIVE_DBS = {
    "microstructure": "data/live/microstructure.duckdb",
    "tail_risk": "data/live/tail_risk.duckdb",
    "btc_shadow": "data/live/btc_micro_scalp_shadow.duckdb",
    "sol_shadow": "data/live/sol_micro_scalp_shadow.duckdb",
    "eth_v4_shadow": "data/live/eth_micro_scalp_v4_shadow.duckdb",
    "eth_lifecycle_shadow": "data/live/eth_micro_scalp_lifecycle_shadow.duckdb",
    "sol_entry_shadow": "data/live/sol_micro_scalp_entry_shadow.duckdb",
}

PSI_ALERT_THRESHOLD = 0.2
PSI_WATCH_THRESHOLD = 0.1
STALE_ROW_AGE_HOURS = 2.0


def connect_retry(path, read_only=True, retries=8, backoff=2.0):
    last_exc = None
    for attempt in range(retries):
        try:
            return duckdb.connect(path, read_only=read_only)
        except duckdb.IOException as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
    raise last_exc


def psi(expected: np.ndarray, actual: np.ndarray, bins=10):
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return None
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return None
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.clip(exp_counts / max(exp_counts.sum(), 1), 1e-6, None)
    act_pct = np.clip(act_counts / max(act_counts.sum(), 1), 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# ---------------- D2: drift monitor ----------------

def run_d2_drift():
    con = connect_retry(LIVE_DBS["microstructure"])
    df = con.execute(
        "select ts, obi, taker_buy_ratio, shadow_toxicity_score, shadow_queue_collapse, "
        "shadow_absorption_score, oi_delta_pct, funding_rate from microstructure_1m order by ts"
    ).df()
    con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cutoff = df["ts"].max() - pd.Timedelta(days=7)
    baseline = df[df["ts"] < cutoff]
    recent = df[df["ts"] >= cutoff]

    cols = ["obi", "taker_buy_ratio", "shadow_toxicity_score", "shadow_queue_collapse",
            "shadow_absorption_score", "oi_delta_pct", "funding_rate"]
    results = {}
    for c in cols:
        p = psi(baseline[c].values, recent[c].values)
        alert = "ALERT" if (p or 0) > PSI_ALERT_THRESHOLD else ("WATCH" if (p or 0) > PSI_WATCH_THRESHOLD else "OK")
        results[c] = {"psi": p, "alert": alert}
    return {
        "baseline_n": int(len(baseline)),
        "recent_n": int(len(recent)),
        "cutoff": str(cutoff),
        "psi_by_column": results,
    }


def run_d2_injection_selftest():
    """자가검증: obi에 +1시그마 합성 주입 후 PSI가 실제로 잡는지, 무조작 시 오탐 없는지 확인.
    매 실행마다 이 자가검증을 같이 돌려서 모니터 자체가 고장나지 않았는지 확인한다."""
    con = connect_retry(LIVE_DBS["microstructure"])
    df = con.execute("select obi from microstructure_1m order by ts").df()
    con.close()
    vals = df["obi"].dropna().values
    n = len(vals)
    baseline = vals[: n // 2]
    unmodified_recent = vals[n // 2:]
    sigma = np.std(baseline)
    shifted_recent = unmodified_recent + 1.0 * sigma

    psi_unmodified = psi(baseline, unmodified_recent)
    psi_shifted = psi(baseline, shifted_recent)
    return {
        "psi_unmodified_recent": psi_unmodified,
        "psi_after_plus1sigma_injection": psi_shifted,
        "detection_pass": bool((psi_shifted or 0) > PSI_ALERT_THRESHOLD and (psi_unmodified or 0) <= PSI_ALERT_THRESHOLD),
    }


# ---------------- D3: shadow performance deviation ----------------

def run_d3_shadow_deviation():
    out = {}
    for name, path in LIVE_DBS.items():
        if "shadow" not in name:
            continue
        try:
            con = connect_retry(path)
            tables = [r[0] for r in con.execute("show tables").fetchall()]
            if "shadow_pnl" not in tables:
                out[name] = {"error": "no shadow_pnl table"}
                con.close()
                continue
            cols = [c[0] for c in con.execute("describe shadow_pnl").fetchall()]
            if "settlement_timestamp" in cols:
                ts_col = "settlement_timestamp"
            elif "decision_timestamp" in cols:
                ts_col = "decision_timestamp"
            else:
                ts_col = next(
                    (c for c in cols if "time" in c.lower() or c.endswith("_kst") or c == "ts"), cols[0]
                )
            pnl_col = next(
                (c for c in cols if c in ("net_return", "pnl", "net_pnl")
                 or "pnl" in c.lower() or "net_return" in c.lower()),
                None,
            )
            df = con.execute(f'select * from shadow_pnl order by "{ts_col}"').df()
            if pnl_col is None or ts_col not in df.columns:
                out[name] = {"error": "pnl/ts column not identified", "columns": cols}
                con.close()
                continue
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.dropna(subset=[ts_col])
            cutoff = df[ts_col].max() - pd.Timedelta(days=7)
            hist = df[df[ts_col] < cutoff]
            recent = df[df[ts_col] >= cutoff]
            latest_age_hours = (
                (pd.Timestamp.now(tz="UTC") - df[ts_col].max()).total_seconds() / 3600.0
            )

            # BENIGN(명시적 비활성/워밍업) vs UNRESOLVED(활성인데 무거래) 자동 분류
            zero_variance_explanation = None
            try:
                om = con.execute("select * from observer_metadata limit 1").df()
                research_enabled = bool(om["research_policy_enabled"].iloc[0]) if "research_policy_enabled" in om.columns else None
                fresh_start = str(om["fresh_start_utc"].iloc[0]) if "fresh_start_utc" in om.columns else None
                if "decisions" in tables:
                    dec = con.execute("select available, target_position from decisions").df()
                    pct_available = float(dec["available"].mean()) if "available" in dec.columns else None
                    pct_nonzero_target = float((dec["target_position"] != 0).mean()) if "target_position" in dec.columns else None
                else:
                    pct_available = pct_nonzero_target = None

                if pct_nonzero_target is not None and pct_nonzero_target > 0:
                    zero_variance_explanation = "not_zero_variance -- has taken positions"
                elif research_enabled is False:
                    zero_variance_explanation = (
                        f"BENIGN: research_policy_enabled=False (explicit gate), fresh_start={fresh_start}"
                    )
                elif pct_available is not None and pct_available < 0.5:
                    zero_variance_explanation = (
                        f"BENIGN: only {pct_available:.0%} of decisions available (warming up), fresh_start={fresh_start}"
                    )
                elif pct_nonzero_target == 0.0:
                    zero_variance_explanation = (
                        f"UNRESOLVED: available/enabled but NEVER took a position, fresh_start={fresh_start}"
                    )
            except Exception as exc:
                zero_variance_explanation = f"metadata_check_failed: {exc}"
            con.close()

            hist_daily = hist.set_index(ts_col)[pnl_col].resample("1D").sum()
            recent_sum = float(recent[pnl_col].sum())
            if len(hist_daily) >= 5:
                rng = np.random.default_rng(int(time.time()))
                boot_sums = [
                    rng.choice(hist_daily.values, size=7, replace=True).sum() for _ in range(2000)
                ]
                ci5, ci95 = float(np.percentile(boot_sums, 5)), float(np.percentile(boot_sums, 95))
                within_ci = bool(ci5 <= recent_sum <= ci95)
            else:
                ci5 = ci95 = within_ci = None

            out[name] = {
                "n_rows": int(len(df)),
                "latest_row_age_hours": float(latest_age_hours),
                "stale_alert": bool(latest_age_hours > STALE_ROW_AGE_HOURS),
                "recent_7d_pnl_sum": recent_sum,
                "hist_daily_bootstrap_ci5": ci5,
                "hist_daily_bootstrap_ci95": ci95,
                "within_ci": within_ci,
                "zero_variance_explanation": zero_variance_explanation,
            }
        except Exception as exc:
            out[name] = {"error": str(exc)}
    return out


# ---------------- D4: collection health ----------------

def run_d4_health():
    out = {}
    for name, path in LIVE_DBS.items():
        try:
            con = connect_retry(path)
            tables = [r[0] for r in con.execute("show tables").fetchall()]
            tbl_info = {}
            for t in tables:
                try:
                    cols = [c[0] for c in con.execute(f'describe "{t}"').fetchall()]
                    ts_col = next(
                        (c for c in cols if c in ("ts", "recorded_at_kst") or "time" in c.lower()), None
                    )
                    n = con.execute(f'select count(*) from "{t}"').fetchone()[0]
                    info = {"n_rows": int(n)}
                    if ts_col:
                        mn, mx = con.execute(f'select min("{ts_col}"), max("{ts_col}") from "{t}"').fetchone()
                        info["ts_col"] = ts_col
                        info["min_ts"] = str(mn)
                        info["max_ts"] = str(mx)
                        if mx is not None:
                            mx_ts = pd.Timestamp(mx)
                            if mx_ts.tzinfo is None:
                                mx_ts = mx_ts.tz_localize("UTC")
                            age_min = (pd.Timestamp.now(tz="UTC") - mx_ts.tz_convert("UTC")).total_seconds() / 60.0
                            info["latest_row_age_minutes"] = float(age_min)
                            # 1m 테이블 기준: 최신행 10분 이상 지연되면 경보
                            info["stale_alert"] = bool(age_min > 10.0) if "1m" in t or "snapshot" in t else None
                    tbl_info[t] = info
                except Exception as exc:
                    tbl_info[t] = {"error": str(exc)}
            con.close()
            out[name] = tbl_info
        except Exception as exc:
            out[name] = {"error": str(exc)}
    return out


CHOP_SHADOW_FAILURE_MARKER = Path("data/live/omega4_6_1_eth_chop_shadow.FAILED.json")


def run_d5_chop_shadow_health():
    """2026-07-19 세션: Omega4.6.1 ETH chop 소프트사이징 shadow(fail-fast 마커) 감시.
    실제 트레이딩 로직과 무관 -- 마커 파일 존재/최신성만 확인."""
    if not CHOP_SHADOW_FAILURE_MARKER.exists():
        return {"status": "no_failures_recorded"}
    try:
        payload = json.loads(CHOP_SHADOW_FAILURE_MARKER.read_text())
        last_failure = pd.Timestamp(payload.get("last_failure_utc"))
        age_hours = (pd.Timestamp.now(tz="UTC") - last_failure).total_seconds() / 3600.0
        return {
            "status": "failure_marker_present",
            "hook": payload.get("hook"),
            "error": payload.get("error"),
            "failure_count": payload.get("failure_count"),
            "last_failure_age_hours": age_hours,
        }
    except Exception as exc:
        return {"status": "marker_unreadable", "error": str(exc)}


def collect_alerts(report: dict) -> list[str]:
    alerts = []
    for col, info in report.get("D2_drift", {}).get("psi_by_column", {}).items():
        if info.get("alert") == "ALERT":
            alerts.append(f"D2 DRIFT: {col} PSI={info['psi']:.3f}")
    if not report.get("D2_injection_selftest", {}).get("detection_pass", True):
        alerts.append("D2 SELFTEST FAILED: injection detection did not pass -- monitor itself may be broken")
    for name, info in report.get("D3_shadow_deviation", {}).items():
        if info.get("stale_alert"):
            alerts.append(f"D3 STALE: {name} latest row {info.get('latest_row_age_hours'):.1f}h old")
        if info.get("within_ci") is False:
            alerts.append(f"D3 PNL DEVIATION: {name} 7d PnL sum {info.get('recent_7d_pnl_sum')} outside bootstrap CI")
    for db_name, tables in report.get("D4_collection_health", {}).items():
        if isinstance(tables, dict):
            for t, info in tables.items():
                if isinstance(info, dict) and info.get("stale_alert"):
                    alerts.append(f"D4 STALE: {db_name}.{t} latest row {info.get('latest_row_age_minutes'):.1f}min old")
    d5 = report.get("D5_chop_shadow_health", {})
    if d5.get("status") == "failure_marker_present":
        alerts.append(
            f"D5 CHOP_SHADOW_FAILURE: hook={d5.get('hook')} count={d5.get('failure_count')} "
            f"age={d5.get('last_failure_age_hours'):.1f}h err={d5.get('error')}"
        )
    return alerts


def main():
    now = pd.Timestamp.now(tz="Asia/Seoul")
    report = {"stage": "live_parity_drift_monitor", "generated_at": now.isoformat()}
    report["D1_note"] = "NOT IMPLEMENTED -- requires full offline feature pipeline recompute (separate task)"

    print("Running D2 (drift)...")
    report["D2_drift"] = run_d2_drift()
    print("Running D2 injection self-test...")
    report["D2_injection_selftest"] = run_d2_injection_selftest()
    print("Running D3 (shadow deviation)...")
    report["D3_shadow_deviation"] = run_d3_shadow_deviation()
    print("Running D4 (collection health)...")
    report["D4_collection_health"] = run_d4_health()
    print("Running D5 (chop shadow health)...")
    report["D5_chop_shadow_health"] = run_d5_chop_shadow_health()

    alerts = collect_alerts(report)
    report["alerts"] = alerts
    report["alert_count"] = len(alerts)

    stamp = now.strftime("%Y%m%d_%H%M%S")
    out_json = OUT_DIR / f"parity_drift_report_{stamp}.json"
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    latest_json = OUT_DIR / "parity_drift_report_latest.json"
    latest_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))

    print(f"WROTE {out_json}")
    print(f"ALERTS: {len(alerts)}")
    for a in alerts:
        print(f"  - {a}")
    if not report["D2_injection_selftest"]["detection_pass"]:
        print("WARNING: monitor self-test failed -- treat all results with suspicion")


if __name__ == "__main__":
    main()
