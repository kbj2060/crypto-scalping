#!/usr/bin/env python3
"""peg-maker 섀도우 **수정본(@trade) 이후** 실효비용 재측정 (2026-09-04).

왜 -- 08-24 1차 실측(peg 3.11bp/leg, 왕복 ~6.2bp)은 체결 스트림이 죽은 채(`trade_msgs=0`) 8일간
쌓인 **편향 표본**이었다(체결이 quote_cross/taker_fallback으로만 폴백, 실제보다 나쁜 쪽으로
치우침). 2026-09-02 18:41:55 KST 워커 재시작(commit 434096d, @aggTrade→@trade) 이후 레그만 유효.
합의된 절차(메모리 `eth_maker_fill_shadow_realized_cost_checkpoint_20260824` "재측정 실행 절차")를
그대로 따른다:

  경계     recorded_at_utc >= 2026-09-02 09:41:55 UTC (= 18:41:55 KST). 이전 레그와 절대 합치지 않는다.
  자기검증 유효 구간에는 mode에 trade_through/queue_exhaust가 있고, 편향 구간에는 없다.
  잠금     워커가 단일 writer라 read_only 접속도 실패 -> .duckdb + .wal을 **함께** 복사해 사본으로 읽는다
           (08-24에 .wal을 빼먹으면 최신 데이터가 통째로 비어 보이는 함정 확인).
  교차검증 워커 텍스트 로그(`logs/supervisor/maker_fill_shadow_20260902.log`)는 재시작 시점에 생성됐으므로
           그 파일의 'leg done' 줄은 전부 수정본 이후다 -> 건수·정책별 평균이 duckdb와 일치해야 한다.

판정에 넣는 것: 진입만 maker(peg)로 바꿨을 때의 절감분. 트레일링 청산은 taker 고정이므로
왕복 = peg 1다리 + taker 1다리. 표준비용(왕복 10bp)·08-24 편향치(6.2bp)·시뮬 밴드와 나란히 둔다.

사용: python scripts/analyze_eth_maker_fill_shadow_postfix_20260904.py [--db PATH] [--log PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_UTC = "2026-09-02 09:41:55"           # 수정본 워커 시작 (18:41:55 KST)
TAKER_LEG_BP = 5.03                            # 시뮬 실측 taker 즉시체결 (수수료 5 + 슬리피지)
SIM_REF = {"peg": {"calm": (3.09, 3.26), "extreme_days": (3.61, 3.75), "adverse_cond": (3.8, 4.0)},
           "static": {"calm": (3.35, 3.40)}}
BIASED_0824 = {"peg": 3.11, "static": 3.40}    # 08-24 1차(편향) 실측
LOG_RE = re.compile(r"leg done: (\w+) (\w+) (\w+) T(\d+) cost=([-\d.]+)bp mode=(\w+) t=(\d+)ms repegs=(\d+)")


def log(m): print(f"[postfix] {m}", flush=True)


def snapshot_db(db: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    dst = out / db.name
    shutil.copy2(db, dst)
    wal = db.with_name(db.name + ".wal")
    if wal.exists():
        shutil.copy2(wal, out / wal.name)
        log(f"스냅샷: {db.name} {db.stat().st_size/1e6:.1f}MB + .wal {wal.stat().st_size/1e3:.0f}KB")
    else:
        log(f"스냅샷: {db.name} {db.stat().st_size/1e6:.1f}MB (.wal 없음 = 체크포인트 완료 상태)")
    return dst


def stats(v: pd.Series) -> dict:
    v = pd.to_numeric(v, errors="coerce").dropna()
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": round(float(v.mean()), 3), "median": round(float(v.median()), 3),
            "p90": round(float(v.quantile(0.9)), 3), "p99": round(float(v.quantile(0.99)), 3),
            "max": round(float(v.max()), 3)}


def hour_block_ci(df: pd.DataFrame, col="cost_bp", B=2000, seed=0) -> tuple[float, float]:
    """레그는 5분 간격이라 인접 상관 -> 시간(hour) 블록 부트스트랩."""
    g = [x[col].to_numpy(float) for _, x in df.groupby(df["recorded_at_utc"].dt.floor("h"))]
    g = [x[np.isfinite(x)] for x in g if len(x)]
    if len(g) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); means = []
    for _ in range(B):
        pick = rng.integers(0, len(g), len(g))
        arr = np.concatenate([g[i] for i in pick]); means.append(arr.mean())
    return (round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data/live/maker_fill_shadow.duckdb"))
    ap.add_argument("--log-glob", default="logs/supervisor/maker_fill_shadow_2026090[2-9]*.log",
                    help="수정본 이후 워커 로그(재시작으로 여러 파일일 수 있음, *_manual_restart 제외)")
    ap.add_argument("--out", default=str(ROOT / "tmp/eth_maker_fill_shadow_postfix_20260904"))
    a = ap.parse_args()
    import duckdb
    out = Path(a.out)
    snap = snapshot_db(Path(a.db), out)
    con = duckdb.connect(str(snap))               # 사본이므로 read-write로 열어 .wal 재생을 허용한다

    L = con.execute("select * from maker_fill_shadow_legs").df()
    H = con.execute("select * from maker_fill_shadow_heartbeat order by recorded_at_utc").df()
    con.close()
    L["recorded_at_utc"] = pd.to_datetime(L["recorded_at_utc"]).dt.tz_localize(None)
    H["recorded_at_utc"] = pd.to_datetime(H["recorded_at_utc"]).dt.tz_localize(None)
    # ⚠️컬럼 이름은 _utc지만 값은 서버 세션 타임존(KST)으로 저장돼 있다 -- 워커가 tz-aware datetime을
    # TIMESTAMP(naive)에 넣을 때 DuckDB가 세션 TimeZone으로 변환한다. 로그(KST)와 DB의 마지막 레그
    # 시각이 초 단위까지 같은 것으로 확인(2026-09-04). 경계는 DB 시계에 맞춰 적용해야 한다.
    log_files_all = sorted(p for p in ROOT.glob(a.log_glob) if "manual_restart" not in p.name)
    log_first = log_last = None
    for lp in log_files_all:
        legs_ts = [pd.Timestamp(ln[:19]) for ln in lp.read_text(errors="ignore").splitlines() if "leg done" in ln]
        if legs_ts:
            log_first = min(log_first or pd.Timestamp.max, legs_ts[0]); log_last = max(log_last or pd.Timestamp.min, legs_ts[-1])
    db_last = L.recorded_at_utc.max()
    offset_h = int(round((db_last - log_last).total_seconds() / 3600)) if log_last is not None else 0
    # 경계 = 수정본 프로세스의 **첫 leg 로그 시각**(데이터 유도, 1초 여유). 메모리의 18:41:55 KST는 확인용으로만 찍는다
    # (실측: 첫 trade_through가 18:40:07에 이미 있었다 -- 18:41:55는 워커 자체 로그 기준이라 약간 늦다).
    nominal = pd.Timestamp(BOUNDARY_UTC) + pd.Timedelta(hours=9 + offset_h)
    b = (log_first - pd.Timedelta(seconds=1) + pd.Timedelta(hours=offset_h)) if log_first is not None else nominal
    log(f"DB 시계 진단: DB 마지막 {db_last} vs 로그(KST) 마지막 {log_last} → 오프셋 {offset_h:+d}h "
        f"({'DB=KST' if offset_h == 0 else 'DB=UTC' if offset_h == -9 else '⚠️예상밖'})")
    log(f"경계(DB 시계) = 수정본 첫 leg {log_first} − 1s = {b}   (메모리 명목치 18:41:55 KST → {nominal}, 차이 {(nominal - b).total_seconds()/60:+.1f}분)")
    rep_clock = {"db_last": str(db_last), "log_last_kst": str(log_last), "log_first_kst": str(log_first),
                 "db_minus_log_hours": offset_h, "boundary_db_clock": str(b), "nominal_boundary_db_clock": str(nominal)}
    pre, post = L[L.recorded_at_utc < b], L[L.recorded_at_utc >= b].copy()
    log(f"legs 전체 {len(L):,} · 경계 이전(편향) {len(pre):,} · 이후(유효) {len(post):,} "
        f"({post.recorded_at_utc.min()} ~ {post.recorded_at_utc.max()} UTC, "
        f"{(post.recorded_at_utc.max()-post.recorded_at_utc.min()).total_seconds()/3600:.1f}h)")

    rep: dict = {"boundary_utc": BOUNDARY_UTC, "db_clock": rep_clock, "run_utc": datetime.now(timezone.utc).isoformat(),
                 "n_pre_biased": int(len(pre)), "n_post_valid": int(len(post)),
                 "post_window_utc": [str(post.recorded_at_utc.min()), str(post.recorded_at_utc.max())],
                 "post_hours": round((post.recorded_at_utc.max() - post.recorded_at_utc.min()).total_seconds() / 3600, 2)}

    # ── 자기검증 ① mode 분포로 경계가 맞는가
    mode_pre = pre.fill_mode.value_counts().to_dict(); mode_post = post.fill_mode.value_counts().to_dict()
    ok_pre = not any(m in mode_pre for m in ("trade_through", "queue_exhaust"))
    ok_post = any(m in mode_post for m in ("trade_through", "queue_exhaust"))
    log(f"mode 이전 {mode_pre}")
    log(f"mode 이후 {mode_post}")
    log(f"경계 자기검증: 이전에 trade모드 없음 {'✅' if ok_pre else '❌'} · 이후에 trade모드 있음 {'✅' if ok_post else '❌'}")
    rep["boundary_selfcheck"] = {"mode_pre": mode_pre, "mode_post": mode_post, "pass": bool(ok_pre and ok_post)}

    # ── 자기검증 ② heartbeat -- 체결 스트림이 살아 있는가
    hp = H[H.recorded_at_utc >= b]
    if len(hp):
        tm = hp.trade_msgs.to_numpy(); bm = hp.book_msgs.to_numpy()
        alive = bool(len(tm) > 1 and tm[-1] > 0 and (np.diff(tm) > 0).mean() > 0.9)   # 재시작 리셋 1회 허용
        resets = int((np.diff(bm) < 0).sum())
        now_db = datetime.now(timezone.utc).replace(tzinfo=None) + pd.Timedelta(hours=9 + offset_h)
        age_min = (now_db - hp.recorded_at_utc.max()).total_seconds() / 60
        log(f"heartbeat 이후 {len(hp)}건 · trade_msgs {tm[0]:,}→{tm[-1]:,} · book_msgs {bm[0]:,}→{bm[-1]:,} "
            f"(카운터 리셋 {resets}회=재시작) · 마지막 {age_min:.0f}분 전 · 스트림 생존 {'✅' if alive else '❌'}")
        rep["heartbeat"] = {"n": int(len(hp)), "trade_msgs_first_last": [int(tm[0]), int(tm[-1])],
                            "book_msgs_first_last": [int(bm[0]), int(bm[-1])], "counter_resets": resets,
                            "last_age_min": round(age_min, 1), "alive": alive}
    else:
        log("⚠️경계 이후 heartbeat 없음"); rep["heartbeat"] = {"n": 0, "alive": False}

    # ── 자기검증 ②' 재시작/공백 -- legs 간격 > 15분이면 데이터 공백(재부팅·스트림 정지)으로 기록
    ts_sorted = post.recorded_at_utc.sort_values().to_numpy()
    gaps = np.diff(ts_sorted) / np.timedelta64(1, "m") if len(ts_sorted) > 1 else np.array([])
    big = np.flatnonzero(gaps > 15)
    rep["gaps_over_15min"] = [{"from": str(ts_sorted[i]), "to": str(ts_sorted[i + 1]), "minutes": round(float(gaps[i]), 1)} for i in big]
    if len(big):
        log(f"⚠️유효 구간 안 공백 {len(big)}건: " + "; ".join(f"{g['from'][5:16]}→{g['to'][5:16]} ({g['minutes']:.0f}분)" for g in rep["gaps_over_15min"]))
    else:
        log("유효 구간 안 15분 초과 공백 없음")

    # ── 자기검증 ③ 로그 교차검증 (수정본 로그 파일들은 재시작 시점에 생성됨 = 전부 수정본 이후)
    files = sorted(p for p in ROOT.glob(a.log_glob) if "manual_restart" not in p.name)
    xcheck = {"log_files": [str(p.relative_to(ROOT)) for p in files], "available": bool(files)}
    if files:
        rows = []
        for lp in files:
            rows += [LOG_RE.search(ln) for ln in lp.read_text(errors="ignore").splitlines()]
        rows = [m.groups() for m in rows if m]
        lg = pd.DataFrame(rows, columns=["trigger", "policy", "side", "timeout", "cost_bp", "mode", "t_ms", "repegs"])
        lg["cost_bp"] = pd.to_numeric(lg.cost_bp, errors="coerce")
        cmp = {}
        for pol in sorted(post.policy.unique()):
            d_db = post[(post.policy == pol) & post.cost_bp.notna()]; d_lg = lg[lg.policy == pol]
            cmp[pol] = {"db_n": int(len(d_db)), "log_n": int(len(d_lg)),
                        "db_mean": round(float(d_db.cost_bp.mean()), 4), "log_mean": round(float(d_lg.cost_bp.mean()), 4) if len(d_lg) else None}
        match = all(abs(v["db_n"] - v["log_n"]) <= 8 and v["log_mean"] is not None and abs(v["db_mean"] - v["log_mean"]) < 0.02 for v in cmp.values())
        log(f"로그 교차검증 {cmp} → {'✅ 일치' if match else '⚠️불일치(활성 레그/로테이션 확인)'}")
        xcheck.update({"per_policy": cmp, "match": bool(match), "log_legs": int(len(lg))})
    else:
        log(f"⚠️로그 없음 {a.log_glob} -- duckdb 단독")
    rep["log_crosscheck"] = xcheck

    # ── 본 결과: 정책 × 트리거
    post["hour_utc"] = post.recorded_at_utc.dt.hour
    med_spread = float(post.spread_bp.median())
    tables = {}
    def summarize(d: pd.DataFrame) -> dict:
        d = d[d.fill_mode != "aborted_stale"]
        filled = d[d.filled == True]  # noqa: E712
        fb = d[d.fill_mode == "taker_fallback"]
        r = {"n": int(len(d)), "fill_rate": round(float((d.filled == True).mean()), 4) if len(d) else None,  # noqa: E712
             "cost_all": stats(d.cost_bp), "cost_filled": stats(filled.cost_bp), "cost_fallback": stats(fb.cost_bp),
             "modes": d.fill_mode.value_counts().to_dict(), "repegs_mean": round(float(d.repegs.mean()), 3) if len(d) else None,
             "fill_t_ms_median": round(float(filled.fill_t_ms.median()), 0) if len(filled) else None,
             "ci95_hourblock": hour_block_ci(d) if len(d) else None}
        return r
    print("\n=== 유효 구간 정책 × 트리거 (bp/leg, 수수료 포함: maker 2 / taker 5) ===")
    print(f"{'policy':8s}{'trigger':10s}{'n':>6s}{'fill%':>7s}{'mean':>7s}{'med':>7s}{'p90':>7s}{'filled':>8s}{'fallbk':>8s}{'CI95(hour-block)':>20s}  modes")
    for (pol, trg), d in post.groupby(["policy", "trigger"]):
        s = summarize(d); tables[f"{pol}/{trg}"] = s
        ca, cf, cb = s["cost_all"], s["cost_filled"], s["cost_fallback"]
        print(f"{pol:8s}{trg:10s}{s['n']:6d}{(s['fill_rate'] or 0)*100:6.1f}%{ca.get('mean', float('nan')):7.2f}{ca.get('median', float('nan')):7.2f}"
              f"{ca.get('p90', float('nan')):7.2f}{cf.get('mean', float('nan')):8.2f}{cb.get('mean', float('nan')):8.2f}"
              f"{str(s['ci95_hourblock']):>20s}  {s['modes']}")
    rep["by_policy_trigger"] = tables

    # ── 정책 × 방향 / 스프레드 레짐 / 시간대 (비대칭·레짐 의존 확인)
    sched = post[post.trigger == "schedule"]
    rep["by_policy_side"] = {f"{p}/{s}": stats(d.cost_bp) for (p, s), d in sched.groupby(["policy", "side"])}
    sched = sched.assign(spread_regime=np.where(sched.spread_bp > med_spread, "wide", "tight"))
    rep["by_policy_spread"] = {f"{p}/{s}": stats(d.cost_bp) for (p, s), d in sched.groupby(["policy", "spread_regime"])}
    rep["median_spread_bp"] = round(med_spread, 4)
    hb = sched[sched.policy == "peg"].groupby(sched.hour_utc // 4 * 4).cost_bp.agg(["count", "mean", "median"]).round(3)
    rep["peg_by_4h_utc"] = {str(int(k)): v for k, v in hb.to_dict("index").items()}
    print("\n=== peg (schedule) 방향/스프레드/시간대 ===")
    for k, v in rep["by_policy_side"].items():
        if k.startswith("peg"): print(f"  {k:10s} n={v.get('n')} mean={v.get('mean')} med={v.get('median')} p90={v.get('p90')}")
    for k, v in rep["by_policy_spread"].items():
        if k.startswith("peg"): print(f"  {k:10s} n={v.get('n')} mean={v.get('mean')} med={v.get('median')} p90={v.get('p90')}")
    print("  4h(UTC) 블록: " + " ".join(f"{k}h:{v['mean']:.2f}({int(v['count'])})" for k, v in rep["peg_by_4h_utc"].items()))

    # ── 편향 구간과 대조 (같은 정책, schedule)
    pre_s = pre[(pre.trigger == "schedule") & (pre.fill_mode != "aborted_stale")]
    rep["biased_pre_for_contrast"] = {p: stats(d.cost_bp) for p, d in pre_s.groupby("policy")}
    print("\n=== 편향 구간(경계 이전, 참고용) ===")
    for p, v in rep["biased_pre_for_contrast"].items():
        print(f"  {p:8s} n={v.get('n')} mean={v.get('mean')} med={v.get('median')} p90={v.get('p90')}")

    # ── 왕복 비용 함의
    peg = tables.get("peg/schedule", {}).get("cost_all", {}); stc = tables.get("static/schedule", {}).get("cost_all", {})
    peg_m = peg.get("mean", float("nan")); peg_p90 = peg.get("p90", float("nan")); ci = tables.get("peg/schedule", {}).get("ci95_hourblock")
    rt = {"standard_taker_taker": 10.0, "biased_0824_peg_peg": 2 * BIASED_0824["peg"],
          "peg_entry_plus_taker_exit": round(peg_m + TAKER_LEG_BP, 2), "peg_entry_plus_taker_exit_p90": round(peg_p90 + TAKER_LEG_BP, 2),
          "peg_entry_plus_taker_exit_ci95": [round(ci[0] + TAKER_LEG_BP, 2), round(ci[1] + TAKER_LEG_BP, 2)] if ci else None,
          "peg_peg_both_maker": round(2 * peg_m, 2), "taker_leg_assumed_bp": TAKER_LEG_BP}
    rep["roundtrip_bp"] = rt
    print("\n=== 왕복 비용 (bp) ===")
    print(f"  표준 가정 taker+taker            10.00")
    print(f"  08-24 편향 peg+peg                {rt['biased_0824_peg_peg']:.2f}")
    print(f"  ⭐진입 peg + 청산 taker(5.03)     {rt['peg_entry_plus_taker_exit']:.2f}   (p90 {rt['peg_entry_plus_taker_exit_p90']:.2f} · CI95 {rt['peg_entry_plus_taker_exit_ci95']})")
    print(f"  peg+peg (양다리 maker 가능할 때)  {rt['peg_peg_both_maker']:.2f}")
    print("\n시뮬 밴드(peg): calm 3.09~3.26 · extreme 3.61~3.75 · adverse 3.8~4.0  |  static calm 3.35~3.40")
    band = "calm" if peg_m <= 3.3 else ("extreme" if peg_m <= 3.75 else ("adverse" if peg_m <= 4.0 else "위"))
    log(f"peg 평균 {peg_m:.2f}bp → 시뮬 {band} 밴드 · static 평균 {stc.get('mean', float('nan')):.2f}bp")
    rep["verdict_hint"] = {"peg_mean_bp": peg_m, "sim_band": band, "static_mean_bp": stc.get("mean")}

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    post.to_csv(out / "legs_post_fix.csv", index=False)
    log(f"산출: {out}/report.json · legs_post_fix.csv ({len(post):,}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
