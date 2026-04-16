#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import random
import shutil
from pathlib import Path
from urllib import parse, request

import numpy as np

try:
    from scripts.optimize_duckdb_quant_formula import load_merged, run_sim, sample_params
    from scripts.backtest_param_ensemble import _ensemble_backtest
except ModuleNotFoundError:
    from optimize_duckdb_quant_formula import load_merged, run_sim, sample_params
    from backtest_param_ensemble import _ensemble_backtest

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _score_result(res: dict, trades_target: int = 40) -> float:
    pnl = float(res.get("pnl_pct", 0.0))
    mdd = abs(float(res.get("mdd_pct", 0.0)))
    sharpe = float(res.get("sharpe", 0.0))
    trades = float(res.get("trades", 0.0))
    trade_pen = abs(trades - trades_target) * 0.03
    return pnl + 0.35 * sharpe - 0.25 * mdd - trade_pen


def _search_candidates(train, iters: int, rng: random.Random, mode: str) -> list[tuple[float, object, dict]]:
    out: list[tuple[float, object, dict]] = []
    for _ in range(iters):
        p = sample_params(rng)
        if mode == "balanced":
            p["cooldown"] = rng.randint(2, 30)
            p["entry"] = rng.uniform(0.50, 0.84)
            p["exit"] = rng.uniform(0.22, 0.60)
            p["volr_min"] = rng.uniform(0.15, 0.85)
            p["vpin_max"] = rng.uniform(0.80, 0.999)
            p["lev"] = rng.uniform(2.5, 10.0)
            if p["exit"] >= p["entry"]:
                p["exit"] = max(0.20, p["entry"] - 0.06)
            min_trades = 8
            max_dd = -12.0
            trades_target = 45
        else:
            p["cooldown"] = rng.randint(24, 96)
            p["entry"] = rng.uniform(0.68, 0.90)
            p["exit"] = rng.uniform(0.30, 0.70)
            p["volr_min"] = rng.uniform(0.25, 1.00)
            p["vpin_max"] = rng.uniform(0.82, 0.995)
            p["lev"] = rng.uniform(2.0, 6.0)
            if p["exit"] >= p["entry"]:
                p["exit"] = max(0.25, p["entry"] - 0.10)
            min_trades = 4
            max_dd = -10.0
            trades_target = 18

        r = run_sim(train, p)
        if float(r.mdd_pct) < max_dd or int(r.trades) < min_trades:
            continue
        rr = {
            "pnl_pct": float(r.pnl_pct),
            "mdd_pct": float(r.mdd_pct),
            "trades": int(r.trades),
            "win_rate": float(r.win_rate),
            "sharpe": float(r.sharpe),
            "equity": float(r.equity),
        }
        score = _score_result(rr, trades_target=trades_target)
        out.append((score, r, p))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _extract_params(arr: list[dict], key: str) -> list[dict]:
    out = []
    for x in arr:
        p = dict((x or {}).get(key, {}) or {})
        if p:
            out.append(p)
    return out


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            j = json.load(f)
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}


def _eval_incumbent_oos(oos, balanced_path: Path, lowfreq_params_path: Path, lowfreq_grid_path: Path) -> dict:
    out = {
        "balanced": None,
        "lowfreq": None,
    }
    bj = _load_json(balanced_path)
    bparams = _extract_params(list(bj.get("top_params", []) or []), "params")[:10]
    bvotes = int((bj.get("search", {}) or {}).get("min_votes", 6) or 6)
    if bparams:
        out["balanced"] = _ensemble_backtest(oos, bparams, min_votes=bvotes)

    lj = _load_json(lowfreq_params_path)
    lparams = _extract_params(list(lj.get("top10_singles", []) or []), "params")[:10]
    gj = _load_json(lowfreq_grid_path)
    lvotes = int((gj.get("best", {}) or {}).get("votes", 7) or 7)
    if lparams:
        out["lowfreq"] = _ensemble_backtest(oos, lparams, min_votes=lvotes)
    return out


def _promotion_pass(candidate_oos: dict, incumbent_oos: dict | None, min_trades: int, max_mdd_abs: float, min_pnl: float, improve_eps: float) -> tuple[bool, str]:
    trades = int(candidate_oos.get("trades", 0) or 0)
    mdd_abs = abs(float(candidate_oos.get("mdd_pct", 0.0) or 0.0))
    pnl = float(candidate_oos.get("pnl_pct", 0.0) or 0.0)
    if trades < min_trades:
        return False, f"trades<{min_trades}"
    if mdd_abs > max_mdd_abs:
        return False, f"mdd>{max_mdd_abs}"
    if pnl < min_pnl:
        return False, f"pnl<{min_pnl}"

    cand_score = _score_result(candidate_oos)
    if incumbent_oos:
        inc_score = _score_result(incumbent_oos)
        if cand_score < inc_score + improve_eps:
            return False, f"score_not_improved({cand_score:.3f}<{inc_score + improve_eps:.3f})"
    return True, "pass"


def _atomic_swap_many(payloads: dict[Path, dict], backup_root: Path) -> None:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = backup_root / ts
    bdir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    backups: dict[Path, Path] = {}
    try:
        for path, payload in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                bpath = bdir / path.name
                shutil.copy2(path, bpath)
                backups[path] = bpath

            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            written.append(path)
    except Exception:
        for path in written:
            b = backups.get(path)
            if b and b.exists():
                shutil.copy2(b, path)
        raise


def _send_telegram(text: str) -> bool:
    token = str(os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = parse.urlencode({
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        with request.urlopen(req, timeout=8) as resp:
            return int(resp.status) == 200
    except Exception:
        return False


def _build_notify_text(report: dict) -> str:
    st = str(report.get("status", "-"))
    t = str(report.get("time", "-"))
    b = dict(report.get("balanced", {}) or {})
    l = dict(report.get("lowfreq", {}) or {})
    bo = dict(b.get("oos", {}) or {})
    lo = dict((l.get("best", {}) or {}).get("oos", {}) or {})
    lines = [
        f"[주간 앙상블 업데이트] {st}",
        f"time={t}",
        f"balanced pass={b.get('pass')} reason={b.get('reason')}",
        f"balanced oos pnl={bo.get('pnl_pct')} mdd={bo.get('mdd_pct')} trades={bo.get('trades')}",
        f"lowfreq pass={l.get('pass')} reason={l.get('reason')}",
        f"lowfreq oos pnl={lo.get('pnl_pct')} mdd={lo.get('mdd_pct')} trades={lo.get('trades')}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Weekly ensemble parameter updater (walk-forward + promotion gate + rollback)")
    ap.add_argument("--days", type=int, default=36500)
    ap.add_argument("--oos-days", type=int, default=7)
    ap.add_argument("--search-iters", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=20260414)
    ap.add_argument("--price-csv", default="binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
    ap.add_argument("--balanced-votes", type=int, default=6)
    ap.add_argument("--lowfreq-votes-min", type=int, default=6)
    ap.add_argument("--lowfreq-votes-max", type=int, default=9)
    ap.add_argument("--require-weekday", type=int, default=0, help="0=Mon ... 6=Sun")
    ap.add_argument("--skip-weekday-check", action="store_true")
    ap.add_argument("--promote-min-trades", type=int, default=5)
    ap.add_argument("--promote-max-mdd", type=float, default=12.0)
    ap.add_argument("--promote-min-pnl", type=float, default=0.0)
    ap.add_argument("--promote-improve-eps", type=float, default=0.05)
    ap.add_argument("--notify-telegram", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    now_kst = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    if (not args.skip_weekday_check) and now_kst.weekday() != int(args.require_weekday):
        print(json.dumps({
            "status": "skipped",
            "reason": f"weekday_mismatch(now={now_kst.weekday()}, required={args.require_weekday})",
            "now_kst": now_kst.isoformat(),
        }, ensure_ascii=False, indent=2))
        return

    rng = random.Random(args.seed)
    m = load_merged(args.price_csv, args.days)
    m = m.sort_values("ts").reset_index(drop=True)
    if len(m) < 1000:
        raise RuntimeError("Not enough merged rows")

    cutoff = m["ts"].max() - dt.timedelta(days=int(args.oos_days))
    train = m[m["ts"] <= cutoff].reset_index(drop=True)
    oos = m[m["ts"] > cutoff].reset_index(drop=True)
    if len(train) < 500 or len(oos) < 50:
        split_idx = int(len(m) * 0.8)
        train = m.iloc[:split_idx].reset_index(drop=True)
        oos = m.iloc[split_idx:].reset_index(drop=True)
        cutoff = train["ts"].max() if len(train) else m["ts"].min()
    if len(train) < 500 or len(oos) < 50:
        raise RuntimeError(f"split too small train={len(train)} oos={len(oos)}")

    c_bal = _search_candidates(train, args.search_iters, rng, mode="balanced")
    c_low = _search_candidates(train, max(4000, args.search_iters // 3), rng, mode="lowfreq")
    if len(c_bal) < 10 or len(c_low) < 10:
        raise RuntimeError(f"candidate shortage balanced={len(c_bal)} lowfreq={len(c_low)}")

    bal_top = c_bal[:10]
    bal_params = [x[2] for x in bal_top]
    bal_train = _ensemble_backtest(train, bal_params, min_votes=args.balanced_votes)
    bal_oos = _ensemble_backtest(oos, bal_params, min_votes=args.balanced_votes)

    low_top = c_low[:10]
    low_params = [x[2] for x in low_top]
    low_grid = []
    best_low = None
    best_low_score = -1e18
    for v in range(args.lowfreq_votes_min, args.lowfreq_votes_max + 1):
        tr_res = _ensemble_backtest(train, low_params, min_votes=v)
        oo_res = _ensemble_backtest(oos, low_params, min_votes=v)
        row = {
            "k": 10,
            "votes": int(v),
            "train": tr_res,
            "oos": oo_res,
        }
        low_grid.append(row)
        sc = _score_result(oo_res, trades_target=20)
        if sc > best_low_score:
            best_low_score = sc
            best_low = row

    metrics_dir = Path("data/ensemble/metrics")
    bal_path = metrics_dir / "param_ensemble_result.json"
    low_params_path = metrics_dir / "param_ensemble_lowfreq_highpnl.json"
    low_grid_path = metrics_dir / "param_ensemble_lowfreq_grid.json"

    incumbent = _eval_incumbent_oos(oos, bal_path, low_params_path, low_grid_path)
    bal_pass, bal_reason = _promotion_pass(
        candidate_oos=bal_oos,
        incumbent_oos=incumbent.get("balanced"),
        min_trades=args.promote_min_trades,
        max_mdd_abs=args.promote_max_mdd,
        min_pnl=args.promote_min_pnl,
        improve_eps=args.promote_improve_eps,
    )
    low_pass, low_reason = _promotion_pass(
        candidate_oos=dict((best_low or {}).get("oos", {}) or {}),
        incumbent_oos=incumbent.get("lowfreq"),
        min_trades=args.promote_min_trades,
        max_mdd_abs=args.promote_max_mdd,
        min_pnl=args.promote_min_pnl,
        improve_eps=args.promote_improve_eps,
    )

    update_date = now_kst.strftime("%Y-%m-%d")
    common_meta = {
        "param_updated_at": update_date,
        "update_cycle": "주 1회 업데이트",
        "generated_at": now_kst.isoformat(),
        "run_policy": {
            "require_weekday": int(args.require_weekday),
            "oos_days": int(args.oos_days),
            "search_iters": int(args.search_iters),
        },
        "dataset": {
            "rows": int(len(m)),
            "train_rows": int(len(train)),
            "oos_rows": int(len(oos)),
            "start": str(m["ts"].min()),
            "end": str(m["ts"].max()),
            "cutoff": str(cutoff),
        },
    }

    bal_payload = {
        **common_meta,
        "search": {
            "iters": int(args.search_iters),
            "pool_size": int(len(c_bal)),
            "top_k": 10,
            "min_votes": int(args.balanced_votes),
        },
        "ensemble_result": bal_train,
        "oos_result": bal_oos,
        "promotion": {
            "pass": bool(bal_pass),
            "reason": str(bal_reason),
            "incumbent_oos": incumbent.get("balanced"),
        },
        "top_params": [
            {
                "rank": i + 1,
                "score": float(bal_top[i][0]),
                "single": {
                    "pnl_pct": float(bal_top[i][1].pnl_pct),
                    "mdd_pct": float(bal_top[i][1].mdd_pct),
                    "trades": int(bal_top[i][1].trades),
                    "win_rate": float(bal_top[i][1].win_rate),
                    "sharpe": float(bal_top[i][1].sharpe),
                },
                "params": bal_top[i][2],
            }
            for i in range(10)
        ],
    }

    low_params_payload = {
        **common_meta,
        "trials": int(max(4000, args.search_iters // 3)),
        "pool_size": int(len(c_low)),
        "top10_singles": [
            {
                "rank": i + 1,
                "score": float(low_top[i][0]),
                "pnl_pct": float(low_top[i][1].pnl_pct),
                "mdd_pct": float(low_top[i][1].mdd_pct),
                "trades": int(low_top[i][1].trades),
                "win_rate": float(low_top[i][1].win_rate),
                "sharpe": float(low_top[i][1].sharpe),
                "params": low_top[i][2],
            }
            for i in range(10)
        ],
        "ensemble": {
            "train": (best_low or {}).get("train", {}),
            "oos": (best_low or {}).get("oos", {}),
            "votes": int((best_low or {}).get("votes", 7)),
        },
        "promotion": {
            "pass": bool(low_pass),
            "reason": str(low_reason),
            "incumbent_oos": incumbent.get("lowfreq"),
        },
    }

    low_grid_payload = {
        **common_meta,
        "best": {
            **dict((best_low or {}).get("oos", {}) or {}),
            "k": int((best_low or {}).get("k", 10)),
            "votes": int((best_low or {}).get("votes", 7)),
            "train": dict((best_low or {}).get("train", {}) or {}),
        },
        "all": low_grid,
        "promotion": {
            "pass": bool(low_pass),
            "reason": str(low_reason),
            "incumbent_oos": incumbent.get("lowfreq"),
        },
    }

    report = {
        "status": "computed",
        "dry_run": bool(args.dry_run),
        "time": now_kst.isoformat(),
        "balanced": {
            "pass": bool(bal_pass),
            "reason": str(bal_reason),
            "train": bal_train,
            "oos": bal_oos,
            "incumbent_oos": incumbent.get("balanced"),
        },
        "lowfreq": {
            "pass": bool(low_pass),
            "reason": str(low_reason),
            "best": (best_low or {}),
            "incumbent_oos": incumbent.get("lowfreq"),
        },
        "paths": {
            "balanced": str(bal_path),
            "lowfreq_params": str(low_params_path),
            "lowfreq_grid": str(low_grid_path),
        },
    }

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.notify_telegram:
            _send_telegram(_build_notify_text(report))
        return

    payloads: dict[Path, dict] = {}
    if bal_pass:
        payloads[bal_path] = bal_payload
    if low_pass:
        payloads[low_params_path] = low_params_payload
        payloads[low_grid_path] = low_grid_payload

    if payloads:
        _atomic_swap_many(payloads, backup_root=metrics_dir / "weekly_backups")
        report["status"] = "updated"
    else:
        report["status"] = "kept_incumbent"

    out_report = metrics_dir / "weekly_param_update_report.json"
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.notify_telegram:
        _send_telegram(_build_notify_text(report))


if __name__ == "__main__":
    main()
