#!/usr/bin/env python3
"""max pain 1h·6h 표본 밖 판정 — docs/experiments/eth_maxpain_1h_oos_ledger_20261003.md.
판정 = 평균 > 0 ∧ 날짜 부트스트랩 95% CI 0 배제 · 전체 / 이벤트일(사전) 제외 두 집계 · 표본 밖 만기 40개(주 = 처음 40개, 참고 = 전부).
입력(전부 읽기 전용으로 받은 사본):
  ledger.jsonl   서버 data/live/maxpain_oos_ledger.jsonl 사본
  chain.csv      서버 DB 에서 10-10·10-11·10-12 만기 체인(01:00~08:00) -- 장부에 빠진 10-11·10-12 를 backfill 과 같은 식으로
                 사본에만 채운다(서버 장부는 안 고침). 10-10 은 재현 대조(pain·1h bp 가 서버 장부와 같아야 함).
손 장부(문서 표) 09-29~10-04 1시간 규칙 6건 + 6시간 규칙 10-04(자동 정의 +0.12%) 를 더한다.
사용: python scripts/verdict_eth_maxpain_oos_20261118.py <ledger.jsonl> <chain.csv> <작업 폴더>
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAN1H = {"2026-09-29": -38.5, "2026-09-30": 48.1, "2026-10-01": 139.8, "2026-10-02": -42.36, "2026-10-03": 3.7, "2026-10-04": 34.5}
MAN6H, MANCTL = {"2026-10-04": 0.12}, {"2026-10-04": -0.12}   # 문서: 10-04 손 +0.06%(01:50 스냅샷) vs 자동 정의 +0.12% -- 자동 정의로 통일


def backfill(ledger: Path, chain: Path, wd: Path) -> Path:
    spec = importlib.util.spec_from_file_location("M", ROOT / "scripts/live_eth_maxpain_oos_ledger_20261007.py")
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    M.LEDGER = wd / "ledger_local.jsonl"; shutil.copy(ledger, M.LEDGER)
    g0 = M._get     # get_delivery_prices count=10 은 한 달 전까지 안 닿는다
    M._get = lambda method, **p: g0(method, **({**p, "count": 100} if method == "get_delivery_prices" else p))
    ch = pd.read_csv(chain); ch["t"] = pd.to_datetime(ch.recorded_at_utc, utc=True); ch["exp"] = pd.to_datetime(ch.expiration_ts, utc=True)
    for day in ("2026-10-10", "2026-10-11", "2026-10-12"):
        c = ch[ch.exp == M.exp_of(day)]
        for kind, h in M.WINDOWS.items():
            g = c[[M.in_window(t, day, h) for t in c.t]]; g = g[g.t == g.t.min()]
            row = {"kind": kind, "day": day, "snap_utc": g.t.iloc[0].isoformat(), "pain": M.max_pain(g), "source": "db_local_20261118"}
            if day == "2026-10-10":
                assert row["pain"] == next(r for r in M.rows() if r["kind"] == kind and r["day"] == day)["pain"]
            elif not M.has(kind, day):
                M.append(row)
    for day in ("2026-10-11", "2026-10-12"):
        M.settle(day, now=pd.Timestamp("2026-11-18 12:00", tz="UTC").to_pydatetime())
    p = M.perp_1m(M.exp_of("2026-10-10") - pd.Timedelta(hours=31), M.exp_of("2026-10-10"))
    r = next(r for r in M.rows() if r["kind"] == "pre1h" and r["day"] == "2026-10-10")
    P0, s, _ = M.leg(p, pd.Timestamp(r["snap_utc"]), r["pain"])
    bp = round(s * (float(p.c.asof(M.exp_of("2026-10-10") - pd.Timedelta(minutes=1))) / P0 - 1) * 1e4, 2)
    assert bp == next(r for r in M.rows() if r["kind"] == "settle" and r["day"] == "2026-10-10")["r1h_bp"], bp
    return M.LEDGER


def stat(v, rng):
    v = np.asarray(v, float); b = rng.choice(v, (10000, len(v))).mean(1); lo, hi = np.percentile(b, [2.5, 97.5])
    return {"n": len(v), "mean": round(v.mean(), 3), "ci95": [round(lo, 3), round(hi, 3)], "hit": round(float((v > 0).mean()), 3),
            "pass": bool(v.mean() > 0 and lo > 0)}


def main(ledger: Path, chain: Path, wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    by = {}
    for r in (json.loads(x) for x in backfill(ledger, chain, wd).read_text().splitlines() if x.strip()):
        by.setdefault(r["day"], {})[r["kind"]] = r
    pre_ev = {d: (x.get("event") or {}).get("event") for d, x in by.items()}   # True 만 «이벤트일(사전)» -- 사후·미상은 제외 집계에 남김
    one, six, ctl = dict(MAN1H), dict(MAN6H), dict(MANCTL)
    for d, x in by.items():
        s = x.get("settle") or {}
        if "r1h_bp" in s: one[d] = s["r1h_bp"]
        if "r6h_pct" in s: six[d], ctl[d] = s["r6h_pct"], s["ctl_pct"]
    rng = np.random.default_rng(20261118)
    out = {}
    for nm, d in (("1h_bp", one), ("6h_pct", six), ("6h_control_pct", ctl)):
        days = sorted(d)
        for tag, ds in (("first40", days[:40]), ("all", days)):
            v = np.array([d[x] for x in ds])
            out[f"{nm}.{tag}"] = {"range": [ds[0], ds[-1]], "full": stat(v, rng),
                                  "ex_event_pre": stat([d[x] for x in ds if pre_ev.get(x) is not True], rng),
                                  "n_event_pre": sum(pre_ev.get(x) is True for x in ds),
                                  "mean_without_top3": round(float(np.sort(v)[:-3].mean()), 3)}
    (wd / "verdict_maxpain.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
