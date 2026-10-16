#!/usr/bin/env python3
"""**우리 입력 전체(윈도우+규칙 사건+청산맵 1d/7d/30d+풋프린트 1h/2h/4h/1d)로 로컬 LLM 방향 예측** — 10-01~10-15, 2시간 간격 (2026-10-17).

왜: 사용자 «우리가 만든 입력 텍스트로 예측을 잘 하는 로컬 모델» → «방향 예측에 중점». 400 시점 시험(피쳐 99칸)엔 청산맵·풋프린트·윈도우가 없었다.
입력 재구성 = research_llm_haiku_liqfp_read_20261017 의 함수 그대로(청산맵 compute_tier_profile · 서버 테이프 /tmp 덤프 · 일봉 API · REST 윈도우).
정답 = 결정 T 다음 5분봉부터 1h·4h 뒤 종가 로그수익(bp). 🔴정답 창이 겹친다(2h 간격 < 4h) → 신뢰구간은 UTC 일 블록 부트스트랩.
기준선 = 되돌림(−직전 1h)·모멘텀(직전 1h)·직전 4h 되돌림. 표본 ≈175 → 참고용(검정력 낮음).
  python scripts/research_local_llm_brief_direction_20261017.py build | eval
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_llm_haiku_liqfp_read_20261017 as F  # noqa: E402
from dashboard import llm_brief as lb  # noqa: E402
from scripts import live_llm_brief_20261017 as live  # noqa: E402

OUT = ROOT / "tmp/local_llm_briefdir_20261017"
T0, T1, STEP_H = "2026-10-01 06:00", "2026-10-15 20:00", 2
PROMPT = """You are given one snapshot of the ETH perpetual futures market (Binance ETHUSDT) at a 5-minute bar close (times are UTC bar closes).
It has: w5 (last 2h, 5-minute bars), w1h (last 24h, 1-hour bars), rule-detected events, a liquidation map summary
(1d/7d/30d: estimated liquidation bands above price = shorts, below = longs; model estimates) and a footprint summary
(1h/2h/4h windows of 5-minute taker volume by price; 1d = last complete days).
Estimate the probability (0-100) that the ETH price will be HIGHER than now: p_up_1h = 1 hour later, p_up_4h = 4 hours later.
Use your knowledge of crypto market microstructure. 50 means no edge. Be honest; extreme values only when the evidence is strong. Do not try to guess the date.
Output ONLY one JSON object: {"id": "<the snapshot id>", "p_up_1h": int, "p_up_4h": int}

Snapshot:
"""


def times() -> list[int]:
    t = pd.date_range(T0, T1, freq=f"{STEP_H}h", tz="UTC")
    return [int(x.value // 10**6) for x in t]


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "prompts").mkdir(exist_ok=True)
    T = times()
    dump_path = OUT / "dumps.json"
    if not dump_path.exists():
        (OUT / "dump.py").write_text(F.DUMP)
        subprocess.run(["scp", "-q", "-P", "2222", str(OUT / "dump.py"), "llewyn@192.168.1.89:/tmp/llmfp_dump.py"], check=True)
        res = subprocess.run(F.SSH + [f"cd ~/crypto-scalping && nice -n 15 ~/miniconda3/envs/quant_ai/bin/python /tmp/llmfp_dump.py '{json.dumps([t // 1000 for t in T])}' > /tmp/llmfp_dump_out.json"],
                             check=True, timeout=1800)
        subprocess.run(["scp", "-q", "-P", "2222", "llewyn@192.168.1.89:/tmp/llmfp_dump_out.json", str(dump_path)], check=True)
    dumps = json.loads(dump_path.read_text())
    base = "http://127.0.0.1:18787"
    k5 = F.kl("5m", T[-1] + 5 * 3600_000, 5000)                    # 정답용 5분봉
    close = pd.Series(k5.close.to_numpy(), index=k5.close_time.to_numpy() + 1)   # 마감 직후 시각 → 종가
    gold = []
    for j, t in enumerate(T):
        sid = f"b{j:03d}"
        if (OUT / "prompts" / f"dir_{sid}.txt").exists():
            continue
        try:
            lm, px = F.liqmaps(t)
            fpb = F.fp_bars(dumps[str(t // 1000)], px)
            w5, w1h, close5 = F.windows_hist(t, fpb)
            day = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
            d_from, d_to = (day - timedelta(days=3)).strftime("%Y-%m-%d"), (day - timedelta(days=1)).strftime("%Y-%m-%d")
            fpd = live.get(f"{base}/api/footprint-daily", {"asset": "eth", "since": d_from})
            cols = {k: [x for x, d in zip(v, fpd["cols"]["d"]) if d <= d_to][-3:] for k, v in fpd["cols"].items()}
            cols["part"] = [0] * len(cols["d"])
            cells = live.get(f"{base}/api/footprint-daily/cells", {"asset": "eth", "from": d_from, "to": d_to, "row": 2})
        except Exception as e:  # noqa: BLE001
            print(sid, "skip", type(e).__name__, str(e)[:120]); continue
        brief = {"id": sid, "time_utc": day.strftime("%H:%M %a"), "w5": w5, "w1h": w1h, "events": lb.rule_events(w5),
                 "liquidation_map": {k: lb.liqmap_summary(v, px) for k, v in lm.items()},
                 "footprint": {"1h": lb.footprint_summary(fpb, 12, px, close5), "2h": lb.footprint_summary(fpb, 24, px, close5, group=3),
                               "4h": lb.footprint_summary(fpb, 48, px, close5, group=3),
                               "1d": lb.footprint_daily_summary(cols, cells.get("cells"), 2.0, px)}}
        (OUT / "prompts" / f"dir_{sid}.txt").write_text(PROMPT + json.dumps(brief, separators=(",", ":"), ensure_ascii=False))
        c0 = close.get(t)
        gold.append({"id": sid, "T": t, "day": day.strftime("%Y-%m-%d"),
                     "y_1h": np.log(close.get(t + 3600_000) / c0) * 1e4 if c0 else np.nan,
                     "y_4h": np.log(close.get(t + 4 * 3600_000) / c0) * 1e4 if c0 else np.nan,
                     "r1h": -w5["px_pct"][-13] * 100 if w5["px_pct"][-13] is not None else np.nan,          # 직전 1h 수익(bp) = −(1h 전 가격의 지금 대비 %)
                     "r4h": -w1h["px_pct"][-5] * 100 if w1h["px_pct"][-5] is not None else np.nan})
        print(sid, day, flush=True)
        time.sleep(0.3)
    old = pd.read_csv(OUT / "gold.csv") if (OUT / "gold.csv").exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(gold)]).drop_duplicates("id").to_csv(OUT / "gold.csv", index=False)
    print("prompts", len(list((OUT / "prompts").glob("*.txt"))))


def evaluate() -> None:
    from scipy.stats import spearmanr
    g = pd.read_csv(OUT / "gold.csv").dropna(subset=["y_1h", "y_4h"])
    for md in sorted((OUT / "out").iterdir()):
        ans = {}
        for f in md.glob("*.json"):
            r = json.loads(f.read_text())
            try:
                c = r["message"]["content"]; j = json.loads(c[c.index("{"): c.rindex("}") + 1])
                ans[f.stem[4:]] = (float(j["p_up_1h"]), float(j["p_up_4h"]))
            except Exception:  # noqa: BLE001
                pass
        d = g[g.id.isin(ans)].copy()
        d["p1"] = [ans[i][0] for i in d.id]; d["p4"] = [ans[i][1] for i in d.id]
        days = d.day.unique(); rng = np.random.default_rng(0)
        print(f"[{md.name}] 유효 {len(d)}/{len(g)} · {len(days)}일")
        for h, p in (("1h", "p1"), ("4h", "p4")):
            y = d[f"y_{h}"].to_numpy()
            for nm, s in ((md.name, d[p].to_numpy(float)), ("되돌림(−직전 1h)", -d.r1h.to_numpy(float)),
                          ("되돌림(−직전 4h)", -d.r4h.to_numpy(float)), ("모멘텀(직전 1h)", d.r1h.to_numpy(float))):
                a = F_auc(y, s)
                bs = []
                for _ in range(1000):
                    pick = rng.choice(days, len(days)); m = np.concatenate([np.flatnonzero(d.day.to_numpy() == x) for x in pick])
                    bs.append(F_auc(y[m], s[m]))
                print(f"  {h} {nm:20s} AUC {a:.3f} [{np.nanpercentile(bs, 2.5):.3f}, {np.nanpercentile(bs, 97.5):.3f}] IC {spearmanr(s, y).correlation:+.3f}")
            s = d[p].to_numpy(float)
            print(f"  {h} 답: 평균 {s.mean():.1f} 표준편차 {s.std():.1f} · ρ(직전 1h 수익) {spearmanr(s, d.r1h).correlation:+.2f} · ρ(직전 4h 수익) {spearmanr(s, d.r4h).correlation:+.2f}")


def F_auc(y, s):
    from scipy.stats import rankdata
    pos = y > 0
    if pos.all() or (~pos).all():
        return np.nan
    rk = rankdata(s)
    return float((rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


if __name__ == "__main__":
    {"build": build, "eval": evaluate}[sys.argv[1]]()
