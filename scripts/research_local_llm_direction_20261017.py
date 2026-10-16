#!/usr/bin/env python3
"""**서버 로컬 LLM(qwen3:8b) 방향 예측** — Haiku 방향 시험과 같은 400 시점·같은 입력(피쳐 99칸), 시점마다 1 질의 (2026-10-17).

왜: 사용자 «언어 시험보다는 방향 예측에 중점». 해석 읽기 시험에서 형식을 지킨 로컬 후보는 qwen3:8b 하나(EXAONE 1.2B·qwen3:4b 탈락).
입력·정답·기준선 = scripts/research_llm_haiku_direction_20261016.py(gold.csv · 2025-01~2026-08 · ≥8h 간격 · HGB/되돌림/모멘텀) 그대로.
실행 = 서버 시험 전용 Ollama(11436, bwrap) · 뉴스 판정 SIGSTOP·jevk5 내림 동안만(run_all.sh 의 trap 복구) · format=json · think=false.
  python scripts/research_local_llm_direction_20261017.py prompts   # tmp/local_llm_dir_20261017/prompts
  python scripts/research_local_llm_direction_20261017.py eval      # tmp/local_llm_dir_20261017/out/<model> 채점
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "tmp/local_llm_dir_20261017"
SRC = sorted((ROOT / "tmp/haiku_direction_20261016_inputs").glob("batch_*.txt"))
HAIKU = ROOT / "tmp/haiku_direction_20261016"


def prompts() -> None:
    pd_ = OUT / "prompts"; pd_.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in SRC:
        head, body = f.read_text().split("Snapshots:\n", 1)
        head = re.sub(r"You are given \d+ independent snapshots", "You are given one snapshot", head)
        head = head.replace("Snapshots are from different, unrelated days (at least 8 hours apart) and are shuffled; do not compare them to each other or try to guess dates.",
                            "Do not try to guess the date.").replace("For EACH snapshot", "For the snapshot")
        head = re.sub(r"Output ONLY one JSON array.*", 'Output ONLY one JSON object: {"id": "<the snapshot id>", "p_up_1h": int, "p_up_4h": int}', head)
        for line in body.splitlines():
            if line.strip():
                sid = json.loads(line)["id"]
                (pd_ / f"dir_{sid}.txt").write_text(head + "Snapshot:\n" + line); n += 1
    print(f"{n} 질의 → {pd_}")


def auc(y, s):
    from scipy.stats import rankdata
    pos = y > 0; rk = rankdata(s)
    return float((rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def evaluate() -> None:
    from scipy.stats import spearmanr
    gold = pd.read_csv(HAIKU / "scored.csv")                  # Haiku 시험이 채점하며 남긴 표(정답·Haiku 답·HGB·직전 수익)
    for md in sorted((OUT / "out").iterdir()):
        ans = {}
        for f in md.glob("*.json"):
            r = json.loads(f.read_text())
            try:
                c = r["message"]["content"]; j = json.loads(c[c.index("{"): c.rindex("}") + 1])
                ans[f.stem[4:]] = (float(j["p_up_1h"]), float(j["p_up_4h"]))
            except Exception:  # noqa: BLE001
                pass
        g = gold[gold.id.isin(ans)].copy()
        g["loc_1h"] = [ans[i][0] for i in g.id]; g["loc_4h"] = [ans[i][1] for i in g.id]
        print(f"\n[{md.name}] 유효 답 {len(g)}/{len(gold)}")
        rng = np.random.default_rng(0); B = [rng.integers(0, len(g), len(g)) for _ in range(2000)]
        for h in ("1h", "4h"):
            y = g[f"y_{h}"].to_numpy()
            print(f"  ── {h}            AUC  [95% CI]           IC")
            for nm, s in ((f"{md.name}", g[f"loc_{h}"].to_numpy(float)), ("Haiku(같은 시점)", g[f"llm_{h}"].to_numpy(float)),
                          ("되돌림(−직전 1h)", -g.P_r1h.to_numpy()), ("HGB 2024 학습", g[f"hgb24_{h}"].to_numpy())):
                a = auc(y, s); ci = np.percentile([auc(y[b], s[b]) for b in B], [2.5, 97.5])
                print(f"   {nm:24s} {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  {spearmanr(s, y).correlation:+.3f}")
            s = g[f"loc_{h}"].to_numpy(float)
            X = np.column_stack([np.ones(len(g))] + [pd.Series(v).rank().to_numpy() for v in (g.P_r1h, g.P_r4h, g.P_r1d)])
            res = pd.Series(s).rank().to_numpy() - X @ np.linalg.lstsq(X, pd.Series(s).rank().to_numpy(), rcond=None)[0]
            print(f"   답 성질: 평균 {s.mean():.1f} · 표준편차 {s.std():.1f} · ρ(직전 1h) {spearmanr(s, g.P_r1h).correlation:+.2f} · "
                  f"ρ(직전 4h) {spearmanr(s, g.P_r4h).correlation:+.2f} · ρ(Haiku) {spearmanr(s, g[f'llm_{h}']).correlation:+.2f} · "
                  f"직전 수익 뺀 잔차 IC {spearmanr(res, y).correlation:+.3f}")


if __name__ == "__main__":
    {"prompts": prompts, "eval": evaluate}[sys.argv[1]]()
