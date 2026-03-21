#!/usr/bin/env python3
"""rl_training_data_full.csv 레짐 분포 출력"""
import pandas as pd
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_root, "data", "rl_training_data_full.csv")
regime_cols = ["regime_chop", "regime_whipsaw", "regime_bull", "regime_bear", "regime_normal"]

df = pd.read_csv(csv_path, usecols=regime_cols)
total = len(df)

rows = []
for col in regime_cols:
    n = (df[col] == 1.0).sum()
    pct = 100 * n / total
    rows.append((col, n, pct))
rows.sort(key=lambda x: -x[1])

out = []
out.append("")
out.append("  ┌────────────────┬─────────┬───────┐")
out.append("  │      레짐      │  행수   │ 비율  │")
out.append("  ├────────────────┼─────────┼───────┤")
for name, n, pct in rows:
    out.append(f"  │ {name:<14} │ {n:>7,} │ {pct:>5.1f}% │")
out.append("  ├────────────────┼─────────┼───────┤")
out.append(f"  │ 총 행 수       │ {total:>7,} │ 100.0% │")
out.append("  └────────────────┴─────────┴───────┘")
out.append("")
text = "\n".join(out)
print(text)
# 결과 저장 (확인용)
result_path = os.path.join(_root, "data", "ensemble", "regime_dist_result.txt")
with open(result_path, "w", encoding="utf-8") as f:
    f.write(text)
