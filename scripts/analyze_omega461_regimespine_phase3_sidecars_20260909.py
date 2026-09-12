"""Phase 3-a 집계 — 부모별 리스크 사이드카의 선택된 매핑 성과.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
Phase 2 는 BASE_TEMPLATE 고정 사이징(notional 0.45/leverage 2.0)이었다. 여기서는 각 부모에
적합된 사이드카가 고른 매핑의 VAL/OOS 성과를 본다 -- 즉 **봉별 margin/leverage 가 들어간** 수치다.

⚠️ 이건 아직 컴포넌트 개별 성과다. 실제 결정 경로는 h48qual > zig075 우선순위 라우터가
결합한 단일계좌이며 그건 Phase 3-b(greedy_replay)에서 낸다.
⚠️ 사이드카 선택 제약이 validation_mdd >= -25% 로 관대하다 -- 통과가 곧 좋은 리스크 프로필은
아니므로 실제 MDD 값을 함께 본다.
"""
from __future__ import annotations
import glob, json, os, re, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAT = str(ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_regimespine_*_20260909")
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]


def main() -> int:
    rows = {}
    for d in sorted(glob.glob(PAT)):
        p = os.path.join(d, "report.json")
        if not os.path.exists(p):
            continue
        m = re.search(r"regimespine_(balnobb|wide24)_(zig075|h48qual)_s(\d+)_", d)
        if not m:
            continue
        sel = (json.load(open(p)).get("selected") or {})
        v, o = sel.get("validation") or {}, sel.get("oos") or {}
        rows[(m.group(2), m.group(1), int(m.group(3)))] = (v, o)

    print(f"{'comp':9s}{'arm':9s}{'seed':>11s} | {'VALpnl':>8s}{'mdd':>8s}{'tr':>4s}{'notl':>6s}{'lev':>6s} | "
          f"{'OOSpnl':>8s}{'mdd':>8s}{'tr':>4s}", flush=True)
    for k in sorted(rows):
        v, o = rows[k]
        print(f"{k[0]:9s}{k[1]:9s}{k[2]:>11d} | {v.get('pnl',0):+8.2f}{v.get('mdd',0):+8.2f}"
              f"{v.get('trades',0):>4d}{v.get('avg_notional',0):6.2f}{v.get('avg_leverage',0):6.2f} | "
              f"{o.get('pnl',0):+8.2f}{o.get('mdd',0):+8.2f}{o.get('trades',0):>4d}", flush=True)
    print(f"\n완료 {len(rows)}/20", flush=True)

    for comp in ("zig075", "h48qual"):
        dv, dm, do, n = [], [], [], 0
        for s in SEEDS:
            a, b = rows.get((comp, "balnobb", s)), rows.get((comp, "wide24", s))
            if not a or not b:
                continue
            n += 1
            dv.append(a[0]["pnl"] - b[0]["pnl"])
            dm.append(a[0]["mdd"] - b[0]["mdd"])
            do.append(a[1]["pnl"] - b[1]["pnl"])
        if not n:
            continue
        both = sum(1 for p, q in zip(dv, dm) if p > 0 and q > 0)
        print(f"\n[{comp}] 짝 {n}개 (사이드카 사이징 기준)", flush=True)
        print(f"  VAL PnL 개선 {sum(1 for x in dv if x>0)}/{n} · MDD 개선 {sum(1 for x in dm if x>0)}/{n}"
              f" · 둘 다 {both}/{n}", flush=True)
        print(f"  Δ VAL PnL 중앙 {statistics.median(dv):+.2f}pp  ({min(dv):+.2f}~{max(dv):+.2f})", flush=True)
        print(f"  Δ VAL MDD 중앙 {statistics.median(dm):+.2f}pp  ({min(dm):+.2f}~{max(dm):+.2f})", flush=True)
        print(f"  [참고] Δ OOS PnL 중앙 {statistics.median(do):+.2f}pp — 선택 근거로 쓰지 않음", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
