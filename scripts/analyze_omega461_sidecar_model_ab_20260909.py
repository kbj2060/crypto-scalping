"""리스크 사이드카 회귀기 A/B 집계 — HGB vs TabPFN(mean/q25/...).

하위 프로젝트: `omega461_regimegbm_rebuild_20260909` 의 후속(TabPFN 축 2번 항목).
같은 부모·같은 라벨·같은 매핑 격자·같은 선택 제약에서 **회귀기만** 다른 짝을 비교한다.
⚠️ TabPFN 은 sample_weight 를 지원하지 않아 행 복제로 근사했다 — 순수 회귀기 교체가 아니다.
"""
from __future__ import annotations
import glob, json, os, re, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tmp/causal_regen_20260516"
HGB = "omega4_2_trade_risk_sidecar_20260622_regimespine_{arm}_{comp}_s{seed}_20260909"
TPF = "omega4_2_trade_risk_sidecar_20260622_tabpfn{v}_{out}_{arm}_{comp}_s{seed}_20260909"


def read(d: str):
    p = BASE / d / "report.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    s = r.get("selected") or {}
    return {"val": s.get("validation") or {}, "oos": s.get("oos") or {},
            "kind": (r.get("risk_model") or {}).get("model_kind"),
            "nest": (r.get("risk_model") or {}).get("tabpfn_n_estimators"),
            "out": (r.get("risk_model") or {}).get("tabpfn_output_type")}


def main() -> int:
    variant = sys.argv[1] if len(sys.argv) > 1 else "A"
    outtype = sys.argv[2] if len(sys.argv) > 2 else "mean"
    seeds = [615372041, 208844917, 933105268, 471926350, 862017594]
    rows, dv, dm, do = [], [], [], []
    print(f"{'comp':9s}{'arm':9s}{'seed':>11s} | {'HGB VAL':>17s} | {'TabPFN VAL':>17s} | "
          f"{'ΔPnL':>8s}{'ΔMDD':>8s} | {'ΔOOS':>8s}", flush=True)
    for comp in ("zig075", "h48qual"):
        for arm in ("balnobb", "wide24"):
            for s in seeds:
                a = read(HGB.format(arm=arm, comp=comp, seed=s))
                b = read(TPF.format(v=variant, out=outtype, arm=arm, comp=comp, seed=s))
                if not a or not b:
                    continue
                p = b["val"]["pnl"] - a["val"]["pnl"]
                m = b["val"]["mdd"] - a["val"]["mdd"]
                o = b["oos"]["pnl"] - a["oos"]["pnl"]
                dv.append(p); dm.append(m); do.append(o)
                rows.append((comp, arm, s))
                print(f"{comp:9s}{arm:9s}{s:>11d} | {a['val']['pnl']:+8.2f}/{a['val']['mdd']:+7.2f} | "
                      f"{b['val']['pnl']:+8.2f}/{b['val']['mdd']:+7.2f} | {p:+8.2f}{m:+8.2f} | {o:+8.2f}",
                      flush=True)
    n = len(dv)
    if not n:
        print("\n짝이 없다 — TabPFN 산출물이 아직 없거나 접두사가 다르다", flush=True)
        return 0
    both = sum(1 for x, y in zip(dv, dm) if x > 0 and y > 0)
    print(f"\n짝 {n}개 (TabPFN variant={variant} output={outtype})", flush=True)
    print(f"  VAL PnL 개선 {sum(1 for x in dv if x>0)}/{n} · MDD 개선 {sum(1 for x in dm if x>0)}/{n}"
          f" · **둘 다 {both}/{n}**", flush=True)
    print(f"  Δ VAL PnL 중앙 {statistics.median(dv):+.2f}pp ({min(dv):+.2f}~{max(dv):+.2f})", flush=True)
    print(f"  Δ VAL MDD 중앙 {statistics.median(dm):+.2f}pp ({min(dm):+.2f}~{max(dm):+.2f})", flush=True)
    print(f"  [참고] Δ OOS PnL 중앙 {statistics.median(do):+.2f}pp — 선택 근거로 쓰지 않음", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
