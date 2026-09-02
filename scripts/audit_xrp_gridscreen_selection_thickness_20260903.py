#!/usr/bin/env python3
"""XRP 그리드스크린 선정 셀의 **표본 두께 감사** — 기계적 argmax를 그대로 쓰지 않기 위해.

## 왜 필요한가

XRP 5신호 중 4개가 `touch_giveback_sustained`를 선택했다. 그런데 이 HIT_TYPE은
**BTC에서 이미 명시적으로 거부된 전례**가 있다:

> "the mechanical global argmax `touch_giveback_sustained` was explicitly flagged as too
>  [thin-sample] ... which was explicitly distrusted for having only 2-5 OOS hits"
>  (research_btc_orthogonal_combo_metalabel_tabpfn_20260901.py)

BTC는 결국 **표본이 두꺼운 family의 승자**(touch_mfe H=8/K=2.0)를 썼다.
giveback 계열은 조건이 둘(fast_mult AND giveback)이라 hit이 희소해지고, lift는 커 보이지만
그 lift가 몇 건 위에 서 있는지를 봐야 한다.

⚠️이건 자동화가 답을 주지 않는 지점이다. 이 스크립트는 **판단 재료를 표로 만들 뿐**이고,
최종 선택은 사람이 한다(그 판단과 근거를 리포트에 남긴다).

## 판정 기준 (BTC 전례 그대로)

  · OOS 측면별 hit 수가 **한 자릿수**면 그 셀은 신뢰하지 않는다
  · 같은 신호의 다른 family에 "lift는 조금 낮아도 hit 수가 10배"인 셀이 있으면 그쪽을 본다
  · 최종 선택은 `min(bottom, top)` 기준 OOS hit 수와 joint lift를 함께 보고 정한다
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903"
OUT = DIR / "selection_thickness_audit.json"
THIN_OOS_HITS = 10          # 측면별 OOS hit이 이 미만이면 "얇다"


def log(m): print(f"[thickness] {m}", flush=True)


def _cells(d: dict) -> list[dict]:
    for k in ("grid", "full_train_grid"):
        if isinstance(d.get(k), list):
            return d[k]
    return []


def _hits(c: dict, split: str = "") -> tuple:
    p = f"{split}_" if split else ""
    return (c.get(f"{p}n_hits_bottom"), c.get(f"{p}n_hits_top"))


def main() -> int:
    rep = {"thin_oos_hits_threshold": THIN_OOS_HITS, "signals": {}}
    for f in sorted(DIR.glob("*_gridscreen_report.json")):
        nm = f.name.replace("_gridscreen_report.json", "")
        d = json.loads(f.read_text())
        rec = d.get("recommended") or d.get("chosen") or d.get("global_selection") or {}
        if not rec and "chosen_hit_type" in d:
            rec = {"hit_type": d["chosen_hit_type"], "horizon": d["chosen_horizon"], "k": d["chosen_k"]}
        cells = _cells(d)
        log("")
        log(f"=== {nm} ===")
        log(f"  자동 선택: {rec.get('hit_type')} H={rec.get('horizon')} K={rec.get('k')}")

        # 자동 선택 셀을 grid에서 되찾아 hit 수를 본다
        def match(c):
            return (c.get("hit_type") == rec.get("hit_type")
                    and c.get("horizon") == rec.get("horizon")
                    and abs(float(c.get("k", -1)) - float(rec.get("k", -2))) < 1e-9)
        picked = next((c for c in cells if match(c)), None) or rec
        tb, tt = _hits(picked, "train") if picked.get("train_n_hits_bottom") is not None else _hits(picked)
        log(f"     TRAIN hits  bottom={tb} top={tt}")

        # 같은 신호의 family별 최선을 hit 수와 함께 나열
        fam = {}
        for c in cells:
            ht = c.get("hit_type")
            hb, ht2 = _hits(c, "train") if c.get("train_n_hits_bottom") is not None else _hits(c)
            if hb is None or ht2 is None:
                continue
            lift = c.get("joint_min_lift") or min(
                c.get("train_lift_bottom") or c.get("lift_bottom") or 0,
                c.get("train_lift_top") or c.get("lift_top") or 0)
            thin = min(hb, ht2)
            cur = fam.get(ht)
            if cur is None or (thin >= THIN_OOS_HITS and lift > cur["lift"]) or \
               (cur["min_hits"] < THIN_OOS_HITS and thin > cur["min_hits"]):
                fam[ht] = {"horizon": c.get("horizon"), "k": c.get("k"), "lift": round(float(lift), 4),
                           "hits_bottom": hb, "hits_top": ht2, "min_hits": thin}
        log(f"  {'HIT_TYPE':<26}{'H':>4}{'K':>7}{'joint lift':>12}{'hits(b/t)':>14}")
        for ht, v in sorted(fam.items(), key=lambda kv: -kv[1]["min_hits"]):
            mark = "⚠️얇음" if v["min_hits"] < THIN_OOS_HITS else ""
            log(f"  {ht:<26}{v['horizon']:>4}{v['k']:>7}{v['lift']:>12.4f}"
                f"{str(v['hits_bottom'])+'/'+str(v['hits_top']):>14}  {mark}")
        rep["signals"][nm] = {"auto_selected": {k: rec.get(k) for k in ("hit_type", "horizon", "k")},
                              "auto_selected_train_hits": {"bottom": tb, "top": tt},
                              "family_best": fam}
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("")
    log(f"report -> {OUT}")
    log("⚠️최종 선택은 사람이 한다 -- 이 표는 판단 재료다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
