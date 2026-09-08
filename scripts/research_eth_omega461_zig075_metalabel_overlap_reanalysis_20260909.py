"""Stage 0 후속 — 겹침 표본 보정 재분석. 재학습 없음(저장된 점수 재사용).

왜 필요한가
-----------
Stage 0(`research_eth_omega461_zig075_quality_metalabel_tabpfn_20260909.py`)의 후보는 5분 간격인데
라벨 호라이즌은 7일이다(중앙 보유 656봉 ≈ 2.3일). 즉 인접 후보들의 라벨은 거의 같은 미래 구간을
공유한다 — n=12,029 은 **명목 표본이지 독립 표본이 아니다**. Stage 0 의 i.i.d. 부트스트랩 CI 는
그래서 실제보다 훨씬 좁고, 그대로 읽으면 유의성을 크게 과장한다(저장소의 반복 함정: 겹침표본).

이 스크립트는 같은 점수·같은 라벨에 대해 **두 가지 정직한 관점**을 추가한다.

  (A) 블록 부트스트랩 — 라벨 호라이즌(7일)과 같거나 긴 시간 블록 단위로 재표집한다.
      W2 는 59일이라 7일 블록이 ~8개뿐 — 이 축의 진짜 검정력 상한이 그대로 드러난다.

  (B) 비겹침 탐욕 선택 — 시간순으로 훑되 직전 선택 트레이드가 해소되기 전에 시작하는 후보는
      건너뛴다. 실제 단일슬롯 라이브(greedy_replay)가 할 수 있는 것과 같은 구조이고, 남는
      트레이드들은 **진짜로 서로 겹치지 않는다**. 여기서의 정밀도/측면매칭 초과가 Stage 0 의
      명목 수치보다 훨씬 보수적이면, Stage 0 숫자는 겹침 인공물이었다는 뜻이다.

두 관점 모두 측면 매칭 귀무(선택 집합의 롱/숏 구성 보존)를 유지한다 — 두 창 다 하락장이라
pooled 기저율은 잘못된 귀무다.

준수: 신규 학습 없음, 신규 하이퍼파라미터/임계값 탐색 없음, 라이브 파일 미변경.
      Stage 0 이 저장한 `scores_*.npz` + `candidates_*.csv` 만 입력으로 쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/eth_zig075_quality_metalabel_tabpfn_20260909"

BLOCK_BARS = 2016      # 7일 = 라벨 호라이즌
N_BOOT = 2000
COVERAGE_KEY = "incumbent"


def side_matched_null(sel: np.ndarray, y: np.ndarray, side: np.ndarray) -> float:
    k = int(sel.sum())
    if not k:
        return float("nan")
    is_s = side < 0
    n_s = int((sel & is_s).sum())
    n_l = k - n_s
    return float((n_s * y[is_s].mean() + n_l * y[~is_s].mean()) / k)


def block_bootstrap_lift(sel: np.ndarray, y: np.ndarray, side: np.ndarray,
                         block_id: np.ndarray, rng: np.random.Generator) -> dict:
    """블록 단위 재표집으로 (정밀도 - 측면매칭귀무) 의 CI 를 낸다."""
    blocks = np.unique(block_id)
    obs = float(y[sel].mean()) - side_matched_null(sel, y, side)
    draws = []
    for _ in range(N_BOOT):
        take = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([np.flatnonzero(block_id == b) for b in take])
        s, yy, sd = sel[idx], y[idx], side[idx]
        if s.sum() < 5 or len(np.unique(sd[s])) == 0:
            continue
        nul = side_matched_null(s, yy, sd)
        if np.isfinite(nul):
            draws.append(float(yy[s].mean()) - nul)
    draws = np.array(draws)
    lo, hi = np.quantile(draws, [0.025, 0.975]) if len(draws) > 50 else (np.nan, np.nan)
    return {"n_blocks": int(len(blocks)), "lift_pp": obs * 100.0,
            "boot_ci_pp": [float(lo * 100.0), float(hi * 100.0)],
            "excludes_zero": bool(lo > 0.0) if np.isfinite(lo) else False,
            "n_draws": int(len(draws))}


def greedy_nonoverlap(score: np.ndarray, row: np.ndarray, hold: np.ndarray,
                      k_budget: int) -> np.ndarray:
    """점수 내림차순으로 훑되, 이미 선택된 트레이드의 보유구간과 겹치면 건너뛴다.

    실제 단일슬롯 라이브가 할 수 있는 선택 구조이고, 남는 트레이드는 서로 겹치지 않는다.
    """
    order = np.argsort(-score, kind="mergesort")
    taken, spans = [], []
    for i in order:
        s, e = int(row[i]), int(row[i]) + int(hold[i])
        if any(not (e < a or s > b) for a, b in spans):
            continue
        taken.append(i)
        spans.append((s, e))
        if len(taken) >= k_budget:
            break
    sel = np.zeros(len(score), dtype=bool)
    sel[taken] = True
    return sel


def main() -> int:
    rng = np.random.default_rng(20260909)
    report = {}
    cands = {t: pd.read_csv(OUT / f"candidates_{t}.csv") for t in ("W1", "W2")}

    for npz_path in sorted(OUT.glob("scores_*.npz")):
        tag = npz_path.stem.replace("scores_", "")
        direction, fs = tag.split("_", 2)[0] + "->" + tag.split("_")[2], tag.split("_", 3)[3]
        te_tag = tag.split("_")[2]
        z = np.load(npz_path)
        scores, y, side, inc = z["scores"], z["y"], z["side"], z["incumbent"]
        cand = cands[te_tag]
        row, hold = cand["row"].to_numpy(), cand["hold_bars"].to_numpy()
        block_id = row // BLOCK_BARS

        k = int(round((inc >= 0.75).mean() * len(y)))
        entry = {"test_window": te_tag, "featureset": fs, "n_nominal": int(len(y)),
                 "k_at_incumbent_coverage": k,
                 "block_bars": BLOCK_BARS, "n_blocks": int(len(np.unique(block_id)))}

        # --- incumbent ---
        inc_sel = inc >= 0.75
        entry["incumbent_block"] = block_bootstrap_lift(inc_sel, y, side, block_id, rng)
        inc_no = greedy_nonoverlap(inc, row, hold, k)
        entry["incumbent_nonoverlap"] = {
            "n_trades": int(inc_no.sum()), "precision": float(y[inc_no].mean()),
            "side_matched_null": side_matched_null(inc_no, y, side),
            "short_share": float((side[inc_no] < 0).mean()),
            "lift_pp": (float(y[inc_no].mean()) - side_matched_null(inc_no, y, side)) * 100.0}

        # --- TabPFN 시드별 ---
        blk, non = [], []
        for si in range(scores.shape[0]):
            p = scores[si]
            sel = np.zeros(len(y), dtype=bool)
            sel[np.argsort(-p, kind="mergesort")[:k]] = True
            blk.append(block_bootstrap_lift(sel, y, side, block_id, rng))
            no = greedy_nonoverlap(p, row, hold, k)
            nul = side_matched_null(no, y, side)
            non.append({"n_trades": int(no.sum()), "precision": float(y[no].mean()),
                        "side_matched_null": nul, "short_share": float((side[no] < 0).mean()),
                        "lift_pp": (float(y[no].mean()) - nul) * 100.0})
        entry["tabpfn_block"] = {"seeds": blk,
                                 "lift_pp_mean": float(np.mean([b["lift_pp"] for b in blk])),
                                 "ci_excludes_zero_seeds": int(sum(b["excludes_zero"] for b in blk))}
        entry["tabpfn_nonoverlap"] = {"seeds": non,
                                      "lift_pp_mean": float(np.mean([n["lift_pp"] for n in non])),
                                      "precision_mean": float(np.mean([n["precision"] for n in non])),
                                      "n_trades_mean": float(np.mean([n["n_trades"] for n in non]))}
        report[tag] = entry

        print(f"\n=== {direction} / {fs}  (평가 {te_tag}, 명목 n={len(y)}, "
              f"블록 {entry['n_blocks']}개×{BLOCK_BARS}봉) ===", flush=True)
        ib, ino = entry["incumbent_block"], entry["incumbent_nonoverlap"]
        print(f"  incumbent  블록CI: 초과={ib['lift_pp']:+.2f}pp "
              f"CI[{ib['boot_ci_pp'][0]:+.2f},{ib['boot_ci_pp'][1]:+.2f}] "
              f"{'✅' if ib['excludes_zero'] else '❌'}", flush=True)
        print(f"  incumbent  비겹침: {ino['n_trades']}건 정밀도={ino['precision']*100:.1f}% "
              f"귀무={ino['side_matched_null']*100:.1f}% 초과={ino['lift_pp']:+.1f}pp "
              f"숏비중={ino['short_share']*100:.0f}%", flush=True)
        tb, tn = entry["tabpfn_block"], entry["tabpfn_nonoverlap"]
        print(f"  TabPFN     블록CI: 초과평균={tb['lift_pp_mean']:+.2f}pp  "
              f"CI가 0 배제한 시드={tb['ci_excludes_zero_seeds']}/5", flush=True)
        print(f"  TabPFN     비겹침: {tn['n_trades_mean']:.0f}건 정밀도={tn['precision_mean']*100:.1f}% "
              f"초과평균={tn['lift_pp_mean']:+.1f}pp", flush=True)

    (OUT / "overlap_reanalysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    print(f"\n산출물: {OUT}/overlap_reanalysis.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
