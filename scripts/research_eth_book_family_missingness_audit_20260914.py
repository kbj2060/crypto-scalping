"""**book 군 결측 감사** — 신호인가 결측 패턴인가 (2026-09-14, 사용자 *"book 결측 3가지 확인해줘"*).

펀딩 제거 후 3%/3% 방향 라벨에서 `book`(호가깊이 7열)이 IC 증분·순손익 증분 **둘 다 3창 양수**로
유일하게 사전등록 규칙을 통과했다. 그런데 학습창 커버리지가 **76%** 이고 이 저장소는 2026-09-08 에
bookDepth 데이터 결함을 기록했다. 세 가지를 본다:

① **시기 편중** — 결측이 특정 구간에 몰렸나. 몰렸다면 「덮인 구간이 좋은 구간」일 뿐이다.
② **결측 지시자만으로 재현되나** — `build_state` 가 결측을 `nan_to_num(0)` = **정규화 중앙값**으로
   채우므로 결측 행은 「정확히 0」인 상수가 되고 트리가 그걸로 분기할 수 있다. 지시자 1열만 넣어
   같은 증분이 나오면 신호가 아니라 **달력**이다.
   🔴지시자가 **학습창에서 분산 0** 이면 그 검사는 「효과 없음」이 아니라 **검사 불능**이다 -- assert 로 막는다.
③ **짝지은 씨드의 SE** — 증분이 씨드 잡음보다 큰가. 단일 씨드 비교는 이 저장소에서 두 번 뒤집혔다.
④ (보너스, ①②를 가르는 결정타) **덮인 행만으로 다시** — 결측 행을 아예 빼고 base 와 base+book 을
   같은 행에서 비교한다. 여기서도 남으면 달력이 아니라 호가 자체다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"
EXT = ROOT / "data/research/eth_rl_gym_direction_20260914/ext_features.parquet"
FAMS = ROOT / "data/research/eth_rl_gym_direction_20260914/ext_families.json"
EVAL = ("VAL", "OOS", "TEST")


def fit_eval(Xtr, ytr, Xev: dict, lab: dict, seeds) -> dict:
    """짝지은 씨드. 창마다 IC 와 **순손익**(= m + sign(pred)·y) 을 낸다."""
    ic = {w: [] for w in Xev}; net = {w: [] for w in Xev}
    for sd in seeds:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=sd)
        m.fit(Xtr, ytr)
        for w, X in Xev.items():
            pr = m.predict(X); y = lab[w]["y"]
            ic[w].append(float(spearmanr(pr, y).statistic))
            s = np.sign(pr); s[s == 0] = 1.0
            net[w].append(float((lab[w]["m"] + s * y).mean()))
    return {w: {"ic": float(np.mean(ic[w])), "ic_se": float(np.std(ic[w], ddof=1) / np.sqrt(len(seeds))),
                "net": float(np.mean(net[w])), "net_se": float(np.std(net[w], ddof=1) / np.sqrt(len(seeds)))}
            for w in Xev}


def show(name: str, res: dict, base: dict | None = None) -> None:
    for w in EVAL:
        r = res[w]
        if base is None:
            print(f"  {name:<22}{w:>5}  IC {r['ic']:>+7.4f}±{r['ic_se']:.4f}   "
                  f"순손익 {r['net']:>+8.2f}±{r['net_se']:.2f}bp")
        else:
            b = base[w]
            dic = r["ic"] - b["ic"]; dse = np.hypot(r["ic_se"], b["ic_se"])
            dn = r["net"] - b["net"]; dnse = np.hypot(r["net_se"], b["net_se"])
            flag = "⭐" if dn > 2 * dnse else ("" if dn > 0 else "✘")
            print(f"  {name:<22}{w:>5}  ΔIC {dic:>+7.4f}±{dse:.4f}   "
                  f"Δ순손익 {dn:>+8.2f}±{dnse:.2f}bp  ({dn/dnse if dnse > 0 else 0:>+5.1f}σ) {flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(OUT / "labels_3_3_nofund.npz"))
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    seeds = [int(x) for x in np.random.default_rng(20260914).integers(1, 1_000_000, size=a.seeds)]
    d, sm, win, S43, cols43, _ = P.prepare(G.DEFAULT_FAMILIES)
    ext = pd.read_parquet(EXT)
    book = json.load(open(FAMS))["book"]
    assert (ext.timestamp.to_numpy() == d.timestamp.to_numpy()).all(), "정렬 불일치"
    z = np.load(a.labels, allow_pickle=True)
    lab = {w: {"idx": z[f"{w}_idx"], "y": z[f"{w}_y"], "m": z[f"{w}_m"]} for w in z["names"]}
    # 🔴학습창 분산 0 인 열은 **배울 수 없다** -- 그런 열이 평가창에만 값이 생기면 그건 신호가
    # 아니라 **달력**이다. 2026-09-14 실측: `bk_obi_0p2` 는 TRAIN·VAL 0% · OOS 84% · TEST 100%.
    tl, th = win["TRAIN"]
    dead = [c for c in book
            if not np.isfinite(ext[c].to_numpy(float)[tl:th]).any()
            or float(np.nanstd(ext[c].to_numpy(float)[tl:th])) < 1e-12]
    live = [c for c in book if c not in dead]
    print(f"학습창 분산 0 이라 제외: {dead or '없음'}\n쓰는 열 {len(live)}: {live}\n")
    raw = ext[live].to_numpy(float)
    miss = ~np.isfinite(raw).all(1)                     # 봉 단위 「book 결측」(쓸 수 있는 열 기준)

    # ① 시기 편중 ------------------------------------------------------------
    print("① 결측의 시기 편중")
    print(f"  {'창':>6} {'앵커':>9} {'결측률':>7}   월별 커버리지")
    ts = pd.to_datetime(d.timestamp)
    for w in ("TRAIN",) + EVAL:
        i_ = lab[w]["idx"]; mr = float(miss[i_].mean())
        mon = pd.Series(~miss[i_], index=ts.iloc[i_].values).groupby(
            pd.Grouper(freq="MS")).mean()
        s = " ".join(f"{k:%y-%m}:{100*v:.0f}%" for k, v in mon.items())
        print(f"  {w:>6} {len(i_):>9,} {mr:>7.1%}   {s}")

    # 상태행렬 (base43 + book) — 정규화는 학습창에서만
    norm = G.fit_normalizer(ext, live, *win["TRAIN"])
    SB = G.build_state(ext, norm)
    ind = miss.astype(np.float32).reshape(-1, 1)        # ② 결측 지시자 1열
    tr = lab["TRAIN"]
    v = float(ind[tr["idx"]].std())
    assert v > 1e-6, f"결측 지시자가 학습창에서 분산 {v} -- 이 검사는 «불능»이지 «효과 없음»이 아니다"
    print(f"\n② 결측 지시자 학습창 분산 {v:.4f} (결측률 {float(ind[tr['idx']].mean()):.1%}) — 검사 가능")

    ev = {w: lab[w]["idx"] for w in EVAL}
    base = fit_eval(S43[tr["idx"]], tr["y"], {w: S43[i] for w, i in ev.items()}, lab, seeds)
    bookarm = fit_eval(np.hstack([S43, SB])[tr["idx"]], tr["y"],
                       {w: np.hstack([S43, SB])[i] for w, i in ev.items()}, lab, seeds)
    indarm = fit_eval(np.hstack([S43, ind])[tr["idx"]], tr["y"],
                      {w: np.hstack([S43, ind])[i] for w, i in ev.items()}, lab, seeds)
    deadarm = None
    if dead:                       # 달력 열의 기여를 따로 잰다
        nd = G.fit_normalizer(ext, dead, *win["TRAIN"]); SD = G.build_state(ext, nd)
        deadarm = fit_eval(np.hstack([S43, SD])[tr["idx"]], tr["y"],
                           {w: np.hstack([S43, SD])[i] for w, i in ev.items()}, lab, seeds)
    print("\n③ 짝지은 씨드 {}개 — 기준 43열".format(a.seeds))
    show("기준43", base)
    print("\n  기준 대비 증분")
    show(f"+book({len(live)}열)", bookarm, base)
    show("+결측지시자(1열)", indarm, base)
    if deadarm is not None:
        show(f"+학습창상수열({len(dead)})", deadarm, base)

    # ④ 덮인 행만 ------------------------------------------------------------
    print("\n④ 결측 행을 빼고 같은 행에서 비교")
    lab_c = {w: {k: vv[~miss[lab[w]['idx']]] for k, vv in lab[w].items()} for w in z["names"]}
    trc = lab_c["TRAIN"]; evc = {w: lab_c[w]["idx"] for w in EVAL}
    print(f"  남은 앵커 TRAIN {len(trc['idx']):,} · " +
          " · ".join(f"{w} {len(evc[w]):,}" for w in EVAL))
    basec = fit_eval(S43[trc["idx"]], trc["y"], {w: S43[i] for w, i in evc.items()}, lab_c, seeds)
    bookc = fit_eval(np.hstack([S43, SB])[trc["idx"]], trc["y"],
                     {w: np.hstack([S43, SB])[i] for w, i in evc.items()}, lab_c, seeds)
    show("기준43(덮인행)", basec)
    print("\n  기준 대비 증분")
    show("+book(덮인행)", bookc, basec)

    rep = {"seeds": seeds, "dead_cols": dead, "live_cols": live,
           "dead_arm": deadarm, "base": base, "book": bookarm, "indicator": indarm,
           "base_covered": basec, "book_covered": bookc,
           "miss_rate": {w: float(miss[lab[w]["idx"]].mean()) for w in z["names"]}}
    (OUT / "book_missingness_audit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"\n저장: {OUT/'book_missingness_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
