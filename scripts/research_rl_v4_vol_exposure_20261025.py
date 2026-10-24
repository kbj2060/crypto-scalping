#!/usr/bin/env python3
"""RL v4 1단계 — 노출 크기 RL 의 교사(변동성 목표 롱 바구니)가 1배 보유를 위험 대비 이기는지. 사전 등록 docs/prereg/rl_v4_vol_exposure_20261025.md.

    python3 scripts/research_rl_v4_vol_exposure_20261025.py      # 4 규칙 × 굴림 3창 + 2022~25 연속 · 관문 (i)(ii)

실주문 없음. 산출 tmp/rl_v4_vol_exposure_20261025/stage1.json.
"""
import json, pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import research_trend_portfolio_lit_20261025 as TP   # noqa: E402

OUT = TP.ROOT / "tmp" / "rl_v4_vol_exposure_20261025"; COINS, N, OE = TP.COINS, TP.N, TP.OE
RULES = [(v, p) for v in ("rv20", "cc20") for p in (1, 2)]; L = 2.0; DEV = ("2022-01-01", "2025-12-31")


def weights(F, vk, p, trp):
    """코인 목표 명목 = min(L, (σ*/σ̂)^p)/N · σ* = 학습 구간 σ̂ 중앙 · 일 d 행은 d−1 까지(shift 1)."""
    W = {}
    for s in COINS:
        sg = TP.vol(F[s], vk); W[s] = np.minimum(L, (sg.loc[trp[0]:trp[1]].median() / sg) ** p) / N
    return pd.DataFrame(W).replace([np.inf, -np.inf], np.nan).fillna(0.0).shift(1).fillna(0.0)


def block_lo(v, q, b=20, n=4000, seed=7):
    rng = np.random.default_rng(seed); T = len(v); nb = T // b + 1
    ms = [np.concatenate([v[s:s + b] for s in rng.integers(0, T - b, nb)])[:T].mean() for _ in range(n)]
    return float(np.percentile(ms, q) * 1e4)


def main():
    F = {s: TP.daily_frame(s)[0] for s in COINS}; OUT.mkdir(parents=True, exist_ok=True)
    hold_W = pd.DataFrame(1.0 / N, index=F["ETH"].index, columns=COINS)
    rep, R, H = {"rules": {}}, {r: [] for r in RULES}, []
    for w, (trp, _, tep) in TP.WIN5.items():
        te = pd.date_range(*tep, freq="D"); h = TP.run(F, hold_W, te); H.append(h)
        for vk, p in RULES:
            Wr = weights(F, vk, p, trp); x = TP.run(F, Wr, te); R[(vk, p)].append(x)
            rep["rules"].setdefault(f"{vk}_p{p}", {})[w] = {"sharpe": TP.sharpe(x), "hold_sharpe": TP.sharpe(h), "mean_gross": float(Wr.loc[te].sum(1).mean())}
    h = pd.concat(H); dev = pd.date_range(*DEV, freq="D"); hd = TP.run(F, hold_W, dev)
    rep.update(hold_pooled_sharpe=TP.sharpe(h), hold_dev_sharpe=TP.sharpe(hd), hold_dev_mdd=OE.mdd(hd)); passed = []
    for vk, p in RULES:
        k = f"{vk}_p{p}"; x = pd.concat(R[(vk, p)]); d = (x * (h.std() / x.std()) - h).values   # 변동성 맞춘 일 수익 차
        xd = TP.run(F, weights(F, vk, p, DEV), dev)
        g = {"pooled_sharpe": TP.sharpe(x), "pooled_minus_hold": TP.sharpe(x) - rep["hold_pooled_sharpe"], "volmatched_diff_bp": float(d.mean() * 1e4),
             "ci_lo_bonf4_bp": block_lo(d, 1.25), "ci_lo95_bp": block_lo(d, 2.5), "dev_sharpe": TP.sharpe(xd), "dev_mdd": OE.mdd(xd)}
        g["pass"] = g["pooled_minus_hold"] > 0 and g["ci_lo_bonf4_bp"] > 0 and g["dev_sharpe"] > rep["hold_dev_sharpe"]
        rep["rules"][k]["pooled"] = g; passed.append(g["pass"]); print(k, json.dumps(g, ensure_ascii=False), flush=True)
    rep["stage1_pass"] = any(passed)
    print("보유 합침 %.2f · 보유 22~25 %.2f · 1단계 통과 %s" % (rep["hold_pooled_sharpe"], rep["hold_dev_sharpe"], rep["stage1_pass"]))
    (OUT / "stage1.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
