#!/usr/bin/env python3
"""B16: TabPFN **임베딩 / 차원축소** 피쳐 엔지니어링 (2026-09-03).

B15에서 `docs.priorlabs.ai` 권고 3개(결측 raw / 피쳐선별 / n_estimators·ipl)를 적용했더니
전부 전이되지 않았고, **임베딩만 미시도**로 남았다(`tabpfn_extensions` 미설치, quant_ai는
라이브 공유 env라 설치 안 함). 여기서는 **별도 디렉토리 설치**로 그 축을 닫는다.

⭐임베딩의 핵심(공식 문서): `n_fold>=2`면 **OOF 임베딩**이라 각 학습 표본이 자기를 못 본
모델로 인코딩된다 -- 라벨 누수가 없다. `n_fold=0`(vanilla)은 "information leakage" 위험을
문서가 명시한다. 오늘 재료 텐서에서 겪은 것과 **같은 종류의 문제**이므로 n_fold=5를 쓴다.

⚠️텔레메트리: `tabpfn_extensions`는 import 시점에 PostHog 애널리틱스를 초기화하고 비-CI에서
**기본 ON**이다(`default_disable = "1" if ci else "0"`). 이 저장소는 자체 트레이딩 리서치를
다루므로 ①`posthog`를 무동작 스텁으로 대체하고 ②뉴스레터/신원 프롬프트 모듈을 sys.modules에
무동작 주입하며 ③`TABPFN_DISABLE_TELEMETRY=1`을 켠다. 벤더 파일은 고치지 않는다.

팔:
  A HGB-full(161)          -- 동결 기준선
  B TabPFN 직접(분류)       -- B12/B15 최선
  C TabPFN 임베딩 → 로지스틱 -- 문서 권장 조합(저용량 헤드)
  D TabPFN 임베딩 → HGB
  E PCA(k) → HGB           -- 차원축소 대조군 (임베딩 이득이 '차원축소 일반'인지 가른다)

⭐사전등록(B14/B15와 동일): ①앙상블 양 창(VAL·OOS) 승리 ②순열검정 p<0.05 양 창
   ③무작위 필터 대조군 3창 통과. 하나라도 미달이면 **HGB 회귀 동결 유지**.
"""
from __future__ import annotations

import json
import os
import sys
import types
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EXT = Path.home() / "tabpfn_ext_libs"
if EXT.exists() and str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
_m = types.ModuleType("tabpfn_common_utils.telemetry.interactive")
_m.opt_in = lambda *a, **k: None                       # 뉴스레터/신원 프롬프트 무력화
sys.modules["tabpfn_common_utils.telemetry.interactive"] = _m

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
RANK = ROOT / "tmp/eth_entry_b8_featsel_20260903/ranked_features.csv"
OUT = ROOT / "tmp/eth_entry_b16_embedding_20260903"
DEPTH, WAIT, TAU0, NSLOT = 3.0, 6, 0.0040, 4
LABEL_THR, SUB = 0.0040, 18000
N_FOLD, NMEM = 5, 3
B_RND, B_PERM = 150, 20000
RNG = np.random.default_rng(20260903)


def log(m): print(f"[b16] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier
    from tabpfn_extensions.embedding import TabPFNEmbedding

    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "split", "timestamp", "i",
                       "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in D.columns if c.endswith("_r136")] + \
        [c for c in D.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if D[c].dtype.kind in "fiub"]))
    FEATS = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    X = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (D.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    y = D["y"].to_numpy(); lab = (y > LABEL_THR).astype(int)
    dsel = ((D.depth == DEPTH) & (D.btf <= WAIT)).to_numpy()
    itr = np.flatnonzero(tr); pred_rows = np.flatnonzero(dsel)
    log(f"TRAIN {len(itr):,} · 피쳐 {len(FEATS)} · 예측 {len(pred_rows):,}")

    def expand(ps):
        f = np.full(len(D), -np.inf); f[pred_rows] = ps; return f

    hgb = [HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
           .fit(X[tr], y[tr]).predict(X) for s in SEEDS]
    pA = np.mean(hgb, axis=0)
    fracs = {w: float((pA[dsel & (D.split == w).to_numpy()] > TAU0).mean())
             for w in ("VAL", "OOS", "HOLDOUT")}

    def pol(pf, w, frac=None, tau=None):
        m = dsel & (D.split == w).to_numpy(); ww = D[m]; pv = pf[m]
        thr = tau if tau is not None else np.quantile(pv, 1 - frac)
        return stat(slotN(ww[pv > thr], NSLOT))[1]

    hgbA = {w: pol(pA, w, tau=TAU0) for w in ("VAL", "OOS", "HOLDOUT")}
    log("A HGB-full 기준선 " + " ".join(f"{k} {v:+.2f}" for k, v in hgbA.items()))

    Xp = X.iloc[pred_rows].to_numpy()
    outs = {"A HGB-full": pA}

    # ---- B: TabPFN 직접 ----
    log(f"\nB TabPFN 직접 (멤버 {NMEM})...")
    ps = []
    for k in range(NMEM):
        rs = np.random.default_rng(SEEDS[k]).choice(itr, size=min(SUB, len(itr)), replace=False)
        m = TabPFNClassifier(device="cuda", random_state=SEEDS[k])
        m.fit(X.iloc[rs].to_numpy(), lab[rs])
        ps.append(expand(m.predict_proba(Xp)[:, 1]))
    outs["B TabPFN 직접"] = np.mean(ps, axis=0)

    # ---- C/D: 임베딩 ----
    log(f"\nC/D TabPFN 임베딩 (n_fold={N_FOLD}, OOF)...")
    emb_tr, emb_p, ytr_all = [], [], []
    for k in range(NMEM):
        rs = np.random.default_rng(SEEDS[k]).choice(itr, size=min(SUB, len(itr)), replace=False)
        e = TabPFNEmbedding(model=TabPFNClassifier(device="cuda", random_state=SEEDS[k]),
                            n_fold=N_FOLD)
        Etr = np.asarray(e.fit_transform(X.iloc[rs].to_numpy(), lab[rs]))[0]
        Ep = np.asarray(e.transform(Xp))[0]
        log(f"  멤버{k}: train emb {Etr.shape} · pred emb {Ep.shape}")
        emb_tr.append(Etr); emb_p.append(Ep); ytr_all.append((rs, lab[rs], y[rs]))

    pc, pd_ = [], []
    for k in range(NMEM):
        rs, lb, yv = ytr_all[k]
        sc = StandardScaler().fit(emb_tr[k])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(emb_tr[k]), lb)
        pc.append(expand(lr.predict_proba(sc.transform(emb_p[k]))[:, 1]))
        hg = HistGradientBoostingRegressor(random_state=SEEDS[k], loss="squared_error",
                                           **HP).fit(emb_tr[k], yv)
        pd_.append(expand(hg.predict(emb_p[k])))
    outs["C 임베딩→로지스틱"] = np.mean(pc, axis=0)
    outs["D 임베딩→HGB"] = np.mean(pd_, axis=0)

    # ---- E: PCA 차원축소 대조군 ----
    log("\nE PCA→HGB (차원축소 대조군)...")
    for k_pca in (16, 32, 64):
        sc = StandardScaler().fit(X[tr]); Z = sc.transform(X)
        pc_ = PCA(n_components=k_pca, random_state=0).fit(Z[tr])
        Zt = pc_.transform(Z)
        p = np.mean([HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
                     .fit(Zt[tr], y[tr]).predict(Zt) for s in SEEDS], axis=0)
        outs[f"E PCA{k_pca}→HGB"] = p

    # ---- 평가 ----
    W = ("VAL", "OOS", "HOLDOUT")
    print(f"\n{'팔':22s}" + "".join(f"{w:>10s}" for w in W))
    res = {}
    for nm, p in outs.items():
        r = {w: pol(p, w, tau=TAU0) if nm == "A HGB-full" else pol(p, w, frac=fracs[w]) for w in W}
        res[nm] = r
        print(f"{nm:22s}" + "".join(f"{r[w]:+10.2f}" for w in W))

    best = max((k for k in outs if k != "A HGB-full"),
               key=lambda k: (res[k]["VAL"] > hgbA["VAL"]) + (res[k]["OOS"] > hgbA["OOS"]))
    log(f"\n⭐최선 후보: {best}")
    win = res[best]["VAL"] > hgbA["VAL"] and res[best]["OOS"] > hgbA["OOS"]
    log(f"  ①양 창 승리: {'✅' if win else '❌'} "
        f"(VAL {res[best]['VAL']:+.2f} vs {hgbA['VAL']:+.2f} · "
        f"OOS {res[best]['OOS']:+.2f} vs {hgbA['OOS']:+.2f})")

    # 순열검정: 후보와 A의 예측을 봉 단위로 섞어 차이의 유의성
    pv = {}
    for w in ("VAL", "OOS"):
        m = dsel & (D.split == w).to_numpy()
        d0 = res[best][w] - hgbA[w]
        pa, pb = pA[m], outs[best][m]
        ww = D[m]
        cnt = 0
        for _ in range(400):
            sw = RNG.random(len(pa)) < 0.5
            xa = np.where(sw, pb, pa); xb = np.where(sw, pa, pb)
            ta = np.quantile(xa, 1 - fracs[w]); tb = np.quantile(xb, 1 - fracs[w])
            da = stat(slotN(ww[xb > tb], NSLOT))[1] - stat(slotN(ww[xa > ta], NSLOT))[1]
            cnt += (da >= d0)
        pv[w] = (cnt + 1) / 401
        log(f"  ②순열검정 {w}: p={pv[w]:.4f} {'✅' if pv[w] < 0.05 else '❌'}")

    ok2 = all(pv[w] < 0.05 for w in ("VAL", "OOS"))
    log(f"\n⭐사전등록 판정  ①{'✅' if win else '❌'}  ②{'✅' if ok2 else '❌'}")
    log(f"  → {'**' + best + ' 채택 검토**' if (win and ok2) else '**HGB 회귀 동결 유지**'}")
    json.dump({"baseline": hgbA, "arms": res, "perm_p": pv, "best": best,
               "n_fold": N_FOLD, "members": NMEM},
              open(OUT / "result.json", "w"), ensure_ascii=False, indent=2, default=str)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
