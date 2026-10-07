"""핑퐁(지지·저항 오가며 페이드 + 물타기) 전략을 돕는 «남은 레버» 전수 검정 (2026-10-07).

사용자: «내 지지와 저항을 왔다갔다하는 핑퐁 전략에 도움이 되는 모든 수를 찾아서 연구».
이미 닫힌 축(재실행 안 함): 꼭지점·진입 선별 AUC .50(pingpong_extract) · TP/SL 격자 0 · S/R 반등 0 · 물타기 veto 게이트 = 위약 ·
  ER 레짐 필터 0(pingpong_regime_avgdown) · 일봉 추세 필터 순 0 · 30/60분 방향·크기로 보유 길이 0(같은 날).
남은 축 = 물타기 꼬리(10% × −200bp)가 고변동에 몰린다는 것 + 변동성은 예측된다(크기 AUC .68). 결과 보기 전 고정:
  진입 = 핑퐁 페이드 후보(pingpong_extract, 다리 ≥40bp·극값 10bp 안, 페이드 −d, 진입 = 다음 1분 종가) 2022-01~2026-08.
  기본 V0 = 1단위 진입 · 첫 진입가 대비 D=44bp 역행마다 1단위 물타기(최대 3회, 지정가) · 평균단가 +TP=20bp 지정가 익절 ·
    24h 시간초과 종가 청산 · 물타기 체결 봉엔 익절 판정 안 함(보수) · 한 번에 한 포지션(순차).
  σ̂ = 다음 4시간 실현변동성(5분 수익 제곱합 √) 예측 — (a) HGB 회귀(5분봉 33피쳐, TRAIN) (b) 직전 24h 실현변동성(모델 없음).
  V1 크기 = σ̂ 반비례(TRAIN 후보 중앙 기준, [0.25,3] 클립, TRAIN 평균 1 로 정규화) · V2 쉬기 = σ̂ TRAIN 상위 20% 면 진입 안 함 ·
  V3 격자 = D·TP 를 σ̂ 비례(TRAIN 중앙에서 44/20bp, 하한 15/5) · V4 = V1+V3 · V5 시간대 = TRAIN 시간별 평균 상위 절반만.
  분할: TRAIN 2022-01~2024-06(σ̂ 모델·문턱) · VAL 2024H2 · TEST 2025-01~ (주 판정) · 2022-24 는 보조.
  지표(단위-bp = 첫 진입 1단위 명목 대비 bp × 크기): 건당 · 승률 · 최악 · 건 CVaR1% · 일 샤프(연율) · 누적 MDD.
  판정 = TEST 에서 V − V0 일 샤프 차의 일 블록 부트스트랩 95% CI 가 0 배제 + CVaR 개선. 비용 = 0(USDC 메이커)·4bp/단위 왕복 둘 다.
  원장 141왕복: σ̂(진입 시) 로 V1 크기·V2 쉬기를 실손익($)에 적용 · 위약 = 크기/쉬기를 무작위로 섞은 1,000회 분위.
실행: python scripts/research_eth_pingpong_levers_20261007.py [--selftest]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_eth_trend30_60_hold_policy_20261007 import PREV, SEED, TR_END, VA_END, features, load_1m, ms, to_5m  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/pingpong_levers_20261007"
D0, TP0, NADD, HOLD = 44.0, 20.0, 3, 1440
COST_RT = 4.0
N_BOOT = 2000


@numba.njit(cache=False)
def sim(h, l, c, ei, side, D, TP, nadd_max, H):
    """반환: 단위-bp 손익(1단위 = 첫 진입 명목) · 최대 단위 수 · 청산 1분 인덱스 · 첫 진입가 대비 최대 역행 bp · 종류(1 익절 2 시간)."""
    n = len(ei)
    pnl = np.zeros(n); q_out = np.zeros(n); xi = np.zeros(n, np.int64); mae = np.zeros(n); kind = np.zeros(n, np.int64)
    for k in range(n):
        i = ei[k]; s = side[k]; p0 = c[i]
        q = 1.0; cost = p0; nadd = 0; worst = 0.0
        end = min(i + H, len(c) - 1)
        exit_px = c[end]; kind[k] = 2; xi[k] = end
        for j in range(i + 1, end + 1):
            adv = (l[j] / p0 - 1) * 1e4 if s > 0 else (1 - h[j] / p0) * 1e4
            if adv < worst:
                worst = adv
            added = False
            while nadd < nadd_max:
                lvl = p0 * (1 - s * D[k] * (nadd + 1) / 1e4)
                if (s > 0 and l[j] <= lvl) or (s < 0 and h[j] >= lvl):
                    q += 1.0; cost += lvl; nadd += 1; added = True
                else:
                    break
            if added:
                continue
            tgt = cost / q * (1 + s * TP[k] / 1e4)
            if (s > 0 and h[j] >= tgt) or (s < 0 and l[j] <= tgt):
                exit_px = tgt; kind[k] = 1; xi[k] = j
                break
        pnl[k] = s * (exit_px * q - cost) / p0 * 1e4
        q_out[k] = q; mae[k] = worst
    return pnl, q_out, xi, mae, kind


def sequential(ei: np.ndarray, xi: np.ndarray, take: np.ndarray) -> np.ndarray:
    """한 번에 한 포지션: 직전 청산 뒤에 오는 후보만."""
    out = np.zeros(len(ei), bool); last = -1
    for k in np.argsort(ei, kind="stable"):
        if take[k] and ei[k] > last:
            out[k] = True; last = xi[k]
    return out


def stats(pnl_u, day, rng) -> dict:
    d = pd.Series(pnl_u).groupby(day).sum()
    eq = np.cumsum(pnl_u)
    sh = d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan
    q = np.quantile(pnl_u, 0.01)
    return dict(n=int(len(pnl_u)), per_trade=float(pnl_u.mean()), win=float((pnl_u > 0).mean()), worst=float(pnl_u.min()),
                cvar1=float(pnl_u[pnl_u <= q].mean()), daily_sharpe=float(sh), mdd=float((eq - np.maximum.accumulate(eq)).min()),
                total=float(eq[-1]) if len(eq) else 0.0)


def sharpe_diff_ci(a: pd.Series, b: pd.Series, rng) -> list[float]:
    """일 손익 두 열(같은 날 축)의 연율 샤프 차 · 일 재표집."""
    D = pd.concat([a, b], axis=1).fillna(0.0).to_numpy()
    idx = rng.integers(0, len(D), (N_BOOT, len(D)))
    S = D[idx]
    sh = S.mean(1) / S.std(1) * np.sqrt(365)
    diff = sh[:, 0] - sh[:, 1]
    return [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]


def main() -> None:
    rng = np.random.default_rng(SEED)
    k1 = load_1m(); b = to_5m(k1); X = features(b)
    t5, c5 = b.t.to_numpy(), b.c.to_numpy()
    r5 = np.diff(np.log(c5), prepend=np.nan) * 1e4
    r2 = pd.Series(r5 ** 2)
    rv_fwd = np.sqrt(r2[::-1].rolling(48).sum()[::-1].shift(-1).to_numpy())          # τ+1..τ+48
    rv_past = np.sqrt(r2.rolling(288).sum().to_numpy() / 6)                            # 24h → 4h 척도
    trm = (t5 >= ms("2022-04-01")) & (t5 < TR_END - 48 * 300_000) & np.isfinite(rv_fwd) & (rv_fwd > 0) & np.isfinite(X.ret288.to_numpy())
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, early_stopping=False, random_state=SEED)
    m.fit(X[trm], np.log(rv_fwd[trm]))
    sig_hgb = np.exp(m.predict(X))
    te5 = (t5 >= VA_END) & np.isfinite(rv_fwd) & (rv_fwd > 0)
    res: dict = {"sigma_corr_test": dict(
        hgb=float(np.corrcoef(np.log(sig_hgb[te5]), np.log(rv_fwd[te5]))[0, 1]),
        past24h=float(pd.Series(np.log(rv_past[te5])).corr(pd.Series(np.log(rv_fwd[te5])))))}
    print("σ̂ 로그 상관(TEST):", res["sigma_corr_test"], flush=True)

    C = pd.read_parquet(PREV / "tmp/pingpong_extract_20261007/candidates.parquet").sort_values("t")
    t1, c1, h1, l1 = (k1[x].to_numpy() for x in ("t", "c", "h", "l"))
    ei = np.searchsorted(t1, C.t.to_numpy("int64") + 60_000)
    tdec = C.t.to_numpy("int64") + 60_000
    bi = np.searchsorted(t5, tdec // 300_000 * 300_000 - 300_000)
    ok = (ei + HOLD < len(c1)) & (bi < len(t5)) & np.isfinite(rv_past[np.minimum(bi, len(t5) - 1)])
    C, ei, bi, tdec = C[ok].reset_index(drop=True), ei[ok], bi[ok], tdec[ok]
    side = -C.d.to_numpy(np.int64); day = tdec // 86_400_000; hour = (tdec // 3_600_000) % 24
    per = np.where(tdec < TR_END, "TRAIN", np.where(tdec < VA_END, "VAL", "TEST"))
    print(f"후보 {len(C)} · TRAIN {(per == 'TRAIN').sum()} VAL {(per == 'VAL').sum()} TEST {(per == 'TEST').sum()}", flush=True)

    def run(D, TP, size, take):
        pnl, q, xi, mae, kind = sim(h1, l1, c1, ei, side, D.astype(float), TP.astype(float), NADD, HOLD)
        sel = sequential(ei, xi, take)
        out = {}
        for cost in (0.0, COST_RT):
            pu = (pnl - cost * q) * size
            for p in ("TRAIN", "VAL", "TEST"):
                mm = sel & (per == p)
                out[f"{p}_c{int(cost)}"] = stats(pu[mm], day[mm], rng)
            out[f"_daily_c{int(cost)}"] = pd.Series(pu[sel & (per == "TEST")]).groupby(day[sel & (per == "TEST")]).sum()
        out["_raw"] = dict(pnl=pnl, q=q, mae=mae, kind=kind, sel=sel)
        return out

    ones = np.ones(len(C)); allt = np.ones(len(C), bool); fD, fT = np.full(len(C), D0), np.full(len(C), TP0)
    V: dict = {"V0": run(fD, fT, ones, allt)}
    raw0 = V["V0"]["_raw"]
    tail = (raw0["mae"] <= -200) | ((raw0["kind"] == 2) & (raw0["pnl"] < 0))
    trc = per == "TRAIN"
    for sname, sig in (("hgb", sig_hgb[bi]), ("past24h", rv_past[bi])):
        tm = per == "TEST"
        res[f"tail_auc_{sname}"] = float(roc_auc_score(tail[tm], sig[tm]))
        med = np.median(sig[trc])
        size = np.clip(med / sig, 0.25, 3.0); size /= size[trc].mean()
        skip = sig < np.quantile(sig[trc], 0.8)
        Dg, TPg = np.maximum(D0 * sig / med, 15.0), np.maximum(TP0 * sig / med, 5.0)
        V[f"V1_size_{sname}"] = run(fD, fT, size, allt)
        V[f"V2_skip_{sname}"] = run(fD, fT, ones, skip)
        V[f"V3_grid_{sname}"] = run(Dg, TPg, ones, allt)
        V[f"V4_size+grid_{sname}"] = run(Dg, TPg, size, allt)
        if sname == "hgb":
            res["ledger_sizes"] = dict(med=float(med), lo=float(np.quantile(sig[trc], 0.8)))
    hp = pd.Series(raw0["pnl"][trc & raw0["sel"]]).groupby(hour[trc & raw0["sel"]]).mean()
    good_h = set(hp[hp >= hp.median()].index)
    res["V5_hours_list"] = sorted(int(x) for x in good_h)
    V["V5_hours"] = run(fD, fT, ones, np.isin(hour, list(good_h)))

    print(f"\n꼬리(MAE≤−200bp 또는 시간초과 손실) 비율 TEST {tail[per == 'TEST'].mean():.3f} · σ̂ 로 꼬리 AUC hgb {res['tail_auc_hgb']:.3f} · "
          f"past24h {res['tail_auc_past24h']:.3f}")
    for cost in (0, int(COST_RT)):
        print(f"\n=== 비용 {cost}bp/단위 왕복 ===")
        base = V["V0"][f"_daily_c{cost}"]
        for nm, v in V.items():
            s = v[f"TEST_c{cost}"]; s22 = v[f"TRAIN_c{cost}"]
            ci = sharpe_diff_ci(v[f"_daily_c{cost}"], base, rng) if nm != "V0" else [0.0, 0.0]
            res.setdefault(nm, {})[f"c{cost}"] = dict(TEST=s, VAL=v[f"VAL_c{cost}"], TRAIN=s22, sharpe_minus_v0_ci=ci)
            print(f"{nm:22s} TEST n{s['n']:5d} 건당 {s['per_trade']:+6.2f} 승률 {s['win']:.3f} 최악 {s['worst']:8.1f} CVaR1% {s['cvar1']:7.1f} "
                  f"샤프 {s['daily_sharpe']:+.2f} MDD {s['mdd']:8.0f} · V0 대비 샤프 [{ci[0]:+.2f},{ci[1]:+.2f}] · "
                  f"22-24 샤프 {s22['daily_sharpe']:+.2f} CVaR {s22['cvar1']:.0f}")

    # 원장
    T = pd.read_parquet(PREV / "tmp/ledger_micro_20261007/trips.parquet").sort_values("t1").reset_index(drop=True)
    tb = np.searchsorted(t5, T.t0.to_numpy("int64") // 300_000 * 300_000 - 300_000)
    sg = sig_hgb[tb]; med, lo = res["ledger_sizes"]["med"], res["ledger_sizes"]["lo"]
    size = np.clip(med / sg, 0.25, 3.0); size /= size.mean()                               # 원장 평균 크기 유지
    net = T.net.to_numpy()

    def led(x):
        eq = np.cumsum(x); return dict(total=float(eq[-1]), mdd=float((eq - np.maximum.accumulate(eq)).min()), worst=float(x.min()))
    L = {"actual": led(net), "V1_size": led(net * size), "V2_skip": led(net * (sg < lo)), "skip_share": float((sg >= lo).mean())}
    pl_mdd = np.array([led(net * rng.permutation(size))["mdd"] for _ in range(1000)])
    pl_tot = np.array([led(net * rng.permutation(size))["total"] for _ in range(1000)])
    L["V1_mdd_pctile_vs_shuffle"] = float((pl_mdd < L["V1_size"]["mdd"]).mean() * 100)
    L["V1_total_pctile_vs_shuffle"] = float((pl_tot < L["V1_size"]["total"]).mean() * 100)
    L["tail_auc_trip_loss"] = float(roc_auc_score(net < 0, sg))
    res["ledger"] = L
    print("\n[실원장 141왕복 $]", json.dumps(L, ensure_ascii=False))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))


def selftest() -> None:
    # 롱 100 → 99.55 저가(44bp 물타기 1회 at 99.56) → 100.25 고가가 평균 99.78 +20bp 위 → 4번째 봉 익절
    c = np.array([100.0, 99.6, 99.5, 99.7, 100.2]); h = c + 0.05; l = c - 0.05
    pnl, q, xi, mae, kind = sim(h, l, c, np.array([0]), np.array([1]), np.array([44.0]), np.array([20.0]), 3, 10)
    tgt = (100 + 99.56) / 2 * 1.002
    assert q[0] == 2 and kind[0] == 1 and xi[0] == 4 and np.isclose(pnl[0], (tgt * 2 - 199.56) / 100 * 1e4), (pnl, q, xi, kind)
    # 숏 대칭 · 시간초과
    c2 = np.array([100.0, 100.1, 100.2, 100.3]); pnl2, q2, _, _, k2 = sim(c2 + .01, c2 - .01, c2, np.array([0]), np.array([-1]),
                                                                          np.array([44.0]), np.array([20.0]), 3, 3)
    assert k2[0] == 2 and q2[0] == 1 and np.isclose(pnl2[0], -30.0)
    assert sequential(np.array([0, 5, 12]), np.array([10, 11, 20]), np.ones(3, bool)).tolist() == [True, False, True]
    print("selftest OK")


def grid() -> None:
    """격자 설계 탐색(사후 추가, 사전등록 밖): D×물타기 횟수×TP 를 TRAIN 일 샤프로 고르고 TEST 확인. + 크기 ∝ σ(V1 의 거울, 탐색)."""
    rng = np.random.default_rng(SEED)
    k1 = load_1m(); b = to_5m(k1)
    t5 = b.t.to_numpy(); r2 = pd.Series((np.diff(np.log(b.c.to_numpy()), prepend=np.nan) * 1e4) ** 2)
    rv_past = np.sqrt(r2.rolling(288).sum().to_numpy() / 6)
    C = pd.read_parquet(PREV / "tmp/pingpong_extract_20261007/candidates.parquet").sort_values("t")
    t1, c1, h1, l1 = (k1[x].to_numpy() for x in ("t", "c", "h", "l"))
    tdec = C.t.to_numpy("int64") + 60_000; ei = np.searchsorted(t1, tdec)
    bi = np.searchsorted(t5, tdec // 300_000 * 300_000 - 300_000)
    ok = (ei + HOLD < len(c1)) & np.isfinite(rv_past[np.minimum(bi, len(t5) - 1)])
    C, ei, bi, tdec = C[ok], ei[ok], bi[ok], tdec[ok]
    side = -C.d.to_numpy(np.int64); day = tdec // 86_400_000
    per = np.where(tdec < TR_END, "TRAIN", np.where(tdec < VA_END, "VAL", "TEST"))
    rows = []
    for D in (30.0, 44.0, 60.0, 90.0, 130.0):
        for na in (0, 1, 2, 3, 5):
            for TP in (10.0, 20.0, 40.0, 80.0):
                pnl, q, xi, mae, kind = sim(h1, l1, c1, ei, side, np.full(len(ei), D), np.full(len(ei), TP), na, HOLD)
                sel = sequential(ei, xi, np.ones(len(ei), bool))
                r = dict(D=D, nadd=na, TP=TP)
                for cost in (0, 4):
                    pu = pnl - cost * q
                    for pn in ("TRAIN", "TEST"):
                        mm = sel & (per == pn); st = stats(pu[mm], day[mm], rng)
                        r[f"{pn}_c{cost}_sh"], r[f"{pn}_c{cost}_bp"], r[f"{pn}_c{cost}_cvar"] = st["daily_sharpe"], st["per_trade"], st["cvar1"]
                rows.append(r)
    G = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True); G.to_csv(OUT / "grid.csv", index=False)
    for cost in (0, 4):
        top = G.sort_values(f"TRAIN_c{cost}_sh", ascending=False).head(5)
        print(f"\n[비용 {cost}] TRAIN 샤프 상위 5 → TEST")
        print(top[["D", "nadd", "TP", f"TRAIN_c{cost}_sh", f"TEST_c{cost}_sh", f"TEST_c{cost}_bp", f"TEST_c{cost}_cvar"]].round(2).to_string(index=False))
        print(f"  TEST 샤프 > 0 인 조합 {(G[f'TEST_c{cost}_sh'] > 0).mean():.2f} · TRAIN↔TEST 샤프 순위상관 "
              f"{G[f'TRAIN_c{cost}_sh'].corr(G[f'TEST_c{cost}_sh'], method='spearman'):+.2f} · TEST 최고 {G[f'TEST_c{cost}_sh'].max():+.2f}")
    # 크기 ∝ σ (V1 거울, 탐색)
    pnl, q, xi, mae, kind = sim(h1, l1, c1, ei, side, np.full(len(ei), D0), np.full(len(ei), TP0), NADD, HOLD)
    sel = sequential(ei, xi, np.ones(len(ei), bool)); sig = rv_past[bi]; trc = per == "TRAIN"
    size = np.clip(sig / np.median(sig[trc]), 0.25, 3.0); size /= size[trc].mean()
    for cost in (0, 4):
        pu = pnl - cost * q
        a = pd.Series((pu * size)[sel & (per == "TEST")]).groupby(day[sel & (per == "TEST")]).sum()
        b0 = pd.Series(pu[sel & (per == "TEST")]).groupby(day[sel & (per == "TEST")]).sum()
        print(f"[크기∝σ 탐색 비용 {cost}] TEST 샤프 {a.mean() / a.std() * np.sqrt(365):+.2f} vs V0 {b0.mean() / b0.std() * np.sqrt(365):+.2f} · "
              f"차 CI {sharpe_diff_ci(a, b0, rng)}")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else grid() if "--grid" in sys.argv else main()
