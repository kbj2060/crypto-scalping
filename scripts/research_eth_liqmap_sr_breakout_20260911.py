#!/usr/bin/env python3
"""**청산맵 S/R 을 돌파/되돌림 배리어로** -- 사용자 가설 검정 (2026-09-11).

사용자: "돌파와 되돌림에서 중요한 건 저항과 지지를 돌파하는가 돌아오는가."
철회된 돌파/되돌림은 배리어가 ±0.8×ATR **대칭**이었다. 이 가설은 배리어를 **실제 S/R** 로
바꾸자는 것이다. 패널: build_eth_liqmap_sr_panel_5m_20260911.py.

⚠️선행: 청산맵 S/R «레벨이 유효한가» 축은 2026-08-24~25 십수 차례 검정 후 전부 기각됐다
  (memory eth_liquidation_map_sr_research_2026_rollup). 여기서 묻는 건 다른 질문이다 --
  «레벨을 배리어로 삼으면 방향이 예측되는가».

## 두 개의 함정과 그 대조군
1. **기하가 답을 거의 정한다.** «어느 쪽에 먼저 닿나»는 대부분 거리비로 정해진다.
   그래서 «거리비 단독» 점수를 귀무로 놓고, 레벨 강도가 그 위에 얹는 게 있는지만 본다.
2. **«그 가격이 특별한가»는 같은 거리의 아무 선과 비교해야 안다.** 순환이동으로 거리를
   다른 시점 것과 바꿔친 플라시보 배리어를 함께 잰다(거리 분포·군집 보존, 위치만 무관).

라벨은 2026-09-10 A/B 감사 규약을 따른다: 기준가 = 그 봉 **종가**, 창은 **다음 봉부터**.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT if (ROOT / "binance_data").exists() else Path(subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
    capture_output=True, text=True).stdout.strip()).parent
PANEL = DATA / "tmp/eth_liqmap_sr_panel_20260911/sr_panel_5m.parquet"
KL5 = DATA / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
H = 12                      # 1시간 -- 철회된 돌파/되돌림과 같은 지평
SEEDS = [20260911, 7, 131, 977, 20250401]
SHIFT = 20_000              # 순환이동 폭(약 69일) -- 국소 군집을 깨되 레짐은 보존
WINDOWS = {"VAL 25-09~12": ("2025-09-01", "2026-01-01"),
           "OOS 26-01~03": ("2026-01-01", "2026-04-01"),
           "최근 26-04~": ("2026-04-01", "2100-01-01")}
TRAIN_END = "2025-09-01"


def first_touch(hi, lo, start, up_px, dn_px, span):
    """다음 봉부터 span 봉 안에서 위/아래 중 **먼저 닿는 쪽**. 1=위 · 0=아래 · -1=미해소."""
    out = np.full(len(start), -1, np.int8)
    n = len(hi)
    for k in range(len(start)):
        a = start[k]
        if a < 1 or a + span >= n or not (up_px[k] > 0 and dn_px[k] > 0):
            continue
        h, l = hi[a:a + span], lo[a:a + span]
        iu = np.flatnonzero(h >= up_px[k]); idn = np.flatnonzero(l <= dn_px[k])
        u = iu[0] if len(iu) else 1 << 30
        d = idn[0] if len(idn) else 1 << 30
        if u == d == 1 << 30:
            continue
        out[k] = int(u < d)
    return out


DELTA, TAU, COST_BP, MAKER_BP = 0.0015, 0.002, 10.0, 7.8   # peg 메이커 진입+테이커 청산 실측      # 근접 0.15% · 판정 0.2%


def breakout_test(kl, d, cl, hi, lo, ts, dR, roll):
    """저항에 **다가간** 봉에서 돌파/반등 비율. 강한 레벨일수록 덜 돌파해야 사용자 가설이 맞다.

    돌파 = 레벨 위 TAU 를 먼저 침 · 반등 = 레벨 아래 TAU 를 먼저 침. 둘 다 아니면 제외.
    기준선은 **순환이동 거리로 그은 플라시보 라인** -- 레벨이 아닌 «그냥 그 자리»의 돌파율.
    """
    print("\n=== 저항 근접시 돌파 vs 반등 (근접 %.2f%% · 판정 %.2f%%) ===" % (DELTA * 100, TAU * 100))
    n = len(cl); got = {}
    for tag, dist in (("실제 저항", dR), ("플라시보 라인", dR[roll])):
        lvl = cl * (1 + dist)
        near = np.flatnonzero((np.abs(lvl - cl) / cl <= DELTA) & np.isfinite(lvl))
        near = near[(near >= 1) & (near + H < n)]
        brk = np.full(len(near), -1, np.int8)
        for k, i in enumerate(near):
            h, l = hi[i + 1:i + 1 + H], lo[i + 1:i + 1 + H]
            iu = np.flatnonzero(h >= lvl[i] * (1 + TAU)); idn = np.flatnonzero(l <= lvl[i] * (1 - TAU))
            u = iu[0] if len(iu) else 1 << 30
            dn = idn[0] if len(idn) else 1 << 30
            if u == dn == 1 << 30:
                continue
            brk[k] = int(u < dn)
        r = brk >= 0
        print(f"[{tag}] 근접 {len(near):,}건 · 해소 {r.mean():.3f} · **돌파율 {brk[r].mean():.3f}**")
        # 경제성: 저항 근접에서 **되돌림에 건다**(숏). 진입은 레벨이 아니라 그 봉 **종가**다
        #   -- 레벨에 지정가를 걸어두는 건 별개 축이고 여기선 체결 가능한 값만 쓴다.
        px_tp, px_sl = lvl[near][r] * (1 - TAU), lvl[near][r] * (1 + TAU)
        ent = cl[near][r]
        exit_px = np.where(brk[r] == 1, px_sl, px_tp)          # 돌파면 손절 · 반등이면 익절
        bp = (ent - exit_px) / ent * 1e4 - COST_BP             # 숏이므로 진입-청산
        print(f"  되돌림 베팅(숏, 종가진입, 비용 {COST_BP}bp): 건당 {bp.mean():+.2f}bp")
        got[tag] = pd.DataFrame({"ts": ts.to_numpy()[near][r], "brk": brk[r], "bp": bp})

        # 반론: «레벨에 지정가를 걸면 대칭이 되지 않나». 체결된 것만 세되 **체결 봉은 크레딧하지
        #   않는다**(같은 봉 해소는 철회된 돌파/되돌림이 저지른 바로 그 죄다). 메이커 진입+테이커 청산.
        fbp, filled = [], 0
        for i in near:
            L = lvl[i]
            j = next((t for t in range(i + 1, min(i + 1 + H, n)) if hi[t] >= L), None)
            if j is None or j + 1 >= n:
                continue
            filled += 1
            a, b = hi[j + 1:j + 1 + H], lo[j + 1:j + 1 + H]
            iu = np.flatnonzero(a >= L * (1 + TAU)); idn = np.flatnonzero(b <= L * (1 - TAU))
            u = iu[0] if len(iu) else 1 << 30
            dn = idn[0] if len(idn) else 1 << 30
            if u == dn == 1 << 30:
                fbp.append((L - cl[min(j + H, n - 1)]) / L * 1e4 - MAKER_BP)   # 시간청산
            else:
                fbp.append((TAU if dn < u else -TAU) * 1e4 - MAKER_BP)
        fbp = np.array(fbp)
        print(f"  레벨 지정가 진입(체결 {filled:,}/{len(near):,} = {filled/len(near):.2f}, "
              f"체결봉 제외, 비용 {MAKER_BP}bp): 건당 {fbp.mean():+.2f}bp")
        if tag == "실제 저항":
            w = d.r1_w.to_numpy()[near][r]
            q = pd.qcut(w, 4, labels=["약", "중하", "중상", "강"], duplicates="drop")
            g = pd.DataFrame({"q": q, "brk": brk[r]}).groupby("q", observed=True)["brk"]
            print("  강도 사분위별 돌파율(가설: 강할수록 낮아야):",
                  " · ".join(f"{k}={v:.3f}(n={c:,})" for (k, v), c in zip(g.mean().items(), g.size())))

    # 두 이벤트 집합은 **서로 다른 시점**에 몰린다 -- 원시 격차는 레짐 구성 효과일 수 있다.
    A, B = got["실제 저항"].copy(), got["플라시보 라인"].copy()
    for f in (A, B):
        f["m"] = pd.to_datetime(f.ts).dt.to_period("M")
    j = A.groupby("m").brk.agg(["mean", "size"]).join(
        B.groupby("m").brk.agg(["mean", "size"]), lsuffix="_a", rsuffix="_b").dropna()
    j = j[(j["size_a"] >= 30) & (j["size_b"] >= 30)]
    dm = j["mean_a"] - j["mean_b"]
    print(f"  월 매칭({len(j)}개월, 양쪽 30건 이상): 실제-플라시보 평균 {dm.mean()*100:+.2f}pp · "
          f"음수月 {int((dm < 0).sum())}/{len(j)}")
    # 일 군집 부트스트랩 -- 근접 이벤트는 하루 안에 뭉치므로 행 단위 부트는 과신한다.
    rng = np.random.default_rng(20260911)
    A["d"] = pd.to_datetime(A.ts).dt.date; B["d"] = pd.to_datetime(B.ts).dt.date
    da, db = [g.brk.to_numpy() for _, g in A.groupby("d")], [g.brk.to_numpy() for _, g in B.groupby("d")]
    boot = [np.concatenate([da[i] for i in rng.integers(0, len(da), len(da))]).mean()
            - np.concatenate([db[i] for i in rng.integers(0, len(db), len(db))]).mean()
            for _ in range(2000)]
    q = np.percentile(boot, [2.5, 97.5])
    print(f"  일군집 부트(2000회, 일수 {len(da)}/{len(db)}): 격차 95%CI [{q[0]*100:+.2f}, {q[1]*100:+.2f}]pp")
    pa = [g.bp.to_numpy() for _, g in A.groupby("d")]
    bb = [np.concatenate([pa[i] for i in rng.integers(0, len(pa), len(pa))]).mean() for _ in range(2000)]
    qb = np.percentile(bb, [2.5, 97.5])
    print(f"  실제 저항 되돌림 베팅 건당 손익 95%CI [{qb[0]:+.2f}, {qb[1]:+.2f}]bp "
          f"(하루 {len(A)/len(da):.1f}건)")


def evaluate(X, y, ts, cols, tag, rows):
    tr = (ts < TRAIN_END).to_numpy() & (y >= 0)
    p = np.mean([HistGradientBoostingClassifier(random_state=s, max_iter=300)
                 .fit(X[tr][:, cols], y[tr]).predict_proba(X[:, cols])[:, 1] for s in SEEDS], axis=0)
    for name, (a, b) in WINDOWS.items():
        m = ((ts >= a) & (ts < b)).to_numpy() & (y >= 0)
        rows.append((tag, name, int(m.sum()), float(y[m].mean()),
                     roc_auc_score(y[m], p[m]) if len(set(y[m])) > 1 else np.nan))
    return p


def main() -> int:
    d = pd.read_parquet(PANEL)
    kl = (pd.read_csv(KL5, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
          .sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    kl = kl[kl.timestamp.isin(set(d.timestamp))].reset_index(drop=True)
    assert len(kl) == len(d), (len(kl), len(d))
    hi, lo, cl = kl.high.to_numpy(float), kl.low.to_numpy(float), kl.close.to_numpy(float)
    ts, idx = d.timestamp, np.arange(len(d))

    ok = d.s1_px.notna().to_numpy() & d.r1_px.notna().to_numpy()
    print(f"패널 {len(d):,}행 · 양쪽 레벨 존재 {ok.mean():.3f}")
    dR = (d.r1_px.to_numpy() - cl) / cl                       # 저항까지 (+)
    dS = (cl - d.s1_px.to_numpy()) / cl                       # 지지까지 (+)
    print(f"저항거리 중앙 {np.nanmedian(dR)*100:.2f}% · 지지거리 중앙 {np.nanmedian(dS)*100:.2f}%")

    # ── 실제 배리어 vs 순환이동 플라시보(거리만 다른 시점 것으로 바꿔친다) ──────────────
    roll = np.roll(np.arange(len(d)), SHIFT)
    arms = {"실제 S/R": (dR, dS), "플라시보(순환이동 거리)": (dR[roll], dS[roll])}
    labels = {}
    for name, (u, dn) in arms.items():
        y = first_touch(hi, lo, idx + 1, cl * (1 + u), cl * (1 - dn), H)
        labels[name] = y
        res = y >= 0
        print(f"[{name}] 해소 {res.mean():.3f} · 위 먼저 {y[res].mean():.3f}")

    # ── 피쳐: 기하만 / 기하+강도 / 기하+강도+가격 ────────────────────────────────
    r5 = pd.Series(cl).pct_change()
    F = pd.DataFrame({
        "d_res": dR, "d_sup": dS, "ratio": dR / (dR + dS),      # 기하
        "w_res": d.r1_w, "w_sup": d.s1_w, "n_res": d.n_res, "n_sup": d.n_sup,
        "w_gap": d.r1_w.to_numpy() - d.s1_w.to_numpy(),
        "d_res2": (d.r2_px.to_numpy() - cl) / cl, "d_sup2": (cl - d.s2_px.to_numpy()) / cl,
        "ret12": pd.Series(cl).pct_change(12), "ret48": pd.Series(cl).pct_change(48),
        "atr_pct": (pd.Series(hi - lo).rolling(14).mean() / cl),
        "vol48": r5.rolling(48).std(),
    })
    X = np.nan_to_num(F.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    G = [F.columns.get_loc(c) for c in ("d_res", "d_sup", "ratio")]
    GW = G + [F.columns.get_loc(c) for c in ("w_res", "w_sup", "n_res", "n_sup", "w_gap",
                                             "d_res2", "d_sup2")]
    GWP = GW + [F.columns.get_loc(c) for c in ("ret12", "ret48", "atr_pct", "vol48")]

    rows = []
    y = labels["실제 S/R"]
    # 모델 없는 기준: 거리비 하나로 «위가 먼저»를 예측한다(가까울수록 먼저 닿는다).
    score = -np.nan_to_num(F["ratio"].to_numpy(float), nan=0.5)
    for name, (a, b) in WINDOWS.items():
        m = ((ts >= a) & (ts < b)).to_numpy() & (y >= 0)
        rows.append(("거리비 단독(모델 없음)", name, int(m.sum()), float(y[m].mean()),
                     roc_auc_score(y[m], score[m])))
    evaluate(X, y, ts, G, "모델·기하만", rows)
    evaluate(X, y, ts, GW, "모델·기하+레벨강도", rows)
    evaluate(X, y, ts, GWP, "모델·기하+강도+가격", rows)
    # 플라시보는 **자기 기하로** 채점해야 비교가 된다(실제 레벨 피쳐로 플라시보 라벨을 맞추라는
    # 건 다른 배리어를 맞히라는 뜻이라 무의미하다). 거리 3개만 플라시보 것으로 갈아끼운다.
    Xp = X.copy()
    Xp[:, F.columns.get_loc("d_res")] = dR[roll]
    Xp[:, F.columns.get_loc("d_sup")] = dS[roll]
    Xp[:, F.columns.get_loc("ratio")] = dR[roll] / (dR[roll] + dS[roll])
    Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)
    evaluate(Xp, labels["플라시보(순환이동 거리)"], ts, G, "플라시보·자기기하만", rows)

    # ── 사용자의 문자 그대로의 질문: 레벨에 다가갔을 때 «돌파»인가 «반등»인가 ──────────
    breakout_test(kl, d, cl, hi, lo, ts, dR, roll)

    print(f"\n{'팔':<26}{'창':<14}{'n':>8}{'기저':>8}{'AUC':>8}")
    print("-" * 66)
    for tag, win, n, base, auc in rows:
        print(f"{tag:<26}{win:<14}{n:>8,}{base:>8.3f}{auc:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
