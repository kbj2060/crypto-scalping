#!/usr/bin/env python3
"""선행갭 라벨 — «경보가 울리고 **L분 뒤에** 전환이 온다» (2026-09-11, 사용자 요구).

현행 라벨은 `(t, t+24]` 안에 큰 이동이면 양성이라, **t+1(5분 뒤)에 터지는 자리도 양성**이다.
경보로 쓰려면 그건 이미 늦었다. 그래서 창을 둘로 가른다.

    공백 (t, t+L]        — 여기선 아직 조용해야 한다. 사용자가 대응할 시간.
    판정 (t+L, t+L+W]    — 여기서 큰 이동이 **시작**되면 양성.

기준가는 `close[t+L]` 이다 — `close[t]` 로 재면 공백 안의 움직임이 판정 이탈폭에 섞인다.
ATR 정규화 분모는 `atr[t]`(결정 시점에 알려진 값)를 쓴다. 피쳐는 봉 t 까지만 본다.

⭐**선행 L 은 고정하지 않고 스윕한다** — 배포 신호등 3종의 최적 지평이 각각 2시간·5~15분·
  30분으로 달라서, 규칙마다 최적 선행이 다를 수 있다(사용자 지적, 2026-09-11).
⭐**판정창 길이 W 는 고정한다.** 선행이 길어질수록 창이 짧아지게 두면 «선행이 나쁜 건지
  창이 짧아진 건지»를 못 가린다.
⭐**선택은 VAL 에서만** 한다. 15칸 격자에서 OOS/FWD 까지 보고 고르면 그게 선택편향이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import alert_base_20260911 as R  # noqa: E402   ← 브랜치 판의 사본.
# 🔴서버의 `research_eth_breakout_alert_tabpfn_20260911.py` 는 커밋된 두 판과 md5 가 달랐다
#   (다른 세션의 미커밋 변경). 덮어쓰지 않으려고 **이름을 바꾼 사본**을 쓴다.

OUT = ROOT / "docs/charts/eth_alert_lead_sweep_labels_20260911.png"
KF = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KF.exists():
    fm.fontManager.addfont(str(KF))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KF)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "figure.facecolor": "white"})
INK, HOT, COOL, GREY = "#1b1f24", "#d73027", "#2c7fb8", "#9aa3ad"
LEADS = (0, 3, 6, 9, 12)          # 선행 0·15·30·45·60분 (5분봉)
WS = (3, 6, 12, 24)               # 판정창 15·30·60·120분 — 규칙마다 최적 지평이 다르다
QUIET = {"없음": None, "q70": 0.70, "q50(중앙값)": 0.50}   # 공백이 «조용»할 것의 강도
BACK, PAD = 60, 30       # 차트에서 판정 시점 앞뒤로 보여줄 봉 수
REQ_LEAD = 6             # 사용자 요구: 경보는 최소 30분 앞서야 한다
SEED = 20260911


def _fwd_extreme(logc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(t, t+k] 구간의 로그가격 최대·최소. 자기 봉 t 는 뺀다."""
    rmax = pd.Series(logc[::-1]).rolling(k, min_periods=1).max()[::-1].to_numpy()
    rmin = pd.Series(logc[::-1]).rolling(k, min_periods=1).min()[::-1].to_numpy()
    return np.r_[rmax[1:], np.nan], np.r_[rmin[1:], np.nan]


def lead_move(logc: np.ndarray, atr: np.ndarray, lead: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(공백 이탈폭, 판정 이탈폭) — 둘 다 atr[t] 로 정규화한 ATR 배수.

    공백  = (t, t+lead]  를 close[t] 기준으로
    판정  = (t+lead, t+h] 를 **close[t+lead]** 기준으로 (기준가를 옮겨야 공백이 안 섞인다)
    """
    a = np.maximum(atr, 1e-9)
    if not lead:
        fmax, fmin = _fwd_extreme(logc, h)
        return np.zeros(len(logc)), np.maximum(fmax - logc, logc - fmin) / a
    gmax, gmin = _fwd_extreme(logc, lead)
    gap = np.maximum(gmax - logc, logc - gmin) / a
    kmax, kmin = _fwd_extreme(logc, h - lead)
    sh = lambda x: np.r_[x[lead:], np.full(lead, np.nan)]        # noqa: E731  t -> t+lead
    ref = sh(logc)
    return gap, np.maximum(sh(kmax) - ref, ref - sh(kmin)) / a


def label_windows(p: pd.DataFrame, val: np.ndarray, ok: np.ndarray, topq: float = R.TOPQ):
    """창별 상위 (1-topq) 를 양성으로. 창마다 변동성 수준이 달라 전역 임계는 못 쓴다."""
    y = np.zeros(len(p), bool)
    thr = {}
    for nm, _, _ in R.WINDOWS:
        m = ok & (p.win == nm).to_numpy() & np.isfinite(val)
        if m.sum() < 500:
            continue
        thr[nm] = float(np.nanquantile(val[m], topq))
        y[m] = val[m] >= thr[nm]
    return y, thr


def screen(p, feats, y, idx):
    """이 라벨을 **누가** 맞히는가 — 배포 규칙 3종과 최고 단변량. 모델 없이.

    🔴창별로 **행을 잘라서** 넘긴다. 마스킹만 하고 전체 배열을 넘기면 `lift_at` 의 기저
      `y.mean()` 이 전 구간에서 계산돼 lift 가 이론 최대(1/기저=20)를 훌쩍 넘는다.
      첫 판에서 32~72 가 나왔고 그게 신호였다.
    """
    out = {}
    for nm in [f"rule::{d[0]}" for d in R.DEPLOYED]:
        out[nm] = {w: R.lift_at(p[nm].to_numpy(float)[idx[w]], y[idx[w]], 0.01)[0]
                   for w in R.EVAL_WINS}
    best = {}
    for w in R.EVAL_WINS:
        i = idx[w]
        sc = []
        for f in feats:
            x = p[f].to_numpy(float)[i]
            sc.append((float(np.nanmax([R.lift_at(x, y[i], 0.01)[0],
                                        R.lift_at(-x, y[i], 0.01)[0]])), f))
        best[w] = max(s for s in sc if np.isfinite(s[0]))
    out["best"] = best
    out["base"] = {w: float(y[idx[w]].mean()) for w in R.EVAL_WINS}
    return out


def panel(ax, x0, c, t, h, lead, title):
    a, b = t - BACK, t + h + PAD
    x = np.arange(a, b) - t
    ax.plot(x, c[a:b], color=INK, lw=1.3, zorder=3)
    if lead:
        ax.axvspan(0, lead, color=GREY, alpha=0.22, zorder=0)      # 공백 — 조용해야 하는 곳
    ax.axvspan(lead, h, color=HOT, alpha=0.12, zorder=0)           # 판정창
    ax.axvline(0, color=INK, lw=1.2, ls="--", zorder=4)
    ax.axhline(c[t + lead], color=COOL, lw=1.0, ls=":", zorder=1)  # 기준가 close[t+lead]
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel("가격", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("봉 (0 = 경보 판정 시점, 1봉 = 5분)", fontsize=9)
    ax.set_title(title, fontsize=10.2, color=INK, loc="left", pad=4)


def main() -> int:
    p = R.build()
    feats = R.feature_cols(p)
    c = pd.read_csv(R.KL5, usecols=["close"]).close.to_numpy(float)
    logc = np.log(np.maximum(c, 1e-12))
    atr = p["atr_pct"].to_numpy()
    p["win"] = ""
    for nm, a, b in R.WINDOWS:
        p.loc[(p.timestamp >= a) & (p.timestamp <= b), "win"] = nm

    # ── 선행 × 공백조용 스윕. 판정창 길이 W 는 고정한다 — 선행마다 창 길이가 달라지면
    #    «선행이 나쁜 건지 창이 짧아진 건지»를 못 가린다.
    fin = np.isfinite(atr) & np.isfinite(p[feats]).all(axis=1).to_numpy() & (p.win != "").to_numpy()
    print(f"[패널] {len(p):,}봉 · 판정창 {[f'{x*5}분' for x in WS]} · "
          f"선행 {[f'{l*5}분' for l in LEADS]} · 공백 조용 {list(QUIET)}"
          f"  = {len(LEADS)*len(WS)*len(QUIET)}칸")
    lab: dict[tuple, tuple] = {}
    for lead in LEADS:
        for w_ in WS:
            gap, mv = lead_move(logc, atr, lead, lead + w_)
            ok = fin & np.isfinite(mv)
            y, thr = label_windows(p, mv, ok)
            for qn, qq in QUIET.items():
                if lead == 0 and qq is not None:
                    continue                              # 공백이 없으면 조용 조건도 없다
                quiet = np.ones(len(p), bool)
                if qq is not None:
                    for nm, _, _ in R.WINDOWS:
                        m = ok & (p.win == nm).to_numpy()
                        if m.sum() > 500:
                            quiet[m] = gap[m] <= float(np.nanquantile(gap[m], qq))
                lab[(lead, w_, qn)] = (y & quiet, ok & quiet, gap, mv, thr)

    res = {}
    for key, (y, ok, *_) in lab.items():
        idx = {w: np.flatnonzero(ok & (p.win == w).to_numpy()) for w in R.EVAL_WINS}
        res[key] = screen(p, feats, y, idx)
    arms = [f"rule::{d[0]}" for d in R.DEPLOYED]
    # 🔴기저가 다른 셀끼리 lift 를 직접 비교하면 안 된다 — 기저가 낮을수록 이론 최대(1/기저)가
    #   커져 **기계적으로** 유리하다. 그래서 정밀도(= lift x 기저)를 같이 찍는다.
    for qn in QUIET:
        for w_ in WS:
            rows = sorted(k for k in res if k[1] == w_ and k[2] == qn)
            if not rows:
                continue
            print(f"\n[판정창 {w_*5}분 · 공백 조용 {qn}]  커버리지 1% 매칭 "
                  f"lift VAL/OOS/FWD  ·  정밀도는 VAL")
            print(f"  {'선행':>5s} {'기저':>7s}  {'최고 규칙':>22s} {'정밀도':>7s} {'(누구)':>14s}"
                  f"  {'최고 단변량':>22s} {'정밀도':>7s} {'(누구)':>11s}")
            for k in rows:
                r = res[k]
                ba = max(arms, key=lambda a: r[a]["VAL"])
                rl = "/".join(f"{r[ba][w]:4.2f}" for w in R.EVAL_WINS)
                bu = "/".join(f"{r['best'][w][0]:4.2f}" for w in R.EVAL_WINS)
                print(f"  {k[0]*5:3d}분 {r['base']['VAL']*100:6.2f}%  {rl:>22s} "
                      f"{r[ba]['VAL']*r['base']['VAL']*100:6.2f}% {ba[6:]:>14s}"
                      f"  {bu:>22s} {r['best']['VAL'][0]*r['base']['VAL']*100:6.2f}%"
                      f" {r['best']['VAL'][1][:11]:>11s}")

    # ⭐선택은 **VAL 에서만** 한다. OOS/FWD 는 보고용이지 고르는 데 쓰지 않는다.
    #   그리고 기준은 lift 가 아니라 **정밀도** — 셀마다 기저가 다르기 때문이다.
    prec = lambda k: max(res[k][a]["VAL"] for a in arms) * res[k]["base"]["VAL"]  # noqa: E731
    pick = max(res, key=prec)
    print(f"\n[VAL 정밀도 기준 최적] 선행 {pick[0]*5}분 · 판정창 {pick[1]*5}분 · 공백 조용 {pick[2]}"
          f"  → 규칙 정밀도 {prec(pick)*100:.2f}% (기저 {res[pick]['base']['VAL']*100:.2f}%)")
    for lo in (1, REQ_LEAD):
        bk = max((k for k in res if k[0] >= lo), key=prec)
        print(f"[선행 >= {lo*5}분 제한] 선행 {bk[0]*5}분 · 판정창 {bk[1]*5}분 · 공백 조용 {bk[2]}"
              f" → 규칙 정밀도 {prec(bk)*100:.2f}% (기저 {res[bk]['base']['VAL']*100:.2f}%) · "
              f"lift " + "/".join(f"{max(res[bk][a][w] for a in arms):.2f}" for w in R.EVAL_WINS))
        if lo == REQ_LEAD:
            LEAD, W = bk[0], bk[1]        # ⭐차트는 **사용자 요구(>=30분)** 안의 최선으로 그린다
    gap6, mv6 = lead_move(logc, atr, LEAD, LEAD + W)
    base_ok = fin & np.isfinite(mv6)
    y6, thr6 = label_windows(p, mv6, base_ok)
    _, mv0 = lead_move(logc, atr, 0, W)
    y0, _ = label_windows(p, mv0, base_ok & np.isfinite(mv0))
    QK = "q50(중앙값)"                    # 차트의 «공백 조용»은 항상 중앙값 기준으로 보인다
    assert (LEAD, W, QK) in lab, sorted(lab)  # 키 이름이 바뀌면 조용히 전부-조용이 되어버린다
    quiet = lab[(LEAD, W, QK)][1]
    quiet = quiet | ~base_ok                              # 마스크를 «조용 여부»로 되돌린다
    y6q = y6 & quiet
    a_, b_ = y0 & base_ok, y6 & base_ok
    print(f"\n[차트 구성] 선행 {LEAD*5}분 · 양성 자카드 lead0 ∩ lead{LEAD*5} = "
          f"{(a_ & b_).sum()/(a_ | b_).sum()*100:.1f}%  ·  "
          f"lead0 양성 중 첫 {LEAD*5}분이 더 크게 움직인 비율 "
          f"{(a_ & (gap6 >= np.where(np.isfinite(mv6), mv6, 0))).sum()/a_.sum()*100:.1f}%")
    h = LEAD + W

    # ── 차트
    ts = p.timestamp
    sel = base_ok & (np.arange(len(p)) > BACK + 10) & (np.arange(len(p)) < len(p) - h - PAD - 2)
    rng = np.random.default_rng(SEED)
    groups = [
        (f"A. 요구사항 그대로 — {LEAD*5}분 조용하다가 그 뒤에 간다", sel & y6q),
        ("B. 양성이지만 공백에서 이미 움직였다 — 공백 조용 조건이 거르는 자리", sel & y6 & ~quiet),
        (f"C. 선행 요구가 버리는 자리 — 경보 직후 터지고 그 뒤는 잠잠", sel & y0 & ~y6),
        (f"D. 음성 — {LEAD*5}분 조용하고 그 뒤에도 안 간다", sel & quiet & ~y6 & ~y0),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(15.5, 15.5))
    for r, (nm, m) in enumerate(groups):
        idx = np.flatnonzero(m)
        print(f"  {nm[:46]:48s} {len(idx):7,}건")
        pick = rng.choice(idx, min(2, len(idx)), replace=False) if len(idx) else []
        for k in range(2):
            ax = axes[r, k]
            if k >= len(pick):
                ax.axis("off")
                continue
            t = int(pick[k])
            panel(ax, None, c, t, h, LEAD,
                  f"{nm}\n{ts.iloc[t]:%Y-%m-%d %H:%M}  ·  {p['win'].iloc[t]}  ·  "
                  f"공백 {LEAD*5}분 이탈 {gap6[t]:.2f}×ATR  ·  "
                  f"판정창 이탈 {mv6[t]:.2f}×ATR (임계 {thr6.get(p['win'].iloc[t], np.nan):.2f})")
    fig.suptitle(f"학습 정답 라벨 — «경보 뒤 최소 {LEAD*5}분은 조용하고, 그 다음에 간다»"
                 f"  (요구 «선행 >= {REQ_LEAD*5}분» 안의 VAL 최적: 판정창 {W*5}분)\n"
                 f"세로 점선 = 경보 판정 시점(피쳐는 여기까지만) · 회색 = {LEAD*5}분 공백 · "
                 f"빨강 = 판정창 (t+{LEAD}, t+{LEAD+W}] · 파란 점선 = 기준가 close[t+{LEAD}]",
                 fontsize=13.5, color=INK, y=0.996)
    fig.tight_layout(rect=(0, 0.02, 1, 0.962))
    fig.text(0.5, 0.005, f"판정 이탈폭은 **close[t+{LEAD}] 기준**이다 — close[t] 로 재면 공백 "
             "안의 움직임이 섞여 «이미 간 자리»가 양성이 된다.",
             ha="center", fontsize=11, color=GREY)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight")
    print(f"\n저장: {OUT}")
    return 0


def _self_check() -> None:
    """선행갭 라벨이 «공백 안의 움직임을 안 센다»를 실제로 지키는가."""
    n, h, lead = 200, 24, 6
    lc = np.zeros(n)
    lc[50:] = 0.10                               # t=49 직후(공백 안)에 계단 +10%
    atr = np.full(n, 0.01)
    gap, mv = lead_move(lc, atr, lead, h)
    t = 49                                       # 이동이 공백 (t, t+6] 안에서 끝난다
    assert gap[t] > 9.0, gap[t]                  # 공백 이탈폭은 크게 잡히고
    assert mv[t] < 1e-6, mv[t]                   # 판정창은 0 — 기준가가 옮겨갔으므로
    _, mv_old = lead_move(lc, atr, 0, h)
    assert mv_old[t] > 9.0, mv_old[t]            # 현행 라벨이면 같은 자리가 양성이다
    t2 = 40                                      # 이동이 판정창 (t+6, t+24] 안에 들어온다
    assert gap[t2] < 1e-6 and mv[t2] > 9.0, (gap[t2], mv[t2])
    lc2 = np.zeros(n)
    lc2[60:] = -0.05                             # 하락도 같은 크기로 잡혀야 한다(절대 이탈)
    _, mvd = lead_move(lc2, atr, lead, h)
    assert mvd[50] > 4.0, mvd[50]
    assert not np.isfinite(lead_move(lc, atr, lead, h)[1][-1]), "끝부분은 라벨이 없어야 한다"
    print("self-check OK  (공백 움직임 제외 · 기준가 이동 · 하락 대칭 · 꼬리 NaN)")


# ────────────────────────────────────────────── 모델 팔 (선행 15분 · 30분 두 후보)
CANDS = ((3, 3), (6, 3))          # (선행, 판정창) 봉 — 사용자가 둘 다 들고 가기로 결정
NSEED = 5                         # ⭐무작위 추출. 고정 증분은 Seed-Diversity 게이트가 금지한다


def shift_null(score, y, B=400, rng=None):
    """순환이동 귀무 — 라벨을 통째로 굴려 자기상관(군집)을 보존한 채 정렬만 깬다.

    무작위 부분표집 귀무는 양성이 시간에 뭉쳐 있으면 폭을 과소평가한다. 이 라벨은
    큰 이동이 뭉쳐 오므로 순환이동이라야 한다.
    """
    obs = R.lift_at(score, y, 0.01)[0]
    n = len(y)
    v = [R.lift_at(score, np.roll(y, int(rng.integers(n // 20, n - n // 20))), 0.01)[0]
         for _ in range(B)]
    v = np.array([x for x in v if np.isfinite(x)])
    return obs, float(np.quantile(v, 0.95)), float((v >= obs).mean())


def model_arm() -> int:
    p = R.build()
    feats = R.feature_cols(p)
    X = p[feats].to_numpy(np.float32)
    logc = np.log(np.maximum(pd.read_csv(R.KL5, usecols=["close"]).close.to_numpy(float), 1e-12))
    atr = p["atr_pct"].to_numpy()
    p["win"] = ""
    for nm, a, b in R.WINDOWS:
        p.loc[(p.timestamp >= a) & (p.timestamp <= b), "win"] = nm
    fin = np.isfinite(atr) & np.isfinite(p[feats]).all(axis=1).to_numpy() & (p.win != "").to_numpy()
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(1, 10**6, NSEED).tolist()
    print(f"[모델] HGB · 무작위 시드 {seeds} · 피쳐 {len(feats)}개")
    arms = [f"rule::{d[0]}" for d in R.DEPLOYED]

    for lead, w_ in CANDS:
        _, mv = lead_move(logc, atr, lead, lead + w_)
        ok = fin & np.isfinite(mv)
        y, _ = label_windows(p, mv, ok)
        # 금지대 — 라벨이 창 경계를 넘어 다음 창을 보지 않게. 라벨 길이의 2배.
        emb = 2 * (lead + w_)
        for nm, _, _ in R.WINDOWS:
            i = np.flatnonzero((p.win == nm).to_numpy())
            if len(i) > emb:
                ok[i[-emb:]] = False
        idx = {w: np.flatnonzero(ok & (p.win == w).to_numpy()) for w in R.EVAL_WINS}
        tr = np.flatnonzero(ok & (p.win == "TRAIN").to_numpy())
        print(f"\n=== 선행 {lead*5}분 · 판정창 {w_*5}분 ===  TRAIN {len(tr):,}행 · "
              + " · ".join(f"{w} {len(idx[w]):,}" for w in R.EVAL_WINS))
        base = {w: float(y[idx[w]].mean()) for w in R.EVAL_WINS}
        rl = {w: max(R.lift_at(p[a].to_numpy(float)[idx[w]], y[idx[w]], 0.01)[0] for a in arms)
              for w in R.EVAL_WINS}
        print("  최고 규칙        " + "  ".join(
            f"{w} {rl[w]:5.2f} ({rl[w]*base[w]*100:5.2f}%)" for w in R.EVAL_WINS))
        sc = np.zeros((len(seeds), len(p)))
        for si, sd in enumerate(seeds):
            m = R.hgb(sd).fit(X[tr], y[tr])
            sc[si] = m.predict_proba(X)[:, 1]
            lf = {w: R.lift_at(sc[si][idx[w]], y[idx[w]], 0.01)[0] for w in R.EVAL_WINS}
            print(f"  HGB seed {sd:<7d} " + "  ".join(
                f"{w} {lf[w]:5.2f} ({lf[w]*base[w]*100:5.2f}%)" for w in R.EVAL_WINS), flush=True)
        # ⭐판정은 **시드 최악값**으로 한다 — 평균은 시드 운을 성적으로 바꾼다
        for tag, agg in (("시드 최악", np.min), ("시드 평균", np.mean)):
            lf = {w: float(agg([R.lift_at(sc[si][idx[w]], y[idx[w]], 0.01)[0]
                                for si in range(len(seeds))])) for w in R.EVAL_WINS}
            print(f"  *{tag}       " + "  ".join(
                f"{w} {lf[w]:5.2f} ({lf[w]*base[w]*100:5.2f}%)" for w in R.EVAL_WINS)
                + "   규칙대비 " + "/".join(f"{lf[w]-rl[w]:+5.2f}" for w in R.EVAL_WINS))
        ens = sc.mean(axis=0)
        for w in R.EVAL_WINS:
            o, q95, pv = shift_null(ens[idx[w]], y[idx[w]], rng=rng)
            ba = max(arms, key=lambda a: rl[w])
            ro, rq, rp = shift_null(p[ba].to_numpy(float)[idx[w]], y[idx[w]], rng=rng)
            print(f"  [순환이동 귀무 {w}] 모델앙상블 {o:5.2f} vs q95 {q95:4.2f} p={pv:.3f}"
                  f"   ·  규칙 {ro:5.2f} vs q95 {rq:4.2f} p={rp:.3f}", flush=True)
        cov_table(sc, y, idx, p, arms, rl)
        # 점수를 남긴다 — 커버리지를 다시 자르는 데 재학습이 필요 없게
        od = ROOT / "tmp/alert_tabpfn_20260911"
        od.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(od / f"hgb_scores_L{lead}_W{w_}.npz", score=sc.astype(np.float32),
                            y=y, **{f"idx_{w}": idx[w] for w in R.EVAL_WINS})
    return 0


# ────────────────────────────────────────────── 피쳐 정리 + BTC 축 추가
KL_BTC = ROOT / "binance_data" / "klines" / "BTCUSDT" / "BTCUSDT-5m-api.csv"
# 🔴빼는 것 — 둘 다 «없어도 되는» 이유가 서로 다르다
DROP = ("hod_sin", "hod_cos",      # 시각 축은 이 저장소가 이미 닫았다(1h intraday 스크린)
        "comp_depth")              # = 0.70 - volexp. 트리에게는 volexp 의 아핀 복제라 정보 0


def btc_feats(ts: pd.Series) -> pd.DataFrame:
    """BTC 축 — ETH 와 **같은 공식**을 BTC 봉에 적용한다. 방향은 만들지 않는다.

    BTC 봉 t 는 ETH 봉 t 와 같은 시각에 마감하므로 결정 시점에 알려진 값이다.
    ⚠️BTC 아카이브가 ETH 보다 짧다 — 비는 구간은 NaN 으로 두고, 비교는 같은 행에서만 한다.
    """
    d = pd.read_csv(KL_BTC, usecols=["timestamp", "high", "low", "close", "quote_volume",
                                     "trades"], parse_dates=["timestamp"])
    d = d.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    lr = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(c[0], 1e-12)))
    S = pd.Series(lr, index=d.index)
    tr = np.maximum.reduce([hi - lo, np.abs(hi - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))])
    batr = pd.Series(tr, index=d.index).rolling(96).mean().to_numpy() / np.maximum(c, 1e-9)
    F = {}
    for nm, ss in (("btcn", d.trades.astype(float)), ("btcqv", d.quote_volume.astype(float))):
        for W in (96, 288, 864, 2016):
            F[f"z_{nm}_{W}"] = (ss - ss.rolling(W).mean()) / ss.rolling(W).std()
    F["btc_volexp"] = S.rolling(12).std() / S.rolling(288).std()
    F["btc_absret_atr"] = pd.Series(np.abs(lr) / np.maximum(batr, 1e-9), index=d.index)
    # BTC 가 지난 1시간 얼마나 갔나 — «BTC 가 먼저 가고 ETH 가 따라간다» 가설의 직접 측정
    lg = pd.Series(np.log(np.maximum(c, 1e-12)), index=d.index)
    F["btc_move12_atr"] = (lg.rolling(12).max() - lg.rolling(12).min()) / np.maximum(batr, 1e-9)
    F["btc_rv12"] = S.rolling(12).std()
    out = pd.DataFrame(F).reindex(ts.to_numpy())
    out.index = ts.index
    return out


def cross_feats(p: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """ETH-BTC 관계 — 동조 강도와 상대 변동성. 어느 쪽으로 가는지는 만들지 않는다."""
    c = pd.read_csv(R.KL5, usecols=["close"]).close.to_numpy(float)
    e = pd.Series(np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(c[0], 1e-12))))
    erv12 = e.rolling(12).std()
    bb = pd.read_csv(KL_BTC, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    bb = bb.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    blr = pd.Series(np.diff(np.log(np.maximum(bb.close.to_numpy(float), 1e-12)),
                            prepend=0.0), index=bb.index).reindex(p.timestamp.to_numpy())
    blr.index = p.index
    return pd.DataFrame({
        "eth_btc_corr96": e.rolling(96).corr(blr),          # 동조가 강한 국면인가
        "eth_btc_rvratio": erv12 / np.maximum(b["btc_rv12"], 1e-12),   # 누가 더 뛰고 있나
    })


def feature_arm() -> int:
    """A 현행 33 · B 정리 30 · C 정리+BTC — **같은 행**에서 HGB 5시드로 비교한다."""
    p = R.build()
    base_f = R.feature_cols(p)
    b = btc_feats(p.timestamp)
    x = cross_feats(p, b)
    for col in b.columns:
        p[col] = b[col].to_numpy()
    for col in x.columns:
        p[col] = x[col].to_numpy()
    keep = [f for f in base_f if f not in DROP]
    btc_cols = [c for c in b.columns if c != "btc_rv12"] + list(x.columns)
    SETS = {"A 현행": base_f, "B 정리": keep, "C 정리+BTC": keep + btc_cols}
    print("[피쳐 팔] " + " · ".join(f"{k} {len(v)}개" for k, v in SETS.items()))
    print(f"  뺀 것: {DROP}\n  더한 것({len(btc_cols)}): {btc_cols}")

    logc = np.log(np.maximum(pd.read_csv(R.KL5, usecols=["close"]).close.to_numpy(float), 1e-12))
    atr = p["atr_pct"].to_numpy()
    p["win"] = ""
    for nm, a_, b_ in R.WINDOWS:
        p.loc[(p.timestamp >= a_) & (p.timestamp <= b_), "win"] = nm
    # ⭐세 팔을 **같은 행**에서 잰다 — BTC 가 있는 행으로 전부 맞춘다. 안 그러면
    #   «BTC 를 넣어서 좋아졌다»와 «표본이 달라졌다»를 못 가린다.
    allf = base_f + btc_cols
    fin = (np.isfinite(atr) & np.isfinite(p[allf]).all(axis=1).to_numpy()
           & (p.win != "").to_numpy())
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(1, 10**6, NSEED).tolist()
    print(f"  공통 유효행 {fin.sum():,} (BTC 아카이브가 짧아 ETH 단독 {len(p):,}봉보다 준다)")
    arms = [f"rule::{d[0]}" for d in R.DEPLOYED]

    for lead, w_ in [c for c in CANDS if not only or c[0] == only]:
        _, mv = lead_move(logc, atr, lead, lead + w_)
        ok = fin & np.isfinite(mv)
        y, _ = label_windows(p, mv, ok)
        emb = 2 * (lead + w_)
        for nm, _, _ in R.WINDOWS:
            i = np.flatnonzero((p.win == nm).to_numpy())
            if len(i) > emb:
                ok[i[-emb:]] = False
        idx = {w: np.flatnonzero(ok & (p.win == w).to_numpy()) for w in R.EVAL_WINS}
        tr = np.flatnonzero(ok & (p.win == "TRAIN").to_numpy())
        base = {w: float(y[idx[w]].mean()) for w in R.EVAL_WINS}
        rl = {w: max(R.lift_at(p[a].to_numpy(float)[idx[w]], y[idx[w]], 0.01)[0] for a in arms)
              for w in R.EVAL_WINS}
        print(f"\n=== 선행 {lead*5}분 · 판정창 {w_*5}분 ===  TRAIN {len(tr):,} · "
              + " · ".join(f"{w} {len(idx[w]):,}" for w in R.EVAL_WINS))
        print("  최고 규칙        " + "  ".join(
            f"{w} {rl[w]:5.2f} ({rl[w]*base[w]*100:5.2f}%)" for w in R.EVAL_WINS), flush=True)
        for nm_, cols in SETS.items():
            Xs = p[cols].to_numpy(np.float32)
            lfs = {w: [] for w in R.EVAL_WINS}
            for sd in seeds:
                m = R.hgb(sd).fit(Xs[tr], y[tr])
                sc = m.predict_proba(Xs)[:, 1]
                for w in R.EVAL_WINS:
                    lfs[w].append(R.lift_at(sc[idx[w]], y[idx[w]], 0.01)[0])
            print(f"  {nm_:12s}" + "  ".join(
                f"{w} {min(lfs[w]):5.2f} ({min(lfs[w])*base[w]*100:5.2f}%)" for w in R.EVAL_WINS)
                + "   [시드최악]  규칙대비 "
                + "/".join(f"{min(lfs[w])-rl[w]:+5.2f}" for w in R.EVAL_WINS), flush=True)
    return 0


# ────────────────────────────────────────────── 안 들어간 축들 (사용자 지시)
DATA = Path("/home/kbj20/crypto-scalping")
METR = [DATA / "data/TOTAL_ETHUSDT_metrics_2021_2023.csv",
        DATA / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv"]
LIQP = DATA / "tmp/eth_liqmap_sr_panel_20260911/sr_panel_5m.parquet"
BOOKP = DATA / "data/research/eth_bookdepth_30s_20260908.parquet"


def _z(s: pd.Series, W: int) -> pd.Series:
    return (s - s.rolling(W).mean()) / s.rolling(W).std()


def metr_feats(ts: pd.Series) -> pd.DataFrame:
    """OI · 롱숏비 · 테이커비 — 5분 간격 아카이브. 방향 예측이 아니라 **포지션 상태**다."""
    d = pd.concat([pd.read_csv(f, parse_dates=["create_time"]) for f in METR])
    d = d.sort_values("create_time").drop_duplicates("create_time").set_index("create_time")
    oi = d.sum_open_interest.astype(float)
    F = {f"z_oi_{W}": _z(oi, W) for W in (96, 288, 864)}
    for k in (12, 96):                       # OI 증감률 — 포지션이 쌓이나 풀리나
        F[f"doi_{k}"] = oi / oi.shift(k) - 1.0
    for nm, col in (("ttlsr", "sum_toptrader_long_short_ratio"),
                    ("lsr", "count_long_short_ratio"),
                    ("tkr", "sum_taker_long_short_vol_ratio")):
        x = d[col].astype(float)
        for W in (96, 864):
            F[f"z_{nm}_{W}"] = _z(x, W)
    out = pd.DataFrame(F).reindex(ts.to_numpy())
    out.index = ts.index
    return out


def liq_feats(ts: pd.Series) -> pd.DataFrame:
    """청산맵 S/R — 벽까지의 **거리**와 **두께**. 어느 쪽으로 갈지는 만들지 않는다.

    ⚠️이 축은 방향 라벨에서는 정보 0 이었다(2026-09-11). 여기 타깃은 «얼마나 멀리 가나»라
      다른 질문이므로 다시 잰다 — 같은 결론이 나와도 그건 새 정보다.
    """
    d = pd.read_parquet(LIQP).sort_values("timestamp").drop_duplicates("timestamp")
    d = d.set_index("timestamp")
    c = d.close.astype(float)
    ds = (c - d.s1_px.astype(float)) / c          # 가장 가까운 지지까지 (아래로 %)
    dr = (d.r1_px.astype(float) - c) / c          # 가장 가까운 저항까지 (위로 %)
    F = {"liq_dist_s1": ds, "liq_dist_r1": dr,
         "liq_dist_min": np.minimum(ds.abs(), dr.abs()),     # 가장 가까운 «벽»
         "liq_w_s1": d.s1_w.astype(float), "liq_w_r1": d.r1_w.astype(float),
         "liq_n_sup": d.n_sup.astype(float), "liq_n_res": d.n_res.astype(float),
         "liq_w_sum": sum(d[f"{k}_w"].astype(float).fillna(0.0)
                          for k in ("s1", "s2", "s3", "r1", "r2", "r3"))}
    out = pd.DataFrame(F).reindex(ts.to_numpy())
    out.index = ts.index
    return out


def book_feats(ts: pd.Series) -> pd.DataFrame:
    """호가 depth 30초 → 5분. **bd_ok 필터 필수**(2025-10/11 밴드 상수화 결함).

    🔴밴드 선택이 곧 표본 범위다 — `dm0p2`/`d0p2`(±0.2%)는 **2026-01부터만** 존재해
      그걸 쓰면 TRAIN 이 0 이 된다. ±1% 밴드는 2024-04 부터 있다. 첫 판이 그 함정을 밟았다.
    """
    d = pd.read_parquet(BOOKP)
    d = d[d.bd_ok.astype(bool)].copy()
    d["bar"] = pd.to_datetime(d.ts).dt.floor("5min")
    g = d.groupby("bar")[["dm1p0", "d1p0", "dm2p0", "d2p0"]].last()
    b, a = g.dm1p0.astype(float), g.d1p0.astype(float)
    s_ = b + a
    F = {"bk_near_sum": s_,
         "bk_imb": (b - a).abs() / np.maximum(s_, 1e-9),     # 방향 없이 «불균형 크기»
         "bk_z_near_288": _z(s_, 288),
         "bk_thin": s_ / np.maximum(s_.rolling(288).median(), 1e-9),
         "bk_wide_ratio": s_ / np.maximum(g.dm2p0.astype(float) + g.d2p0.astype(float), 1e-9)}
    out = pd.DataFrame(F).reindex(ts.to_numpy())
    out.index = ts.index
    return out


AXES = {"BTC 동조": None, "메트릭 OI/롱숏": metr_feats, "청산맵 S/R": liq_feats, "호가 depth": book_feats}


def axes_arm() -> int:
    """축별로 **같은 행**에서 기준(현행 33) 대비 증분을 잰다. 시드 3개 · 선행 30분만.

    ⭐축마다 커버리지가 달라 행 수가 다르다 — 그래서 축마다 **자기 기준선을 다시 깐다**.
      한 기준선에 여러 축을 대면 «표본이 달라진 것»이 «축이 좋은 것»으로 읽힌다.
    """
    p = R.build()
    base_f = R.feature_cols(p)
    logc = np.log(np.maximum(pd.read_csv(R.KL5, usecols=["close"]).close.to_numpy(float), 1e-12))
    atr = p["atr_pct"].to_numpy()
    p["win"] = ""
    for nm, a_, b_ in R.WINDOWS:
        p.loc[(p.timestamp >= a_) & (p.timestamp <= b_), "win"] = nm
    lead, w_ = CANDS[1]                                   # 선행 30분 — 모델이 값을 내는 쪽
    _, mv = lead_move(logc, atr, lead, lead + w_)
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(1, 10**6, 3).tolist()
    arms = [f"rule::{d[0]}" for d in R.DEPLOYED]
    print(f"[축 스크린] 선행 {lead*5}분 · 판정창 {w_*5}분 · HGB 시드 {seeds}")

    only = [a for a in sys.argv[1:] if a in AXES]
    built = {}
    for nm, fn in (AXES.items() if not only else [(k, AXES[k]) for k in only]):
        if fn is None:
            b = btc_feats(p.timestamp)
            x = cross_feats(p, b)
            cols = pd.concat([b.drop(columns=["btc_rv12"]), x], axis=1)
        else:
            cols = fn(p.timestamp)
        built[nm] = cols
        print(f"  {nm:14s} {cols.shape[1]:2d}개 · 유효 {np.isfinite(cols).all(axis=1).sum():,}행")

    for nm, cols in built.items():
        for c in cols.columns:
            p[c] = cols[c].to_numpy(float)
        add_f = list(cols.columns)
        ok = (np.isfinite(atr) & np.isfinite(mv) & (p.win != "").to_numpy()
              & np.isfinite(p[base_f]).all(axis=1).to_numpy()
              & np.isfinite(p[add_f]).all(axis=1).to_numpy())
        y, _ = label_windows(p, mv, ok)
        emb = 2 * (lead + w_)
        for wn, _, _ in R.WINDOWS:
            i = np.flatnonzero((p.win == wn).to_numpy())
            if len(i) > emb:
                ok[i[-emb:]] = False
        idx = {w: np.flatnonzero(ok & (p.win == w).to_numpy()) for w in R.EVAL_WINS}
        tr = np.flatnonzero(ok & (p.win == "TRAIN").to_numpy())
        if len(tr) < 20000 or min(len(idx[w]) for w in R.EVAL_WINS) < 3000:
            print(f"\n=== {nm} ===  표본 부족(TRAIN {len(tr):,}) — 판정 불가")
            continue
        base = {w: float(y[idx[w]].mean()) for w in R.EVAL_WINS}
        rl = {w: max(R.lift_at(p[a].to_numpy(float)[idx[w]], y[idx[w]], 0.01)[0] for a in arms)
              for w in R.EVAL_WINS}
        print(f"\n=== {nm} ({len(add_f)}개) ===  TRAIN {len(tr):,} · "
              + " · ".join(f"{w} {len(idx[w]):,}" for w in R.EVAL_WINS))
        print("  최고 규칙        " + "  ".join(f"{w} {rl[w]:5.2f}" for w in R.EVAL_WINS), flush=True)
        got = {}
        for tag, cs in (("기준 33", base_f), (f"기준+{nm}", base_f + add_f)):
            Xs = p[cs].to_numpy(np.float32)
            lfs = {w: [] for w in R.EVAL_WINS}
            for sd in seeds:
                m = R.hgb(sd).fit(Xs[tr], y[tr])
                sc = m.predict_proba(Xs)[:, 1]
                for w in R.EVAL_WINS:
                    lfs[w].append(R.lift_at(sc[idx[w]], y[idx[w]], 0.01)[0])
            got[tag] = {w: min(lfs[w]) for w in R.EVAL_WINS}
            print(f"  {tag:16s}" + "  ".join(
                f"{w} {got[tag][w]:5.2f} ({got[tag][w]*base[w]*100:5.2f}%)"
                for w in R.EVAL_WINS) + "  [시드최악]", flush=True)
        k0, k1 = list(got)
        print("  ⇒ 축 증분      " + "  ".join(
            f"{w} {got[k1][w]-got[k0][w]:+5.2f}" for w in R.EVAL_WINS)
            + ("   세 창 모두 양수" if all(got[k1][w] > got[k0][w] for w in R.EVAL_WINS)
               else "   ← 세 창 일관성 없음"), flush=True)
    return 0


# ────────────────────────────────────────────── TabPFN 팔 (서버 GPU)
CAP = 10000        # 컨텍스트 — 선행 연구에서 20k 는 OOS 6.60→2.6 으로 반토막났다(사전학습 분포 밖)
CHUNK = 20000      # 추론 분할 — 8GB VRAM
# 커버리지별로 정밀도와 **재현율**을 같이 낸다. 재현율 = lift x 커버리지 이므로
# 커버리지 1% 에서는 lift 4.5 여도 큰 이동 100건 중 4~5건만 잡는다.
# 사용자 손실함수가 «놓치는 쪽이 더 비싸다» 이면 1% 는 너무 조인 설정이다.
COVS = (0.01, 0.02, 0.03, 0.05, 0.10)


def pr_at(score, y, cov):
    """(정밀도, 재현율) — 상위 cov 비율만 발동시켰을 때."""
    ok = np.isfinite(score)
    k = max(int(round(ok.sum() * cov)), 1)
    cut = np.partition(score[ok], -k)[-k]
    fire = ok & (score >= cut)
    npos = max(int(y.sum()), 1)
    return float(y[fire].mean()), float((y & fire).sum() / npos)


def _arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def cov_table(allsc, y, idx, p, arms, rl):
    """커버리지별 정밀도/재현율 — 시드 최악값(= 각 셀의 하한)으로 찍는다.

    ⭐재현율 = lift x 커버리지 다. 커버리지 1% 에서는 lift 4.5 여도 큰 이동 100건 중 4~5건만
      잡는다. «놓치는 쪽이 비싸다» 면 1% 는 너무 조인 설정이라는 뜻이다.
    """
    print("  -- 커버리지별 정밀도 / 재현율 (시드최악) --", flush=True)
    print(f"  {'커버':>5s} {'회/일':>6s}  " + "  ".join(f"{w:>18s}" for w in R.EVAL_WINS), flush=True)
    for cv in COVS:
        cells = []
        for w in R.EVAL_WINS:
            pr = [pr_at(allsc[si][idx[w]], y[idx[w]], cv) for si in range(len(allsc))]
            cells.append(f"{min(x[0] for x in pr)*100:5.1f}% / {min(x[1] for x in pr)*100:5.1f}%")
        print(f"  {cv*100:4.0f}% {cv*288:6.1f}  " + "  ".join(f"{c:>18s}" for c in cells), flush=True)
    rcell = []
    for w in R.EVAL_WINS:
        ba = max(arms, key=lambda a: rl[w])
        pr = pr_at(p[ba].to_numpy(float)[idx[w]], y[idx[w]], 0.01)
        rcell.append(f"{pr[0]*100:5.1f}% / {pr[1]*100:5.1f}%")
    print(f"  (규칙 1%)      " + "  ".join(f"{c:>18s}" for c in rcell), flush=True)


def tabpfn_arm() -> int:
    """두 후보(선행 15분·30분)에 TabPFN. 기준선은 규칙과 HGB 둘 다 옆에 둔다."""
    kl = Path(_arg("--klines", str(R.KL5)))
    dev = _arg("--device", "cuda")
    nest = int(_arg("--nest", "8"))
    only = int(_arg("--only-lead", "0"))          # 봉 단위. 0 이면 두 후보 다
    # ⚠️Seed-Diversity 게이트는 **승격 주장**에 N>=5 를 요구한다. 탐색 단계에서 3 으로 줄이는
    #   건 되지만, 이 수치로 승격을 주장할 수는 없다. 3 개는 5 개 열의 앞 3 개와 같다.
    nseed = int(_arg("--seeds", str(NSEED)))
    p = R.build(kl=kl)
    feats = R.feature_cols(p)
    X = p[feats].to_numpy(np.float32)
    logc = np.log(np.maximum(pd.read_csv(kl, usecols=["close"]).close.to_numpy(float), 1e-12))
    atr = p["atr_pct"].to_numpy()
    p["win"] = ""
    for nm, a_, b_ in R.WINDOWS:
        p.loc[(p.timestamp >= a_) & (p.timestamp <= b_), "win"] = nm
    fin = np.isfinite(atr) & np.isfinite(p[feats]).all(axis=1).to_numpy() & (p.win != "").to_numpy()
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(1, 10**6, nseed).tolist()
    print(f"[TabPFN] {kl.name} · {len(p):,}봉 · 피쳐 {len(feats)} · 컨텍스트 {CAP:,} · "
          f"n_estimators {nest} · 무작위 시드 {seeds} · device {dev}", flush=True)
    print(f"  창: " + " · ".join(f"{n} {a}~{b}" for n, a, b in R.WINDOWS), flush=True)
    arms = [f"rule::{d[0]}" for d in R.DEPLOYED]

    for lead, w_ in [c for c in CANDS if not only or c[0] == only]:
        _, mv = lead_move(logc, atr, lead, lead + w_)
        ok = fin & np.isfinite(mv)
        y, _ = label_windows(p, mv, ok)
        emb = 2 * (lead + w_)
        for nm, _, _ in R.WINDOWS:
            i = np.flatnonzero((p.win == nm).to_numpy())
            if len(i) > emb:
                ok[i[-emb:]] = False
        idx = {w: np.flatnonzero(ok & (p.win == w).to_numpy()) for w in R.EVAL_WINS}
        tr = np.flatnonzero(ok & (p.win == "TRAIN").to_numpy())
        base = {w: float(y[idx[w]].mean()) for w in R.EVAL_WINS}
        rl = {w: max(R.lift_at(p[a].to_numpy(float)[idx[w]], y[idx[w]], 0.01)[0] for a in arms)
              for w in R.EVAL_WINS}
        print(f"\n=== 선행 {lead*5}분 · 판정창 {w_*5}분 ===  TRAIN {len(tr):,} · "
              + " · ".join(f"{w} {len(idx[w]):,}" for w in R.EVAL_WINS), flush=True)
        print("  최고 규칙        " + "  ".join(
            f"{w} {rl[w]:5.2f} ({rl[w]*base[w]*100:5.2f}%)" for w in R.EVAL_WINS), flush=True)
        ev = np.concatenate([idx[w] for w in R.EVAL_WINS])
        allsc = np.zeros((len(seeds), len(p)))
        for si, sd in enumerate(seeds):
            srng = np.random.default_rng(int(sd))
            sub = tr[R.sub_idx(len(tr), CAP, srng)]          # 전 기간 무작위 — 최근만 쓰면 무너진다
            m = R.tabpfn(int(sd), dev, CAP, nest).fit(X[sub], y[sub])
            out = np.zeros(len(ev))
            for k in range(0, len(ev), CHUNK):
                out[k:k + CHUNK] = m.predict_proba(X[ev[k:k + CHUNK]])[:, 1]
            allsc[si][ev] = out
            lf = {w: R.lift_at(allsc[si][idx[w]], y[idx[w]], 0.01)[0] for w in R.EVAL_WINS}
            print(f"  TabPFN seed {sd:<7d}" + "  ".join(
                f"{w} {lf[w]:5.2f} ({lf[w]*base[w]*100:5.2f}%)" for w in R.EVAL_WINS), flush=True)
        for tag, agg in (("시드 최악", np.min), ("시드 평균", np.mean)):
            lf = {w: float(agg([R.lift_at(allsc[si][idx[w]], y[idx[w]], 0.01)[0]
                                for si in range(len(seeds))])) for w in R.EVAL_WINS}
            print(f"  *{tag}       " + "  ".join(
                f"{w} {lf[w]:5.2f} ({lf[w]*base[w]*100:5.2f}%)" for w in R.EVAL_WINS)
                + "   규칙대비 " + "/".join(f"{lf[w]-rl[w]:+5.2f}" for w in R.EVAL_WINS), flush=True)
        ens = allsc.mean(axis=0)
        for w in R.EVAL_WINS:
            o, q95, pv = shift_null(ens[idx[w]], y[idx[w]], rng=rng)
            print(f"  [순환이동 귀무 {w}] 앙상블 {o:5.2f} vs q95 {q95:4.2f} p={pv:.3f}", flush=True)
        cov_table(allsc, y, idx, p, arms, rl)
        np.savez_compressed(ROOT / f"tmp/alert_tabpfn_20260911/scores_L{lead}_W{w_}_n{nest}.npz",
                            score=allsc.astype(np.float32), y=y,
                            **{f"idx_{w}": idx[w] for w in R.EVAL_WINS})
    return 0


# ────────────────────────── 경보 + 탐지기 결합 (사용자 질문: 탐지기가 보험이 되는가)
def _trail_q(x, q, mask=None, win=2016):
    """후행 분위. mask 가 있으면 **그 봉만 모아** 재고 비마스크 봉엔 직전 확정값을 쓴다
    (배포 탐지기 `_thr` 와 같은 규약). shift(1) 로 자기 봉을 안 본다."""
    if mask is None:
        return pd.Series(x).rolling(win, min_periods=200).quantile(q).shift(1).to_numpy()
    i = np.flatnonzero(mask & np.isfinite(x))
    if len(i) < 200:
        return np.full(len(x), np.inf)
    qs = pd.Series(x[i]).rolling(win, min_periods=200).quantile(q).shift(1).to_numpy()
    out = np.full(len(x), np.nan)
    out[i] = qs
    return pd.Series(out).ffill().to_numpy()


def joint_arm() -> int:
    """사건마다 «경보가 미리 잡았나 / 탐지기가 도중에 잡았나 / 둘 다 놓쳤나».

    ⭐핵심 수치는 **P(탐지 | 경보 놓침)** 이다. 이게 무조건부 탐지율과 비슷하면 두 시스템이
      독립이라 탐지기가 진짜 보험이고, 훨씬 낮으면 같은 사건에서 같이 실패하는 것이다.
    """
    lead, w_ = CANDS[1]
    z = np.load(ROOT / f"tmp/alert_tabpfn_20260911/hgb_scores_L{lead}_W{w_}.npz")
    sc, y = z["score"], z["y"]
    idx = {w: z[f"idx_{w}"] for w in R.EVAL_WINS}
    p = R.build()
    c = p["atr_pct"].to_numpy()                       # 길이 확인용
    assert len(p) == len(y), (len(p), len(y))
    lr = np.diff(np.log(pd.read_csv(R.KL5, usecols=["close"]).close.to_numpy(float)), prepend=0.0)
    S = pd.Series(lr)
    volexp = (S.rolling(12).std() / S.rolling(288).std()).to_numpy()
    comp = (volexp < 0.70) & np.isfinite(volexp)
    watch = pd.Series(comp).rolling(12, min_periods=1).max().to_numpy() == 1
    zq, zn = p["z_qv_288"].to_numpy(), p["z_n_288"].to_numpy()
    det = {
        "탐지 배포판(압축게이트)": watch & (zq >= _trail_q(zq, 0.90, comp))
                                      & (zn >= _trail_q(zn, 0.90, comp)),
        "탐지 게이트제거": (zq >= _trail_q(zq, 0.90)) & (zn >= _trail_q(zn, 0.90)),
    }
    for k, v in det.items():
        det[k] = np.where(np.isfinite(zq) & np.isfinite(zn), v, False)
        print(f"[{k}] 발동 {det[k].mean()*288:.1f}회/일")

    for w in R.EVAL_WINS:
        i = idx[w]
        yy = y[i]
        ev = np.flatnonzero(yy & ~np.r_[False, yy[:-1]])      # 사건 = 상승 엣지
        print(f"\n=== {w} · 사건 {len(ev)}건 · 선행 {lead*5}분 · 판정창 {w_*5}분 ===")
        # 탐지기는 **이동이 일어나는 동안**(t+lead ~ t+lead+w_) 켜지면 잡은 것으로 센다
        dcaught = {}
        for k, f in det.items():
            got = []
            for t in ev:
                a = i[t] + lead
                got.append(bool(f[a:a + w_ + 1].any()))
                                                            # 전역 인덱스로 창을 본다
            dcaught[k] = np.array(got)
            print(f"  {k:22s} 사건 포착 {dcaught[k].mean()*100:5.1f}%")
        print(f"  {'커버':>5s} {'경보포착':>8s} {'합집합':>8s} {'둘다놓침':>8s}"
              f"   {'P(탐지|경보놓침)':>16s}  {'P(탐지|경보잡음)':>16s}")
        for cv in (0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30):
            rows = []
            for si in range(len(sc)):
                x = sc[si][i]
                ok = np.isfinite(x)
                k_ = max(int(round(ok.sum() * cv)), 1)
                cut = np.partition(x[ok], -k_)[-k_]
                fire = ok & (x >= cut)
                acaught = np.array([bool(fire[t]) for t in ev])
                d = dcaught["탐지 게이트제거"]
                rows.append((acaught.mean(), (acaught | d).mean(), (~acaught & ~d).mean(),
                             d[~acaught].mean() if (~acaught).any() else np.nan,
                             d[acaught].mean() if acaught.any() else np.nan))
            m = np.nanmin(np.array(rows), axis=0)       # 시드최악
            print(f"  {cv*100:4.0f}% {m[0]*100:7.1f}% {m[1]*100:7.1f}% {m[2]*100:7.1f}%"
                  f"   {m[3]*100:15.1f}%  {m[4]*100:15.1f}%")
    return 0


# ────────────────────────── 탐지기 HGB 모델 (사용자 지시: 규칙만 있고 모델이 없었다)
DET_W = 3          # 판정창 15분. 탐지는 **선행 0** — «지금 가고 있나»를 본다


def detector_arm() -> int:
    """배포된 탐지 규칙(2종 AND, 게이트 제거판) vs HGB 모델.

    ⭐**발동 빈도를 맞춰** 겨룬다. 규칙이 창마다 실제로 켜지는 비율을 그대로 모델 커버리지로
      쓴다 — 더 자주 켜는 쪽이 그냥 많이 잡아 보이는 걸 막는다.
    ⭐탐지는 lift 만으로 못 판정한다. 같은 포착률이라도 **늦게** 켜지면 쓸모가 없다.
      그래서 사건 단위로 지연(봉)과 진행률(켜진 시점 이동폭 / 사건 전체 이동폭)을 같이 낸다.
    """
    global DET_W
    DET_W = int(_arg("--det-w", str(DET_W)))     # 판정창을 바꿔가며 본다(15/30/60분)
    p = R.build()
    feats = R.feature_cols(p)
    X = p[feats].to_numpy(np.float32)
    c = pd.read_csv(R.KL5, usecols=["close"]).close.to_numpy(float)
    logc = np.log(np.maximum(c, 1e-12))
    atr = p["atr_pct"].to_numpy()
    p["win"] = ""
    for nm, a_, b_ in R.WINDOWS:
        p.loc[(p.timestamp >= a_) & (p.timestamp <= b_), "win"] = nm
    _, mv = lead_move(logc, atr, 0, DET_W)                  # 선행 0 = 탐지
    ok = (np.isfinite(atr) & np.isfinite(mv) & (p.win != "").to_numpy()
          & np.isfinite(p[feats]).all(axis=1).to_numpy())
    y, _ = label_windows(p, mv, ok)
    emb = 2 * DET_W
    for nm, _, _ in R.WINDOWS:
        i = np.flatnonzero((p.win == nm).to_numpy())
        if len(i) > emb:
            ok[i[-emb:]] = False

    # 배포된 탐지 규칙 재현 — 게이트 제거판(2026-09-11 배포). 전 봉 후행 2016 분위 q90 AND.
    rule = np.ones(len(p), bool)
    for col in ("qv", "n"):
        x = p[f"z_{col}_288"].to_numpy(float)
        thr = pd.Series(x).rolling(2016, min_periods=200).quantile(0.90).shift(1).to_numpy()
        rule &= np.isfinite(x) & np.isfinite(thr) & (x >= thr)

    idx = {w: np.flatnonzero(ok & (p.win == w).to_numpy()) for w in R.EVAL_WINS}
    tr = np.flatnonzero(ok & (p.win == "TRAIN").to_numpy())
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(1, 10**6, NSEED).tolist()
    print(f"[탐지 모델] HGB · 무작위 시드 {seeds} · 피쳐 {len(feats)} · 판정창 {DET_W*5}분 · 선행 0")
    sc = np.zeros((len(seeds), len(p)))
    for si, sd in enumerate(seeds):
        sc[si] = R.hgb(int(sd)).fit(X[tr], y[tr]).predict_proba(X)[:, 1]
        print(f"  seed {sd} 학습 완료", flush=True)

    def ev_metrics(fire, i, yy):
        """사건 단위 포착률·지연·진행률. 사건 = 라벨의 상승 엣지, 창 = [t, t+DET_W]."""
        ev = np.flatnonzero(yy & ~np.r_[False, yy[:-1]])
        caught, delays, prog = 0, [], []
        for t in ev:
            hi = min(t + DET_W, len(i) - 1)
            w = np.flatnonzero(fire[t:hi + 1])
            if not len(w):
                continue
            caught += 1
            delays.append(int(w[0]))
            seg = c[i[t]:i[hi] + 1]
            tot = float(np.max(np.abs(seg - seg[0]))) if len(seg) else 0.0
            prog.append(abs(c[i[t + int(w[0])]] - c[i[t]]) / tot if tot > 0 else np.nan)
        return (caught / max(len(ev), 1), float(np.median(delays)) if delays else np.nan,
                float(np.nanmedian(prog)) if prog else np.nan, len(ev))

    print(f"\n  {'창':>4s} {'사건':>5s} {'팔':>10s} {'발동/일':>7s} {'정밀도':>7s} "
          f"{'사건포착':>8s} {'지연':>6s} {'진행률':>7s}")
    for w in R.EVAL_WINS:
        i = idx[w]
        yy = y[i]
        rf = rule[i]
        cov = float(rf.mean())                       # ⭐모델 커버리지를 규칙 발동률에 맞춘다
        rp = float(yy[rf].mean()) if rf.any() else np.nan
        rr = ev_metrics(rf, i, yy)
        print(f"  {w:>4s} {rr[3]:5d} {'규칙(배포)':>10s} {cov*288:7.1f} {rp*100:6.1f}% "
              f"{rr[0]*100:7.1f}% {rr[1]*5:5.0f}분 {rr[2]*100:6.1f}%")
        rows = []
        for si in range(len(seeds)):
            x = sc[si][i]
            k = max(int(round(np.isfinite(x).sum() * cov)), 1)
            cut = np.partition(x[np.isfinite(x)], -k)[-k]
            f = np.isfinite(x) & (x >= cut)
            m_ = ev_metrics(f, i, yy)
            rows.append((float(yy[f].mean()), m_[0], m_[1], m_[2]))
        a = np.array(rows)
        print(f"  {'':>4s} {'':>5s} {'HGB 최악':>10s} {cov*288:7.1f} {a[:, 0].min()*100:6.1f}% "
              f"{a[:, 1].min()*100:7.1f}% {a[:, 2].max()*5:5.0f}분 {a[:, 3].max()*100:6.1f}%")
        print(f"  {'':>4s} {'':>5s} {'HGB 평균':>10s} {cov*288:7.1f} {a[:, 0].mean()*100:6.1f}% "
              f"{a[:, 1].mean()*100:7.1f}% {a[:, 2].mean()*5:5.0f}분 {a[:, 3].mean()*100:6.1f}%")
        # ⭐반대 방향 맞춤 — **규칙과 같은 포착률**을 내려면 모델이 몇 번 울려야 하나.
        #   발동 빈도를 맞추면 «덜 잡는다»가 되고, 포착률을 맞추면 «더/덜 울린다»가 된다.
        #   둘 다 봐야 교체 판단이 선다.
        need = []
        for si in range(len(seeds)):
            x = sc[si][i]
            fin_ = np.isfinite(x)
            lo, hi = cov, 1.0
            for _ in range(24):                       # 이분 탐색: 목표 포착률을 내는 최소 커버리지
                mid = (lo + hi) / 2
                k = max(int(round(fin_.sum() * mid)), 1)
                cut = np.partition(x[fin_], -k)[-k]
                f = fin_ & (x >= cut)
                if ev_metrics(f, i, yy)[0] >= rr[0]:
                    hi = mid
                else:
                    lo = mid
            k = max(int(round(fin_.sum() * hi)), 1)
            cut = np.partition(x[fin_], -k)[-k]
            f = fin_ & (x >= cut)
            m_ = ev_metrics(f, i, yy)
            need.append((hi, float(yy[f].mean()), m_[0], m_[2]))
        nd = np.array(need)
        print(f"  {'':>4s} {'':>5s} {'HGB@포착맞춤':>10s} {nd[:, 0].max()*288:7.1f} "
              f"{nd[:, 1].min()*100:6.1f}% {nd[:, 2].min()*100:7.1f}% {'':>5s}  "
              f"진행률 {nd[:, 3].max()*100:5.1f}%  (규칙 포착 {rr[0]*100:.1f}% 를 내는 최소 발동)")
    return 0


# ────────── 탐지기 발동을 라벨로 쓰는 경보 (사용자 제안 2026-09-11)
def predetect_arm() -> int:
    """«30분 뒤 15분 안에 **탐지기가 켜지나**» 를 맞히는 모델.

    ⭐장점: 라벨이 촘촘하다(기저 ~24% vs 큰이동 5%) — 학습 신호가 5배다.
    🔴함정: 탐지기는 정답이 아니다. 큰 이동 기준 정밀도가 16~18% 라 발동의 80%+ 는
      큰 이동으로 안 이어진다. 탐지기를 라벨로 쓰면 **그 헛발동까지 배운다**.
      ⇒ 학습은 탐지기로 하되 **채점은 큰 이동으로** 한다. 두 지표를 나란히 찍는다.
    """
    lead, w_ = CANDS[1]                       # 선행 30분 · 판정창 15분
    p = R.build()
    feats = R.feature_cols(p)
    X = p[feats].to_numpy(np.float32)
    c = pd.read_csv(R.KL5, usecols=["close"]).close.to_numpy(float)
    logc = np.log(np.maximum(c, 1e-12))
    atr = p["atr_pct"].to_numpy()
    p["win"] = ""
    for nm, a_, b_ in R.WINDOWS:
        p.loc[(p.timestamp >= a_) & (p.timestamp <= b_), "win"] = nm

    # 배포된 탐지 규칙(게이트 제거판) 재현
    fire = np.ones(len(p), bool)
    for col in ("qv", "n"):
        x = p[f"z_{col}_288"].to_numpy(float)
        thr = pd.Series(x).rolling(2016, min_periods=200).quantile(0.90).shift(1).to_numpy()
        fire &= np.isfinite(x) & np.isfinite(thr) & (x >= thr)
    # 라벨 A(사용자 제안): (t+lead, t+lead+w_] 안에 탐지기가 한 번이라도 켜지나
    fwd_fire = pd.Series(fire[::-1]).rolling(w_, min_periods=1).max()[::-1].to_numpy().astype(bool)
    y_det = np.r_[fwd_fire[lead + 1:], np.zeros(lead + 1, bool)]
    # 라벨 B(현행 경보): 같은 창의 큰 이동 — **채점 기준**
    _, mv = lead_move(logc, atr, lead, lead + w_)
    ok = (np.isfinite(atr) & np.isfinite(mv) & (p.win != "").to_numpy()
          & np.isfinite(p[feats]).all(axis=1).to_numpy())
    y_mv, _ = label_windows(p, mv, ok)
    emb = 2 * (lead + w_)
    for nm, _, _ in R.WINDOWS:
        i = np.flatnonzero((p.win == nm).to_numpy())
        if len(i) > emb:
            ok[i[-emb:]] = False
    idx = {w: np.flatnonzero(ok & (p.win == w).to_numpy()) for w in R.EVAL_WINS}
    tr = np.flatnonzero(ok & (p.win == "TRAIN").to_numpy())
    print(f"[탐지기-라벨 경보] 선행 {lead*5}분 · 판정창 {w_*5}분 · TRAIN {len(tr):,}")
    print(f"  기저  탐지기라벨 {y_det[tr].mean()*100:5.2f}%"
          f"   큰이동라벨 {y_mv[tr].mean()*100:5.2f}%")
    # ⭐라벨 자명성 — 최고 단변량이 높을수록 «자명한» 라벨이다(저장소 규율)
    for tag, yy in (("탐지기라벨", y_det), ("큰이동라벨", y_mv)):
        i = idx["VAL"]
        best = max((float(np.nanmax([R.lift_at(p[f].to_numpy(float)[i], yy[i], 0.10)[0],
                                     R.lift_at(-p[f].to_numpy(float)[i], yy[i], 0.10)[0]])), f)
                   for f in feats)
        print(f"  {tag} 최고 단변량(cov 10%) lift {best[0]:.2f} ({best[1]})")

    rng = np.random.default_rng(SEED)
    seeds = rng.integers(1, 10**6, NSEED).tolist()
    res = {}
    for tag, ytr in (("A 탐지기로 학습", y_det), ("B 큰이동으로 학습", y_mv)):
        sc = np.zeros((len(seeds), len(p)))
        for si, sd in enumerate(seeds):
            sc[si] = R.hgb(int(sd)).fit(X[tr], ytr[tr]).predict_proba(X)[:, 1]
        res[tag] = sc
        print(f"  {tag} 학습 완료", flush=True)

    print(f"\n  {'팔':>16s} " + "  ".join(f"{w:^26s}" for w in R.EVAL_WINS))
    print(f"  {'':>16s} " + "  ".join(f"{'탐지기적중':>10s} {'큰이동정밀':>14s}" for _ in R.EVAL_WINS))
    for tag, sc in res.items():
        cells = []
        for w in R.EVAL_WINS:
            i = idx[w]
            a = []
            for si in range(len(seeds)):
                x = sc[si][i]
                k = max(int(round(np.isfinite(x).sum() * 0.10)), 1)
                cut = np.partition(x[np.isfinite(x)], -k)[-k]
                f = np.isfinite(x) & (x >= cut)
                a.append((float(y_det[i][f].mean()), float(y_mv[i][f].mean())))
            a = np.array(a)
            cells.append(f"{a[:, 0].min()*100:9.1f}% {a[:, 1].min()*100:13.1f}%")
        print(f"  {tag:>16s} " + "  ".join(cells))
    for w in R.EVAL_WINS:
        i = idx[w]
        print(f"  (기저 {w}) 탐지기 {y_det[i].mean()*100:.1f}% · 큰이동 {y_mv[i].mean()*100:.1f}%",
              end="   ")
    print()
    return 0


# ────────── «앞으로 N분 이내에 탐지기가 발동하나» (사용자 지시 2026-09-11, 누적창)
def prewarn_arm() -> int:
    """라벨 = (t, t+H] **안에 한 번이라도** 탐지 발동. H = 6봉(30분) · 12봉(1시간).

    앞판은 고정 슬라이스((t+lead, t+lead+3])였다. 사용자가 «N분 이내»로 바꾸라고 해서
    누적창으로 고쳤다 — 결정 시점 바로 다음 봉부터 센다(피쳐는 t 까지, 라벨은 t+1 부터).
    ⚠️누적이라 **기저가 크게 오른다**. 정밀도를 기저와 나란히 봐야 한다.
    ⭐«N분 이내»는 실제 리드타임을 숨긴다 — 경보가 몇 분 전에 울렸는지 중앙값을 같이 낸다.
    """
    p = R.build()
    feats = R.feature_cols(p)
    X = p[feats].to_numpy(np.float32)
    p["win"] = ""
    for nm, a_, b_ in R.WINDOWS:
        p.loc[(p.timestamp >= a_) & (p.timestamp <= b_), "win"] = nm
    fire = np.ones(len(p), bool)
    for col in ("qv", "n"):
        x = p[f"z_{col}_288"].to_numpy(float)
        thr = pd.Series(x).rolling(2016, min_periods=200).quantile(0.90).shift(1).to_numpy()
        fire &= np.isfinite(x) & np.isfinite(thr) & (x >= thr)
    ev_all = fire & ~np.r_[False, fire[:-1]]                 # 발동 묶음의 시작
    base_ok = np.isfinite(p[feats]).all(axis=1).to_numpy() & (p.win != "").to_numpy()
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(1, 10**6, NSEED).tolist()

    for H in (6, 12):
        fwd = pd.Series(fire[::-1]).rolling(H, min_periods=1).max()[::-1].to_numpy().astype(bool)
        y = np.r_[fwd[1:], False]                            # (t, t+H] 안에 발동
        ok = base_ok.copy()
        for nm, _, _ in R.WINDOWS:
            i2 = np.flatnonzero((p.win == nm).to_numpy())
            if len(i2) > 2 * H:
                ok[i2[-2 * H:]] = False
        idx = {w: np.flatnonzero(ok & (p.win == w).to_numpy()) for w in R.EVAL_WINS}
        tr = np.flatnonzero(ok & (p.win == "TRAIN").to_numpy())
        sc = np.zeros((len(seeds), len(p)))
        for si, sd in enumerate(seeds):
            sc[si] = R.hgb(int(sd)).fit(X[tr], y[tr]).predict_proba(X)[:, 1]
        print(f"\n=== 앞으로 {H*5}분 이내에 탐지 발동 ===  TRAIN {len(tr):,}", flush=True)
        print("  " + " · ".join(f"{w} 기저 {y[idx[w]].mean()*100:.1f}% "
                                f"(발동묶음 {int(ev_all[idx[w]].sum())}건)" for w in R.EVAL_WINS))
        print(f"  {'커버':>5s} {'경보/일':>7s}  " + "  ".join(f"{w:^30s}" for w in R.EVAL_WINS))
        print(f"  {'':>5s} {'':>7s}  " + "  ".join(
            f"{'정밀도':>7s} {'묶음예고':>8s} {'리드중앙':>10s}" for _ in R.EVAL_WINS))
        for cv in (0.05, 0.10, 0.15, 0.20, 0.30):
            cells = []
            for w in R.EVAL_WINS:
                i2 = idx[w]
                yy = y[i2]
                ev = np.flatnonzero(ev_all[i2])
                per = []
                for si in range(len(seeds)):
                    x = sc[si][i2]
                    fin = np.isfinite(x)
                    k = max(int(round(fin.sum() * cv)), 1)
                    cut = np.partition(x[fin], -k)[-k]
                    f = fin & (x >= cut)
                    leads = []
                    for e in ev:
                        lo = max(e - H, 0)
                        w_ = np.flatnonzero(f[lo:e])          # 창 안 경보들
                        if len(w_):
                            leads.append(e - (lo + int(w_[0])))   # **가장 이른** 경보 기준
                    per.append((float(yy[f].mean()), len(leads) / max(len(ev), 1),
                                float(np.median(leads)) if leads else np.nan))
                a = np.array(per)
                cells.append(f"{a[:, 0].min()*100:6.1f}% {a[:, 1].min()*100:7.1f}% "
                             f"{np.nanmin(a[:, 2])*5:8.0f}분")
            print(f"  {cv*100:4.0f}% {cv*288:7.1f}  " + "  ".join(cells), flush=True)
    return 0


# ────────── 라그/변화율 피쳐 + **누수 검사 우선** (사용자 지시 2026-09-11)
LAG_BASE = ("z_n_96", "z_n_288", "z_qv_96", "z_qv_288")


def lag_feats(p: pd.DataFrame) -> pd.DataFrame:
    """같은 피쳐의 **과거값과의 차이**. 전부 뒤만 본다 — shift(k) 는 t-k 를 가져온다.

    지금 33개 중 시간 변화를 담은 건 7개뿐이라(volexp_d1/d12 · min3 4개 · comp_age)
    «체결이 달아오르는 중인가» 라는 방향성이 거의 안 들어간다. 그걸 직접 넣는다.
    """
    F = {}
    for b in LAG_BASE:
        x = p[b].astype(float)
        for k in (1, 3, 12):
            F[f"{b}_d{k}"] = x - x.shift(k)          # t 와 t-k 만 쓴다
        # 12봉 회귀 기울기 — 닫힌형(공분산/분산). rolling 은 t-11..t 만 본다.
        t_ = np.arange(12, dtype=float)
        tc = t_ - t_.mean()
        den = float((tc ** 2).sum())
        F[f"{b}_slope12"] = x.rolling(12).apply(
            lambda v, tc=tc, den=den: float(np.dot(v - v.mean(), tc) / den), raw=True)
    return pd.DataFrame(F, index=p.index)


def lag_arm() -> int:
    """⭐누수 검사를 **먼저** 통과해야 성능을 잰다(사용자 지시)."""
    p = R.build()
    base_f = R.feature_cols(p)
    lf = lag_feats(p)
    for c_ in lf.columns:
        p[c_] = lf[c_].to_numpy()
    new_f = list(lf.columns)
    allf = base_f + new_f
    print(f"[누수 검사] 기존 {len(base_f)} + 신규 {len(new_f)} = {len(allf)}개")

    # ── 검사 1: 절단 불변성. 봉 t 의 피쳐는 t 이후 데이터가 없어도 **같아야** 한다.
    rng = np.random.default_rng(SEED)
    idx_test = rng.choice(np.arange(300_000, len(p) - 100), 3, replace=False)
    bad = 0
    for t in idx_test:
        cut = p.iloc[:t + 1].copy()
        lf_cut = lag_feats(cut)
        for c_ in new_f:
            a, b = float(lf_cut[c_].iloc[-1]), float(p[c_].iloc[t])
            if not (np.isnan(a) and np.isnan(b)) and abs(a - b) > 1e-9:
                print(f"  🔴절단 불변성 위반 t={t} {c_}: 절단 {a} vs 전체 {b}")
                bad += 1
    print(f"  검사1 절단 불변성 : {'통과' if not bad else 'FAIL'} "
          f"(3지점 x {len(new_f)}피쳐 = {3*len(new_f)}칸)")

    # ── 검사 2: 명시적 미래 변조. t 이후를 난수로 덮어도 t 의 값이 안 변해야 한다.
    q = p.copy()
    t0 = int(idx_test[0])
    for b in LAG_BASE:
        v = q[b].to_numpy(float).copy()
        v[t0 + 1:] = rng.normal(0, 5, len(v) - t0 - 1)
        q[b] = v
    lf_q = lag_feats(q)
    bad2 = sum(1 for c_ in new_f
               if not np.allclose(lf_q[c_].iloc[t0], p[c_].iloc[t0], equal_nan=True, atol=1e-9))
    print(f"  검사2 미래 변조 불변: {'통과' if not bad2 else f'FAIL ({bad2}개 변함)'}")

    # ── 검사 3: 단변량 상한. 라벨(30분 이내 탐지 발동)에 대해 AUC>=0.95 면 누수다.
    fire = np.ones(len(p), bool)
    for col in ("qv", "n"):
        x = p[f"z_{col}_288"].to_numpy(float)
        thr = pd.Series(x).rolling(2016, min_periods=200).quantile(0.90).shift(1).to_numpy()
        fire &= np.isfinite(x) & np.isfinite(thr) & (x >= thr)
    fwd = pd.Series(fire[::-1]).rolling(6, min_periods=1).max()[::-1].to_numpy().astype(bool)
    y = np.r_[fwd[1:], False]
    p["win"] = ""
    for nm, a_, b_ in R.WINDOWS:
        p.loc[(p.timestamp >= a_) & (p.timestamp <= b_), "win"] = nm
    ok = np.isfinite(p[allf]).all(axis=1).to_numpy() & (p.win != "").to_numpy()
    for nm, _, _ in R.WINDOWS:
        i2 = np.flatnonzero((p.win == nm).to_numpy())
        if len(i2) > 12:
            ok[i2[-12:]] = False
    iv = np.flatnonzero(ok & (p.win == "VAL").to_numpy())
    worst = max((abs(R.auc(p[f].to_numpy(float)[iv], y[iv]) - 0.5) + 0.5, f) for f in new_f)
    print(f"  검사3 단변량 상한  : 최고 |AUC| {worst[0]:.4f} ({worst[1]}) "
          f"{'통과' if worst[0] < 0.95 else '🔴FAIL'}")
    # ── 검사 4: 라벨 경계. 피쳐는 t 까지, 라벨은 t+1 부터.
    assert not np.array_equal(y, fire), "라벨이 자기 봉을 그대로 쓰고 있다"
    print(f"  검사4 라벨 경계    : 통과 (피쳐 <= t, 라벨 (t, t+6])")
    if bad or bad2 or worst[0] >= 0.95:
        print("\n🔴누수 검사 실패 — 성능 측정을 하지 않는다.")
        return 1

    # ── 통과했으므로 성능 비교
    idx = {w: np.flatnonzero(ok & (p.win == w).to_numpy()) for w in R.EVAL_WINS}
    tr = np.flatnonzero(ok & (p.win == "TRAIN").to_numpy())
    seeds = np.random.default_rng(SEED).integers(1, 10**6, NSEED).tolist()
    print(f"\n[성능] 라벨 = 30분 이내 탐지 발동 · TRAIN {len(tr):,} · 시드 {len(seeds)}개")
    print("  " + " · ".join(f"{w} 기저 {y[idx[w]].mean()*100:.1f}%" for w in R.EVAL_WINS))
    for tag, cols in (("기존 33", base_f), (f"기존+라그 {len(allf)}", allf)):
        Xs = p[cols].to_numpy(np.float32)
        out = []
        for sd in seeds:
            sc = R.hgb(int(sd)).fit(Xs[tr], y[tr]).predict_proba(Xs)[:, 1]
            row = []
            for w in R.EVAL_WINS:
                i2 = idx[w]
                x = sc[i2]
                k = max(int(round(len(x) * 0.10)), 1)
                cut = np.partition(x, -k)[-k]
                row.append(float(y[i2][x >= cut].mean()))
            out.append(row)
        a = np.array(out)
        print(f"  {tag:16s} " + "  ".join(
            f"{w} {a[:, k].min()*100:5.1f}%" for k, w in enumerate(R.EVAL_WINS))
            + "   [커버 10% 정밀도 · 시드최악]", flush=True)
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    elif "--model" in sys.argv:
        raise SystemExit(model_arm())
    elif "--features" in sys.argv:
        raise SystemExit(feature_arm())
    elif "--axes" in sys.argv:
        raise SystemExit(axes_arm())
    elif "--tabpfn" in sys.argv:
        raise SystemExit(tabpfn_arm())
    elif "--joint" in sys.argv:
        raise SystemExit(joint_arm())
    elif "--detector" in sys.argv:
        raise SystemExit(detector_arm())
    elif "--predetect" in sys.argv:
        raise SystemExit(predetect_arm())
    elif "--prewarn" in sys.argv:
        raise SystemExit(prewarn_arm())
    elif "--lag" in sys.argv:
        raise SystemExit(lag_arm())
    else:
        raise SystemExit(main())
