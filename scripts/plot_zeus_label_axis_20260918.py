#!/usr/bin/env python3
"""Zeus 라벨 축 차트 — 사람이 보는 용도 (2026-09-18).

--kind trades  v3/v4 체결 원장을 VAL·OOS·HOLDOUT 각 «첫 월~일» 주간에 얹는다
       label   지그재그 «정답» 라벨 + v4 진입의 라벨 일치/불일치
       soft    soft-argmax 라벨 — 하드에서 CASH 로 «내려간» 봉을 빗금으로
       tuned   지그재그 현행 1.0% vs 조정 1.6% + 같은 주간의 더블배리어 라벨

🔴창 매핑 주의: 저장소 기본 VAL/OOS 날짜가 Zeus 폴드 격자와 안 맞는다.
   VAL 2025-09~12 ⊂ F5 · OOS 2026-01~03 중 **3월만** CAND TEST(1~2월은 CAND 의 TRAIN) ·
   HOLDOUT 2026-07-01~08-19 = SHADOW. 주간은 각 창의 첫 월~일을 기계적으로 고른다.
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_zeus_fold_replay_20260918 as R           # noqa: E402
import build_wave3_action_labels_20260531 as W           # noqa: E402

KF = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KF.exists():
    fm.fontManager.addfont(str(KF)); plt.rcParams["font.family"] = fm.FontProperties(fname=str(KF)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 12, "axes.titlesize": 13.5,
                     "figure.facecolor": "white", "axes.grid": True, "grid.alpha": .25})
G, RD, INK, MUT, OR, BL = "#1F9D55", "#D94141", "#1F2430", "#9AA0A6", "#E08A1E", "#2A5DB0"
WIN = [("VAL", "F5", "2025-09-01", "2025-09-07"), ("OOS", "CAND", "2026-03-02", "2026-03-08"),
       ("HOLDOUT", "SHADOW", "2026-07-06", "2026-07-12")]
QV = {"v3": 0.98, "v4": 0.85}       # v3 는 건수맞춤이라 절대값이 없다 -- 공표 0.78건/일로 역산
OUT = ROOT / "tmp"


def _win(a, b):
    return pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(hours=23, minutes=55)


def _labels():
    d = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(R.BASE / "zigzag_labels_full/*.csv")))])
    d["timestamp"] = pd.to_datetime(d.timestamp).dt.tz_localize(None)
    return d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _save(fig, name, legend, title):
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False,
               bbox_to_anchor=(.5, .004), fontsize=12)
    fig.suptitle(title, fontsize=18, y=.985)
    fig.tight_layout(rect=[0, .035, 1, .975])
    p = OUT / name; fig.savefig(p, dpi=105); print("저장", p)


def kind_trades(df):
    LED = {(v, f): R.trades(df, f, v, QV[v])[1:] for v in ("v3", "v4") for _, f, *_ in WIN}
    fig, AX = plt.subplots(3, 2, figsize=(22, 15))
    for r, (tag, fold, a, b) in enumerate(WIN):
        for c, ver in enumerate(("v3", "v4")):
            ax = AX[r, c]; tr, _ = LED[(ver, fold)]
            te, _ = R.candidates(df, fold, ver, QV[ver])
            t0, t1 = _win(a, b); ts = te.timestamp.to_numpy()
            cl = pd.to_numeric(te.close).to_numpy()
            w = (te.timestamp >= t0) & (te.timestamp <= t1)
            ax.plot(ts[w], cl[w], lw=1.0, color=INK, alpha=.55, zorder=1)
            sub = tr[(tr.ts >= t0) & (tr.ts <= t1)]
            for _, x in sub.iterrows():
                xi = min(int(x.exit_i), len(ts) - 1); col = G if x.bp > 0 else RD
                ax.plot([x.ts, ts[xi]], [x.entry, cl[xi]], color=col, lw=2.6, alpha=.85, zorder=3)
                ax.scatter([x.ts], [x.entry], marker="^" if x.side > 0 else "v", s=190,
                           color="white", edgecolor=col, linewidth=2.2, zorder=4)
                ax.scatter([ts[xi]], [cl[xi]], marker="o", s=70, color=col, zorder=4)
                ax.annotate(f"{x.bp:+.0f}", (ts[xi], cl[xi]), textcoords="offset points",
                            xytext=(6, 8 if x.bp > 0 else -16), fontsize=10, color=col, weight="bold")
            n = len(sub)
            ax.set_title(f"{tag} · {ver.upper()} · {a}~{b}   —   체결 {n}건 · "
                         f"합계 {sub.bp.sum() if n else 0:+.0f}bp · 승 {(sub.bp>0).mean()*100 if n else 0:.0f}%")
            ax.set_ylabel("ETHUSDT")
    _save(fig, "zeus_v3v4_week_20260918.png",
          [Line2D([], [], marker="^", color="w", mec=INK, mew=2, ms=13, ls="", label="롱 진입"),
           Line2D([], [], marker="v", color="w", mec=INK, mew=2, ms=13, ls="", label="숏 진입"),
           Line2D([], [], color=G, lw=3, label=f"이익 청산(TP {R.TP*100:g}%)"),
           Line2D([], [], color=RD, lw=3, label=f"손실 청산(SL {R.SL*100:g}%)")],
          f"Zeus Baseline v3 vs v4 — 주간 거래차트 (TP{R.TP*100:g}%/SL{R.SL*100:g}% · 1슬롯 순차 · 비용 0)")


def kind_label(df):
    lab = _labels()
    fig, AX = plt.subplots(3, 1, figsize=(20, 15))
    for r, (tag, fold, a, b) in enumerate(WIN):
        ax = AX[r]; t0, t1 = _win(a, b)
        w = lab[(lab.timestamp >= t0) & (lab.timestamp <= t1)].reset_index(drop=True)
        ts, cl, hi, lo = (w.timestamp.to_numpy(), w.close.to_numpy(), w.high.to_numpy(), w.low.to_numpy())
        act = w.zigzag_action.to_numpy()
        cut = np.where(np.diff(act) != 0)[0] + 1
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(act)]):
            ax.axvspan(ts[s], ts[min(e, len(ts) - 1)], color={1: G, 2: RD, 0: MUT}[act[s]],
                       alpha=.13, lw=0, zorder=0)
        ax.plot(ts, cl, lw=1.1, color=INK, alpha=.7, zorder=2)
        seg = w.zigzag_segment_id.to_numpy()
        bnd = np.r_[0, np.where(np.diff(seg) != 0)[0] + 1, len(seg) - 1]
        px = [lo[i] if act[min(i, len(act)-1)] == 1 else hi[i] if act[min(i, len(act)-1)] == 2 else cl[i]
              for i in bnd]
        ax.plot(ts[bnd], px, color=BL, lw=2.4, marker="o", ms=8, zorder=5, alpha=.9)
        _, tr, _ = R.trades(df, fold, "v4")
        sub = tr[(tr.ts >= t0) & (tr.ts <= t1)]
        A = dict(zip(w.timestamp.to_numpy(), act)); ok = 0
        for _, x in sub.iterrows():
            la = A.get(np.datetime64(pd.Timestamp(x.ts)), 0)
            hit = (x.side > 0 and la == 1) or (x.side < 0 and la == 2); ok += hit
            ax.scatter([x.ts], [x.entry], marker="^" if x.side > 0 else "v", s=200,
                       color="white" if hit else "#FFE2E2", edgecolor=INK if hit else RD,
                       linewidth=2.4 if hit else 2.8, zorder=6)
        sh = pd.Series(act).value_counts(normalize=True)
        ax.set_title(f"{tag} · {a}~{b}   —   라벨 LONG {sh.get(1,0)*100:.0f}% / SHORT {sh.get(2,0)*100:.0f}% "
                     f"/ CASH {sh.get(0,0)*100:.0f}%   ·   파동 {len(bnd)-1}개   ·   "
                     f"v4 진입 {len(sub)}건 중 라벨일치 {ok}건({ok/max(len(sub),1)*100:.0f}%)")
        ax.set_ylabel("ETHUSDT"); ax.set_xlim(ts[0], ts[-1])
    _save(fig, "zeus_zigzag_label_week_20260918.png",
          [Patch(color=G, alpha=.3, label="라벨 LONG"), Patch(color=RD, alpha=.3, label="라벨 SHORT"),
           Patch(color=MUT, alpha=.3, label="라벨 CASH"),
           Line2D([], [], color=BL, lw=2.4, marker="o", ms=8, label="지그재그 파동(꼭짓점=전환)"),
           Line2D([], [], marker="^", color="w", mec=INK, mew=2.4, ms=13, ls="", label="v4 진입 — 라벨 일치"),
           Line2D([], [], marker="^", color="#FFE2E2", mec=RD, mew=2.8, ms=13, ls="", label="v4 진입 — 불일치")],
          "지그재그 «정답» 라벨 — v3/v4 차트와 같은 3주간 (라벨이 곧 학습 타깃)")


def kind_soft(df):
    d = _labels()
    d["soft"] = d[["zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"]].to_numpy().argmax(1)
    fig, AX = plt.subplots(3, 1, figsize=(20, 15))
    for r, (tag, _f, a, b) in enumerate(WIN):
        ax = AX[r]; t0, t1 = _win(a, b)
        w = d[(d.timestamp >= t0) & (d.timestamp <= t1)].reset_index(drop=True)
        ts, cl = w.timestamp.to_numpy(), w.close.to_numpy()
        sf, hd = w.soft.to_numpy(), w.zigzag_action.to_numpy()
        cut = np.where(np.diff(sf) != 0)[0] + 1
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(sf)]):
            ax.axvspan(ts[s], ts[min(e, len(ts) - 1)], color={1: G, 2: RD, 0: MUT}[sf[s]],
                       alpha=.15, lw=0, zorder=0)
        down = (hd != 0) & (sf == 0)                      # 하드=방향 -> soft=CASH 로 내려간 봉
        dc = np.where(np.diff(down.astype(int)) != 0)[0] + 1
        for s, e in zip(np.r_[0, dc], np.r_[dc, len(down)]):
            if down[s]:
                ax.axvspan(ts[s], ts[min(e, len(ts) - 1)], facecolor="none", edgecolor=OR,
                           hatch="///", lw=0, alpha=.9, zorder=1)
        ax.plot(ts, cl, lw=1.2, color=INK, alpha=.75, zorder=3)
        seg = w.zigzag_segment_id.to_numpy()
        bnd = np.r_[0, np.where(np.diff(seg) != 0)[0] + 1, len(seg) - 1]
        ax.plot(ts[bnd], cl[bnd], color=BL, lw=1.6, marker="o", ms=6, alpha=.8, zorder=4)
        sh = pd.Series(sf).value_counts(normalize=True)
        ax.set_title(f"{tag} · {a}~{b}   —   soft LONG {sh.get(1,0)*100:.0f}% / SHORT {sh.get(2,0)*100:.0f}% "
                     f"/ CASH {sh.get(0,0)*100:.0f}%   ·   내려간 봉 {down.sum()}개 "
                     f"({down.sum()/max((hd!=0).sum(),1)*100:.0f}% of 방향봉)")
        ax.set_ylabel("ETHUSDT"); ax.set_xlim(ts[0], ts[-1])
    _save(fig, "zeus_softargmax_label_week_20260918.png",
          [Patch(color=G, alpha=.35, label="soft LONG"), Patch(color=RD, alpha=.35, label="soft SHORT"),
           Patch(color=MUT, alpha=.35, label="soft CASH"),
           Patch(facecolor="white", edgecolor=OR, hatch="///", label="하드=방향 → soft=CASH 로 내려간 봉"),
           Line2D([], [], color=BL, lw=1.6, marker="o", ms=6, label="지그재그 꼭짓점")],
          "soft-argmax 지그재그 라벨 — 파동 끝자락·작은 파동을 CASH 로 내린다 (같은 3주간)")


def kind_tuned(df):
    import research_zeus_label_axis_20260918 as LA
    db = pd.read_parquet(R.BASE / "zeus_double_barrier_labels_20260917/zeus_db_tp15_sl10_usdc3x_3.06bp.parquet",
                         columns=["timestamp", "tb_action", "tb_quality"])
    db["timestamp"] = pd.to_datetime(db.timestamp).dt.tz_localize(None)
    te = R.panel(); TS = te.timestamp.to_numpy(); CL = pd.to_numeric(te.close).to_numpy(float)
    PV = {p: LA.waves(te, p) for p in (0.010, 0.016)}
    fig, AX = plt.subplots(3, 2, figsize=(22, 15))
    for r, (tag, _f, a, b) in enumerate(WIN):
        t0, t1 = _win(a, b); w = np.where((te.timestamp >= t0) & (te.timestamp <= t1))[0]
        ax = AX[r, 0]; ax.plot(TS[w], CL[w], lw=1.0, color=INK, alpha=.5, zorder=2)
        for pct, col, lw, lb in ((0.010, MUT, 1.6, "현행 1.0%"), (0.016, BL, 2.8, "조정 1.6%")):
            ent, _ = PV[pct]; k = (ent >= w[0]) & (ent <= w[-1])
            ax.plot(TS[ent[k]], CL[ent[k]], color=col, lw=lw, marker="o",
                    ms=6 if pct > .012 else 4, zorder=4 if pct > .012 else 3, label=lb)
        n0 = int(((PV[0.010][0] >= w[0]) & (PV[0.010][0] <= w[-1])).sum())
        n1 = int(((PV[0.016][0] >= w[0]) & (PV[0.016][0] <= w[-1])).sum())
        ax.set_title(f"{tag} · 지그재그 파동 — 현행 1.0% {n0}개 vs 조정 1.6% {n1}개 ({a}~{b})")
        ax.set_ylabel("ETHUSDT"); ax.legend(loc="best", fontsize=11); ax.set_xlim(TS[w[0]], TS[w[-1]])
        ax = AX[r, 1]
        dd = db[(db.timestamp >= t0) & (db.timestamp <= t1)].reset_index(drop=True)
        act, ts2 = dd.tb_action.to_numpy(), dd.timestamp.to_numpy()
        cut = np.where(np.diff(act) != 0)[0] + 1
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(act)]):
            ax.axvspan(ts2[s], ts2[min(e, len(ts2) - 1)], color={1: G, 2: RD, 0: MUT}[act[s]],
                       alpha=.16, lw=0, zorder=0)
        ax.plot(TS[w], CL[w], lw=1.1, color=INK, alpha=.7, zorder=2)
        ax2 = ax.twinx(); ax2.plot(ts2, dd.tb_quality, lw=1.3, color="#8A5CF6", alpha=.85)
        ax2.set_ylabel("tb_quality", color="#8A5CF6"); ax2.grid(False)
        sh = pd.Series(act).value_counts(normalize=True)
        ax.set_title(f"{tag} · 더블배리어(TP1.5%/SL1.0%) — LONG {sh.get(1,0)*100:.0f}% / "
                     f"SHORT {sh.get(2,0)*100:.0f}% / CASH {sh.get(0,0)*100:.0f}% · 전환 "
                     f"{int((np.diff(act)!=0).sum())}회")
        ax.set_xlim(TS[w[0]], TS[w[-1]])
    _save(fig, "zeus_tuned_zigzag_vs_db_20260918.png",
          [Line2D([], [], color=MUT, lw=1.6, marker="o", ms=4, label="지그재그 현행(1.0%)"),
           Line2D([], [], color=BL, lw=2.8, marker="o", ms=6, label="지그재그 조정(1.6%)"),
           Patch(color=G, alpha=.35, label="더블배리어 LONG"), Patch(color=RD, alpha=.35, label="더블배리어 SHORT"),
           Patch(color=MUT, alpha=.35, label="더블배리어 CASH"),
           Line2D([], [], color="#8A5CF6", lw=1.6, label="tb_quality")],
          "조정 지그재그(반전 1.6%) vs 더블배리어 라벨 — 같은 3주간")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="trades", choices=["trades", "label", "soft", "tuned"])
    a = ap.parse_args()
    df = R.panel()
    {"trades": kind_trades, "label": kind_label, "soft": kind_soft, "tuned": kind_tuned}[a.kind](df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
