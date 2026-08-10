"""Charts for the lag audit of the frozen theta=0.5% classifier (2026-08-08).

Three panels: (1) lag profile -- agreement vs oracle shifted by k bars, frozen vs czz05;
(2) agreement by quintile of position inside the oracle wave; (3) ablation -- which feature
block actually carries the edge.  Reads tmp/regime_theta005_20260808/lag_audit.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "tmp/regime_theta005_20260808"
INK, C_A, C_B, C_C = "#1F2430", "#2563EB", "#D9542B", "#0E7C66"

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def main() -> int:
    d = json.loads((OUT_DIR / "lag_audit.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    for det, color, label in (("frozen", C_A, "고정 앙상블"), ("czz05", C_B, "czz05 (기계적 지연)")):
        prof = d["oos_2026Q1"]["A_lag_profile"][det]["profile"]
        ks = sorted(int(k) for k in prof if prof[k] is not None)
        vs = [prof[str(k)] if str(k) in prof else prof[k] for k in ks]
        ax.plot(ks, vs, color=color, linewidth=1.8, label=label)
        pk = d["oos_2026Q1"]["A_lag_profile"][det]["peak_lag_bars"]
        ax.plot(pk, d["oos_2026Q1"]["A_lag_profile"][det]["peak_agreement_pct"],
                marker="o", markersize=6, color=color)
    ax.axvline(0, color="#9AA0A6", linewidth=1.0, linestyle="--")
    ax.set_title("① 지연 프로파일 (OOS)\n오라클을 k바 시프트했을 때 일치율", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("k (바) — 양수 = 오라클을 과거로 밀었을 때", fontsize=9)
    ax.set_ylabel("일치율 %", fontsize=9)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    qs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    fr = [d["oos_2026Q1"]["C_by_wave_quintile"]["frozen"][q] for q in qs]
    cz = [d["oos_2026Q1"]["C_by_wave_quintile"]["czz05"][q] for q in qs]
    xs = np.arange(5)
    ax.bar(xs - 0.19, fr, width=0.36, color=C_A, label="고정 앙상블")
    ax.bar(xs + 0.19, cz, width=0.36, color=C_B, label="czz05")
    ax.axhline(50, color="#9AA0A6", linewidth=1.0, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(["파동\n초반", "Q2", "Q3", "Q4", "파동\n후반"], fontsize=9)
    ax.set_title("② 파동 내 위치별 일치율 (OOS)\n지연 복사면 Q1이 0 근처여야 함", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    abl = d["E_ablation_4seed"]
    names = list(abl)
    ko = {"full (panel+czz)": "패널+지그재그\n(현 고정안)", "panel_only (no czz)": "패널만\n(지그재그 제거)",
          "czz_only": "지그재그만\n(패널 제거)"}
    xs = np.arange(len(names))
    ax.bar(xs - 0.19, [abl[n]["val"] for n in names], width=0.36, color=C_C, label="VAL")
    ax.bar(xs + 0.19, [abl[n]["oos"] for n in names], width=0.36, color=C_A, label="OOS")
    ax.axhline(60.8, color=C_B, linewidth=1.2, linestyle="--")
    ax.annotate("czz05 기계적 지연 OOS 60.8", xy=(len(names) - 0.5, 61.1), fontsize=8, color=C_B, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([ko[n] for n in names], fontsize=9)
    ax.set_ylim(55, 72)
    ax.set_title("③ Ablation — 엣지는 어디서 오는가\n(4시드, 진단용)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)

    for a in axes:
        a.grid(axis="y", color="#000000", alpha=0.08, linewidth=0.8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle("고정 분류기는 오라클의 지연 복사인가? — 세 각도 검사", fontsize=13, y=1.02)
    out = OUT_DIR / "lag_audit.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
