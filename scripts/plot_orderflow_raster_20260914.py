"""**호가 히트맵 육안 점검** — 래스터가 실제로 무엇을 담고 있는지 본다 (2026-09-14).

P0 종료 기준의 마지막 칸: UI 를 한 줄도 쓰기 전에 수집된 바이트를 그림으로 확인한다.
데이터를 안 보고 캔버스부터 만들면, 나중에 화면이 이상할 때 수집기 탓인지 렌더러 탓인지
가를 수 없다.

수집기의 read_window() 를 **그대로** 부른다(재구현 금지 -- 대시보드 엔드포인트가 P2 에서
쓸 바로 그 함수를 여기서 먼저 검증하는 게 목적이다).

  python scripts/plot_orderflow_raster_20260914.py --minutes 30 --out tmp/raster.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_korean_font_20260909 import use_korean_font  # noqa: E402
from scripts.live_orderflow_raster_collector_20260914 import SYMBOL, read_window  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--agg", type=int, default=1, help="초를 이만큼 접는다(가로 축소)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="tmp/orderflow_raster.png")
    a = ap.parse_args()
    use_korean_font()

    cols = int(a.minutes * 60 / a.agg)
    w = read_window(a.symbol, int(time.time() * 1000), cols=cols, agg=a.agg,
                    root=Path(a.root) if a.root else None)
    if w["n_bins"] == 0:
        print("유효 행이 없다 — 수집기가 돌고 있는지, --root 가 맞는지 확인할 것.")
        return

    q = w["qty"]
    # 색: log1p 를 그 창의 p99 로 정규화. 선형이면 벽 하나가 전부를 먹어 아무것도 안 보인다.
    mag = np.abs(q)
    scale = np.percentile(mag[mag > 0], 99) if (mag > 0).any() else 1.0
    img = np.sign(q) * np.log1p(mag / max(scale, 1e-9)) / np.log(2.0)
    img = np.clip(img, -1, 1)

    price_lo = w["bin_lo"] * w["bin_size"]
    price_hi = (w["bin_lo"] + w["n_bins"]) * w["bin_size"]
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(img.T, origin="lower", aspect="auto", interpolation="nearest",
              cmap="RdBu", vmin=-1, vmax=1,   # +bid→파랑, −ask→빨강 (RdBu_r 이면 반대가 된다)
              extent=[0, w["cols"], price_lo, price_hi])
    ax.plot(np.arange(w["cols"]) + 0.5, w["mid"], lw=0.8, color="black", label="mid")

    gaps = ~np.isfinite(w["mid"])
    for i in np.nonzero(gaps)[0]:  # 수집 공백은 회색으로. 보간하지 않는다.
        ax.axvspan(i, i + 1, color="0.6", alpha=0.55, lw=0)

    ax.set_title(f"{a.symbol.upper()} 호가 히트맵 · {a.minutes:.0f}분 (1열={a.agg}초, "
                 f"빈={w['bin_size']}$, 유효 {w['valid_ratio']*100:.1f}%) "
                 f"· 파랑=매수호가 빨강=매도호가 회색=수집공백")
    ax.set_xlabel("열 (왼쪽=과거)")
    ax.set_ylabel("가격 (USDT)")
    ax.legend(loc="upper left")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"{out} · 유효 {w['valid_ratio']*100:.1f}% · 가격폭 {price_lo:.1f}~{price_hi:.1f} "
          f"· 최대잔량 {mag.max():.1f} (p99 {scale:.2f})")


if __name__ == "__main__":
    main()
