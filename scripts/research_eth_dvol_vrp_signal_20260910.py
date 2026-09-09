#!/usr/bin/env python3
"""**새 신호** 후보 — 변동성 위험 프리미엄(VRP), Deribit DVOL 기반 (2026-09-10, 사용자 지시).

사용자 *"새로운 신호를 만들어줘. 표본이 얇아도 되니까 적은 데이터라도 증명해줘"*.

## 왜 이 정보원인가
2026-09-10 b축 스크린이 **"같은 정보를 다르게 변환하는 축은 소진됐다"**는 결론을 냈다
(96셀, 초과분이 호라이즌·변동성 양쪽으로 감소). 새 신호가 나오려면 **새 변환이 아니라
새 정보집합**이 필요하다. Deribit DVOL(30일 내재변동성 지수)은 **옵션시장의 위험 가격**이라
바이낸스 perp klines 에서 파생될 수 없다 — 이 저장소가 "데이터는 이미 있는데 아직 안 했다"고만
적어둔 유일한 축이다.

## 가설 (문헌 기반, 데이터 보기 전에 고정)
VRP = IV − RV 는 **위험회피의 가격**이다. VRP 가 높다 = 옵션 매도자가 큰 프리미엄을 요구한다
= 시장이 겁먹었다 → 이후 위험자산 수익률 양수(프리미엄 수취). 낮거나 음수면 안일 → 이후 저조.
주식(Bollerslev-Tauchen-Zhou 2009)과 암호자산 문헌에서 반복 확인된 관계.
⭐기존 8종 증거신호는 전부 가격/체결 기반 **평균회귀**다 — VRP 는 그것과 직교하는 축이다.

## 사전등록 (실행 전에 고정)
- **후보**: vrp · vrp_z · dvol · dvol_z · dvol_chg24 · rv · rv_z  (전부 인과적)
- ⚠️호메로스 5.12절: "정규화된 비율 후보는 **그 분모를 대조군으로 먼저** 돌릴 것" →
  vrp(=DVOL−RV) 를 주장하려면 **RV 단독·DVOL 단독**이 대조군으로 같이 돌아야 한다.
- **규칙**: 인과적 롤링 분위(90일)에서 상위 q → 롱, 하위 q → 숏. 전건을 센다.
- **호라이즌**: 6 · 12 · 24 · 48 시간
- **귀무**: 순환이동(발동 군집·개수·측면 보존, 가격 정렬만 파괴) B=600 → 드리프트까지 뺀다
- **대조군**: 방향 뒤집기 · 무작위 방향 · 항상 롱
- **분할**: TRAIN 2024-01-01~2026-03-31 · OOS 2026-04-01~2026-08-04
  🔒**신선 홀드아웃 2026-08-04~2026-09-09** — 로컬 DVOL 파일에 **존재하지 않던** 구간이다
  (파일이 08-04 에서 끝난다). API 로 새로 받았으므로 어떤 이전 작업도 이 구간을 못 봤다.
- **판정**: 세 창 전부 초과분 > 0 이고 OOS·홀드아웃에서 비용선(테이커 10bp)을 넘을 것.
  하나라도 실패하면 기각한다. 홀드아웃은 **한 번만** 본다.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from pathlib import Path
import numpy as np, pandas as pd, requests, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/eth_dvol_vrp_20260910"
DVOL_CSV = ROOT / "data/derivatives/deribit_dvol/ETH_dvol_hourly.csv"
KLCACHE = ROOT / "tmp/eth_signal_map_20260909/klcache"
KL5 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
HS = (6, 12, 24, 48)                 # 시간
QS = (0.20, 0.10)                    # 상/하위 분위 (커버리지 두 단계)
ROLL_Q = 90 * 24                     # 인과 분위 창 90일
RV_W = 24                            # 실현변동성 창 24시간
Z_W = 30 * 24                        # z 창 30일
NSHIFT = 600
COST = 10.0
RNG = np.random.default_rng(20260910)
TRAIN_END = pd.Timestamp("2026-03-31")
OOS_END = pd.Timestamp("2026-08-04 10:00")     # 로컬 DVOL 파일의 마지막 행


def fetch_dvol_tail(t0: pd.Timestamp) -> pd.DataFrame:
    """로컬 파일 끝 이후를 Deribit 공개 API 로 받는다(신선 홀드아웃 확보)."""
    end = int(time.time() * 1000)
    start = int(t0.timestamp() * 1000)
    r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                     params={"currency": "ETH", "start_timestamp": start,
                             "end_timestamp": end, "resolution": "3600"}, timeout=30)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["result"]["data"], columns=["ts", "open", "high", "low", "close"])
    d["timestamp"] = pd.to_datetime(d["ts"], unit="ms")
    return d[["timestamp", "close"]].sort_values("timestamp")


def load_px5() -> pd.DataFrame:
    best, bn = None, 0
    for f in glob.glob(str(KLCACHE / "ETHUSDT_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    hist = pd.read_csv(KL5, usecols=["timestamp", "open", "high", "low", "close"],
                       parse_dates=["timestamp"])
    d = pd.concat([hist, best[["timestamp", "open", "high", "low", "close"]]], ignore_index=True)
    return d.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--holdout", action="store_true",
                                                    help="🔒신선 홀드아웃까지 본다(한 번만)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    dv = pd.read_csv(DVOL_CSV, parse_dates=["timestamp"])[["timestamp", "close"]]
    print(f"로컬 DVOL {dv.timestamp.min()} ~ {dv.timestamp.max()} ({len(dv):,}행)")
    tail = fetch_dvol_tail(dv.timestamp.max())
    dv = pd.concat([dv, tail], ignore_index=True).drop_duplicates("timestamp", keep="first")
    dv = dv.sort_values("timestamp").reset_index(drop=True).rename(columns={"close": "dvol"})
    print(f"API 로 연장  → {dv.timestamp.max()} (신선 구간 {len(tail):,}행 중 "
          f"{int((tail.timestamp > OOS_END).sum()):,}행이 홀드아웃)")

    px5 = load_px5()
    print(f"5분봉 {px5.timestamp.min()} ~ {px5.timestamp.max()} ({len(px5):,}행)")
    # 시간봉 집계 (인과: 시각 t 봉은 [t, t+1h) 를 담고 t+1h 에 확정 -> DVOL_t 와 같은 시각에 맞춘다)
    p = px5.set_index("timestamp")
    h = pd.DataFrame({"open": p["open"].resample("1h").first(),
                      "high": p["high"].resample("1h").max(),
                      "low": p["low"].resample("1h").min(),
                      "close": p["close"].resample("1h").last()}).dropna()
    r5 = np.log(p["close"]).diff()
    # 실현변동성: 지난 24시간 5분 수익률의 연율화 표준편차 (%)
    rv = (r5.rolling(RV_W * 12, min_periods=RV_W * 6).std() * np.sqrt(288 * 365) * 100)
    h["rv"] = rv.resample("1h").last()
    d = h.join(dv.set_index("timestamp")["dvol"], how="inner").dropna(subset=["dvol", "rv"])
    d = d.reset_index().rename(columns={"index": "timestamp"})
    print(f"조인 후 {d.timestamp.min()} ~ {d.timestamp.max()} ({len(d):,}시간)\n")

    # --- 인과 피쳐 (전부 시각 t 종가에 알려진 값만) --------------------------------
    d["vrp"] = d["dvol"] - d["rv"]
    for c in ("vrp", "dvol", "rv"):
        m = d[c].rolling(Z_W, min_periods=Z_W // 3).mean()
        s = d[c].rolling(Z_W, min_periods=Z_W // 3).std()
        d[f"{c}_z"] = (d[c] - m) / s.replace(0, np.nan)
    d["dvol_chg24"] = d["dvol"].diff(24)
    FEATS = ["vrp", "vrp_z", "dvol", "dvol_z", "dvol_chg24", "rv", "rv_z"]

    op = d["open"].to_numpy(float); cl = d["close"].to_numpy(float)
    ts = pd.to_datetime(d["timestamp"]); n = len(d)
    warm = max(ROLL_Q, Z_W) + 48
    hi_i = n - max(HS) - 2
    base = np.zeros(n, bool); base[warm:hi_i] = True
    span = hi_i - warm

    def ret_bp(idx, H, long):
        r = (cl[idx + H] - op[idx + 1]) / op[idx + 1] * 1e4
        return np.where(long, r, -r)

    def excess(idx, longv, H, w0, w1):
        """🔴순환이동은 **평가창 안에서만** 돌린다.

        전체 표본으로 돌리면 귀무의 평균이 '전 기간 드리프트'가 되는데 실제 값은 '이 창의
        드리프트'라, 초과분에 **창끼리의 드리프트 차이**가 통째로 섞여 들어온다. 2026-09-10
        첫 실행에서 TRAIN +2.25bp / OOS +99.2bp 라는 10배 격차가 그 인공물이었다 --
        신호가 아니라 평가창이 강세장이었던 것이다. 창 안에서 돌리면 롱/숏 배정과 개수·군집을
        그대로 둔 채 그 창 자신의 드리프트만 귀무에 담긴다."""
        if len(idx) < 25: return (np.nan,) * 4
        g = float(np.mean(ret_bp(idx, H, longv)))
        L = w1 - w0
        if L < 400: return g, np.nan, np.nan, np.nan
        nul = np.empty(NSHIFT)
        for b in range(NSHIFT):
            sh = RNG.integers(50, L - 50)
            nul[b] = np.mean(ret_bp(w0 + ((idx - w0 + sh) % L), H, longv))
        return g, float(nul.mean()), g - float(nul.mean()), float((nul >= g).mean())

    win = {"TRAIN": base & (ts <= TRAIN_END).to_numpy(),
           "OOS": base & (ts > TRAIN_END).to_numpy() & (ts <= OOS_END).to_numpy()}
    if a.holdout:
        win["🔒HOLDOUT"] = base & (ts > OOS_END).to_numpy()
    for k, m in win.items():
        print(f"  {k:<10} {ts[m].min():%Y-%m-%d} ~ {ts[m].max():%Y-%m-%d}  ({m.sum():,}시간, "
              f"{(ts[m].max()-ts[m].min()).days}일)")

    rows = []
    print("\n" + "=" * 116)
    print("후보 스크린 -- 인과 롤링분위(90일) 상/하위. 창별로 [원시 gross · 귀무 · 초과 · p]")
    print("⭐귀무는 **평가창 안에서** 순환이동한다 -- 창끼리의 드리프트 차이가 초과분에 안 섞이게")
    print("=" * 116)
    print(f"{'':<12}{'':<5}" + "".join(f"{w:>30}" for w in win))
    for f in FEATS:
        v = d[f].to_numpy(float)
        qh = d[f].rolling(ROLL_Q, min_periods=ROLL_Q // 3).quantile(1 - QS[0]).to_numpy()
        ql = d[f].rolling(ROLL_Q, min_periods=ROLL_Q // 3).quantile(QS[0]).to_numpy()
        ok = base & np.isfinite(v) & np.isfinite(qh) & np.isfinite(ql)
        for H in HS:
            cells = {}
            for wname, wm in win.items():
                m = ok & wm
                hi_m = m & (v >= qh); lo_m = m & (v <= ql)
                idx = np.flatnonzero(hi_m | lo_m)
                if len(idx) < 25: cells[wname] = (np.nan,) * 4; continue
                longv = (v[idx] >= qh[idx])          # 가설: 높은 VRP -> 롱
                wi = np.flatnonzero(wm)
                cells[wname] = excess(idx, longv, H, int(wi.min()), int(wi.max()) + 1)
            rows.append(dict(feat=f, H=H,
                             **{f"{w}_g": cells[w][0] for w in win},
                             **{f"{w}_null": cells[w][1] for w in win},
                             **{f"{w}_ex": cells[w][2] for w in win},
                             **{f"{w}_p": cells[w][3] for w in win},
                             n_oos=int((ok & win["OOS"] & ((v >= qh) | (v <= ql))).sum())))
            r = rows[-1]
            line = f"{f:<12}H{H:<4}" + "".join(
                f"{r[w+'_g']:>8.1f}{r[w+'_null']:>8.1f}{r[w+'_ex']:>8.1f}{r[w+'_p']:>6.2f}"
                for w in win)
            print(line + f"  n={r['n_oos']}")
    dd = pd.DataFrame(rows); dd.to_csv(OUT / "screen.csv", index=False)
    print(f"\n저장 {OUT/'screen.csv'}")

    ok = dd[(dd.TRAIN_ex > 0) & (dd.OOS_ex > COST)]
    print(f"\n★ TRAIN 초과분>0 이고 OOS 초과분>{COST}bp: {len(ok)}/{len(dd)}셀")
    if len(ok):
        print(ok[["feat", "H", "TRAIN_ex", "OOS_ex", "OOS_p", "n_oos"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
