"""호가 방아쇠가 «계속 바뀐다» — 안정화 규칙 후보를 같은 1초 패널에서 재생해 안정성과 정보량을 같이 잰다.

깜빡임은 두 종류다. ①경계 채터: 값이 임계선 근처에서 떨어 라벨만 오간다(정보 손실 없이 없앨 수 있다).
②신호 자체가 빠름: QI 반감기 1초(ACF 0.61). 이건 정보와 맞바꿔야 한다.
그래서 규칙마다 **둘을 같이** 낸다 — 시간당 라벨 변경·중앙 체류초 / 그 상태에서의 앞 5·15·60초 수익(bp).

판정 기준: «같은 정보량을 유지하면서 변경 횟수를 가장 많이 줄이는 규칙». 정보량은
  edge15 = (동조매수일 때 앞 15초 평균) − (동조매도일 때 평균), 1시간 블록 평균±SE.
🔴패널의 OFI 는 depth diff 리플레이(±50bp)이고 배포 카드는 래스터(±40bp)라 같은 가족이지만 같은 값은 아니다.
   규칙 «모양»을 고르는 데는 충분하고, 임계값 자체는 배포 쪽에서 분위로 다시 잡는다.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = pd.read_parquet(ROOT / "tmp/rt_probe_20260920/panel_1s.parquet")
OUT = ROOT / "tmp/rt_probe_20260920/trigger_stability.json"
QI_ABS = 0.56           # 배포본과 같은 값(|QI| p50)
mid = P.bt_mid.ffill(limit=5)
for h in (5, 15, 60):
    P[f"fwd{h}"] = (np.log(mid.shift(-h)) - np.log(mid)) * 1e4
P["blk"] = P.index // 3600

qi_raw = P.bt_qi_last.to_numpy(dtype="float64")          # 배포 카드는 1초에 한 번 점 표본 -> last
# 패널에는 추가/취소 원자료만 있다 -- analyze 와 같은 정의로 OFI 를 만들고 10초로 접는다(배포 카드와 같은 창).
P["dd_ofi"] = ((P.dd_add_b - P.dd_rem_b) - (P.dd_add_a - P.dd_rem_a)).where(P.dd_valid == 1)
P["dd_ofi10"] = P.dd_ofi.rolling(10, min_periods=5).sum()
ofi = P.dd_ofi10.to_numpy(dtype="float64")
# 배포본의 임계: 최근 10분 |OFI10| 중앙값(롤링). 같은 규약으로 재현한다.
ofi_thr = pd.Series(np.abs(ofi)).rolling(600, min_periods=60).median().to_numpy()
ok = np.isfinite(qi_raw) & np.isfinite(ofi) & np.isfinite(ofi_thr)


def side(x: np.ndarray, thr: np.ndarray | float) -> np.ndarray:
    t = np.full_like(x, thr, dtype="float64") if np.isscalar(thr) else thr
    return np.where(x >= t, 1, np.where(x <= -t, -1, 0)).astype(np.int8)


def agree(qs: np.ndarray, os_: np.ndarray) -> np.ndarray:
    """둘이 같은 쪽일 때만 그 쪽. 하나라도 중립이거나 갈리면 0."""
    return np.where((qs != 0) & (qs == os_), qs, 0).astype(np.int8)


def schmitt(x: np.ndarray, hi: float | np.ndarray, lo: float | np.ndarray) -> np.ndarray:
    """|x|>=hi 에서 그 부호로 들어가고, |x|<lo 가 되어야 나온다(반대쪽으로 가려면 반대 hi 를 넘어야).
    경계 채터만 없앤다 -- 값이 확실히 한쪽이면 즉시 들어가므로 «느려지지» 않는다."""
    n = len(x)
    hi_a = np.full(n, hi, dtype="float64") if np.isscalar(hi) else hi
    lo_a = np.full(n, lo, dtype="float64") if np.isscalar(lo) else lo
    out = np.zeros(n, dtype=np.int8)
    s = 0
    for i in range(n):
        v, h, l = x[i], hi_a[i], lo_a[i]
        if not np.isfinite(v) or not np.isfinite(h):
            out[i] = s
            continue
        if s == 0:
            s = 1 if v >= h else (-1 if v <= -h else 0)
        elif s == 1:
            s = -1 if v <= -h else (0 if v < l else 1)
        else:
            s = 1 if v >= h else (0 if v > -l else -1)
        out[i] = s
    return out


def min_dwell(state: np.ndarray, n: int) -> np.ndarray:
    """라벨이 바뀌면 최소 n 초 붙든다. 늦게 바뀌는 대신 되돌아가지 않는다."""
    out = np.empty_like(state)
    cur, frozen = state[0], 0
    for i, s in enumerate(state):
        if frozen > 0:
            frozen -= 1
        elif s != cur:
            cur, frozen = s, n - 1
        out[i] = cur
    return out


def confirm(state: np.ndarray, k: int) -> np.ndarray:
    """k 초 연속 같은 값일 때만 채택한다(그 전에는 직전 라벨 유지)."""
    out = np.empty_like(state)
    cur, run, prev = np.int8(0), 0, None
    for i, s in enumerate(state):
        run = run + 1 if s == prev else 1
        prev = s
        if run >= k:
            cur = s
        out[i] = cur
    return out


def ewma(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def measure(state: np.ndarray, name: str) -> dict:
    s = pd.Series(np.where(ok, state, np.nan), index=P.index)
    v = s.dropna()
    hours = len(v) / 3600.0
    chg = (v != v.shift()).fillna(False)
    flips = int(chg.sum())
    rev = int(((v == 1) & (v.shift() == -1) | (v == -1) & (v.shift() == 1)).sum())   # 매수<->매도 직접 반전
    run_id = chg.cumsum()
    runs = v.groupby(run_id).agg(["first", "size"])
    dir_runs = runs[runs["first"] != 0]["size"]
    cov = float((v != 0).mean())
    out = {"name": name, "flips_per_hour": round(flips / hours, 1), "reversals_per_hour": round(rev / hours, 2),
           "dwell_median_s": float(dir_runs.median()) if len(dir_runs) else 0.0,
           "dwell_p90_s": float(dir_runs.quantile(0.9)) if len(dir_runs) else 0.0,
           "coverage": round(cov, 3), "n_dir_runs": int(len(dir_runs))}
    P["_st"] = s
    for h in (5, 15, 60):
        vals = []
        for _, g in P.groupby("blk"):
            up, dn = g.loc[g._st == 1, f"fwd{h}"], g.loc[g._st == -1, f"fwd{h}"]
            if len(up) >= 30 and len(dn) >= 30:
                vals.append(up.mean() - dn.mean())
        a = np.array(vals)
        out[f"edge{h}"] = round(float(a.mean()), 3) if len(a) > 2 else None
        out[f"edge{h}_se"] = round(float(a.std(ddof=1) / np.sqrt(len(a))), 3) if len(a) > 2 else None
    return out


qs_base, os_base = side(qi_raw, QI_ABS), side(ofi, ofi_thr)
base = agree(qs_base, os_base)
rules: list[tuple[str, np.ndarray]] = [("현행 (|QI|≥0.56 · OFI≥중앙값)", base)]
# ① 경계 채터만 제거 -- 히스테리시스
for hi, lo in ((0.56, 0.35), (0.65, 0.40), (0.75, 0.45)):
    rules.append((f"히스테리시스 QI {hi}/{lo}", agree(schmitt(qi_raw, hi, lo), os_base)))
rules.append(("히스테리시스 QI 0.65/0.40 + OFI 1.0/0.6×", agree(schmitt(qi_raw, 0.65, 0.40), schmitt(ofi, ofi_thr, ofi_thr * 0.6))))
# ② 확인·체류 -- 늦어지는 대신 안 되돌아간다
for k in (2, 3, 5):
    rules.append((f"{k}초 연속 확인", confirm(base, k)))
for n in (5, 10, 15, 30):
    rules.append((f"최소 체류 {n}초", min_dwell(base, n)))
# ③ 신호 자체를 느리게 -- 평활(정보 손실 구간)
for span in (3, 5, 10):
    q = ewma(qi_raw, span)
    thr = float(np.nanquantile(np.abs(q[ok]), 0.5))     # 같은 «절반이 중립» 규약
    rules.append((f"QI EWMA {span}초 (임계 p50={thr:.2f})", agree(side(q, thr), os_base)))
# ④ 조합 후보
rules.append(("히스테리시스 0.65/0.40 + 최소 체류 10초", min_dwell(agree(schmitt(qi_raw, 0.65, 0.40), os_base), 10)))
rules.append(("히스테리시스 0.65/0.40 + 3초 확인", confirm(agree(schmitt(qi_raw, 0.65, 0.40), os_base), 3)))
rules.append(("EWMA 5초 + 히스테리시스 0.45/0.25", agree(schmitt(ewma(qi_raw, 5), 0.45, 0.25), os_base)))
rules.append(("EWMA 5초 + 히스테리시스 + 최소 체류 10초",
              min_dwell(agree(schmitt(ewma(qi_raw, 5), 0.45, 0.25), os_base), 10)))

# ⑤ «신호»가 아니라 «표시»를 접는다 -- 1초 판정은 그대로 두고 최근 N초의 동조 점수를 보여준다.
#    정보를 버리지 않는다(원 상태를 그대로 세는 것이므로). 임계는 현행과 같은 점유율이 되도록 분위로 잡는다.
COV = float(np.mean(base[ok] != 0))
for win in (15, 30, 60):
    sc = pd.Series(np.where(ok, base, np.nan)).rolling(win, min_periods=win // 2).mean().to_numpy()
    enter = float(np.nanquantile(np.abs(sc), 1 - COV))
    rules.append((f"최근 {win}초 동조 점수 (임계 {enter:.2f})", schmitt(sc, enter, enter * 0.6)))
rules.append(("최근 30초 점수 + 히스테리시스 0.5×",
              schmitt(pd.Series(np.where(ok, base, np.nan)).rolling(30, min_periods=15).mean().to_numpy(),
                      float(np.nanquantile(np.abs(pd.Series(np.where(ok, base, np.nan)).rolling(30, min_periods=15).mean().to_numpy()), 1 - COV)),
                      float(np.nanquantile(np.abs(pd.Series(np.where(ok, base, np.nan)).rolling(30, min_periods=15).mean().to_numpy()), 1 - COV)) * 0.5)))

res = [measure(st, nm) for nm, st in rules]
hdr = f"{'규칙':44} {'변경/시간':>9} {'반전/시간':>9} {'체류중앙':>8} {'p90':>6} {'점유':>6} {'edge5':>13} {'edge15':>13} {'edge60':>13}"
print(hdr)
print("-" * len(hdr))
for r in res:
    def f(h):
        return f"{r[f'edge{h}']:+.2f}±{r[f'edge{h}_se']:.2f}" if r[f"edge{h}"] is not None else "n/a"
    print(f"{r['name']:44} {r['flips_per_hour']:9.1f} {r['reversals_per_hour']:9.2f} {r['dwell_median_s']:8.0f} "
          f"{r['dwell_p90_s']:6.0f} {r['coverage']:6.3f} {f(5):>13} {f(15):>13} {f(60):>13}")
base_r = res[0]
print(f"\n기준(현행): 변경 {base_r['flips_per_hour']}/시간 · 체류 중앙 {base_r['dwell_median_s']:.0f}초 · edge15 {base_r['edge15']:+.2f}")
print("«같은 edge15 를 SE 안에서 지키면서 변경/시간이 가장 적은 규칙»이 답이다.")
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("->", OUT)

# ── 즉시 신호 × 느린 누적: «깜빡임을 언제 믿을까» ────────────────────────────
# 느린 쪽을 «표시»로 쓰는 건 위에서 값이 깎였다. 그럼 «걸러내는 조건»으로 쓰면 어떤가.
print("\n## 즉시 동조(현행) × 최근 60초 동조 점수 -- 앞 15초 수익(bp, 블록 평균±SE)")
sc60 = pd.Series(np.where(ok, base, np.nan)).rolling(60, min_periods=30).mean().to_numpy()
P["_base"], P["_sc"] = np.where(ok, base, np.nan), sc60
CASES = {
    "동조매수 (전체)": (P._base == 1),
    "  └ 60초 누적도 매수쪽": (P._base == 1) & (P._sc > 0),
    "  └ 60초 누적이 매도쪽/0": (P._base == 1) & (P._sc <= 0),
    "동조매도 (전체)": (P._base == -1),
    "  └ 60초 누적도 매도쪽": (P._base == -1) & (P._sc < 0),
    "  └ 60초 누적이 매수쪽/0": (P._base == -1) & (P._sc >= 0),
}
for name, m in CASES.items():
    m = m.fillna(False)
    vals = [g.loc[m.reindex(g.index).fillna(False), "fwd15"].mean() for _, g in P.groupby("blk")
            if m.reindex(g.index).fillna(False).sum() >= 30]
    a = np.array([v for v in vals if np.isfinite(v)])
    if len(a) > 2:
        print(f"  {name:28} n={int(m.sum()):7,} 점유 {m.mean():5.1%}  fwd15 {a.mean():+6.3f}±{a.std(ddof=1)/np.sqrt(len(a)):.3f}")
    else:
        print(f"  {name:28} n={int(m.sum())} 표본부족")
agree_share = float(((P._base == 1) & (P._sc > 0) | (P._base == -1) & (P._sc < 0)).sum() / max(1, (P._base != 0).sum()))
print(f"\n  즉시 동조가 60초 누적과 같은 쪽인 비율: {agree_share:.1%}")

# ── «마지막 동조 이후 나이»가 몇 초까지 값을 갖는가 -- 표시 눈금을 정한다 ──────
# 깜빡임의 정체는 «지금 상태»를 discrete 라벨로 그린 것이다. 대신 «마지막 방아쇠 + 나이»를 보여주면
# 라벨은 새 방아쇠에서만 바뀌고 나이는 매끄럽게 올라간다(깜빡임 0). 그 «유효 나이»를 여기서 잰다.
print("\n## 마지막 동조 이후 나이별 앞 15초 수익 (그 방향 기준, bp)")
st = pd.Series(np.where(ok, base, np.nan), index=P.index)
last_dir = st.where(st != 0).ffill(limit=120)
age = st.groupby((st != 0).cumsum()).cumcount().where(last_dir.notna())
P["_ld"], P["_age"] = last_dir, age
for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 60)):
    m = (P._age >= lo) & (P._age <= hi) & P._ld.notna() & (P._ld != 0)
    signed = P.fwd15 * P._ld
    vals = [g.loc[m.reindex(g.index).fillna(False), "_sf"].mean() for _, g in P.assign(_sf=signed).groupby("blk")
            if m.reindex(g.index).fillna(False).sum() >= 30]
    a = np.array([v for v in vals if np.isfinite(v)])
    lbl = f"{lo}초" if lo == hi else f"{lo}~{hi}초"
    print(f"  나이 {lbl:8} n={int(m.sum()):7,} 점유 {m.mean():5.1%}  방향맞춘 fwd15 {a.mean():+6.3f}±{a.std(ddof=1)/np.sqrt(len(a)):.3f}"
          if len(a) > 2 else f"  나이 {lbl:8} 표본부족")
