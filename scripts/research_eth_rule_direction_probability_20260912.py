#!/usr/bin/env python3
"""조건부 **방향 확률 규칙** 채굴 — "천장 신호 + X 면 숏 확률 NN%" (2026-09-12).

사용자: *"차트나 데이터를 보고 예를 들어 «천장 신호에 극점 탐지기까지 하니까 숏일 확률이
80%» 같은 알고리즘들을 만들어줘."*  입력은 어제 만든 재료 패널
(`build_eth_signal_trigger_material_panel_20260912`), 모델은 안 쓴다.

## 규칙의 모양
  방향원자(측면 있음) 1~2개  +  필터원자(측면 없음) 0~2개   (원자 총 ≤3)
  방향은 방향원자들의 측면에서 나온다 — 측면이 엇갈리는 조합은 만들지 않는다.

## 라벨 — 「사건 라벨 경계 계약」
피쳐는 봉 i 종가까지, **라벨 탐색은 i+1부터**, 기준가는 **open[i+1]**(체결 가능한 값 —
저가/고가를 기준가로 쓰면 못 사는 가격이 된다, 2026-09-10 전수점검).
  hit_H    = H봉 뒤 종가가 진입가 대비 규칙 방향인가            (H = 12, 48)
  bar_k    = ±k×ATR 중 **어느 쪽에 먼저 닿는가**(봉 고저, 96봉 안, 라이브 컨벤션)  (k = 0.8, 1.5)

## 판정 규약 (이 저장소 표준)
  기저는 **무조건부 같은 창 전체**다(0.5 아님 — 되돌림·왜도 때문에 실제로 어긋난다).
  선택   VAL·OOS **두 창 모두** (적중률−기저) Wilson 95% 하한 > 0 · 창당 n ≥ 100
  확인   TRAIN 에서도 양수인가 (모델 없는 규칙은 TRAIN 이 확인창, 2026-09-08)
  귀무   통과 규칙만 순환이동 B=400(발동 군집·개수 보존) → p
  가족   라벨을 통째로 순환이동시켜 **전체 화면을 B_FAM 번 재실행** → 귀무하 기대 통과 수
  ⚠️커버리지(건/일·봉비율) 항상 병기. 정확도를 bp 로 환산하지 않는다 — 손익은 배리어로 따로 잰다.

산출 tmp/eth_rule_direction_20260912/{rules.csv,report.json}
자체점검 `--selftest`
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import build_eth_signal_trigger_material_panel_20260912 as P  # noqa: E402

PANEL = P.OUT / "panel_5m.parquet"
OUT = ROOT / "tmp/eth_rule_direction_20260912"
HS = (12, 48)
KS = (0.8, 1.5)
BAR_MAX = 96                 # 배리어 탐색 상한(8시간)
WARM = 900                   # 후행 2016봉 분위의 min_periods=500 + 여유
MIN_N = 100                  # 창당 최소 건수
NSHIFT, B_FAM = 400, 12
COST_BP = {"테이커": 10.0, "메이커": 7.8}
SPLITS = {"TRAIN": (None, "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"), "OOS": ("2026-01-01", None)}
RNG = np.random.default_rng(20260912)


def log(m: str) -> None:
    print(f"[rule {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ 라벨
def first_touch(hi: np.ndarray, lo: np.ndarray, entry: np.ndarray, atr: np.ndarray, k: float) -> np.ndarray:
    """봉 i 의 규칙에 대해 **i+1부터** ±k×ATR 중 먼저 닿는 쪽. +1 위 / −1 아래 / 0 미해결.

    라이브 컨벤션과 같다 — resting TP/SL 은 종가가 아니라 닿는 즉시 체결되고, 이미 확정된
    봉의 고저만 본다(`omega4_6_1_live.py::evaluate_exit`).
    """
    n = len(hi)
    up = entry * (1.0 + k * atr)
    dn = entry * (1.0 - k * atr)
    out = np.zeros(n, np.int8)
    live = np.ones(n, bool)
    for j in range(1, BAR_MAX + 1):
        s = np.arange(n) + j
        ok = live & (s < n)
        if not ok.any():
            break
        idx = np.flatnonzero(ok)
        t = idx + j
        hit_u = hi[t] >= up[idx]
        hit_d = lo[t] <= dn[idx]
        both = hit_u & hit_d                       # 같은 봉에서 양쪽 → 판정 불가, 보수적으로 반대쪽
        res = np.where(hit_u & ~hit_d, 1, np.where(hit_d & ~hit_u, -1, 0)).astype(np.int8)
        res[both] = -1
        done = hit_u | hit_d
        out[idx[done]] = res[done]
        live[idx[done]] = False
    return out


def labels(p: pd.DataFrame) -> dict[str, np.ndarray]:
    op = p["open"].to_numpy(float); cl = p["close"].to_numpy(float)
    hi = p["high"].to_numpy(float); lo = p["low"].to_numpy(float)
    atr = p["atr_pct"].to_numpy(float)
    n = len(p)
    ent = np.roll(op, -1); ent[-1] = np.nan            # 진입가 = 다음 봉 시가
    L: dict[str, np.ndarray] = {"_entry": ent}
    for H in HS:
        fwd = np.full(n, np.nan)
        fwd[: n - H - 1] = cl[H + 1 : n] / ent[: n - H - 1] - 1.0
        L[f"up{H}"] = np.where(np.isfinite(fwd), (fwd > 0).astype(float), np.nan)
        L[f"bp{H}"] = fwd * 1e4
    for k in KS:
        t = first_touch(hi, lo, ent, atr, k)
        L[f"bar{k}"] = np.where(t == 0, np.nan, (t > 0).astype(float))   # 미해결은 제외
        L[f"barbp{k}"] = k * atr * 1e4
    return L


# ------------------------------------------------------------------ 원자
def atoms(p: pd.DataFrame) -> tuple[dict[str, tuple[np.ndarray, int]], dict[str, np.ndarray]]:
    """방향원자 {이름: (마스크, 측면 +1롱/−1숏)} · 필터원자 {이름: 마스크}"""
    D: dict[str, tuple[np.ndarray, int]] = {}
    for s in P.B.SIGNALS:
        v = p[f"ev_{P.B.ABBR[s]}"].to_numpy()
        D[f"바닥:{P.B.ABBR[s]}"] = (v > 0, +1)
        D[f"천장:{P.B.ABBR[s]}"] = (v < 0, -1)
    for t in ("any3w3", "any2w3", "first_fire", "vrev"):
        v = p[f"trg_{t}"].to_numpy()
        D[f"바닥:{t}"] = (v > 0, +1)
        D[f"천장:{t}"] = (v < 0, -1)
    D["바닥:2종이상"] = (p["ev_n_bottom"].to_numpy() >= 2, +1)
    D["천장:2종이상"] = (p["ev_n_top"].to_numpy() >= 2, -1)

    F: dict[str, np.ndarray] = {}
    F["돌파탐지"] = p["trg_breakout"].to_numpy() > 0
    F["변동성확장"] = p["trg_vol_expand"].to_numpy() > 0
    F["추세레짐"] = p["trg_regime_trend"].to_numpy() > 0
    F["횡보레짐"] = p["trg_regime_trend"].to_numpy() == 0
    F["압축"] = p["volexp"].to_numpy() < 0.70
    ap = p["atr_pctile"].to_numpy()
    F["고변동성"] = ap >= 0.70
    F["저변동성"] = ap <= 0.30
    pr = p["pos_in_range48"].to_numpy()
    F["레인지상단"] = pr >= 0.80
    F["레인지하단"] = pr <= 0.20
    b12 = p["btc_ret12"].to_numpy()
    F["BTC상승"] = b12 > 0
    F["BTC하락"] = b12 < 0
    h = p["hour"].to_numpy()
    F["아시아장"] = (h >= 0) & (h < 7)
    F["유럽장"] = (h >= 7) & (h < 14)
    F["미국장"] = h >= 14
    rg = p["mt_regime_eth"].to_numpy()
    for c, nm in ((0, "레짐0"), (1, "레짐1"), (2, "레짐2")):
        F[nm] = rg == c
    return D, F


def enumerate_rules(D: dict, F: dict) -> list[tuple[tuple[str, ...], tuple[str, ...], int]]:
    """방향원자 1~2(측면 일치) + 필터 0~2, 원자 총 ≤3. **이름만** 담는다.

    마스크를 미리 만들어 들고 있으면 7천 규칙 × 277천 봉 = 2GB 다. 필요할 때 만든다.
    """
    out = []
    dk = list(D)
    f2s = list(itertools.combinations(F, 2))
    for a in dk:
        sa = D[a][1]
        out.append(((a,), (), sa))
        out += [((a,), (f,), sa) for f in F]
        out += [((a,), fp, sa) for fp in f2s]
    for a, b in itertools.combinations(dk, 2):
        if D[a][1] != D[b][1]:
            continue
        out.append(((a, b), (), D[a][1]))
        out += [((a, b), (f,), D[a][1]) for f in F]
    return out


def rule_mask(da: tuple, fa: tuple, D: dict, F: dict) -> np.ndarray:
    m = D[da[0]][0].copy()
    for x in da[1:]:
        m &= D[x][0]
    for x in fa:
        m &= F[x]
    return m


def rule_name(da: tuple, fa: tuple) -> str:
    return " + ".join(da + fa)


# ------------------------------------------------------------------ 평가
def sided(base_long: float, side: int) -> float:
    """무조건부 기저는 **측면별로 다르다**. y = P(위 먼저) 이므로 숏의 기저는 1−y 다.

    ⚠️이걸 빼먹으면(초판) 기저 0.483 을 숏 적중률에 그대로 대서 모든 숏 규칙이 +3.4pp
    공짜로 부풀고 통과 236개가 **전부 숏**이 된다. 이 저장소가 반복해서 밟은 함정이다
    (측면별 초과가 거울상이면 신호가 아니라 잔존 베타, 2026-09-05/09-08).
    """
    return base_long if side > 0 else 1.0 - base_long


def wilson_lo(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    r = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return float((c - r) / d)


def evaluate(mask: np.ndarray, side: int, y_long: np.ndarray, win: np.ndarray) -> tuple[int, int, float]:
    """규칙 방향 적중 건수/표본수/적중률. side=−1 이면 적중 = 하락."""
    return rate_at(np.flatnonzero(mask & win), side, y_long)


def rate_at(idx: np.ndarray, side: int, y_long: np.ndarray) -> tuple[int, int, float]:
    y = y_long[idx]
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return 0, 0, float("nan")
    if side < 0:
        y = 1.0 - y
    k = int(y.sum())
    return k, len(y), k / len(y)



# ------------------------------------------------------------------ 검증(--verify)
def verify(label: str, topn: int) -> int:
    """통과 규칙 상위 N개에 **이 저장소가 과거에 죽인 방식** 네 가지를 그대로 적용한다.

    1. 반기 안정성   — OOS 를 둘로 갈라 둘 다 사는가 (smt 천장 H144 은 전반 48.6/후반 1.3 이었다)
    2. 일블록 부트   — 겹침 표본이라 Wilson 은 과대다. 일 단위로 재표집해 CI 를 다시 낸다
    3. 꼬리 의존     — 상위 1% 절사·윈저. 쏠림 페이드는 손익 94%가 꼬리라 여기서 죽었다
    4. 거울상        — 같은 조합의 반대 측면. 둘이 거울이면 신호가 아니라 잔존 베타다
    5. TRAIN·분기    — 규칙은 VAL/OOS 로 골랐으니 **TRAIN 이 유일한 선택 밖 창**이다. 여기가 음수면
                       "최근 몇 분기에만 살아있는 것"이다(2026-09-11 횡단면 교훈). 분기별 양수 비율과
                       **마지막 분기**를 같이 본다 — 이미 죽은 효과를 승격하지 않기 위해.
    """
    p_ = pd.read_parquet(PANEL)
    L = labels(p_)
    H = int(label[2:])
    bp_long = L[f"bp{H}"]
    ts = pd.to_datetime(p_["timestamp"])
    n = len(p_)
    base_valid = np.zeros(n, bool); base_valid[WARM : n - BAR_MAX - 2] = True
    w = base_valid & (ts >= pd.Timestamp(SPLITS["OOS"][0])).to_numpy()
    wtr = base_valid & (ts <= pd.Timestamp(SPLITS["TRAIN"][1])).to_numpy()
    quarter = ts.dt.to_period("Q").astype(str).to_numpy()
    mid = ts[w].min() + (ts[w].max() - ts[w].min()) / 2
    D, F = atoms(p_)
    day = ts.dt.floor("D").to_numpy()
    rng = np.random.default_rng(20260912)

    d = pd.read_csv(OUT / f"rules_{label}.csv").sort_values("net_테이커", ascending=False).head(topn)
    print("=" * 122)
    print(f"검증 — {label} 상위 {len(d)}규칙 (순익순)")
    print("=" * 122)
    print(f"{'규칙':<40}{'방향':<5}{'gross':>8}{'전반':>7}{'후반':>7}{'부트하한':>9}"
          f"{'절사1%':>8}{'거울':>7}{'TRAIN':>7}{'분기+':>7}{'끝분기':>8}{'판정':>6}")
    rows = []
    for r in d.itertuples():
        side = 1 if r.side == "롱" else -1
        da = tuple(x for x in r.rule.split(" + ") if x.startswith(("바닥:", "천장:")))
        fa = tuple(x for x in r.rule.split(" + ") if not x.startswith(("바닥:", "천장:")))
        m = rule_mask(da, fa, D, F)
        bp = bp_long * side
        idx = np.flatnonzero(m & w & np.isfinite(bp))
        v = bp[idx]
        g = float(v.mean())
        h1 = float(bp[idx[ts.to_numpy()[idx] <= mid]].mean())
        h2 = float(bp[idx[ts.to_numpy()[idx] > mid]].mean())
        # 일블록 부트 — 겹침은 같은 날 안에서 가장 심하다
        dd = day[idx]
        udays = np.unique(dd)
        by = {u: v[dd == u] for u in udays}
        boot = np.array([np.concatenate([by[u] for u in rng.choice(udays, len(udays))]).mean()
                         for _ in range(2000)])
        lo95 = float(np.percentile(boot, 2.5))
        cut = np.percentile(np.abs(v), 99)
        trim = float(v[np.abs(v) <= cut].mean())
        # 거울상 — 같은 조합의 반대 측면
        flip = tuple(("바닥:" + x[3:]) if x.startswith("천장:") else ("천장:" + x[3:]) for x in da)
        mm = rule_mask(flip, fa, D, F)
        mi = np.flatnonzero(mm & w & np.isfinite(bp_long))
        mir = float((bp_long * -side)[mi].mean()) if len(mi) >= 30 else np.nan
        itr = np.flatnonzero(m & wtr & np.isfinite(bp))
        gtr = float(bp[itr].mean()) if len(itr) >= 30 else np.nan
        ia = np.flatnonzero(m & base_valid & np.isfinite(bp))
        qt = pd.DataFrame({"q": quarter[ia], "bp": bp[ia]}).groupby("q")["bp"].agg(["size", "mean"])
        qt = qt[qt["size"] >= 40]
        qpos = float((qt["mean"] > 0).mean()) if len(qt) else np.nan
        qlast = float(qt["mean"].iloc[-1]) if len(qt) else np.nan
        ok = (h1 > 0 and h2 > 0 and lo95 > 10.0 and trim > 10.0 and gtr > 0 and qpos >= 0.6 and qlast > 0)
        rows.append(dict(rule=r.rule, side=r.side, gross=g, h1=h1, h2=h2, boot_lo=lo95, trim1=trim,
                         mirror=mir, train_bp=gtr, q_pos=qpos, q_last=qlast,
                         n=len(idx), days=len(udays), pass_all=bool(ok)))
        print(f"{r.rule[:38]:<40}{r.side:<5}{g:>8.2f}{h1:>7.1f}{h2:>7.1f}{lo95:>9.2f}"
              f"{trim:>8.2f}{mir:>7.1f}{gtr:>7.2f}{qpos:>7.2f}{qlast:>8.1f}{'통과' if ok else '탈락':>6}")
    V = pd.DataFrame(rows)
    V.to_csv(OUT / f"verify_{label}.csv", index=False)
    print(f"\n다섯 검증 동시 통과: **{int(V.pass_all.sum())}/{len(V)}**  (기준: 두 반기 양수 · 일블록 부트"
          f" 95% 하한 > 비용 10bp · 상위1% 절사 후 > 10bp · TRAIN 양수 · 분기 60%↑ 양수 · 마지막 분기 양수)")
    print(f"거울상 평균: {V.mirror.mean():.2f}bp — 원 규칙 평균 {V.gross.mean():.2f}bp."
          f" 둘이 부호까지 같으면 측면이 아니라 조건이 일한 것이다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="bar0.8", help="bar0.8 | bar1.5 | up12 | up48")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", type=int, default=0, help="통과 규칙 상위 N개에 검증 4종")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.verify:
        return verify(a.label, a.verify)

    p = pd.read_parquet(PANEL)
    L = labels(p)
    y = L[a.label]
    ts = pd.to_datetime(p["timestamp"])
    n = len(p)
    base_valid = np.zeros(n, bool); base_valid[WARM : n - BAR_MAX - 2] = True
    wins = {}
    for nm, (s, e) in SPLITS.items():
        m = base_valid.copy()
        if s: m &= (ts >= pd.Timestamp(s)).to_numpy()
        if e: m &= (ts <= pd.Timestamp(e)).to_numpy()
        wins[nm] = m
    base = {nm: float(np.nanmean(y[w & np.isfinite(y)])) for nm, w in wins.items()}
    days = {nm: (ts[w].max() - ts[w].min()).total_seconds() / 86400 for nm, w in wins.items()}
    log(f"라벨 {a.label} · 무조건부 기저(롱) " + " ".join(f"{k} {v:.4f}" for k, v in base.items())
        + " | 숏 기저 = 1−롱")

    D, F = atoms(p)
    rules = enumerate_rules(D, F)
    log(f"규칙 {len(rules):,}개 (방향원자 {len(D)} · 필터 {len(F)})")

    def screen(yv: np.ndarray, bs: dict) -> list[dict]:
        keep = []
        for da, fa, side in rules:
            m = rule_mask(da, fa, D, F)
            row = {"rule": rule_name(da, fa), "side": "롱" if side > 0 else "숏", "_da": da, "_fa": fa}
            ok = True
            for nm in ("VAL", "OOS"):
                k, nn, ph = evaluate(m, side, yv, wins[nm])
                row[f"n_{nm}"], row[f"p_{nm}"] = nn, ph
                row[f"base_{nm}"] = sided(bs[nm], side)
                if nn < MIN_N or wilson_lo(k, nn) <= sided(bs[nm], side):
                    ok = False
                    break
            if not ok:
                continue
            k, nn, ph = evaluate(m, side, yv, wins["TRAIN"])
            row["n_TRAIN"], row["p_TRAIN"] = nn, ph
            row["base_TRAIN"] = sided(bs["TRAIN"], side)
            row["train_ok"] = bool(nn >= MIN_N and ph > sided(bs["TRAIN"], side))
            keep.append(row)
        return keep

    passed = screen(y, base)
    log(f"VAL·OOS 두 창 통과 {len(passed)}개 · 그중 TRAIN 확인 {sum(r['train_ok'] for r in passed)}개")

    # 가족 통제 — 라벨을 통째로 순환이동시켜 화면 전체를 다시 돌린다
    fam = []
    span = n - 2 * WARM
    for b in range(B_FAM):
        sh = int(RNG.integers(2000, span - 2000))
        ysh = np.roll(y, sh)
        bs = {nm: float(np.nanmean(ysh[w & np.isfinite(ysh)])) for nm, w in wins.items()}
        fam.append(len(screen(ysh, bs)))
        log(f"  가족귀무 {b+1}/{B_FAM}: 통과 {fam[-1]}")
    fam_mean = float(np.mean(fam))

    # 통과 규칙만 순환이동 귀무(규칙 마스크를 옮긴다 = 발동 군집·개수 보존)
    for r in passed:
        side = 1 if r["side"] == "롱" else -1
        m = rule_mask(r["_da"], r["_fa"], D, F)
        for nm in ("VAL", "OOS"):
            w = wins[nm]
            idx = np.flatnonzero(m & w)
            wi = np.flatnonzero(w)
            lo_i, sp = int(wi[0]), int(wi[-1] - wi[0])
            nul = np.empty(NSHIFT)
            for b in range(NSHIFT):
                sh = int(RNG.integers(1, sp))
                j = lo_i + ((idx - lo_i + sh) % sp)
                nul[b] = rate_at(j, side, y)[2]
            r[f"null_{nm}"] = float(np.nanmean(nul))
            r[f"pval_{nm}"] = float(np.nanmean(nul >= r[f"p_{nm}"]))
        r["per_day_OOS"] = r["n_OOS"] / days["OOS"]
        # ── 경제성: 적중률을 bp 로 **환산하지 않고**(CLAUDE.md) 실현 수익을 직접 잰다.
        if a.label.startswith("bar"):
            bmed = float(np.nanmedian(L[f"barbp{a.label[3:]}"][m & wins["OOS"]]))
            r["move_bp"], r["gross_bp"], r["excess_bp"] = bmed, (2 * r["p_OOS"] - 1) * bmed, np.nan
        else:
            H = int(a.label[2:])
            bp = L[f"bp{H}"] * side                       # 규칙 방향으로 부호를 맞춘 실현 bp
            w = wins["OOS"]
            idx = np.flatnonzero(m & w & np.isfinite(bp))
            g = float(np.mean(bp[idx]))
            wi = np.flatnonzero(w); lo_i, sp = int(wi[0]), int(wi[-1] - wi[0])
            nul = np.empty(NSHIFT)
            for b_ in range(NSHIFT):
                j = lo_i + ((idx - lo_i + int(RNG.integers(1, sp))) % sp)
                v = bp[j]
                nul[b_] = float(np.mean(v[np.isfinite(v)]))
            r["move_bp"] = float(np.nanmedian(np.abs(L[f"bp{H}"][idx])))
            r["gross_bp"] = g
            r["excess_bp"] = g - float(nul.mean())        # 순환이동 귀무 = 그 창의 드리프트 제거
            r["pval_bp"] = float((nul >= g).mean())
        for cn, c in COST_BP.items():
            r[f"net_{cn}"] = r["gross_bp"] - c

    for r in passed:
        r.pop("_da", None); r.pop("_fa", None)
    d = pd.DataFrame(passed).sort_values("p_OOS", ascending=False) if passed else pd.DataFrame()
    OUT.mkdir(parents=True, exist_ok=True)
    if len(d):
        d.to_csv(OUT / f"rules_{a.label}.csv", index=False)
    rep = {"label": a.label, "rules_tested": len(rules), "base": base, "days": days,
           "passed": len(passed), "passed_train_ok": int(sum(r["train_ok"] for r in passed)),
           "family_null_mean_pass": fam_mean, "family_null_runs": fam,
           "min_n": MIN_N, "cost_bp": COST_BP}
    (OUT / f"report_{a.label}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 118)
    print(f"라벨 {a.label} — 검정 {len(rules):,}규칙 · 통과 {len(passed)} · **귀무하 기대 통과 {fam_mean:.1f}**")
    print("=" * 118)
    if len(d):
        nl, ns = int((d.side == "롱").sum()), int((d.side == "숏").sum())
        print(f"측면 구성: 롱 {nl} · 숏 {ns}   (한쪽으로만 쏠리면 신호가 아니라 잔존 베타를 의심한다)")
        print(f"{'규칙':<44}{'방향':<5}{'VAL':>7}{'OOS':>7}{'기저':>7}{'n':>7}{'건/일':>6}"
              f"{'p':>7}{'gross':>8}{'초과':>8}{'순익10':>8}")
        for r in d.head(28).itertuples():
            print(f"{r.rule[:42]:<44}{r.side:<5}{r.p_VAL:>7.3f}{r.p_OOS:>7.3f}{r.base_OOS:>7.3f}"
                  f"{r.n_OOS:>7,}{r.per_day_OOS:>6.2f}{r.pval_OOS:>7.3f}"
                  f"{r.gross_bp:>8.2f}{r.excess_bp:>8.2f}{getattr(r, 'net_테이커'):>8.2f}")
        pos = int((d["net_테이커"] > 0).sum())
        exc = int((d["excess_bp"] > 10).sum()) if d["excess_bp"].notna().any() else 0
        print(f"\n비용선: 테이커 10bp 후 순익>0 **{pos}/{len(d)}** · 순환이동 초과분>10bp **{exc}/{len(d)}**")
    else:
        print("  통과 0")
    return 0


def _selftest() -> int:
    """배리어 라벨을 손으로 만든 봉에서 확인한다 — 규약이 틀리면 전부 틀린다."""
    hi = np.array([10, 10, 12, 10, 10.0])
    lo = np.array([10, 10, 10, 7, 10.0])
    entry = np.array([10.0] * 5)
    atr = np.array([0.1] * 5)
    t = first_touch(hi, lo, entry, atr, 1.0)      # ±10% → 위 11 / 아래 9
    assert t[0] == 1, f"봉0: i+1부터 보면 봉2 고가 12 가 먼저 → +1, got {t[0]}"
    assert t[1] == 1, f"봉1: 봉2 고가 12 → +1, got {t[1]}"
    assert t[2] == -1, f"봉2: 봉3 저가 7 → −1, got {t[2]}"
    assert t[4] == 0, "마지막 봉은 미래가 없다 → 0"
    # 경계 계약: 자기 봉은 절대 안 본다
    hi2 = np.array([99, 10, 10.0]); lo2 = np.array([10, 10, 10.0])
    assert first_touch(hi2, lo2, np.array([10.0] * 3), np.array([0.1] * 3), 1.0)[0] == 0, \
        "봉 0 자신의 고가 99 를 보면 미래참조다"
    assert abs(wilson_lo(80, 100) - 0.7111) < 1e-3, wilson_lo(80, 100)
    print("selftest OK — 배리어 첫터치 4건 · 자기봉 배제 · Wilson 하한")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
