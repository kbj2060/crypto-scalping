"""롱·숏 분리 포지션 프로파일(1·7·30일)이 가격에 주는 영향 — 사전등록 검정 (2026-10-21, 사용자 «이 데이터가 가격에 어떤 영향을 줄 수 있는지 깊게 연구»).

장부(결과 전 고정, dashboard/position_profile.py 와 같은 식): 바이낸스 5분 OI(1봉 지연)·거래량 · 회전 K=.057 · 나이 가중 exp(−나이/2일)+.02 · 30일 링.
  롱·숏 분리 = 10-10 HL 근사: 손바뀜 중 롱 몫 s = clip(.497 − .027·r1h/41.7 − .036·r24h/125.6, .05, .95) (bp) → 롱 장부 회전 2s·K·v · 숏 2(1−s)·K·v
  · OI 증감은 양쪽 같게. 대칭 장부 Y = 롱=숏(현행 화면). 진입가 = 봉 고가~저가 균등.
스냅샷: 매시 정각 봉 마감(i). 피쳐는 i 까지(OI 는 i−1 까지), 라벨은 close[i] → close[i+h]. 모든 양 / OI.
가설(창 w ∈ 1·7·30일, 주 지평 4h) — 증분 = 가격 통제 위 부분 상관(DEV 에서 맞춘 선형 잔차, 스피어만):
  H1 고통 불균형 D = 물린 숏 고통(Σx·E[max(1−e/p,0)]) − 물린 롱 고통(Σx·E[max(e/p−1,0)]) → 4h 수익(양 = 숏 스퀴즈, 음 = 평균회귀; 양방향)
  H2 본전 근처 N = 현재가 아래 0~2% 물린 숏 − 위 0~2% 물린 롱 → 4h 수익
  H3a 총 고통 M = 롱 고통 + 숏 고통 → |4h 수익|/σ (통제 = HAR 류 변동성)
  H3b 고통 쏠림 A = 롱 고통 − 숏 고통 → 4h 하방 꼬리(저가 ≤ −1.5σ4h) 대 상방 꼬리 차(아래 꼬리 1, 위 꼬리 −1, 둘 다/없음 0)
  H4 분리 증분 D − D_sym (통제에 D_sym 추가) → 4h 수익. 위약: s 를 30일 이상 원형 이동 5회 → 같은 증분의 위약 분포.
통제(가격·흐름만): r1h·r4h·r24h·r7d·r30d · VWAP 1·7·30일 거리 · σ1h·σ24h·σ7d · 거래량 24h/7d · ΔOI 1h·24h·7d · 테이커 불균형 1h·24h · |r1h|·|r4h|.
구간: DEV 2022-02~2024-12 · TEST 2025-01~2026-09-14(주 판정, 최근 우선). CI = 주 블록 부트스트랩 2000.
통과: TEST CI 가 0 배제 + Holm(15 주 검정) p<.05 + DEV 같은 부호. 통과분만 BTC·SOL·XRP 복제 + 매매(DEV 분위 문턱, 4h 보유, 왕복 8bp) + HGB 증분.
사용: --build ASSET [--placebo]  →  --test ASSET
"""
import json, os, sys
import numpy as np, pandas as pd
import numba

BV = "/home/kbj20/crypto-scalping/data/binance_vision/panel"
OUT = "tmp/position_side_price_impact_20261021"
K, TAU, RHO, M = 0.057, 576.0, 0.02, 8640
DEV, TEST = ("2022-02-01", "2025-01-01"), ("2025-01-01", "2026-09-15")
NF = 7   # 장부·창마다: qty, above, below, pain_above, pain_below, near_above(0~2%), near_below


@numba.njit(cache=True)
def _step(X, old, i, d, vv, oi_i, w, ring):
    n_ex = min(i, ring) - 1
    rem = max(-d, 0.0) + K * vv
    if rem > 0:
        tw = old * RHO; wmax = RHO
        for a in range(1, n_ex + 1):
            tw += X[(i - a) % ring] * w[a]
        if n_ex >= 1:
            wmax = max(w[1], RHO)
        if tw > 0:
            f = min(rem / tw, 1.0 / wmax)
            for a in range(1, n_ex + 1):
                j = (i - a) % ring
                X[j] -= X[j] * w[a] * f
            old -= old * RHO * f
    j0 = i % ring
    if i >= ring:
        old += X[j0]
    X[j0] = max(d, 0.0) + K * vv
    tot = old
    for a in range(0, min(i + 1, ring)):
        tot += X[(i - a) % ring]
    if tot > 0:
        r = oi_i / tot; old *= r
        for a in range(0, min(i + 1, ring)):
            X[(i - a) % ring] *= r
    return old


@numba.njit(cache=True)
def _feats(X, i, h, l, p, oi_i, ring, out):
    """out[wi, k] — 균등 [l,h] 진입가 분포로 위·아래 양·고통·0~2% 근처."""
    for a in range(0, min(i + 1, ring)):
        x = X[(i - a) % ring]
        if x <= 0:
            continue
        lo = l[i - a] / p - 1.0; hi = max(h[i - a] / p - 1.0, lo + 1e-9); wd = hi - lo
        fa = 1.0 if lo >= 0 else (0.0 if hi <= 0 else hi / wd)                              # P(e>p)
        pa = (lo + hi) / 2 if lo >= 0 else (hi * hi / (2 * wd) if hi > 0 else 0.0)        # E[max(e/p−1,0)]
        pb = -(lo + hi) / 2 if hi <= 0 else (lo * lo / (2 * wd) if lo < 0 else 0.0)       # E[max(1−e/p,0)]
        na = max(0.0, min(hi, 0.02) - max(lo, 0.0)) / wd
        nb = max(0.0, min(hi, 0.0) - max(lo, -0.02)) / wd
        for wi in range(3):
            span = 288 if wi == 0 else (2016 if wi == 1 else ring)
            if a < span:
                out[wi, 0] += x; out[wi, 1] += x * fa; out[wi, 2] += x * (1 - fa)
                out[wi, 3] += x * pa; out[wi, 4] += x * pb; out[wi, 5] += x * na; out[wi, 6] += x * nb
    for wi in range(3):
        for k in range(NF):
            out[wi, k] /= oi_i


@numba.njit(cache=True)
def build_books(oi, v, h, l, c, s, snap, ring):
    n = len(oi); w = np.exp(-np.arange(ring + 1) / TAU) + RHO
    XL = np.zeros(ring); XS = np.zeros(ring); XY = np.zeros(ring)
    oL = oi[0]; oS = oi[0]; oY = oi[0]
    F = np.zeros((len(snap), 3, 3, NF)); si = 0
    for i in range(1, n):
        d = oi[i] - oi[i - 1]
        oL = _step(XL, oL, i, d, 2 * s[i] * v[i], oi[i], w, ring)
        oS = _step(XS, oS, i, d, 2 * (1 - s[i]) * v[i], oi[i], w, ring)
        oY = _step(XY, oY, i, d, v[i], oi[i], w, ring)
        while si < len(snap) and snap[si] == i:
            _feats(XL, i, h, l, c[i], oi[i], ring, F[si, 0])
            _feats(XS, i, h, l, c[i], oi[i], ring, F[si, 1])
            _feats(XY, i, h, l, c[i], oi[i], ring, F[si, 2])
            si += 1
    return F


def load(sym):
    p = pd.read_parquet(f"{BV}/{sym}.parquet").sort_values("timestamp").reset_index(drop=True)
    p["oi"] = p.sum_open_interest.shift(1).ffill().bfill()
    return p


def side_share(c):
    r1 = (c / np.roll(c, 12) - 1) * 1e4; r24 = (c / np.roll(c, 288) - 1) * 1e4
    r1[:12] = 0; r24[:288] = 0
    return np.clip(0.497 - 0.027 * r1 / 41.7 - 0.036 * r24 / 125.6, 0.05, 0.95)


def build(sym, placebo=False):
    os.makedirs(OUT, exist_ok=True)
    p = load(sym); t5 = p.timestamp.astype("int64").to_numpy() // 10**6
    snap = np.flatnonzero(((t5 + 300_000) % 3_600_000 == 0) & (np.arange(len(p)) > M))
    c = p.close.to_numpy(float); s = side_share(c)
    args = (p.oi.to_numpy(float), p.volume.to_numpy(float), p.high.to_numpy(float), p.low.to_numpy(float), c)
    if not placebo:
        F = build_books(*args, s, snap, M)
        np.save(f"{OUT}/{sym}_F.npy", F); np.save(f"{OUT}/{sym}_snap.npy", snap)
        print(sym, "스냅샷", len(snap), "s 평균·표준편차", round(float(s.mean()), 4), round(float(s.std()), 4), flush=True)
    else:
        rng = np.random.default_rng(20261021)
        for k in range(5):
            sh = int(rng.integers(30 * 288, len(s) - 30 * 288))
            F = build_books(*args, np.roll(s, sh), snap, M)
            np.save(f"{OUT}/{sym}_F_placebo{k}.npy", F[:, :2])
            print(sym, "위약", k, "이동", sh // 288, "일", flush=True)


def controls(p):
    c = p.close; lc = np.log(c); r5 = lc.diff()
    tp = (p.high + p.low + c) / 3
    C = pd.DataFrame(index=p.index)
    for nm, k in (("r1h", 12), ("r4h", 48), ("r24h", 288), ("r7d", 2016), ("r30d", 8640)):
        C[nm] = lc - lc.shift(k)
    for nm, k in (("vw1", 288), ("vw7", 2016), ("vw30", 8640)):
        C[nm] = c / ((tp * p.volume).rolling(k).sum() / p.volume.rolling(k).sum()) - 1
    for nm, k in (("s1h", 12), ("s24", 288), ("s7d", 2016)):
        C[nm] = np.sqrt((r5 ** 2).rolling(k).mean())
    C["vol_ratio"] = np.log(p.volume.rolling(288).sum() / (p.volume.rolling(2016).sum() / 7))
    lo = np.log(p.oi)
    for nm, k in (("doi1h", 12), ("doi24", 288), ("doi7d", 2016)):
        C[nm] = lo - lo.shift(k)
    for nm, k in (("tk1h", 12), ("tk24", 288)):
        C[nm] = (2 * p.taker_buy_base.rolling(k).sum() - p.volume.rolling(k).sum()) / p.volume.rolling(k).sum()
    C["ar1h"] = C.r1h.abs(); C["ar4h"] = C.r4h.abs()
    return C


def frame(sym):
    p = load(sym); F = np.load(f"{OUT}/{sym}_F.npy"); snap = np.load(f"{OUT}/{sym}_snap.npy")
    C = controls(p).iloc[snap].reset_index(drop=True)
    c = p.close.to_numpy(); lo = p.low.to_numpy(); hi = p.high.to_numpy(); n = len(c)
    S = C.copy(); S["t"] = pd.to_datetime(p.timestamp.to_numpy()[snap]) + pd.Timedelta(minutes=5)
    for nm, k in (("y1h", 12), ("y4h", 48), ("y24h", 288)):
        j = np.minimum(snap + k, n - 1); S[nm] = np.where(snap + k < n, np.log(c[j] / c[snap]), np.nan)
    s4 = S.s24 * np.sqrt(48)                                            # 4h σ (24h 5분 rv 로)
    S["abs4"] = S.y4h.abs() / s4
    mn = np.array([lo[i + 1:i + 49].min() if i + 49 <= n else np.nan for i in snap]) / c[snap] - 1
    mx = np.array([hi[i + 1:i + 49].max() if i + 49 <= n else np.nan for i in snap]) / c[snap] - 1
    dn = mn <= -1.5 * s4; up = mx >= 1.5 * s4
    S["tail"] = np.where(dn & ~up, 1.0, np.where(up & ~dn, -1.0, 0.0)); S.loc[np.isnan(mn), "tail"] = np.nan
    for wi, w in enumerate((1, 7, 30)):
        L, Sh, Y = F[:, 0, wi], F[:, 1, wi], F[:, 2, wi]
        S[f"D{w}"] = Sh[:, 4] - L[:, 3]
        S[f"Dsym{w}"] = Y[:, 4] - Y[:, 3]
        S[f"N{w}"] = Sh[:, 6] - L[:, 5]
        S[f"M{w}"] = L[:, 3] + Sh[:, 4]
        S[f"A{w}"] = L[:, 3] - Sh[:, 4]
        S[f"Dsp{w}"] = S[f"D{w}"] - S[f"Dsym{w}"]
        S[f"Lup{w}"] = L[:, 1]; S[f"Sdn{w}"] = Sh[:, 2]                 # 서술용: 물린 롱 / 물린 숏 비율(OI 대비)
    S["per"] = np.where(S.t >= TEST[0], "TEST", np.where(S.t >= DEV[0], "DEV", "warm"))
    S.loc[S.t >= TEST[1], "per"] = "warm"
    S["wk"] = S.t.dt.to_period("W").astype(str)
    return S


CTRL = ["r1h", "r4h", "r24h", "r7d", "r30d", "vw1", "vw7", "vw30", "s1h", "s24", "s7d", "vol_ratio",
        "doi1h", "doi24", "doi7d", "tk1h", "tk24", "ar1h", "ar4h"]


def resid(S, cols, ctrl):
    """DEV 에서 맞춘 선형 잔차(순위 변환 후) — TEST 에 그대로 적용."""
    R = S[cols + ctrl].rank(pct=True)
    dev = (S.per == "DEV").to_numpy()
    X = np.column_stack([np.ones(len(S)), R[ctrl].to_numpy()])
    out = {}
    for col in cols:
        y = R[col].to_numpy(); m = dev & np.isfinite(y) & np.isfinite(X).all(1)
        b = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        out[col] = y - X @ b
    return pd.DataFrame(out, index=S.index)


def pic(S, f, y, ctrl, nb=2000, seed=20261021):
    E = resid(S, [f, y], ctrl); res = {}
    for per in ("DEV", "TEST"):
        m = (S.per == per).to_numpy() & np.isfinite(E[f]).to_numpy() & np.isfinite(E[y]).to_numpy()
        a, b, wk = E[f].to_numpy()[m], E[y].to_numpy()[m], S.wk.to_numpy()[m]
        est = float(np.corrcoef(a, b)[0, 1])
        uw = np.unique(wk); g = {u: np.flatnonzero(wk == u) for u in uw}; rng = np.random.default_rng(seed)
        bs = np.array([np.corrcoef(a[ix], b[ix])[0, 1] for ix in
                       (np.concatenate([g[u] for u in rng.choice(uw, len(uw))]) for _ in range(nb))])
        pv = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / nb)
        res[per] = {"ic": round(est, 4), "ci": [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)],
                    "p": round(float(pv), 4), "n": int(m.sum())}
    return res


def tests(sym):
    S = frame(sym); keep = (S.per != "warm").to_numpy(); S = S[keep].reset_index(drop=True)
    vctrl = ["s1h", "s24", "s7d", "ar1h", "ar4h", "vol_ratio", "doi1h", "doi24"]
    plan = []
    for w in (1, 7, 30):
        plan += [(f"H1 D{w}→y4h", f"D{w}", "y4h", CTRL), (f"H2 N{w}→y4h", f"N{w}", "y4h", CTRL),
                 (f"H3a M{w}→|y4h|", f"M{w}", "abs4", vctrl + [c for c in CTRL if c not in vctrl]),
                 (f"H3b A{w}→꼬리", f"A{w}", "tail", CTRL), (f"H4 Dsp{w}→y4h", f"Dsp{w}", "y4h", CTRL + [f"Dsym{w}"])]
    R = {nm: pic(S, f, y, ctrl) for nm, f, y, ctrl in plan}
    ps = sorted((R[k]["TEST"]["p"], k) for k in R); holm, run = {}, 0.0
    for r, (pv, k) in enumerate(ps):
        run = max(run, min(1.0, pv * (len(ps) - r))); holm[k] = run
    for k in R:
        t, d = R[k]["TEST"], R[k]["DEV"]
        R[k]["holm"] = round(holm[k], 4)
        R[k]["pass"] = bool((t["ci"][0] > 0 or t["ci"][1] < 0) and holm[k] < .05 and np.sign(t["ic"]) == np.sign(d["ic"]))
    sec = {}                                                            # 보조: 1h·24h 지평(판정 아님)
    for w in (1, 7, 30):
        for f in (f"D{w}", f"N{w}", f"Dsp{w}"):
            for y in ("y1h", "y24h"):
                sec[f"{f}→{y}"] = pic(S, f, y, CTRL + ([f"Dsym{w}"] if f.startswith("Dsp") else []), nb=500)
    raw = {}                                                            # 원시(통제 없음) 상관 — 통제가 흡수한 것
    for w in (1, 7, 30):
        for f in (f"D{w}", f"N{w}", f"Dsym{w}", f"Dsp{w}"):
            raw[f] = {per: round(float(S.loc[S.per == per, [f, "y4h"]].rank().corr().iloc[0, 1]), 4) for per in ("DEV", "TEST")}
        raw[f"D{w}~vw{w}"] = round(float(S[[f"D{w}", f"vw{w}"]].rank().corr().iloc[0, 1]), 4)
        raw[f"Dsym{w}~vw{w}"] = round(float(S[[f"Dsym{w}", f"vw{w}"]].rank().corr().iloc[0, 1]), 4)
    plac = {}                                                           # H4 위약: 같은 증분, s 원형 이동
    for fn in sorted(f for f in os.listdir(OUT) if f.startswith(f"{sym}_F_placebo")):
        Fp = np.load(f"{OUT}/{fn}")[keep]
        for wi, w in enumerate((1, 7, 30)):
            S[f"Dpl{w}"] = (Fp[:, 1, wi, 4] - Fp[:, 0, wi, 3]) - S[f"Dsym{w}"]
            plac.setdefault(f"Dsp{w}", []).append(pic(S, f"Dpl{w}", "y4h", CTRL + [f"Dsym{w}"], nb=200)["TEST"]["ic"])
    desc = {per: S.loc[S.per == per, [f"{a}{w}" for a in ("Lup", "Sdn", "D", "Dsp") for w in (1, 7, 30)]].describe().round(4).to_dict()
            for per in ("DEV", "TEST")}
    rep = {"asset": sym, "main": R, "secondary": sec, "raw_spearman": raw, "placebo_Dsp_TEST_ic": plac, "describe": desc,
           "n_pass": int(sum(R[k]["pass"] for k in R))}
    json.dump(rep, open(f"{OUT}/{sym}_report.json", "w"), ensure_ascii=False, indent=1)
    for k in R:
        print(f"{k:18s} DEV {R[k]['DEV']['ic']:+.4f} {R[k]['DEV']['ci']}  TEST {R[k]['TEST']['ic']:+.4f} {R[k]['TEST']['ci']} holm {R[k]['holm']} {'PASS' if R[k]['pass'] else ''}")
    print("통과", rep["n_pass"], "/ 15 · 위약", plac)
    return rep


def selftest():
    # 균등 분포 위·아래·고통 = 수치 적분 · s=.5 면 롱=숏=대칭 · 대시보드 Book 1일 창 양 재현
    h = np.array([110.0]); l = np.array([90.0]); X = np.array([1.0]); out = np.zeros((3, NF))
    _feats(X, 0, h, l, 100.0, 1.0, 1, out)
    e = np.linspace(90, 110, 200001)
    assert abs(out[0, 1] - .5) < 1e-6 and abs(out[0, 3] - np.mean(np.maximum(e / 100 - 1, 0))) < 1e-6
    assert abs(out[0, 4] - np.mean(np.maximum(1 - e / 100, 0))) < 1e-6 and abs(out[0, 5] - 0.1) < 1e-6
    rng = np.random.default_rng(1); n = 3000
    oi = 1e6 + np.cumsum(rng.normal(0, 2e3, n)); v = rng.uniform(1e3, 5e3, n); c = 2500 + np.cumsum(rng.normal(0, 2, n))
    F = build_books(oi, v, c + 2, c - 2, c, np.full(n, .5), np.array([n - 1]), 2000)
    assert np.allclose(F[0, 0], F[0, 1]) and np.allclose(F[0, 0], F[0, 2]), "s=.5 이면 롱=숏=대칭"
    sys.path.insert(0, os.getcwd()); from dashboard import position_profile as pp
    b = pp.Book()
    for i in range(n):
        b.push(i * 300, c[i] + 2, c[i] - 2, v[i], oi[i])
    m = b.t >= (n - 1) * 300 - 287 * 300
    assert abs(F[0, 2, 0, 0] * oi[-1] - b.x[m].sum()) / b.x[m].sum() < 1e-3, "대시보드 Book 1일 창 양과 다름"
    print("selftest OK -- 균등 분포 적분 · s=.5 대칭 · 대시보드 Book 재현")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--selftest":
        selftest()
    elif a[0] == "--build":
        build(a[1], "--placebo" in a)
    elif a[0] == "--test":
        tests(a[1])
