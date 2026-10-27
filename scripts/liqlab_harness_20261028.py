"""유동성 전략 실험실(liqlab) — 후보와 무관한 판정 장치. 2026-10-28.

사용자 «바로 테스트하지 말고 테스트 환경을 세세하게 만들어서 완벽하게 불통과/통과».
이 파일은 신호를 만들지 않는다. 신호 배열(봉 τ 끝에 알 수 있는 정보로 만든 −1/0/+1)을 받아
①체결·비용 ②귀무 ③통계 ④관문을 같은 규칙으로 적용한다. 각 후보 팔은 별도 파일이 신호만 만든다.

시각 규약(전 팔 공통 · 사건 라벨 경계 계약):
  행 τ = 5분봉 τ(시작 시각 라벨). 신호 s[τ] 는 봉 τ 종가까지 확정된 값만 쓴다.
  진입은 봉 τ+1 부터, 청산은 봉 τ+H 종가 — 라벨 탐색 시작(τ+1) > 피쳐 끝(τ).
비용(다리당 bp):
  MAKER(주): 사용자 실측 USDC peg 메이커 0.7 · 지정가 = 봉 τ 종가 · 다음 봉이 «관통»해야 체결
            (롱: 저가 < 지정가 · 같으면 미체결) · 미체결 진입 = 거래 없음(역선택이 그대로 반영됨)
            · 청산도 지정가, 다음 봉 관통 못 하면 그다음 봉 종가 테이커 4.0
  TAKER(표준): VIP0 USDT 테이커 5.0 · 진입 open[τ+1] · 청산 close[τ+H]
  손절 = 테이커(4.0 / 5.0) · 같은 봉에서 TP·SL 둘 다 닿으면 SL 먼저(비관)
관문(돈 걸린 매매 신호 = 엄격 등급, feedback_two_tier_significance):
  G1 시험창 거래당 순 bp 일 군집 부트 양측 95% CI 하한 > 0 (MAKER, 코인 묶음)
  G2 같은 건수·측면·시간대·변동성 3분위 무작위 진입 95분위 이상 + 원형 이동 95분위 이상
  G3 검증창·시험창 둘 다 순 평균 > 0
  G4 코인 일관: 시험창 양수 코인 ≥ 2/3
  G5 팔 전체 Holm(α .05) 후에도 G1 기각 유지
  G6 DSR > .95 (시행 수 = 등록 팔 × 변형 전부)
  G7 검정력: 시험창 MDE(80%) 가 사전등록 기대 효과의 2배 초과 + CI 가 0 포함이면 «판정 불능»(불통과와 구분)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

MAKER_BP, MAKER_FALLBACK_BP, TAKER_BP = 0.7, 4.0, 5.0
BARS_PER_DAY = 288


# ───────────────────────────── 체결·비용 ─────────────────────────────
def simulate(sig, o, h, l, c, H, mode="maker", tp_bp=None, sl_bp=None):
    """신호 → 비겹침 거래 목록. 반환 dict(배열): i(결정 봉), side, gross, net, exit_i, kind.

    한 번에 한 포지션(보유 중 신호는 버림). gross/net 은 bp(명목 1 기준).
    """
    n = len(c); out = {k: [] for k in ("i", "side", "gross", "net", "exit_i", "kind")}
    busy_until = -1
    for t in np.flatnonzero(sig):
        if t <= busy_until or t + H + 2 >= n:
            continue
        sd = int(np.sign(sig[t])); e = t + 1
        if mode == "maker":
            px = c[t]
            if not ((sd > 0 and l[e] < px) or (sd < 0 and h[e] > px)):
                continue                                   # 미체결 = 거래 없음
            ent, cost = px, MAKER_BP
        else:
            ent, cost = o[e], TAKER_BP
        ex_i, kind, xp = t + H, "time", None
        if tp_bp is not None or sl_bp is not None:          # 봉내 배리어(비관 순서: SL 먼저)
            for k in range(e, t + H + 1):
                adverse = ((l[k] if sd > 0 else h[k]) / ent - 1) * 1e4 * sd
                favor = ((h[k] if sd > 0 else l[k]) / ent - 1) * 1e4 * sd
                if sl_bp is not None and adverse <= -sl_bp:
                    ex_i, kind, xp = k, "sl", ent * (1 - sd * sl_bp / 1e4); break
                if tp_bp is not None and favor >= tp_bp:
                    ex_i, kind, xp = k, "tp", ent * (1 + sd * tp_bp / 1e4); break
        if kind == "sl":
            cost += MAKER_FALLBACK_BP if mode == "maker" else TAKER_BP
        elif kind == "tp":
            cost += MAKER_BP if mode == "maker" else TAKER_BP
        elif mode == "maker":                                # 시간 청산 지정가 → 미체결이면 다음 봉 테이커
            xp = c[ex_i]; nx = ex_i + 1
            if (sd > 0 and h[nx] > xp) or (sd < 0 and l[nx] < xp):
                cost += MAKER_BP
            else:
                xp, ex_i = c[nx], nx; cost += MAKER_FALLBACK_BP
        else:
            xp = c[ex_i]; cost += TAKER_BP
        g = sd * (xp / ent - 1) * 1e4
        for k_, v in zip(out, (t, sd, g, g - cost, ex_i, kind)):
            out[k_].append(v)
        busy_until = ex_i
    return {k: np.asarray(v) for k, v in out.items()}


# ───────────────────────────── 통계 ─────────────────────────────
def cluster_boot_mean(x, day, B=4000, seed=7):
    """일 군집 부트스트랩 평균 분포(거래가 같은 날 몰리는 것 반영, 이항 SE 금지)."""
    x = np.asarray(x, float); day = np.asarray(day)
    if len(x) == 0:
        return np.array([np.nan])
    u, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, x); m = np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    return s[idx].sum(1) / m[idx].sum(1)


def ci95(dist):
    return float(np.nanquantile(dist, .025)), float(np.nanquantile(dist, .975))


def p_two_sided(dist):
    d = dist[np.isfinite(dist)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / len(d)))


def holm(p: dict, alpha=.05) -> dict:
    out, run, k = {}, 0.0, len(p)
    for j, (nm, v) in enumerate(sorted(p.items(), key=lambda t: t[1])):
        run = max(run, min(1.0, v * (k - j))); out[nm] = {"p": v, "p_holm": run, "reject": run <= alpha}
    return out


def dsr(daily, trial_srs):
    """Bailey·López de Prado 디플레이티드 샤프(일 단위 수익 · trial_srs 도 같은 단위)."""
    x = np.asarray(daily, float); T = len(x); sd = x.std()
    if T < 30 or sd == 0 or len(trial_srs) < 2:
        return float("nan")
    sr = x.mean() / sd; g3 = float(((x - x.mean()) ** 3).mean() / sd ** 3); g4 = float(((x - x.mean()) ** 4).mean() / sd ** 4)
    N = len(trial_srs); V = float(np.var(trial_srs, ddof=1)); em = 0.5772156649
    sr0 = math.sqrt(V) * ((1 - em) * norm.ppf(1 - 1 / N) + em * norm.ppf(1 - 1 / (N * math.e)))
    return float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))))


def mde80(net, day):
    """양측 5%·검정력 80% 최소 검출 효과(bp/거래) = 2.8 × 일 군집 SE."""
    return float(2.8 * np.nanstd(cluster_boot_mean(net, day, B=1000, seed=3)))


# ───────────────────────────── 귀무 ─────────────────────────────
def matched_random_null(sig, strata, eligible, sim_kw, o, h, l, c, B=300, seed=11):
    """같은 건수·같은 측면·같은 층(시간대×변동성 3분위)에서 무작위로 고른 진입의 평균 순 bp 분포."""
    rng = np.random.default_rng(seed); fired = np.flatnonzero(sig)
    pools = {s_: np.flatnonzero(eligible & (strata == s_)) for s_ in np.unique(strata[eligible])}
    res = np.empty(B)
    for b in range(B):
        r = np.zeros_like(sig)
        for t in fired:
            p = pools.get(strata[t])
            if p is not None and len(p):
                r[p[rng.integers(len(p))]] = sig[t]
        tr = simulate(r, o, h, l, c, **sim_kw)
        res[b] = tr["net"].mean() if len(tr["net"]) else np.nan
    return res


def circular_null(sig, sim_kw, o, h, l, c, B=200, seed=13, min_shift=BARS_PER_DAY):
    """신호열을 하루 이상 원형 이동 — 발동 군집·건수 보존, 가격과의 정렬만 끊는다."""
    rng = np.random.default_rng(seed); n = len(sig); res = np.empty(B)
    for b in range(B):
        tr = simulate(np.roll(sig, int(rng.integers(min_shift, n - min_shift))), o, h, l, c, **sim_kw)
        res[b] = tr["net"].mean() if len(tr["net"]) else np.nan
    return res


def strata_of(ts_hour, rv, q=None):
    """시간대(UTC 시) × 실현변동성 3분위 층 번호. q(분위 경계)는 학습창에서 정해 넘긴다."""
    if q is None:
        q = np.nanquantile(rv, [1 / 3, 2 / 3])
    return ts_hour * 3 + np.digitize(np.nan_to_num(rv, nan=q[0]), q)


# ───────────────────────────── 판정 ─────────────────────────────
def evaluate_window(sig, px, day, strata, H, tp_bp=None, sl_bp=None, B_null=300):
    """한 창·한 코인 채점. px = dict(o,h,l,c) · day = 봉별 일 번호."""
    o, h, l, c = px["o"], px["h"], px["l"], px["c"]
    out = {}
    for mode in ("maker", "taker"):
        kw = dict(H=H, mode=mode, tp_bp=tp_bp, sl_bp=sl_bp)
        tr = simulate(sig, o, h, l, c, **kw)
        n = len(tr["net"])
        r = {"n": n, "days": int(len(np.unique(day[tr["i"]]))) if n else 0,
             "net": tr["net"].tolist(), "day": day[tr["i"]].tolist() if n else []}
        if n >= 10:
            d = cluster_boot_mean(tr["net"], day[tr["i"]])
            r.update(mean_net=float(tr["net"].mean()), mean_gross=float(tr["gross"].mean()),
                     ci95=ci95(d), p=p_two_sided(d), mde80=mde80(tr["net"], day[tr["i"]]),
                     hit=float((tr["net"] > 0).mean()),
                     exits={k: int((tr["kind"] == k).sum()) for k in ("time", "tp", "sl")})
            if mode == "maker":
                elig = np.ones(len(c), bool); elig[-H - 3:] = False
                rn = matched_random_null(sig, strata, elig, kw, o, h, l, c, B=B_null)
                cn = circular_null(sig, kw, o, h, l, c, B=max(100, B_null // 2))
                r["null_random_pct"] = float((rn < r["mean_net"]).mean() * 100)
                r["null_circular_pct"] = float((cn < r["mean_net"]).mean() * 100)
                lg = simulate(np.where(sig != 0, 1, 0), o, h, l, c, **kw)["net"]
                sh = simulate(np.where(sig != 0, -1, 0), o, h, l, c, **kw)["net"]
                r["excess_vs_long"] = r["mean_net"] - float(lg.mean()) if len(lg) else None
                r["excess_vs_short"] = r["mean_net"] - float(sh.mean()) if len(sh) else None
        out[mode] = r
    return out


def gates(per_coin_val, per_coin_test, expected_bp, pooled_test_ci, holm_reject, dsr_val):
    """관문 G1~G7 → «통과/불통과/판정 불능» 과 이유. 입력은 evaluate_window 결과(MAKER 기준)."""
    t = [v["maker"] for v in per_coin_test.values()]; vv = [v["maker"] for v in per_coin_val.values()]
    ok_t = [x for x in t if x.get("n", 0) >= 10]
    if not ok_t:
        return {"verdict": "판정 불능", "why": "시험창 거래 < 10"}
    g = {"G1": pooled_test_ci[0] > 0,
         "G2": all(x["null_random_pct"] >= 95 and x["null_circular_pct"] >= 95 for x in ok_t),
         "G3": all(x.get("mean_net", -1) > 0 for x in vv if x.get("n", 0) >= 10) and all(x["mean_net"] > 0 for x in ok_t),
         "G4": sum(x["mean_net"] > 0 for x in ok_t) >= math.ceil(2 / 3 * len(ok_t)),
         "G5": bool(holm_reject),
         "G6": (dsr_val or 0) > .95}
    mde = float(np.median([x["mde80"] for x in ok_t]))
    g["G7_mde80"], g["expected_bp"] = mde, expected_bp
    if all(g[k] for k in ("G1", "G2", "G3", "G4", "G5", "G6")):
        return {"verdict": "통과", **g}
    if mde > 2 * expected_bp and pooled_test_ci[0] <= 0 < pooled_test_ci[1]:
        return {"verdict": "판정 불능", "why": "MDE 가 기대 효과의 2배 초과 · CI 가 0 포함", **g}
    return {"verdict": "불통과", **g}


# ───────────────────────────── 자가 점검 ─────────────────────────────
def _synthetic(n=288 * 120, seed=0, vol_bp=12.0):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n) * vol_bp / 1e4
    c = 2000 * np.exp(np.cumsum(r)); o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.standard_normal(n)) * vol_bp / 2e4 * c * (rng.random(n) < .6)   # 40% 봉은 꼬리 없음 → 메이커 체결이 방향에 의존(역선택)
    return o, np.maximum(o, c) + w, np.minimum(o, c) - w, c, r


def selfcheck():
    # SC1 손계산 — 테이커 롱 1건: 진입 open[1]=100(open[0]=99 를 쓰면 미래참조 아닌 «현재봉» 오류) → close[3]=101
    #   ⇒ gross 100bp, net 90bp
    o = np.array([99, 100, 100, 100, 100, 100, 100, 100, 100, 100.]); c = np.array([100, 100, 100, 101, 101, 101, 101, 101, 101, 101.])
    h, l = np.maximum(o, c) + .5, np.minimum(o, c) - .5
    s = np.zeros(10, int); s[0] = 1
    tr = simulate(s, o, h, l, c, H=3, mode="taker")
    assert abs(tr["gross"][0] - 100) < 1e-9 and abs(tr["net"][0] - 90) < 1e-9, tr
    # SC2 메이커 관통 — 저가 == 지정가면 미체결
    l2 = l.copy(); l2[1] = c[0]
    assert len(simulate(s, o, h, l2, c, H=3, mode="maker")["net"]) == 0
    # SC2b 메이커 시간 청산 미체결 → 다음 봉 종가 테이커: 진입 100(메이커 .7) · 청산 지정가 101 을 봉 4 가 못 넘음
    #   → 봉 4 종가 101.5 테이커 4.0 ⇒ gross 150, net 145.3
    c5 = np.array([100, 100, 100, 101, 101.5, 101.5, 101.5, 101.5, 101.5, 101.5]); o5 = np.r_[100, c5[:-1]]
    h5, l5 = np.maximum(o5, c5) + .5, np.minimum(o5, c5) - .5; h5[4] = 101.0
    tr = simulate(s, o5, h5, l5, c5, H=3, mode="maker")
    assert abs(tr["gross"][0] - 150) < 1e-9 and abs(tr["net"][0] - 145.3) < 1e-9 and tr["exit_i"][0] == 4, tr
    # SC3 같은 봉 TP·SL 동시 → SL 먼저
    o3 = np.full(10, 100.); c3 = np.full(10, 100.); h3 = np.full(10, 100.); l3 = np.full(10, 100.)
    h3[2], l3[2] = 102, 98
    tr = simulate(s, o3, h3, l3, c3, H=3, mode="taker", tp_bp=100, sl_bp=100)
    assert tr["kind"][0] == "sl" and abs(tr["gross"][0] + 100) < 1e-9, tr
    # SC4 비겹침 — 보유 중(봉 1~3) 신호 버림, 청산 뒤(봉 4) 신호는 받음
    s4 = np.zeros(10, int); s4[0] = s4[1] = s4[4] = 1
    assert list(simulate(s4, o, h, l, c, H=3, mode="taker")["i"]) == [0, 4]
    # SC5 음성 대조 — 무작위 신호 40회, CI 하한>0 거짓 양성 ≤ 6/40
    O, Hh, L, C, R = _synthetic()
    day = np.arange(len(C)) // BARS_PER_DAY; fp = 0
    for k in range(40):
        g_ = np.random.default_rng(100 + k)
        sg = np.where(g_.random(len(C)) < .01, g_.choice([-1, 1], len(C)), 0)
        tr = simulate(sg, O, Hh, L, C, H=6, mode="taker")
        fp += ci95(cluster_boot_mean(tr["gross"], day[tr["i"]], B=800))[0] > 0
    assert fp <= 6, f"음성 대조 거짓 양성 {fp}/40"
    # SC6 양성 대조 — 미래 6봉 수익 부호를 70% 맞히는 신호는 메이커 순 CI·무작위 95분위를 통과해야
    #   (60% 는 합성 역선택이 gross 5.3→2.0bp 로 깎아 메이커 순 0 — 지정가는 불리할 때만 체결된다)
    rng = np.random.default_rng(5)
    fwd = np.array([R[t + 1:t + 7].sum() for t in range(len(R))])
    sg = np.zeros(len(C), int); fire = rng.random(len(C)) < .01
    sg[fire] = np.where(rng.random(fire.sum()) < .7, np.sign(fwd[fire]), -np.sign(fwd[fire])).astype(int)
    tr = simulate(sg, O, Hh, L, C, H=6, mode="maker")
    assert ci95(cluster_boot_mean(tr["net"], day[tr["i"]]))[0] > 0, "양성 대조 못 잡음"
    st = strata_of((np.arange(len(C)) % BARS_PER_DAY) // 12, np.abs(R))
    el = np.r_[np.ones(len(C) - 9, bool), np.zeros(9, bool)]
    rn = matched_random_null(sg, st, el, dict(H=6, mode="maker"), O, Hh, L, C, B=60)
    assert (rn < tr["net"].mean()).mean() >= .95, "양성 대조가 무작위 귀무에 묻힘"
    # SC6b 역선택 — 무작위 신호의 메이커 체결은 일부만(<90%)이고, 체결분 gross 는 테이커 gross 보다 낮아야
    g_ = np.random.default_rng(9); sg3 = np.where(g_.random(len(C)) < .01, g_.choice([-1, 1], len(C)), 0)
    tm = simulate(sg3, O, Hh, L, C, H=6, mode="maker"); tt = simulate(sg3, O, Hh, L, C, H=6, mode="taker")
    assert len(tm["net"]) < .9 * len(tt["net"]), "메이커가 무조건 체결 — 역선택 미반영"
    # SC7 미래참조 트립와이어 — 봉 τ+1 수익 부호를 τ 에 알면 gross 가 비현실적으로 커져야(검출 가능)
    leak = np.zeros(len(C), int); leak[fire] = np.sign(np.r_[R[1:], 0][fire]).astype(int)
    assert simulate(leak, O, Hh, L, C, H=1, mode="taker")["gross"].mean() > 5, "트립와이어 둔감"
    # SC8 Holm 단조 · DSR 범위
    hm = holm({"a": .01, "b": .02, "c": .5})
    assert abs(hm["a"]["p_holm"] - .03) < 1e-12 and abs(hm["b"]["p_holm"] - .04) < 1e-12 and not hm["c"]["reject"]
    hm = holm({"a": .01, "b": .012, "c": .5})          # b 원값 .024 < a .03 → 단조 보정으로 .03
    assert abs(hm["b"]["p_holm"] - .03) < 1e-12
    v = dsr(np.random.default_rng(1).standard_normal(400) * .01 + .002, list(np.random.default_rng(2).normal(0, .05, 20)))
    assert 0 <= v <= 1
    print("liqlab selfcheck OK (SC1~SC8)")


if __name__ == "__main__":
    selfcheck()
