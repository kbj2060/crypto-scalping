#!/usr/bin/env python3
"""K — **「이 창에서 숏이었다」를 실력과 분리한다.** 부모 후보의 가장 큰 미해결 경고.

## 왜 지금 증거로는 못 가르는가
`모델−숏`은 **비대칭 귀무**다. 숏이 이긴 창에서 "항상 숏"은 사후에나 고를 수 있는
벤치마크라 이걸 못 이긴다고 실력이 0인 것도, 이긴다고 실력이 있는 것도 아니다.
두 질문이 섞여 있다: ① 어느 봉에 들어갔나 ② 그 봉에서 **어느 쪽**을 골랐나.

## 분해 (항등식, 근사 아님)
후보 통과봉에서 s ∈ {+1,−1}, r = 전방수익(bp) 일 때

    mean(s·r)  =  mean(s)·mean(r)  +  cov(s, r)
      총bp          「기울기」항         「타이밍」항

**기울기 항 = 「이 창에서 숏이었다」를 정확히 정량화한 값**이다(측면비중 × 그 봉들의 드리프트).
모델이 같은 측면비중으로 **무작위로** 측면을 골랐다면 기대값이 정확히 이 값이다.
**타이밍 항만이 측면 선택 실력이다.** 코드로 1e-9 이내 항등식을 assert 한다.

선택 항도 같이 낸다: mean(r|통과봉) − mean(r|전체) = 「어느 봉을 골랐나」의 드리프트 기여.

## 왜 한 창으론 안 되는가 — 인과 워크포워드 6폴드
드리프트 부호가 **반대인 창들에서 타이밍 항이 같이 양수**여야 실력이다. 현행 후보의 TRAIN 이
2025-01~2026-02 뿐이라 **2022~2024 가 통째로 인과 폴드**로 남는다(독립일 122 → 1,000+).
각 폴드는 TRAIN 이 TEST 앞이고 겹치지 않는다(assert).

## 사전등록 판정규칙 (결과 보기 전에 고정)
 · **해소**: 타이밍 항이 6폴드 중 **≥5 양수** & **풀링 CI 하한 > 0** & **드리프트 음수 폴드와
   양수 폴드 양쪽에서 각각 양수**.
 · **확정(경고가 사실)**: 풀링 타이밍 CI 가 0 을 포함하고 총bp 의 과반이 기울기 항.
 · 그 사이면 **미해결** — 부모를 동결하지 않는다.

## 부수 — 시드 측면 일치율
「롱비중 시드폭 16~47%」가 잡음인지 본다. 5시드 쌍별로 **둘 다 비CASH 인 봉에서 측면 일치율**을
재고, 각 시드의 롱비중이 주는 **우연 일치율**과 비교한다. 우연 수준이면 측면 머리는 의견이 없다.

⚠️ 2022~2024 는 README §5 에서 아래층(L4/L5) 재료로 예정된 구간이다. 이 스크립트는
**사전등록된 단일 진단**만 돌리고 구성 선택에 쓰지 않는다. 그래도 접촉 사실은 리포트에 남는다.
⚠️ 경제 수치는 총이익 방향 읽기다(라이브 ATR 배리어·exit·라우터·사이징 아님). 승격 근거 아님.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_parent_quality_calibration_and_bias_20260917 as F  # noqa: E402
import research_omega461_parent_pnl_concentration_20260917 as G  # noqa: E402
import research_omega461_parent_debias_retrain_20260917 as H  # noqa: E402

# 현행 후보 = base/old 가중/대칭게이트/5시드 앙상블. 레시피를 폴드마다 그대로 복제한다.
SEEDS5 = H.SEEDS5
TARGET = H.TARGET
FOLDS = [
    # 이름   TRAIN 시작      TRAIN 끝        TEST 시작       TEST 끝
    ("F1", "2022-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("F2", "2022-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("F3", "2022-01-01", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("F4", "2022-01-01", "2024-12-31", "2025-01-01", "2025-06-30"),
    ("F5", "2022-01-01", "2025-06-30", "2025-07-01", "2025-12-31"),
    ("CAND", "2025-01-01", "2026-02-28", "2026-03-01", "2026-06-30"),   # 현행 후보 창 그대로
]
# 지평. 캐시는 공유라 재학습 없이 바뀐다. `--hz=12h` 또는 `--h4`(구 표기).
BARS = {"1h": 12, "4h": 48, "8h": 96, "12h": 144, "1d": 288}
HZ = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--hz=")),
          "4h" if "--h4" in sys.argv else "1h")
assert HZ in BARS, f"모르는 지평: {HZ}"
E.HORIZONS.setdefault(HZ, BARS[HZ])        # E.load() 가 fwd_{HZ}_bp 를 만들게 한다
SUF = "" if HZ == "1h" else f"_{HZ}"
# 학습창 팔. deep = 2022-01 부터 확장(기본, 캐시 있음) · base = 테스트 직전 14개월 후행.
# 같은 테스트 창에서 깊이만 바꿔 짝지어 비교하기 위한 것이다.
ARM = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--arm=")), "deep")
assert ARM in ("deep", "base"), f"모르는 팔: {ARM}"
ASUF = "" if ARM == "deep" else f"_{ARM}"
_FA = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--folds=")), "")
FSUF = "_" + _FA.replace(",", "") if _FA else ""
OUTJ = E.OUT / f"stageK_side_skill{SUF}{FSUF}{ASUF}.json"
CACHE = E.OUT / "stageK_probs.npz"


def log(*a, **k): print(*a, flush=True)


def decompose(side, fwd):
    """총bp = 기울기 + 타이밍. 항등식이므로 assert 로 지킨다."""
    ok = (side != 0) & np.isfinite(fwd)
    s, r = side[ok], fwd[ok]
    gross, tilt = float((s * r).mean()), float(s.mean() * r.mean())
    timing = gross - tilt
    assert abs(gross - (tilt + timing)) < 1e-9, "분해 항등식 파손"
    assert abs(timing - float(np.cov(s, r, bias=True)[0, 1])) < 1e-6, "타이밍 ≠ cov(s,r)"
    return {"n": int(ok.sum()), "gross_bp": gross, "tilt_bp": tilt, "timing_bp": timing,
            "side_mean": float(s.mean()), "cand_drift_bp": float(r.mean())}, ok


def block_boot(s, r, days, n=2000, seed=17):
    """날짜블록 부트스트랩. 총·기울기·타이밍을 **재표집 안에서 다시 계산**한다."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    idx = {d: np.where(days == d)[0] for d in uniq}
    g = np.empty(n); t = np.empty(n); tl = np.empty(n)
    for i in range(n):
        pick = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        ss, rr = s[pick], r[pick]
        m = float((ss * rr).mean()); b = float(ss.mean() * rr.mean())
        g[i], tl[i], t[i] = m, b, m - b
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"gross": ci(g), "tilt": ci(tl), "timing": ci(t)}, len(uniq)


def seed_side_agreement(Ds):
    """쌍별 측면 일치율 vs 각 시드 롱비중이 주는 «우연 일치율»."""
    acts = [D.argmax(1) for D in Ds]
    obs, chance = [], []
    for i in range(len(acts)):
        for j in range(i + 1, len(acts)):
            m = (acts[i] != 0) & (acts[j] != 0)
            if m.sum() < 100:
                continue
            obs.append(float((acts[i][m] == acts[j][m]).mean()))
            pi = float((acts[i][m] == 1).mean()); pj = float((acts[j][m] == 1).mean())
            chance.append(pi * pj + (1 - pi) * (1 - pj))
    return {"pairs": len(obs), "agree": float(np.mean(obs)) if obs else float("nan"),
            "agree_chance": float(np.mean(chance)) if chance else float("nan")}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"지평={HZ} · device={device} · seeds={SEEDS5} · 레시피=base/old가중/대칭게이트/5시드앙상블")
    df, base_cols = E.load()
    cache = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}
    if cache:
        log(f"⚡확률 캐시 재사용: {CACHE} ({len(cache)}개 폴드)")

    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--folds=")), None)
    rows, pooled = [], []
    for name, t0, t1, v0, v1 in FOLDS:
        if only and name not in only:      # 배포본과 같은 폴드로 맞춰 견줄 때 쓴다
            continue
        if ARM == "base":                  # 후행 14개월 -- 배포 부모의 학습 깊이와 같은 눈금
            t0 = str((pd.Timestamp(v0) - pd.DateOffset(months=14)).date())
        ck = name if ARM == "deep" else f"{name}|{ARM}"
        tm = (df.timestamp >= t0) & (df.timestamp <= t1 + " 23:59:59")
        vm = (df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")
        tr, te = df[tm].reset_index(drop=True), df[vm].reset_index(drop=True)
        assert len(te) > 10_000, f"{name} TEST 가 너무 작다: {len(te)}"
        assert tr.timestamp.max() < te.timestamp.min(), f"{name} TRAIN 이 TEST 를 침범"
        yt = pd.to_numeric(tr["zigzag_action"]).to_numpy(np.int64)
        yv = pd.to_numeric(te["zigzag_action"]).to_numpy(np.int64)
        rt = tabm._route_probs(tr)
        ev_expert = tabm._route_probs(te).argmax(1)
        tdays = te.timestamp.dt.floor("D").to_numpy()
        fwd = te[f"fwd_{HZ}_bp"].to_numpy(np.float64)
        n = len(tr); split = max(int(n * 0.85), min(n - 1, 512))
        log(f"\n{'='*92}\n=== {name}  TRAIN {t0}~{t1} {n:,}행 · TEST {v0}~{v1} {len(te):,}행 "
            f"· 독립일 {len(np.unique(tdays))}\n{'='*92}")

        if ck in cache:
            z = dict(cache[ck].item())
            Ds, Qs, tDm, tQm = list(z["Ds"]), list(z["Qs"]), z["tDm"], z["tQm"]
        else:
            xs, scaler = tabm._standardize_fit(tabm._base_input(tr, base_cols))
            xv_std = tabm._standardize_apply(tabm._base_input(te, base_cols), scaler)
            tail_expert = rt[split:].argmax(1)
            Ds, Qs, tDs, tQs = [], [], [], []
            for seed in SEEDS5:
                D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
                tD = np.zeros((n - split, 3)); tQ = np.zeros((n - split, 3))
                for ei in range(3):
                    w = H.w_old(yt, rt[:, ei].astype(np.float32))
                    m, _ = E.fit_expert(xs[:split], yt[:split], w[:split],
                                        xs[split:], yt[split:], w[split:],
                                        seed=seed, ei=ei, device=device)
                    sel = ev_expert == ei
                    if sel.any():
                        D[sel], Q[sel] = E.heads(m, xv_std[sel], device)
                    ts = tail_expert == ei
                    if ts.any():
                        tD[ts], tQ[ts] = E.heads(m, xs[split:][ts], device)
                Ds.append(D); Qs.append(Q); tDs.append(tD); tQs.append(tQ)
                ls = (D.argmax(1) == 1).sum() / max((D.argmax(1) != 0).sum(), 1)
                log(f"  seed {seed}: 방향 롱비중(게이트 전) {ls:.3f}")
            tDm, tQm = np.mean(tDs, 0), np.mean(tQs, 0)
            cache[ck] = np.array({"Ds": Ds, "Qs": Qs, "tDm": tDm, "tQm": tQm}, dtype=object)
            np.savez(CACHE, **cache)

        Dm, Qm = np.mean(Ds, 0), np.mean(Qs, 0)
        ql, qsh = H.thresholds(tDm, tQm, symmetric=True)        # TRAIN 꼬리에서만, TEST 미사용
        side = H.side_from(Dm, Qm, ql, qsh)

        dec, ok = decompose(side, fwd)
        ci, nd = block_boot(side[ok], fwd[ok], tdays[ok])
        fin = np.isfinite(fwd)
        r = {"fold": name, "train": [t0, t1], "test": [v0, v1], "train_rows": n,
             "q_long": ql, "q_short": qsh, "indep_days": nd,
             "all_bar_drift_bp": float(fwd[fin].mean()),
             "selection_bp": dec["cand_drift_bp"] - float(fwd[fin].mean()),
             **dec, "ci95": ci,
             "seed_agreement": seed_side_agreement(Ds),
             **{k: v for k, v in F.bias_report(name, side, fwd, yv, tdays).items()
                if k in ("long_share", "per_side_bp", "excess_vs_long", "excess_vs_short")},
             **{k: v for k, v in G.day_stats(side, fwd, tdays).items()
                if k in ("median_bp", "pos_day_share", "drop_top1d_bp")}}
        rows.append(r)
        pooled.append((side[ok], fwd[ok], tdays[ok]))
        sa = r["seed_agreement"]
        log(f"  통과율 {dec['n']/len(te):.3f} · 롱 {r['long_share']*100:.1f}% · "
            f"전체봉 드리프트 {r['all_bar_drift_bp']:+.3f} → 통과봉 {dec['cand_drift_bp']:+.3f} "
            f"(선택 {r['selection_bp']:+.3f})")
        log(f"  ⭐총 {dec['gross_bp']:+7.3f} = 기울기 {dec['tilt_bp']:+7.3f} + "
            f"타이밍 {dec['timing_bp']:+7.3f}  CI(타이밍)[{ci['timing'][0]:+.3f},{ci['timing'][1]:+.3f}]"
            f"{'  🟢0배제' if ci['timing'][0] > 0 else '  0포함'}")
        log(f"  모델−롱 {r['excess_vs_long']['bp']:+.2f} · 모델−숏 {r['excess_vs_short']['bp']:+.2f} · "
            f"중앙 {r['median_bp']:+.2f} · 양수일 {r['pos_day_share']*100:.1f}% · "
            f"시드측면일치 {sa['agree']:.3f} (우연 {sa['agree_chance']:.3f})")
        OUTJ.write_text(json.dumps(rows, indent=2, default=float))

    # ── 풀링 판정 ──
    ps = np.concatenate([p[0] for p in pooled]); pr = np.concatenate([p[1] for p in pooled])
    pd_ = np.concatenate([p[2] for p in pooled])
    pdec, _ = decompose(ps, pr)
    pci, pnd = block_boot(ps, pr, pd_)
    pos = sum(1 for r in rows if r["timing_bp"] > 0)
    up = [r for r in rows if r["all_bar_drift_bp"] > 0]
    dn = [r for r in rows if r["all_bar_drift_bp"] <= 0]
    up_ok = bool(up) and all(r["timing_bp"] > 0 for r in up)
    dn_ok = bool(dn) and all(r["timing_bp"] > 0 for r in dn)

    log(f"\n{'='*92}\n=== 폴드 요약 ===")
    log(f"{'폴드':<6}{'독립일':>7}{'드리프트':>10}{'통과':>7}{'롱%':>7}"
        f"{'총bp':>9}{'기울기':>9}{'타이밍':>9}{'CI(타이밍)':>22}")
    for r in rows:
        c = r["ci95"]["timing"]
        log(f"{r['fold']:<6}{r['indep_days']:>7}{r['all_bar_drift_bp']:>+10.3f}{r['n']:>7,}"
            f"{r['long_share']*100:>6.1f}%{r['gross_bp']:>+9.3f}{r['tilt_bp']:>+9.3f}"
            f"{r['timing_bp']:>+9.3f}   [{c[0]:+7.3f},{c[1]:+7.3f}]{' 🟢' if c[0] > 0 else ''}")
    log(f"\n풀링({pnd}일 · n {pdec['n']:,}): 총 {pdec['gross_bp']:+.3f} = "
        f"기울기 {pdec['tilt_bp']:+.3f} + 타이밍 {pdec['timing_bp']:+.3f} "
        f"CI[{pci['timing'][0]:+.3f},{pci['timing'][1]:+.3f}]")
    log(f"타이밍 양수 폴드 {pos}/{len(rows)} · 드리프트양수창 전부양수 {up_ok}({len(up)}개) · "
        f"드리프트음수창 전부양수 {dn_ok}({len(dn)}개)")

    resolved = pos >= 5 and pci["timing"][0] > 0 and up_ok and dn_ok
    tilt_dominates = abs(pdec["tilt_bp"]) > abs(pdec["timing_bp"])
    confirmed = pci["timing"][0] <= 0 <= pci["timing"][1] and tilt_dominates
    verdict = "해소 — 측면 타이밍 실력이 실재" if resolved else \
              ("확정 — 경고가 사실, 성과의 정체는 창의 측면 기울기" if confirmed else
               "미해결 — 부모를 동결하지 않는다")
    log(f"\n⭐사전등록 판정: {verdict}")
    (OUTJ).write_text(json.dumps({"folds": rows, "pooled": {**pdec, "ci95": pci, "indep_days": pnd},
                                  "verdict": verdict, "resolved": resolved, "confirmed": confirmed,
                                  "timing_positive_folds": pos,
                                  "touched_window_2022_2024": True,
                                  "seeds": SEEDS5}, indent=2, default=float))
    log(f"저장: {OUTJ}")
    return 0


def robust() -> int:
    """--robust: 캐시 재사용(재학습 없음). 타이밍 항이 «꼬리 몇 건»인지 «폭»인지 가른다.

    셋을 같이 낸다:
      · 최고1일/최고3일 **제거 후** 타이밍 항
      · 수익을 ±1% 로 **윈저라이즈** 한 뒤의 타이밍 항 (꼬리 의존 제거)
      · ⭐**측면 적중률** = mean(s == sign(r)) -- 크기와 무관한 꼬리 없는 측면 실력 지표.
        귀무는 0.5 가 아니라 **그 창에서 s 를 섞었을 때의 기대 적중률**이다
        (드리프트가 있으면 상수 편향만으로도 0.5 를 넘는다). 그래서 같은 측면비중을 유지한
        채 s 를 날짜블록 안에서 순열해 귀무 분포를 만든다.
    """
    cache = dict(np.load(CACHE, allow_pickle=True))
    df, base_cols = E.load()
    rows, pooled = [], []
    for name, t0, t1, v0, v1 in FOLDS:
        z = dict(cache[name].item())
        tm = (df.timestamp >= t0) & (df.timestamp <= t1 + " 23:59:59")
        vm = (df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")
        te = df[vm].reset_index(drop=True)
        fwd = te[f"fwd_{HZ}_bp"].to_numpy(np.float64)
        days = te.timestamp.dt.floor("D").to_numpy()
        ql, qsh = H.thresholds(z["tDm"], z["tQm"], symmetric=True)
        side = H.side_from(np.mean(list(z["Ds"]), 0), np.mean(list(z["Qs"]), 0), ql, qsh)
        ok = (side != 0) & np.isfinite(fwd)
        s, r, d = side[ok], fwd[ok], days[ok]
        timing = lambda ss, rr: float((ss * rr).mean() - ss.mean() * rr.mean())

        dsum = {u: (s[d == u] * r[d == u]).sum() for u in np.unique(d)}
        order = sorted(dsum, key=lambda u: -dsum[u])
        drops = {}
        for k in (1, 3):
            keep = ~np.isin(d, order[:k])
            drops[f"timing_drop_top{k}d"] = timing(s[keep], r[keep]) if keep.sum() > 50 else float("nan")
        w = np.clip(r, -100.0, 100.0)                      # ±1% 윈저라이즈

        hit = float((s == np.sign(r)).mean())
        rng = np.random.default_rng(11)                    # 날짜블록 안 순열 = 측면비중 보존
        null = np.empty(2000)
        for i in range(2000):
            sp = s.copy()
            for u in np.unique(d):
                m = d == u
                sp[m] = rng.permutation(s[m])
            null[i] = float((sp == np.sign(r)).mean())
        pv = float((null >= hit).mean())
        rows.append({"fold": name, "n": int(ok.sum()), "timing": timing(s, r), **drops,
                     "timing_winsor": timing(s, w), "hit": hit,
                     "hit_null": float(null.mean()), "hit_p": pv})
        pooled.append((s, r, d))
        log(f"{name:<6} 타이밍 {rows[-1]['timing']:+7.3f} → 최고1일제거 {drops['timing_drop_top1d']:+7.3f} "
            f"· 최고3일제거 {drops['timing_drop_top3d']:+7.3f} · 윈저 {rows[-1]['timing_winsor']:+7.3f} "
            f"· 측면적중 {hit:.4f} (귀무 {null.mean():.4f}, p {pv:.4f})")

    s = np.concatenate([p[0] for p in pooled]); r = np.concatenate([p[1] for p in pooled])
    d = np.concatenate([p[2] for p in pooled])
    timing = lambda ss, rr: float((ss * rr).mean() - ss.mean() * rr.mean())
    dsum = {u: (s[d == u] * r[d == u]).sum() for u in np.unique(d)}
    order = sorted(dsum, key=lambda u: -dsum[u])
    hit = float((s == np.sign(r)).mean())
    rng = np.random.default_rng(11); null = np.empty(2000)
    for i in range(2000):
        sp = s.copy()
        for u in np.unique(d):
            m = d == u
            sp[m] = rng.permutation(s[m])
        null[i] = float((sp == np.sign(r)).mean())
    out = {"pooled_timing": timing(s, r),
           "pooled_timing_drop_top1d": timing(s[~np.isin(d, order[:1])], r[~np.isin(d, order[:1])]),
           "pooled_timing_drop_top3d": timing(s[~np.isin(d, order[:3])], r[~np.isin(d, order[:3])]),
           "pooled_timing_drop_top10d": timing(s[~np.isin(d, order[:10])], r[~np.isin(d, order[:10])]),
           "pooled_timing_winsor": timing(s, np.clip(r, -100.0, 100.0)),
           "pooled_hit": hit, "pooled_hit_null": float(null.mean()),
           "pooled_hit_p": float((null >= hit).mean()),
           "pooled_days": int(len(np.unique(d))), "pooled_n": int(len(s)), "folds": rows}
    log(f"\n풀링 {out['pooled_n']:,}건/{out['pooled_days']}일: 타이밍 {out['pooled_timing']:+.3f} → "
        f"최고1일제거 {out['pooled_timing_drop_top1d']:+.3f} · 최고3일제거 {out['pooled_timing_drop_top3d']:+.3f} "
        f"· 최고10일제거 {out['pooled_timing_drop_top10d']:+.3f} · 윈저 {out['pooled_timing_winsor']:+.3f}")
    log(f"⭐측면 적중률 {hit:.4f} vs 귀무 {null.mean():.4f} (측면비중 보존 날짜블록 순열) · p {out['pooled_hit_p']:.4f}")
    (E.OUT / f"stageK_robust{SUF}.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"저장: {E.OUT}/stageK_robust{SUF}.json")
    return 0


# 배포 부모의 TRAIN/VAL/OOS 를 덮는 구간. 이 안과 겹치는 TEST 창은 배포본에겐 표본내다.
DEPLOYED_SEEN = ("2025-01-01", "2026-02-28")


def deployed() -> int:
    """--deployed: **배포 부모에 같은 분해를 건다.** 고정 아티팩트라 추론만 — 학습 없음.

    지금까지 배포본은 창 하나(VAL 2026-03~06)로만 봤고 내 후보만 742일로 쟀다.
    같은 자로 재지 않으면 「배포본이 이겼다」가 아니라 「배포본만 시험을 덜 봤다」가 된다.
    게이트는 **배포본의 실제 임계값 q=0.75 단일**(내 대칭게이트는 내 변경분이라 안 씌운다).
    """
    E.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"지평={HZ} · device={device} · 배포본 단일 q={E.Q_THRESH} · 추론만(학습 없음)")
    df, base_cols = E.load()
    b = torch.load(E.BUNDLE, map_location="cpu", weights_only=False)
    experts = {}
    for ename in ("bull", "bear", "chop"):
        pay = dict(b["models"][ename])
        m = tabm.ThreeHeadTabM(int(pay["n_features"]),
                               cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(device)
        m.load_state_dict(pay["state_dict"]); m.eval()
        experts[ename] = (m, dict(pay["scaler"]))

    rows, pooled = [], []
    for name, _t0, _t1, v0, v1 in FOLDS:
        if not (v1 < DEPLOYED_SEEN[0] or v0 > DEPLOYED_SEEN[1]):
            log(f"\n{name} {v0}~{v1} 건너뜀 — 배포본의 학습/검증 구간과 겹친다(표본내)")
            continue
        te = df[(df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")].reset_index(drop=True)
        yv = pd.to_numeric(te["zigzag_action"]).to_numpy(np.int64)
        xr = tabm._base_input(te, base_cols)
        ev = tabm._route_probs(te).argmax(1)
        days = te.timestamp.dt.floor("D").to_numpy()
        fwd = te[f"fwd_{HZ}_bp"].to_numpy(np.float64)
        D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
        for ei, ename in enumerate(("bull", "bear", "chop")):
            m, sc = experts[ename]
            sel = ev == ei
            if sel.any():
                D[sel], Q[sel] = E.heads(m, tabm._standardize_apply(xr[sel], sc), device)
        _, side = F.gate(D, Q, E.Q_THRESH)

        dec, ok = decompose(side, fwd)
        ci, nd = block_boot(side[ok], fwd[ok], days[ok])
        s_, r_, d_ = side[ok], fwd[ok], days[ok]
        dsum = {u: (s_[d_ == u] * r_[d_ == u]).sum() for u in np.unique(d_)}
        order = sorted(dsum, key=lambda u: -dsum[u])
        tim = lambda ss, rr: float((ss * rr).mean() - ss.mean() * rr.mean())
        keep3 = ~np.isin(d_, order[:3])
        fin = np.isfinite(fwd)
        r = {"fold": name, "test": [v0, v1], "indep_days": nd, **dec, "ci95": ci,
             "pass_rate": dec["n"] / len(te),
             "all_bar_drift_bp": float(fwd[fin].mean()),
             "timing_drop_top3d": tim(s_[keep3], r_[keep3]) if keep3.sum() > 50 else float("nan"),
             **{k: v for k, v in F.bias_report(name, side, fwd, yv, days).items()
                if k in ("long_share", "excess_vs_long", "excess_vs_short")},
             **{k: v for k, v in G.day_stats(side, fwd, days).items()
                if k in ("median_bp", "pos_day_share")}}
        rows.append(r); pooled.append((s_, r_, d_))
        log(f"  {name} {v0}~{v1}: 통과율 {r['pass_rate']:.3f}({dec['n']:,}건/{nd}일) · 롱 {r['long_share']*100:.1f}%")
        log(f"    총 {dec['gross_bp']:+7.3f} = 기울기 {dec['tilt_bp']:+7.3f} + 타이밍 {dec['timing_bp']:+7.3f}"
            f"  CI[{ci['timing'][0]:+.3f},{ci['timing'][1]:+.3f}]{' 🟢' if ci['timing'][0] > 0 else ' 0포함'}"
            f" · 최고3일제거 {r['timing_drop_top3d']:+.3f}")
        log(f"    모델−롱 {r['excess_vs_long']['bp']:+.2f} · 모델−숏 {r['excess_vs_short']['bp']:+.2f} · "
            f"중앙 {r['median_bp']:+.2f} · 양수일 {r['pos_day_share']*100:.1f}%")

    s_ = np.concatenate([x[0] for x in pooled]); r_ = np.concatenate([x[1] for x in pooled])
    d_ = np.concatenate([x[2] for x in pooled])
    pdec, _ = decompose(s_, r_); pci, pnd = block_boot(s_, r_, d_)
    dsum = {u: (s_[d_ == u] * r_[d_ == u]).sum() for u in np.unique(d_)}
    order = sorted(dsum, key=lambda u: -dsum[u])
    tim = lambda ss, rr: float((ss * rr).mean() - ss.mean() * rr.mean())
    k3 = ~np.isin(d_, order[:3]); k10 = ~np.isin(d_, order[:10])
    log(f"\n=== 배포본 풀링 ({pnd}일 · n {pdec['n']:,} · 지평 {HZ}) ===")
    log(f"총 {pdec['gross_bp']:+.3f} = 기울기 {pdec['tilt_bp']:+.3f} + 타이밍 {pdec['timing_bp']:+.3f} "
        f"CI[{pci['timing'][0]:+.3f},{pci['timing'][1]:+.3f}]")
    log(f"최고3일제거 {tim(s_[k3], r_[k3]):+.3f} · 최고10일제거 {tim(s_[k10], r_[k10]):+.3f} · "
        f"윈저 {tim(s_, np.clip(r_, -100.0, 100.0)):+.3f}")
    out = {"horizon": HZ, "gate": "single q=0.75 (배포본 실제)", "folds": rows,
           "pooled": {**pdec, "ci95": pci, "indep_days": pnd,
                      "timing_drop_top3d": tim(s_[k3], r_[k3]),
                      "timing_drop_top10d": tim(s_[k10], r_[k10]),
                      "timing_winsor": tim(s_, np.clip(r_, -100.0, 100.0))}}
    (E.OUT / f"stageK_deployed{SUF}.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"저장: {E.OUT}/stageK_deployed{SUF}.json")
    return 0


# 청산 격자. None = 그 배리어 없음(현행 측정 = TP·SL 둘 다 None + 시간청산만).
TP_GRID = [None, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040]
SL_GRID = [None, 0.004, 0.007, 0.010, 0.015, 0.020, 0.040]


def _gain_matrices(entry_i, side, hi, lo, cl, horizon):
    """진입봉 **다음** 봉부터 horizon 봉까지의 (최선, 최악) 가격변동 행렬을 한 번만 만든다.

    라이브 컨벤션(`omega4_6_1_live.py::evaluate_exit`)이 intrabar 고가/저가다 — resting TP/SL 은
    종가가 아니라 닿는 즉시 체결되고, 이미 확정된 봉만 쓰므로 lookahead 가 아니다.
    반환 gh/gl 은 **side 를 이미 적용한** 유리/불리 변동(소수), gc 는 시간청산 수익.
    """
    j = entry_i[:, None] + np.arange(1, horizon + 1)[None, :]
    e = cl[entry_i][:, None]
    up, dn = (hi[j] - e) / e, (lo[j] - e) / e
    sd = side[entry_i][:, None]
    gh = np.where(sd > 0, up, -dn)          # 그 봉에서 갈 수 있었던 최선
    gl = np.where(sd > 0, dn, -up)          # 최악
    gc = (side[entry_i] * (cl[entry_i + horizon] - cl[entry_i]) / cl[entry_i])
    return gh, gl, gc


def _apply_barriers(gh, gl, gc, tp, sl):
    """TP/SL 을 걸어 건당 수익(bp). 같은 봉에서 둘 다 닿으면 **SL 우선**(보수적).

    tp/sl 은 스칼라(정적 배리어) 또는 **건당 배열**(변동성 적응 배리어) 둘 다 받는다.
    """
    big = gh.shape[1] + 1
    col = lambda v: None if v is None else np.broadcast_to(np.asarray(v, float).reshape(-1, 1), gh.shape)
    tpc, slc = col(tp), col(sl)
    t_sl = np.where((gl <= -slc).any(1), (gl <= -slc).argmax(1), big) if sl is not None else np.full(len(gh), big)
    t_tp = np.where((gh >= tpc).any(1), (gh >= tpc).argmax(1), big) if tp is not None else np.full(len(gh), big)
    r = gc.copy()
    hit_sl = t_sl <= t_tp
    r = np.where(t_sl < big, np.where(hit_sl, -np.asarray(sl, float) if sl is not None else r, r), r)
    r = np.where((t_tp < big) & ~hit_sl, np.asarray(tp, float) if tp is not None else r, r)
    return r * 1e4


def _atr_price_move(frame):
    """`build_omega1_2_triple_barrier_labels._atr_price_move` 를 그대로 옮긴다 -- 인과적(과거만).

    ⚠️**전체 프레임에서 한 번** 계산해야 한다. 폴드 조각마다 계산하면 앞 24봉이 NaN 이고
    경계마다 웜업 불연속이 생긴다(라벨 빌더는 전체에서 한 번 낸다).
    """
    hi = pd.to_numeric(frame["high"]).astype(float)
    lo = pd.to_numeric(frame["low"]).astype(float)
    cl = pd.to_numeric(frame["close"]).astype(float)
    pc = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return (tr / cl.replace(0.0, np.nan)).rolling(96, min_periods=24).mean().shift(1).to_numpy()


def _h48_barriers(atr_te, idx):
    """h48_conservative: tp = max(0.6%, 1.2·atr) · sl = max(0.4%, 0.8·atr) · horizon 48봉(4h)."""
    v = atr_te[idx]
    return np.maximum(0.006, 1.2 * v), np.maximum(0.004, 0.8 * v), v


def exitgrid() -> int:
    """--exitgrid: **청산을 붙이면 그 숫자가 살아남는가.**

    지금까지의 모든 측정은 «H시간 뒤 무조건 청산» 이다 — 손절도 익절도 exit 머리도 없다.
    이 엣지는 꼬리에 살기 때문에(최고10일 제거 시 반토막, 윈저 시 감소) **익절이 그 꼬리를
    자를 위험**이 크다. TP·SL 격자를 intrabar 로 실제 시뮬해 전부 보고한다(한 칸만 고르면 선택이다).
    대상은 배포 부모(고정 아티팩트, 추론만) · 그 학습구간과 안 겹치는 폴드.
    """
    E.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = BARS[HZ]
    log(f"지평={HZ}({horizon}봉) · 배포본 q={E.Q_THRESH} · intrabar 고가/저가 · SL 우선")
    df, base_cols = E.load()
    b = torch.load(E.BUNDLE, map_location="cpu", weights_only=False)
    experts = {}
    for ename in ("bull", "bear", "chop"):
        pay = dict(b["models"][ename])
        m = tabm.ThreeHeadTabM(int(pay["n_features"]),
                               cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(device)
        m.load_state_dict(pay["state_dict"]); m.eval()
        experts[ename] = (m, dict(pay["scaler"]))

    atr_full = _atr_price_move(df)          # ⭐전체 프레임에서 한 번
    seg = []
    for name, _t0, _t1, v0, v1 in FOLDS:
        if not (v1 < DEPLOYED_SEEN[0] or v0 > DEPLOYED_SEEN[1]):
            continue
        mask = ((df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")).to_numpy()
        te = df[mask].reset_index(drop=True)
        atr_te = atr_full[mask]
        xr = tabm._base_input(te, base_cols); ev = tabm._route_probs(te).argmax(1)
        D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
        for ei, ename in enumerate(("bull", "bear", "chop")):
            m, sc = experts[ename]
            sel = ev == ei
            if sel.any():
                D[sel], Q[sel] = E.heads(m, tabm._standardize_apply(xr[sel], sc), device)
        _, side = F.gate(D, Q, E.Q_THRESH)
        idx = np.where(side != 0)[0]
        idx = idx[idx + horizon < len(te)]
        seg.append((name, te, side, idx, atr_te))
        log(f"  {name} {v0}~{v1}: 후보 {len(idx):,}건")

    mats = []
    for name, te, side, idx, atr_te in seg:  # 격자 밖에서 한 번만 만든다
        hi = pd.to_numeric(te["high"]).to_numpy(np.float64)
        lo = pd.to_numeric(te["low"]).to_numpy(np.float64)
        cl = pd.to_numeric(te["close"]).to_numpy(np.float64)
        gh, gl, gc = _gain_matrices(idx, side, hi, lo, cl, horizon)
        h48tp, h48sl, v = _h48_barriers(atr_te, idx)
        assert np.isfinite(v).all(), f"{name} ATR 에 NaN -- 웜업 96봉 확인"
        mats.append((gh, gl, gc, te.timestamp.dt.floor("D").to_numpy()[idx], h48tp, h48sl))
    assert all(np.isfinite(m[2]).all() for m in mats), "시간청산 수익에 NaN"
    av = np.concatenate([m[4] for m in mats])
    log(f"h48 배리어 실측: TP 중앙 {np.median(av)*100:.3f}% · 바닥(0.6%) 비율 "
        f"{float((av <= 0.006).mean())*100:.1f}% · SL 중앙 "
        f"{np.median(np.concatenate([m[5] for m in mats]))*100:.3f}%")

    # 격자 + **사전 지정된 h48 행들**(격자에서 고른 게 아니라 라벨 정의를 그대로 옮긴 것)
    specs = [(f"TP {str(tp):>6} SL {str(sl):>6}", tp, sl) for tp in TP_GRID for sl in SL_GRID]
    specs += [("⭐h48 실제(TP&SL)", "h48", "h48"), ("⭐h48 SL 만", None, "h48"),
              ("⭐h48 TP 만", "h48", None)]

    rows = []
    for label, tp, sl in specs:
            pick = lambda spec, i, j: (mats[i][j] if spec == "h48" else spec)
            pnl = np.concatenate([_apply_barriers(m[0], m[1], m[2], pick(tp, i, 4), pick(sl, i, 5))
                                  for i, m in enumerate(mats)])
            days = np.concatenate([m[3] for m in mats])
            lo_, hi_, nd = E.block_ci(pnl, days)
            dsum = {u: pnl[days == u].sum() for u in np.unique(days)}
            order = sorted(dsum, key=lambda u: -dsum[u])
            k10 = ~np.isin(days, order[:10])
            rows.append({"label": label, "tp": tp, "sl": sl, "n": int(len(pnl)), "indep_days": nd,
                         "gross_bp": float(pnl.mean()), "ci95": [lo_, hi_],
                         "median_bp": float(np.median(pnl)),
                         "win_rate": float((pnl > 0).mean()),
                         "drop_top10d_bp": float(pnl[k10].mean()),
                         "p05_bp": float(np.percentile(pnl, 5)),
                         "worst_bp": float(pnl.min())})
            r = rows[-1]
            log(f"  {label:<20} | 총 {r['gross_bp']:+8.2f} "
                f"CI[{lo_:+7.2f},{hi_:+7.2f}]{'🟢' if lo_ > 0 else '  '} · 중앙 {r['median_bp']:+7.2f} "
                f"· 승률 {r['win_rate']*100:4.1f}% · top10제거 {r['drop_top10d_bp']:+7.2f} "
                f"· 5%분위 {r['p05_bp']:+8.1f} · 최악 {r['worst_bp']:+9.1f}")
    base = next(r for r in rows if r["tp"] is None and r["sl"] is None)
    for r in rows:
        if str(r["label"]).startswith("⭐"):
            log(f"  {r['label']:<20} Δ기준 {r['gross_bp'] - base['gross_bp']:+.2f}bp")
    log(f"\n기준(청산 없음·시간청산만): {base['gross_bp']:+.2f}bp · "
        f"CI[{base['ci95'][0]:+.2f},{base['ci95'][1]:+.2f}] · n {base['n']:,} · {base['indep_days']}일")
    better = [r for r in rows if r["gross_bp"] > base["gross_bp"]]
    log(f"기준을 넘는 칸: {len(better)}/{len(rows)}  "
        f"(넘는다면 청산이 «더한다»는 뜻, 없으면 청산이 이 엣지를 «깎는다»)")
    (E.OUT / f"stageK_exitgrid{SUF}.json").write_text(
        json.dumps({"horizon": HZ, "bars": horizon, "rows": rows}, indent=2, default=float))
    log(f"저장: {E.OUT}/stageK_exitgrid{SUF}.json")
    return 0


# 라벨 활성률을 볼 배리어 후보. (tp, sl) -- None 은 그 배리어 없음.
LABEL_SPECS = [("h48 현행 0.6/0.4", 0.006, 0.004),
               ("TP1%/SL1%", 0.010, 0.010), ("TP1.5%/SL1%", 0.015, 0.010),
               ("TP2%/SL1%", 0.020, 0.010), ("TP3%/SL1%", 0.030, 0.010),
               ("TP4%/SL1%", 0.040, 0.010), ("TP없음/SL1%", None, 0.010),
               ("TP2%/SL0.7%", 0.020, 0.007), ("TP4%/SL2%", 0.040, 0.020)]
FEE_LEVELS = {"taker3x_42bp": 0.0042, "usdc3x_3.06bp": 0.000306}
MAE_PEN, SL_PEN = 0.20, 0.003


def _side_outcome(gh, gl, gc, tp, sl):
    """배리어 결과와 «이유»·MAE 를 같이 낸다. 같은 봉 동시터치는 SL 우선."""
    big = gh.shape[1] + 1
    col = lambda v: None if v is None else np.full(gh.shape, float(v))
    t_sl = np.where((gl <= -col(sl)).any(1), (gl <= -col(sl)).argmax(1), big) if sl is not None else np.full(len(gh), big)
    t_tp = np.where((gh >= col(tp)).any(1), (gh >= col(tp)).argmax(1), big) if tp is not None else np.full(len(gh), big)
    hit_sl = (t_sl < big) & (t_sl <= t_tp)
    hit_tp = (t_tp < big) & ~hit_sl
    ret = np.where(hit_sl, -(sl or 0.0), np.where(hit_tp, (tp or 0.0), gc))
    # ⚠️MAE 는 **전체 창**에서 낸다 -- 라벨 빌더(`_reason_and_return`)가 청산 전에
    # future_high/low 전체로 mae/mfe 를 먼저 계산하기 때문이다. 청산 시점까지로 자르면
    # 벌칙이 작아져 활성률이 부풀려진다.
    return ret, hit_sl, hit_tp, np.minimum(gl.min(axis=1), 0.0)


def labelrate() -> int:
    """--labelrate: **이 배리어로 라벨을 만들면 하루 몇 건이 나오나.**

    라벨 빌더(`build_omega1_2_triple_barrier_labels`)와 같은 품질식을 쓴다:
        quality = ret - fee - 0.20·max(-mae,0) - 0.003·(reason=="sl")
        action  = LONG if lq>0 and lq>=sq · SHORT if sq>0 · else CASH
    전 봉(2022-01~2026-08)에서 양쪽 다 계산해 **활성률**과 **후보/일**을 낸다.
    ⚠️이건 라벨 활성률이지 «게이트 통과 후 거래수»가 아니다 -- 품질 머리가 다시 거른다.
    """
    E.OUT.mkdir(parents=True, exist_ok=True)
    horizon = BARS[HZ]
    df, _ = E.load()
    hi = pd.to_numeric(df["high"]).to_numpy(np.float64)
    lo = pd.to_numeric(df["low"]).to_numpy(np.float64)
    cl = pd.to_numeric(df["close"]).to_numpy(np.float64)
    days = df.timestamp.dt.floor("D").to_numpy()
    n_days = len(np.unique(days))
    n = len(cl) - horizon - 1
    log(f"지평={HZ}({horizon}봉) · 전 봉 {n:,} · 달력일 {n_days}")

    acc = {(lab, fee): np.zeros(3, np.int64) for lab, _, _ in LABEL_SPECS for fee in FEE_LEVELS}
    reason = {lab: np.zeros(3, np.int64) for lab, _, _ in LABEL_SPECS}   # sl · tp · timeout
    for st in range(0, n, 50_000):                       # 청크 -- 490k×48 행렬을 한 번에 안 만든다
        idx = np.arange(st, min(st + 50_000, n))
        out = {}
        for sd in (1.0, -1.0):
            gh, gl, gc = _gain_matrices(idx, np.full(len(cl), sd), hi, lo, cl, horizon)
            out[sd] = (gh, gl, gc)
        for lab, tp, sl in LABEL_SPECS:
            res = {}
            for sd in (1.0, -1.0):
                gh, gl, gc = out[sd]
                res[sd] = _side_outcome(gh, gl, gc, tp, sl)
            reason[lab] += np.array([res[1.0][1].sum(), res[1.0][2].sum(),
                                     len(idx) - res[1.0][1].sum() - res[1.0][2].sum()])
            for fname, fee in FEE_LEVELS.items():
                q = {}
                for sd in (1.0, -1.0):
                    ret, hsl, _htp, mae = res[sd]
                    q[sd] = ret - fee - MAE_PEN * np.maximum(-mae, 0.0) - SL_PEN * hsl
                a = np.where((q[1.0] > 0) & (q[1.0] >= q[-1.0]), 1, np.where(q[-1.0] > 0, 2, 0))
                acc[(lab, fname)] += np.bincount(a, minlength=3)

    log(f"\n=== 청산 이유 분포 (롱 기준, 전 봉) ===")
    log(f"{'배리어':<18}{'SL':>9}{'TP':>9}{'시간청산':>10}")
    for lab, _, _ in LABEL_SPECS:
        c = reason[lab]; t = c.sum()
        log(f"{lab:<18}{c[0]/t*100:>8.1f}%{c[1]/t*100:>8.1f}%{c[2]/t*100:>9.1f}%")

    log(f"\n=== 라벨 활성률 & 하루 후보수 ===")
    log(f"{'배리어':<18}{'비용':<16}{'CASH':>8}{'LONG':>8}{'SHORT':>8}{'활성률':>8}{'후보/일':>9}")
    rows = []
    for lab, tp, sl in LABEL_SPECS:
        for fname in FEE_LEVELS:
            c = acc[(lab, fname)]; t = c.sum()
            act = 1.0 - c[0] / t
            rows.append({"barrier": lab, "tp": tp, "sl": sl, "fee": fname,
                         "cash": float(c[0]/t), "long": float(c[1]/t), "short": float(c[2]/t),
                         "active_rate": float(act), "cand_per_day": float(act * t / n_days)})
            log(f"{lab:<18}{fname:<16}{c[0]/t*100:>7.1f}%{c[1]/t*100:>7.1f}%{c[2]/t*100:>7.1f}%"
                f"{act*100:>7.1f}%{act*t/n_days:>9.1f}")
    (E.OUT / f"stageK_labelrate{SUF}.json").write_text(
        json.dumps({"horizon": HZ, "bars": horizon, "rows": rows,
                    "reason": {k: v.tolist() for k, v in reason.items()}}, indent=2, default=float))
    log(f"저장: {E.OUT}/stageK_labelrate{SUF}.json")
    return 0


MAXBARS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--maxbars=")), 2016))


def _first_touch_open(entry_i, side, hi, lo, cl, tp, sl, maxbars, block=288):
    """**시간 배리어 없는 이중 배리어.** TP/SL 중 먼저 닿는 쪽까지 들고 간다.

    전체 창을 한 행렬로 만들면 메모리가 터지므로 288봉(=1일) 블록으로 전진하며
    아직 안 닫힌 거래만 계속 본다. 같은 봉 동시터치는 **SL 우선**(보수적).
    반환: ret(소수·side 적용) · hold(봉) · resolved(bool) · reason(0=SL,1=TP,2=미해소) ·
          mae(진입~해소 시점까지의 최대 불리 변동, ≤0). MAE 는 블록마다 누적한다 --
          2차 패스로 (건수 × MAXBARS) 행렬을 다시 만들면 메모리가 터진다.
    """
    n = len(cl)
    m = len(entry_i)
    e = cl[entry_i]
    sd = side[entry_i]
    ret = np.full(m, np.nan)
    hold = np.full(m, maxbars, np.int64)
    reason = np.full(m, 2, np.int8)
    mae = np.zeros(m)
    alive = np.ones(m, bool)
    for st in range(0, maxbars, block):
        if not alive.any():
            break
        a = np.where(alive)[0]
        j = entry_i[a][:, None] + np.arange(st + 1, min(st + block, maxbars) + 1)[None, :]
        okj = j < n
        j = np.clip(j, 0, n - 1)
        up = (hi[j] - e[a][:, None]) / e[a][:, None]
        dn = (lo[j] - e[a][:, None]) / e[a][:, None]
        gh = np.where(sd[a][:, None] > 0, up, -dn)
        gl = np.where(sd[a][:, None] > 0, dn, -up)
        gh = np.where(okj, gh, -np.inf); gl = np.where(okj, gl, np.inf)
        big = gh.shape[1] + 1
        tpa = None if tp is None else np.asarray(tp, float)[a][:, None] if np.ndim(tp) else float(tp)
        sla = None if sl is None else np.asarray(sl, float)[a][:, None] if np.ndim(sl) else float(sl)
        t_sl = np.where((gl <= -sla).any(1), (gl <= -sla).argmax(1), big) if sl is not None else np.full(len(a), big)
        t_tp = np.where((gh >= tpa).any(1), (gh >= tpa).argmax(1), big) if tp is not None else np.full(len(a), big)
        hs = (t_sl < big) & (t_sl <= t_tp)
        ht = (t_tp < big) & ~hs
        done = hs | ht
        # 이 블록에서 «해소 시점까지만» 본 불리 변동을 기존 누적값과 합친다
        lim = np.where(done, np.where(hs, t_sl, t_tp), gh.shape[1] - 1)
        jj = np.arange(gh.shape[1])[None, :]
        mae[a] = np.minimum(mae[a], np.where(jj <= lim[:, None], gl, 0.0).min(axis=1))
        if done.any():
            k = a[done]
            sk = (np.asarray(sl, float)[k] if (sl is not None and np.ndim(sl)) else (sl or 0.0))
            tk = (np.asarray(tp, float)[k] if (tp is not None and np.ndim(tp)) else (tp or 0.0))
            ret[k] = np.where(hs[done], -np.asarray(sk, float), np.asarray(tk, float))
            hold[k] = st + 1 + np.where(hs[done], t_sl[done], t_tp[done])
            reason[k] = np.where(hs[done], 0, 1)
            alive[k] = False
        # 데이터 끝에 닿은 건 미해소로 확정한다
        ran_out = a[~done & ~okj[:, -1]]
        alive[ran_out] = False
    un = np.isnan(ret)
    if un.any():                                   # 미해소는 maxbars(또는 데이터 끝) 종가로 청산
        k = np.where(un)[0]
        jend = np.minimum(entry_i[k] + maxbars, n - 1)
        ret[k] = sd[k] * (cl[jend] - e[k]) / e[k]
        hold[k] = jend - entry_i[k]
    return ret, hold, ~un, reason, np.minimum(mae, 0.0)


def nocap() -> int:
    """--nocap: **시간 배리어를 뺀 이중 배리어.** 보유시간이 결과가 되므로 빈도도 결과가 된다.

    핵심 산출: 해소까지 보유봉 분위 · 미해소율 · 건당 수익 · **한 슬롯 기준 하루 거래수**.
    하루 거래수 = 288봉 / 평균보유봉 (슬롯이 비는 시간은 무시한 **상한**이다 --
    실제로는 후보가 그때 없을 수 있어 이보다 적다).
    """
    E.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"이중 배리어(시간청산 없음) · 최대 {MAXBARS}봉({MAXBARS/288:.1f}일) · 배포본 q={E.Q_THRESH}")
    df, base_cols = E.load()
    b = torch.load(E.BUNDLE, map_location="cpu", weights_only=False)
    experts = {}
    for ename in ("bull", "bear", "chop"):
        pay = dict(b["models"][ename])
        m = tabm.ThreeHeadTabM(int(pay["n_features"]),
                               cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(device)
        m.load_state_dict(pay["state_dict"]); m.eval()
        experts[ename] = (m, dict(pay["scaler"]))
    seg = []
    for name, _t0, _t1, v0, v1 in FOLDS:
        if not (v1 < DEPLOYED_SEEN[0] or v0 > DEPLOYED_SEEN[1]):
            continue
        te = df[(df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")].reset_index(drop=True)
        xr = tabm._base_input(te, base_cols); ev = tabm._route_probs(te).argmax(1)
        D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
        for ei, ename in enumerate(("bull", "bear", "chop")):
            m, sc = experts[ename]
            sel = ev == ei
            if sel.any():
                D[sel], Q[sel] = E.heads(m, tabm._standardize_apply(xr[sel], sc), device)
        _, side = F.gate(D, Q, E.Q_THRESH)
        idx = np.where(side != 0)[0]
        seg.append((name, te, side, idx))
        log(f"  {name} {v0}~{v1}: 후보 {len(idx):,}건")

    log(f"\n{'배리어':<16}{'건수':>8}{'중앙보유':>9}{'평균보유':>9}{'p90':>7}{'미해소':>8}"
        f"{'SL%':>7}{'TP%':>7}{'총bp':>9}{'CI':>20}{'건/일상한':>10}")
    rows = []
    for lab, tp, sl in LABEL_SPECS:
        pnl, hold, res, rsn, days = [], [], [], [], []
        for name, te, side, idx in seg:
            hi = pd.to_numeric(te["high"]).to_numpy(np.float64)
            lo = pd.to_numeric(te["low"]).to_numpy(np.float64)
            cl = pd.to_numeric(te["close"]).to_numpy(np.float64)
            r, h, rs, rn, _mae = _first_touch_open(idx, side, hi, lo, cl, tp, sl, MAXBARS)
            pnl.append(r * 1e4); hold.append(h); res.append(rs); rsn.append(rn)
            days.append(te.timestamp.dt.floor("D").to_numpy()[idx])
        pnl = np.concatenate(pnl); hold = np.concatenate(hold).astype(float)
        res = np.concatenate(res); rsn = np.concatenate(rsn); days = np.concatenate(days)
        lo_, hi_, nd = E.block_ci(pnl, days)
        per_day = 288.0 / max(hold.mean(), 1e-9)
        rows.append({"barrier": lab, "tp": tp, "sl": sl, "n": int(len(pnl)),
                     "median_hold_bars": float(np.median(hold)), "mean_hold_bars": float(hold.mean()),
                     "p90_hold_bars": float(np.percentile(hold, 90)),
                     "unresolved": float((~res).mean()),
                     "sl_share": float((rsn == 0).mean()), "tp_share": float((rsn == 1).mean()),
                     "gross_bp": float(pnl.mean()), "ci95": [lo_, hi_], "indep_days": nd,
                     "median_bp": float(np.median(pnl)), "win_rate": float((pnl > 0).mean()),
                     "trades_per_day_cap": float(per_day)})
        r = rows[-1]
        log(f"{lab:<16}{r['n']:>8,}{r['median_hold_bars']:>9.0f}{r['mean_hold_bars']:>9.0f}"
            f"{r['p90_hold_bars']:>7.0f}{r['unresolved']*100:>7.1f}%{r['sl_share']*100:>6.1f}%"
            f"{r['tp_share']*100:>6.1f}%{r['gross_bp']:>+9.2f}  [{lo_:+7.2f},{hi_:+7.2f}]"
            f"{per_day:>10.2f}")
    log(f"\n※ 보유봉 1봉=5분 · 288봉=1일. 「건/일상한」은 슬롯이 한 번도 안 비는 가정의 **상한**이다.")
    (E.OUT / f"stageK_nocap{SUF}.json").write_text(
        json.dumps({"horizon_cap_bars": MAXBARS, "rows": rows}, indent=2, default=float))
    log(f"저장: {E.OUT}/stageK_nocap{SUF}.json")
    return 0


# ATR 주입 실험. 각 (tp 중앙폭, sl 중앙폭) 목표마다 «정적» 과 «ATR 비례» 를 짝지어 낸다.
# 배수는 mult = 목표폭 / median(atr) 로 잡아 **중앙 폭을 맞춘다** -- 그래야 「수준」이 아니라
# 「적응」의 효과만 남는다. 라이브 공식(tp=12·atr, sl=6·atr)의 비율 2:1 도 사다리에 들어있다.
VOL_TARGETS = [(0.010, 0.010), (0.015, 0.010), (0.020, 0.010), (0.020, 0.007), (0.030, 0.010)]
ATR_WINDOWS = [96, 192]          # 96 = 라벨 빌더 · 192 = 라이브(omega4_6_1_live)


def _atr_win(frame, win):
    hi = pd.to_numeric(frame["high"]).astype(float)
    lo = pd.to_numeric(frame["low"]).astype(float)
    cl = pd.to_numeric(frame["close"]).astype(float)
    pc = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return (tr / cl.replace(0.0, np.nan)).rolling(win, min_periods=max(win // 4, 2)).mean().shift(1).to_numpy()


def volatr() -> int:
    """--volatr: **SL/TP 에 ATR 변동성을 주입하면 나아지는가.**

    ⭐설계의 핵심: 같은 **중앙 폭**을 갖는 정적 배리어를 대조군으로 붙인다. 안 그러면
    「ATR 이 좋다」가 아니라 「폭이 달랐다」를 재게 된다.
    시간 배리어는 없다(이중 배리어). 건당이 아니라 **하루 기준 순bp** 로 판정한다 --
    폭이 넓어지면 건당은 오르지만 거래수가 줄어 상쇄되기 때문이다.
    """
    E.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"이중 배리어 · 최대 {MAXBARS}봉 · 배포본 q={E.Q_THRESH} · ATR 창 {ATR_WINDOWS}")
    df, base_cols = E.load()
    b = torch.load(E.BUNDLE, map_location="cpu", weights_only=False)
    experts = {}
    for ename in ("bull", "bear", "chop"):
        pay = dict(b["models"][ename])
        m = tabm.ThreeHeadTabM(int(pay["n_features"]),
                               cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(device)
        m.load_state_dict(pay["state_dict"]); m.eval()
        experts[ename] = (m, dict(pay["scaler"]))
    atrs = {w: _atr_win(df, w) for w in ATR_WINDOWS}
    allmed = {w: float(np.nanmedian(atrs[w])) for w in ATR_WINDOWS}
    log("전체 봉 ATR 중앙값: " + " · ".join(f"창{w} {allmed[w]*100:.4f}%" for w in ATR_WINDOWS))
    log("  (라이브 배수 12/6 을 쓰면 tp/sl = "
        + " · ".join(f"{12*allmed[w]*100:.2f}%/{6*allmed[w]*100:.2f}%(창{w})" for w in ATR_WINDOWS) + ")")

    seg = []
    for name, _t0, _t1, v0, v1 in FOLDS:
        if not (v1 < DEPLOYED_SEEN[0] or v0 > DEPLOYED_SEEN[1]):
            continue
        mask = ((df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")).to_numpy()
        te = df[mask].reset_index(drop=True)
        xr = tabm._base_input(te, base_cols); ev = tabm._route_probs(te).argmax(1)
        D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
        for ei, ename in enumerate(("bull", "bear", "chop")):
            m, sc = experts[ename]
            sel = ev == ei
            if sel.any():
                D[sel], Q[sel] = E.heads(m, tabm._standardize_apply(xr[sel], sc), device)
        _, side = F.gate(D, Q, E.Q_THRESH)
        idx = np.where(side != 0)[0]
        seg.append((name, te, side, idx, {w: atrs[w][mask] for w in ATR_WINDOWS}))
        log(f"  {name}: 후보 {len(idx):,}건")

    # ⭐배수는 **후보 봉**의 ATR 중앙값으로 잡는다. 전체 봉으로 잡으면 실현 폭이 대조군과
    # 어긋나 「적응」이 아니라 「폭」을 재게 된다(1차 측정에서 실제로 10/10 이 그렇게 어긋났다).
    med = {w: float(np.nanmedian(np.concatenate([a[w][ix] for _n, _t, _s, ix, a in seg])))
           for w in ATR_WINDOWS}
    for w in ATR_WINDOWS:
        log(f"후보 봉 ATR 중앙값 창{w}: {med[w]*100:.4f}% "
            f"(전체 봉 대비 {med[w]/allmed[w]*100:.1f}% -- 부모가 조용한 봉에서 발화한다)")

    def run(tp, sl):
        """tp/sl 은 스칼라(정적) 또는 {폴드명: **건당** 배열}(ATR 비례).

        ⚠️`_first_touch_open` 의 배열 배리어는 **건당** 색인이다(봉당 아님).
        """
        pnl, hold, res, days, widths, widths_sl = [], [], [], [], [], []
        for name, te, side, idx, _a in seg:
            hi = pd.to_numeric(te["high"]).to_numpy(np.float64)
            lo = pd.to_numeric(te["low"]).to_numpy(np.float64)
            cl = pd.to_numeric(te["close"]).to_numpy(np.float64)
            t = tp[name] if isinstance(tp, dict) else tp
            l = sl[name] if isinstance(sl, dict) else sl
            assert not isinstance(t, np.ndarray) or len(t) == len(idx), "배리어 배열은 건당 길이여야 한다"
            r, h, rs, _rn, _mae = _first_touch_open(idx, side, hi, lo, cl, t, l, MAXBARS)
            pnl.append(r * 1e4); hold.append(h); res.append(rs)
            days.append(te.timestamp.dt.floor("D").to_numpy()[idx])
            widths.append(np.full(len(idx), t) if np.isscalar(t) else np.asarray(t, float))
            widths_sl.append(np.full(len(idx), l) if np.isscalar(l) else np.asarray(l, float))
        pnl = np.concatenate(pnl); hold = np.concatenate(hold).astype(float)
        res = np.concatenate(res); days = np.concatenate(days)
        lo_, hi_, nd = E.block_ci(pnl, days)
        per_day = 288.0 / max(hold.mean(), 1e-9)
        wid = np.concatenate(widths) if widths else np.array([np.nan])
        wsl = np.concatenate(widths_sl) if widths_sl else np.array([np.nan])
        return {"n": int(len(pnl)), "gross_bp": float(pnl.mean()), "ci95": [lo_, hi_],
                "tp_width_med": float(np.median(wid)), "tp_width_mean": float(np.mean(wid)),
                "sl_width_med": float(np.median(wsl)),
                "indep_days": nd, "median_hold": float(np.median(hold)),
                "mean_hold": float(hold.mean()), "unresolved": float((~res).mean()),
                "win_rate": float((pnl > 0).mean()), "trades_per_day": per_day,
                "net_day_usdc": (float(pnl.mean()) - 1.02) * per_day,
                "net_day_peg": (float(pnl.mean()) - 5.52) * per_day}

    log(f"\n{'배리어':<30}{'실현TP폭':>9}{'실현SL폭':>9}{'중앙보유':>9}{'평균보유':>9}{'건당bp':>9}"
        f"{'CI':>20}{'건/일':>7}{'순/일USDC':>10}{'순/일peg':>9}")
    rows = []
    for tpm, slm in VOL_TARGETS:
        lab = f"정적 TP{tpm*100:g}%/SL{slm*100:g}%"
        r = run(tpm, slm); r.update({"kind": "static", "tp_med": tpm, "sl_med": slm, "atr_win": None})
        rows.append({"label": lab, **r})
        log(f"{lab:<30}{r['tp_width_med']*100:>8.3f}%{r['sl_width_med']*100:>8.3f}%"
            f"{r['median_hold']:>9.0f}{r['mean_hold']:>9.0f}{r['gross_bp']:>+9.2f}"
            f"  [{r['ci95'][0]:+7.2f},{r['ci95'][1]:+7.2f}]{r['trades_per_day']:>7.2f}"
            f"{r['net_day_usdc']:>10.1f}{r['net_day_peg']:>9.1f}")
        for w in ATR_WINDOWS:
            mt, ms = tpm / med[w], slm / med[w]
            # 건당 배열: 그 후보 «진입봉»의 ATR × 배수. 웜업 NaN 은 중앙값으로 채운다.
            per = {nm: np.nan_to_num(a[w][ix], nan=med[w]) for nm, _te, _sd, ix, a in seg}
            r2 = run({k: v * mt for k, v in per.items()}, {k: v * ms for k, v in per.items()})
            r2.update({"kind": "atr", "tp_med": tpm, "sl_med": slm, "atr_win": w,
                       "tp_mult": mt, "sl_mult": ms})
            lab2 = f"  ATR창{w} ×{mt:.1f}/×{ms:.1f}"
            rows.append({"label": lab2, **r2})
            log(f"{lab2:<30}{r2['tp_width_med']*100:>8.3f}%{r2['sl_width_med']*100:>8.3f}%"
                f"{r2['median_hold']:>9.0f}{r2['mean_hold']:>9.0f}{r2['gross_bp']:>+9.2f}"
                f"  [{r2['ci95'][0]:+7.2f},{r2['ci95'][1]:+7.2f}]{r2['trades_per_day']:>7.2f}"
                f"{r2['net_day_usdc']:>10.1f}{r2['net_day_peg']:>9.1f}"
                f"   Δ {r2['net_day_usdc'] - r['net_day_usdc']:+.1f}"
                f"{'  🔴폭 불일치' if abs(r2['tp_width_med']/max(r['tp_width_med'],1e-12) - 1) > 0.05 else ''}")
    wins = sum(1 for i, x in enumerate(rows) if x["kind"] == "atr" and
               x["net_day_usdc"] > next(y for y in rows[:i][::-1] if y["kind"] == "static")["net_day_usdc"])
    tot = sum(1 for x in rows if x["kind"] == "atr")
    log(f"\n⭐ATR 주입이 같은 중앙폭 정적 대조군을 이긴 칸: {wins}/{tot} (USDC 순bp/일 기준)")
    (E.OUT / f"stageK_volatr{SUF}.json").write_text(
        json.dumps({"atr_median": med, "max_bars": MAXBARS, "rows": rows}, indent=2, default=float))
    log(f"저장: {E.OUT}/stageK_volatr{SUF}.json")
    return 0


# ── Zeus Baseline v1 라벨 ────────────────────────────────────────────────────
BASE_TP, BASE_SL = 0.015, 0.010          # 사용자 결정(2026-09-17): 더블 배리어, 시간청산 없음
LABEL_OUT = E.OUT.parent / "zeus_double_barrier_labels_20260917"


def buildlabel() -> int:
    """--buildlabel: **Zeus Baseline v1 의 더블 배리어 라벨**을 전 봉에 만든다.

    TP 1.5% / SL 1% · **시간 배리어 없음**(최대 MAXBARS 까지 추적, 그 뒤는 미해소로 표시).
    품질식은 라벨 빌더(`build_omega1_2_triple_barrier_labels`)와 동일하게 간다:
        quality = ret - fee - 0.20·max(-mae,0) - 0.003·(reason=="sl")
        action  = LONG if lq>0 and lq>=sq · SHORT if sq>0 · else CASH
    MAE 는 **해소 시점까지**가 아니라 빌더 규약대로 잡되, 더블 배리어에서는 해소 시점이 곧
    창의 끝이므로 그 시점까지로 본다(시간 배리어가 없어 「청산 후」가 존재하지 않는다).

    ⚠️미해소 건은 `tb_resolved=False` 로 표시하고 action 계산에서 **CASH 로 둔다** --
    14일 안에 어느 배리어도 못 닿은 봉을 「좋은 진입」이라 가르치지 않는다.
    """
    E.OUT.mkdir(parents=True, exist_ok=True)
    df, _ = E.load()
    hi = pd.to_numeric(df["high"]).to_numpy(np.float64)
    lo = pd.to_numeric(df["low"]).to_numpy(np.float64)
    cl = pd.to_numeric(df["close"]).to_numpy(np.float64)
    ts = df["timestamp"].to_numpy()
    n = len(cl)
    log(f"더블 배리어 라벨: TP {BASE_TP*100:g}% / SL {BASE_SL*100:g}% · 시간청산 없음 · "
        f"최대 {MAXBARS}봉({MAXBARS/288:.1f}일) · 전 봉 {n:,}")

    out = {}
    for sd, nm in ((1.0, "long"), (-1.0, "short")):
        ret = np.full(n, np.nan); hold = np.full(n, -1, np.int64)
        res = np.zeros(n, bool); rsn = np.full(n, 2, np.int8); mae = np.full(n, np.nan)
        side = np.full(n, sd)
        for st in range(0, n, 40_000):
            idx = np.arange(st, min(st + 40_000, n))
            idx = idx[idx + 1 < n]
            if not len(idx):
                continue
            r, h, rs, rn, mm = _first_touch_open(idx, side, hi, lo, cl, BASE_TP, BASE_SL, MAXBARS)
            ret[idx], hold[idx], res[idx], rsn[idx], mae[idx] = r, h, rs, rn, mm
            log(f"  {nm} {st:,}~{idx[-1]:,} 완료 (해소 {rs.mean()*100:.1f}%)")
        out[nm] = {"ret": ret, "hold": hold, "resolved": res, "reason": rsn, "mae": mae}

    # ── v2: **봉당 수익률** 라벨 ──────────────────────────────────────────────
    # v1(어느 배리어가 먼저 닿는가)은 실패했다: 이기면 항상 +TP, 지면 항상 −SL 이라
    # **등급이 없고** CASH 가 19.45% 뿐이라 품질 머리가 걸러낼 게 없었다(stageM: 게이트를
    # 켜면 오히려 나빠짐). 빠진 등급은 «얼마나 빨리»다 -- 슬롯이 하나면 목적함수는
    # 건당 bp 가 아니라 **시간당 bp** 이고, 그게 평가에 쓰는 「하루 순bp」와 같은 것이다.
    LABEL_OUT.mkdir(parents=True, exist_ok=True)
    fee_r = FEE_LEVELS["usdc3x_3.06bp"]
    rate = {}
    for nm in ("long", "short"):
        o_ = out[nm]
        q = (o_["ret"] - fee_r - MAE_PEN * np.maximum(-o_["mae"], 0.0)
             - SL_PEN * (o_["reason"] == 0))
        rate[nm] = np.where(o_["resolved"], q / np.maximum(o_["hold"], 1), -np.inf)
    best = np.maximum(rate["long"], rate["short"])
    fin = np.isfinite(best)
    for cash_target in (0.40, 0.50, 0.60):
        tau = float(np.quantile(best[fin], cash_target))
        a = np.where((rate["long"] > tau) & (rate["long"] >= rate["short"]), 1,
                     np.where(rate["short"] > tau, 2, 0))
        sh = np.bincount(a, minlength=3) / len(a)
        hb = np.where(a == 1, out["long"]["hold"], np.where(a == 2, out["short"]["hold"], np.nan))
        frame = pd.DataFrame({
            "timestamp": ts, "tb_action": a.astype(np.int64),
            "tb_rate": best, "tb_tau": tau,
            "tb_long_rate": rate["long"], "tb_short_rate": rate["short"],
            "tb_long_ret": out["long"]["ret"], "tb_short_ret": out["short"]["ret"],
            "tb_long_bars": out["long"]["hold"], "tb_short_bars": out["short"]["hold"],
            "tb_long_reason": out["long"]["reason"], "tb_short_reason": out["short"]["reason"],
        })
        path = LABEL_OUT / f"zeus_rate_tp{int(BASE_TP*1000)}_sl{int(BASE_SL*1000)}_cash{int(cash_target*100)}.parquet"
        frame.to_parquet(path, index=False)
        log(f"v2 봉당수익 τ={tau*1e4:.4f}bp/봉 · CASH/LONG/SHORT {sh.round(4)} · "
            f"선택건 중앙보유 {np.nanmedian(hb):.0f}봉 · 저장 {path.name}")

    for fname, fee in FEE_LEVELS.items():
        q = {}
        for nm in ("long", "short"):
            o = out[nm]
            q[nm] = (o["ret"] - fee - MAE_PEN * np.maximum(-o["mae"], 0.0)
                     - SL_PEN * (o["reason"] == 0))
            q[nm] = np.where(o["resolved"], q[nm], -np.inf)     # 미해소는 후보에서 뺀다
        a = np.where((q["long"] > 0) & (q["long"] >= q["short"]), 1,
                     np.where(q["short"] > 0, 2, 0))
        sh = np.bincount(a, minlength=3) / len(a)
        frame = pd.DataFrame({
            "timestamp": ts, "tb_action": a.astype(np.int64),
            "tb_quality": np.maximum(np.where(np.isfinite(q["long"]), q["long"], -9.99),
                                     np.where(np.isfinite(q["short"]), q["short"], -9.99)),
            "tb_long_ret": out["long"]["ret"], "tb_short_ret": out["short"]["ret"],
            "tb_long_mae": out["long"]["mae"], "tb_short_mae": out["short"]["mae"],
            "tb_long_bars": out["long"]["hold"], "tb_short_bars": out["short"]["hold"],
            "tb_long_reason": out["long"]["reason"], "tb_short_reason": out["short"]["reason"],
            "tb_long_resolved": out["long"]["resolved"], "tb_short_resolved": out["short"]["resolved"],
        })
        path = LABEL_OUT / f"zeus_db_tp{int(BASE_TP*1000)}_sl{int(BASE_SL*1000)}_{fname}.parquet"
        frame.to_parquet(path, index=False)
        log(f"{fname:<16} CASH/LONG/SHORT = {sh.round(4)} · 활성률 {(1-sh[0])*100:.1f}% · 저장 {path.name}")
    meta = {"tp": BASE_TP, "sl": BASE_SL, "max_bars": MAXBARS, "time_barrier": False,
            "fees": FEE_LEVELS, "mae_pen": MAE_PEN, "sl_pen": SL_PEN,
            "unresolved_to_cash": True, "rows": int(n),
            "long_unresolved": float((~out["long"]["resolved"]).mean()),
            "short_unresolved": float((~out["short"]["resolved"]).mean())}
    (LABEL_OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    log(f"미해소율 long {meta['long_unresolved']*100:.2f}% · short {meta['short_unresolved']*100:.2f}%")
    log(f"저장: {LABEL_OUT}")
    return 0


# 비율 사다리. 더블 배리어에서 손익분기 p = (SL+비용)/(TP+SL) 이므로 비율이 바뀌면
# 요구 확률도 같이 움직인다. 「어느 비율이 구조적으로 쉬운가」를 직접 잰다.
RATIO_SPECS = [(0.010, 0.010), (0.0125, 0.010), (0.015, 0.010), (0.020, 0.010),
               (0.030, 0.010), (0.015, 0.0075), (0.020, 0.0075), (0.020, 0.015),
               (0.006, 0.004)]
SUB = 8                                  # 기저확률 추정용 부표집(매 8봉). 비율 추정엔 충분하다.


def baserate() -> int:
    """--baserate: **더블 배리어를 「진입 내기」로 보고 기저확률 대 손익분기를 잰다.**

    EV = p·TP − (1−p)·SL − 비용  ⇒  손익분기 p* = (SL + 비용)/(TP + SL)
    라벨의 무조건부 적중률 p0(= 그 측면이 SL 보다 TP 를 먼저 맞을 확률)와 p* 의 차이가
    **품질 머리가 메워야 할 간격**이다. p0 가 p* 에 가까울수록 「선택」에만 전부를 거는 내기다.
    """
    E.OUT.mkdir(parents=True, exist_ok=True)
    df, _ = E.load()
    hi = pd.to_numeric(df["high"]).to_numpy(np.float64)
    lo = pd.to_numeric(df["low"]).to_numpy(np.float64)
    cl = pd.to_numeric(df["close"]).to_numpy(np.float64)
    n = len(cl)
    idx0 = np.arange(0, n - 1, SUB)
    log(f"이중 배리어 기저확률 · 부표집 매 {SUB}봉 = {len(idx0):,}건/측면 · 최대 {MAXBARS}봉")
    log(f"\n{'TP/SL':<14}{'비율':>6}{'p0롱':>8}{'p0숏':>8}{'둘다실패':>9}"
        f"{'p*(0)':>8}{'p*(USDC)':>10}{'p*(peg)':>9}{'여유(USDC)':>11}{'중앙보유':>9}")
    rows = []
    for tp, sl in RATIO_SPECS:
        out = {}
        for sd, nm in ((1.0, "long"), (-1.0, "short")):
            side = np.full(n, sd)
            rr, hh, res = [], [], []
            for st in range(0, len(idx0), 40_000):
                ix = idx0[st:st + 40_000]
                r, h, rs, rn, _m = _first_touch_open(ix, side, hi, lo, cl, tp, sl, MAXBARS)
                rr.append(rn); hh.append(h); res.append(rs)
            out[nm] = (np.concatenate(rr), np.concatenate(hh), np.concatenate(res))
        p_long = float((out["long"][0] == 1).mean())
        p_short = float((out["short"][0] == 1).mean())
        both_fail = 1.0 - p_long - p_short
        hold = np.median(np.concatenate([out["long"][1], out["short"][1]]))
        be = lambda c: (sl * 1e4 + c) / ((tp + sl) * 1e4)
        blend = (p_long + p_short) / 2.0
        r = {"tp": tp, "sl": sl, "ratio": tp / sl, "p0_long": p_long, "p0_short": p_short,
             "p0_blend": blend, "both_fail": both_fail,
             "be_zero": be(0.0), "be_usdc": be(1.02), "be_peg": be(5.52),
             "slack_usdc": blend - be(1.02), "median_hold": float(hold),
             "unresolved": float((~np.concatenate([out["long"][2], out["short"][2]])).mean())}
        rows.append(r)
        log(f"{f'{tp*100:g}%/{sl*100:g}%':<14}{tp/sl:>6.2f}{p_long*100:>7.2f}%{p_short*100:>7.2f}%"
            f"{both_fail*100:>8.2f}%{be(0.0)*100:>7.2f}%{be(1.02)*100:>9.2f}%{be(5.52)*100:>8.2f}%"
            f"{(blend - be(1.02))*100:>+10.2f}pp{hold:>9.0f}")
    log("\n※ p0 = 그 측면이 SL 보다 TP 를 먼저 맞을 무조건부 확률(= 아무 봉에서나 진입)")
    log("※ 여유 = p0(양측면 평균) − 손익분기. **음수면 「선택」이 그만큼을 메워야 한다.**")
    log("※ 롱+숏+둘다실패 = 1 (배타적). 둘다실패 = ±SL 을 오가되 ±TP 를 못 넘은 구간.")
    (E.OUT / "stageK_baserate.json").write_text(json.dumps(rows, indent=2, default=float))
    log(f"저장: {E.OUT}/stageK_baserate.json")
    return 0


def qcalib() -> int:
    """--qcalib: **L4 의 전제 확인 — 부모의 품질점수 q 가 이미 p 를 아는가.**

    더블 배리어에서 페이오프가 고정(+TP/−SL)이라 건당 분산이 **오직 p 의 함수**다
    (`Var = p(1−p)·(TP+SL)²`). 그러니 L4 가 예측해야 할 것은 변동성이 아니라 **건별 p** 이고
    켈리는 `f* = (p·TP − (1−p)·SL) / (TP·SL) · ...` 대신 고정배당 형태로
    `f* = (p(1+b) − 1)/b`, b = TP/SL 로 쓴다.

    ⭐**새 모델을 만들기 전에**: 배포 부모의 q 가 실현 p 와 단조 관계면 L4 는 **보정**으로
    끝난다(사다리 1단 「안 해도 되는가」). 그래서 q 분위별 실현 p 를 먼저 잰다.
    대조: 방향 확신도(direction softmax max)도 같이 낸다 -- 어느 쪽이 p 를 아는지 가른다.
    """
    E.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b_ = TP_BP, SL_BP = BASE_TP * 1e4, BASE_SL * 1e4
    cost = 1.02
    be = (SL_BP + cost) / (TP_BP + SL_BP)
    # ⭐켈리에 **비용을 넣는다** — 이기면 TP−비용, 지면 SL+비용. 실효 배당
    # b = (TP−비용)/(SL+비용) 를 써야 f* 가 손익분기 p* 에서 정확히 0 이 된다.
    # TP/SL 을 그대로 쓰면 무비용 손익분기(40.00%)에서 0 이 되어 p* 와 어긋난다.
    ratio = (TP_BP - cost) / (SL_BP + cost)
    assert abs((be * (1 + ratio) - 1) / ratio) < 1e-12, "손익분기에서 f* 가 0 이 아니다"
    log(f"더블 배리어 TP{BASE_TP*100:g}%/SL{BASE_SL*100:g}% · 손익분기 p* = {be*100:.2f}% "
        f"(비용 {cost}bp) · 실효배당 b = {ratio:.4f} · 켈리 f* = (p·{1+ratio:.4f} − 1)/{ratio:.4f}")
    df, base_cols = E.load()
    bun = torch.load(E.BUNDLE, map_location="cpu", weights_only=False)
    experts = {}
    for ename in ("bull", "bear", "chop"):
        pay = dict(bun["models"][ename])
        m = tabm.ThreeHeadTabM(int(pay["n_features"]),
                               cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(device)
        m.load_state_dict(pay["state_dict"]); m.eval()
        experts[ename] = (m, dict(pay["scaler"]))

    S, R, Q, DC, DAY, HOLD = [], [], [], [], [], []
    for name, _t0, _t1, v0, v1 in FOLDS:
        if not (v1 < DEPLOYED_SEEN[0] or v0 > DEPLOYED_SEEN[1]):
            continue
        te = df[(df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")].reset_index(drop=True)
        xr = tabm._base_input(te, base_cols); ev = tabm._route_probs(te).argmax(1)
        D = np.zeros((len(te), 3)); Qm = np.zeros((len(te), 3))
        for ei, ename in enumerate(("bull", "bear", "chop")):
            m, sc = experts[ename]
            sel = ev == ei
            if sel.any():
                D[sel], Qm[sel] = E.heads(m, tabm._standardize_apply(xr[sel], sc), device)
        da = D.argmax(1)
        qf = np.where(da > 0, Qm[np.arange(len(Qm)), da], Qm[:, 0])
        side = np.where(da == 1, 1.0, np.where(da == 2, -1.0, 0.0))   # ⭐게이트 «전» 전부
        idx = np.where(side != 0)[0]
        hi = pd.to_numeric(te["high"]).to_numpy(np.float64)
        lo = pd.to_numeric(te["low"]).to_numpy(np.float64)
        cl = pd.to_numeric(te["close"]).to_numpy(np.float64)
        r, h, _res, rn, _m = _first_touch_open(idx, side, hi, lo, cl, BASE_TP, BASE_SL, MAXBARS)
        S.append(np.full(len(idx), name)); R.append(rn); Q.append(qf[idx])
        DC.append(D.max(1)[idx]); HOLD.append(h.astype(float))
        DAY.append(te.timestamp.dt.floor("D").to_numpy()[idx])
        log(f"  {name}: 방향 비CASH {len(idx):,}건(게이트 전)")
    R = np.concatenate(R); Q = np.concatenate(Q); DC = np.concatenate(DC)
    DAY = np.concatenate(DAY); HOLD = np.concatenate(HOLD)
    hit = (R == 1).astype(float)          # TP 를 SL 보다 먼저

    out = {"breakeven_p": be, "tp_bp": TP_BP, "sl_bp": SL_BP, "cost_bp": cost, "bins": {}}
    for nm, v in (("품질점수 q", Q), ("방향 확신도", DC)):
        log(f"\n■ {nm} 10분위별 실현 p (게이트 전 전체 {len(hit):,}건 · 손익분기 {be*100:.2f}%)")
        log(f"{'분위':<6}{'구간':>16}{'건수':>8}{'실현p':>8}{'−손익분기':>10}"
            f"{'켈리f*':>8}{'중앙보유':>9}{'건/일':>7}{'건당bp':>9}")
        qs = pd.qcut(pd.Series(v), 10, labels=False, duplicates="drop")
        rows = []
        for k in sorted(pd.unique(qs)):
            m = (qs == k).to_numpy()
            p = float(hit[m].mean())
            f = (p * (1 + ratio) - 1) / ratio
            mh = float(hold_mean := HOLD[m].mean())
            bp = p * TP_BP - (1 - p) * SL_BP - cost
            rows.append({"bin": int(k), "lo": float(v[m].min()), "hi": float(v[m].max()),
                         "n": int(m.sum()), "p": p, "kelly": f,
                         "median_hold": float(np.median(HOLD[m])), "mean_hold": mh,
                         "per_day": 288.0 / max(mh, 1e-9), "bp": bp})
            log(f"D{k+1:<5}{f'{v[m].min():.3f}~{v[m].max():.3f}':>16}{m.sum():>8,}{p*100:>7.2f}%"
                f"{(p-be)*100:>+9.2f}pp{f:>8.3f}{np.median(HOLD[m]):>9.0f}"
                f"{288.0/max(mh,1e-9):>7.2f}{bp:>+9.2f}")
        out["bins"][nm] = rows
        # 최상 − 최하 분위 차, 날짜블록 CI
        top, bot = (qs == qs.max()).to_numpy(), (qs == qs.min()).to_numpy()
        a = pd.Series(hit[top]).groupby(DAY[top]).mean()
        c = pd.Series(hit[bot]).groupby(DAY[bot]).mean()
        obs = a.mean() - c.mean()
        u = np.unique(DAY); rg = np.random.default_rng(5); bs = []
        for _ in range(2000):
            sm = rg.choice(u, len(u), replace=True)
            bs.append(a.reindex(sm).dropna().mean() - c.reindex(sm).dropna().mean())
        bs = np.array(bs); ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        mono = float(np.corrcoef(np.arange(len(rows)), [r["p"] for r in rows])[0, 1])
        log(f"  ⭐최상−최하 분위 Δp = {obs*100:+.2f}pp  날짜블록 CI95 "
            f"[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]{'  🟢0배제' if ci[0] > 0 else '  0포함'}"
            f" · 분위-p 상관 {mono:+.3f}")
        out["bins"][nm + "_summary"] = {"top_minus_bottom_pp": obs * 100, "ci95_pp": [c_ * 100 for c_ in ci],
                                        "monotonicity": mono}
    (E.OUT / "stageN_q_calibration.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"\n저장: {E.OUT}/stageN_q_calibration.json")
    return 0


def routing() -> int:
    """--routing: **레짐 라우터가 실제로 기여하는가.** 추론만(학습 없음).

    배포 번들의 세 전문가(bull/bear/chop)를 **다르게 배정**해 비교한다. 핵심 귀무는
    ⭐**무작위 라우팅** -- 같은 전문가, 같은 배정 «비율», 틀린 배정. 현행과 차이가 없으면
    전문가들이 서로 교환 가능하다는 뜻이고 라우터는 장식이다.

    사전 지정 팔:
      R0 하드 라우팅(현행, argmax)     R1 확률가중 평균(soft)     R2 균등 평균
      R3 항상 bull / bear / chop       R4 무작위 라우팅(비율 보존, 시드 5개)
    평가는 Zeus 규약: 더블 배리어 TP1.5%/SL1% · 게이트 q=0.75 · 하루 순bp 로 읽는다.
    """
    E.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TPB, SLB, cost = BASE_TP * 1e4, BASE_SL * 1e4, 1.02
    imp = lambda g: (g + SLB) / (TPB + SLB)
    log(f"더블 배리어 TP{BASE_TP*100:g}%/SL{BASE_SL*100:g}% · q={E.Q_THRESH} · 비용 {cost}bp · 추론만")
    df, base_cols = E.load()
    bun = torch.load(E.BUNDLE, map_location="cpu", weights_only=False)
    EN = ("bull", "bear", "chop")
    experts = {}
    for ename in EN:
        pay = dict(bun["models"][ename])
        m = tabm.ThreeHeadTabM(int(pay["n_features"]),
                               cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(device)
        m.load_state_dict(pay["state_dict"]); m.eval()
        experts[ename] = (m, dict(pay["scaler"]))

    segs = []
    for name, _t0, _t1, v0, v1 in FOLDS:
        if not (v1 < DEPLOYED_SEEN[0] or v0 > DEPLOYED_SEEN[1]):
            continue
        te = df[(df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")].reset_index(drop=True)
        xr = tabm._base_input(te, base_cols)
        rp = tabm._route_probs(te)
        # ⭐전문가 «셋 다» 모든 봉에 돌려놓는다 -- 이후 배정만 바꿔 재조합한다(추론 1회로 끝)
        Dall = np.zeros((3, len(te), 3)); Qall = np.zeros((3, len(te), 3))
        for ei, ename in enumerate(EN):
            m, sc = experts[ename]
            Dall[ei], Qall[ei] = E.heads(m, tabm._standardize_apply(xr, sc), device)
        segs.append((name, te, rp, Dall, Qall))
        log(f"  {name}: {len(te):,}봉 · 전문가 3종 전수 추론 완료 · "
            f"라우팅 비율 {np.bincount(rp.argmax(1), minlength=3) / len(te)}")

    def econ(assign_fn, tag, seed=None):
        pnl, hold, days = [], [], []
        for name, te, rp, Dall, Qall in segs:
            a = assign_fn(rp, seed)
            if a.ndim == 1:                       # 봉별 전문가 선택
                D = Dall[a, np.arange(len(te))]; Q = Qall[a, np.arange(len(te))]
            else:                                 # 가중 혼합 (a: (n,3))
                D = np.einsum("en f,n e->n f", Dall, a); Q = np.einsum("en f,n e->n f", Qall, a)
            _, side = F.gate(D, Q, E.Q_THRESH)
            idx = np.where(side != 0)[0]
            if len(idx) < 50:
                continue
            hi = pd.to_numeric(te["high"]).to_numpy(np.float64)
            lo = pd.to_numeric(te["low"]).to_numpy(np.float64)
            cl = pd.to_numeric(te["close"]).to_numpy(np.float64)
            r, h, _rs, _rn, _m = _first_touch_open(idx, side, hi, lo, cl, BASE_TP, BASE_SL, MAXBARS)
            pnl.append(r * 1e4 - cost); hold.append(h.astype(float))
            days.append(te.timestamp.dt.floor("D").to_numpy()[idx])
        pnl = np.concatenate(pnl); hold = np.concatenate(hold); days = np.concatenate(days)
        lo_, hi_, nd = E.block_ci(pnl, days)
        pdy = 288.0 / max(hold.mean(), 1e-9)
        return {"arm": tag, "n": int(len(pnl)), "indep_days": nd, "gross_bp": float(pnl.mean()),
                "ci95": [lo_, hi_], "p": imp(float(pnl.mean()) + cost), "trades_per_day": pdy,
                "median_hold": float(np.median(hold)),
                "net_day_usdc": (float(pnl.mean()) - 0.0) * pdy}

    rows = [econ(lambda rp, s: rp.argmax(1), "R0 하드 라우팅(현행)"),
            econ(lambda rp, s: rp, "R1 확률가중 평균"),
            econ(lambda rp, s: np.full_like(rp, 1 / 3), "R2 균등 평균")]
    for ei, ename in enumerate(EN):
        rows.append(econ(lambda rp, s, _e=ei: np.full(len(rp), _e), f"R3 항상 {ename}"))
    for sd in (11, 22, 33, 44, 55):
        def shuf(rp, s):
            rg = np.random.default_rng(s)
            return rg.permutation(rp.argmax(1))    # ⭐비율 보존 · 배정만 무작위
        rows.append(econ(shuf, f"R4 무작위 라우팅(시드{sd})", seed=sd))

    log(f"\n{'='*106}\n■ 레짐 라우터 기여 (더블배리어 · q=0.75 · 4폴드)")
    log(f"{'팔':<24}{'건수':>8}{'독립일':>7}{'건당bp':>9}{'함축p':>8}{'CI':>20}{'중앙보유':>9}{'건/일':>7}{'순/일':>8}")
    for r in rows:
        log(f"{r['arm']:<24}{r['n']:>8,}{r['indep_days']:>7}{r['gross_bp']:>+9.2f}{r['p']*100:>7.2f}%"
            f"  [{r['ci95'][0]:+7.2f},{r['ci95'][1]:+7.2f}]{r['median_hold']:>9.0f}"
            f"{r['trades_per_day']:>7.2f}{r['net_day_usdc']:>8.1f}")
    r0 = rows[0]; r4 = [r for r in rows if r["arm"].startswith("R4")]
    m4 = float(np.mean([r["gross_bp"] for r in r4]))
    log(f"\n⭐R0(현행) − R4(무작위 라우팅 5시드 평균) = {r0['gross_bp'] - m4:+.2f}bp "
        f"· R4 범위 [{min(r['gross_bp'] for r in r4):+.2f},{max(r['gross_bp'] for r in r4):+.2f}]")
    log("  (차이가 R4 시드 범위 안이면 **라우터는 장식**이다 -- 전문가가 서로 교환 가능하다는 뜻)")
    OUTD = E.OUT; OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "stageO_routing.json").write_text(json.dumps(rows, indent=2, default=float))
    log(f"저장: {OUTD}/stageO_routing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        routing() if "--routing" in sys.argv else
        qcalib() if "--qcalib" in sys.argv else
        baserate() if "--baserate" in sys.argv else
        buildlabel() if "--buildlabel" in sys.argv else
        volatr() if "--volatr" in sys.argv else
        nocap() if "--nocap" in sys.argv else
        labelrate() if "--labelrate" in sys.argv else
        exitgrid() if "--exitgrid" in sys.argv else
        deployed() if "--deployed" in sys.argv else
        robust() if "--robust" in sys.argv else main())
