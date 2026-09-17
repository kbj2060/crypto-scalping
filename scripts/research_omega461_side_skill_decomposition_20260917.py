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


if __name__ == "__main__":
    raise SystemExit(
        labelrate() if "--labelrate" in sys.argv else
        exitgrid() if "--exitgrid" in sys.argv else
        deployed() if "--deployed" in sys.argv else
        robust() if "--robust" in sys.argv else main())
