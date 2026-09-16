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
BARS = {"1h": 12, "4h": 48, "8h": 96, "12h": 144}
HZ = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--hz=")),
          "4h" if "--h4" in sys.argv else "1h")
assert HZ in BARS, f"모르는 지평: {HZ}"
E.HORIZONS.setdefault(HZ, BARS[HZ])        # E.load() 가 fwd_{HZ}_bp 를 만들게 한다
SUF = "" if HZ == "1h" else f"_{HZ}"
OUTJ = E.OUT / f"stageK_side_skill{SUF}.json"
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

    rows, pooled = [], []
    for name, t0, t1, v0, v1 in FOLDS:
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

        if name in cache:
            z = dict(cache[name].item())
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
            cache[name] = np.array({"Ds": Ds, "Qs": Qs, "tDm": tDm, "tQm": tQm}, dtype=object)
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


if __name__ == "__main__":
    raise SystemExit(robust() if "--robust" in sys.argv else main())
