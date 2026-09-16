#!/usr/bin/env python3
"""사이징 규칙 전수 경주 — 같은 원장·같은 프로토콜로 순위를 낸다 (2026-09-16).

사용자 *"크기(사이징)에 적합한 규칙 모델이 이렇게나 많은데 제일 좋은거 순서대로 리스트업해줘"*.

기억에서 나열하지 않는다. **실계좌 72왕복에 전부 같은 방식으로 얹어 잰다**:
진입·청산은 고정하고 **사이징만 반사실**로 갈아끼우며, **평균 명목을 맞춰** 크기 효과만 남긴다.

🔴이 세션이 밝힌 것이 순위의 전제다: 배포 경로에서 **MAE 모델 상한이 묶은 건 72건 중 1건**이고
나머지는 순자산×6·원장중앙×2 가 묶는다([[money_claims_reverified_two_were_wrong_20260916]]).
그래서 «모델»과 «상한»을 **따로** 세워 누가 실제로 일하는지 본다.

지표: bp/일 · **일 샤프**(1차 정렬 — MDD 는 독립일 22 에서 한 왕복이 지배한다) · MDD ·
      Δ(vs 고정)의 일 부트 CI95 · 가중치 변동계수.
🔴검정력: 왕복 72 · **독립일 22**. 이 표는 «순위»이지 «승격»이 아니다.

출력: tmp/eth_sizing_horserace_20260916/
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "tmp/eth_sizing_horserace_20260916"
RNG = np.random.default_rng(20260916)
EQUITY_X, LEDGER_MULT, LEDGER_WIN, LEDGER_MIN = 6.0, 2.0, 30, 10
HOLD_FIXED_MIN, COST_BP, E0 = 240, 5.88, 719.0


def _mod(rel: str, name: str):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp); argv, sys.argv = sys.argv, ["x"]
    try: sp.loader.exec_module(m)
    finally: sys.argv = argv
    return m


def score(bp: np.ndarray, w: np.ndarray, day: pd.Series) -> dict:
    """평균 명목을 맞춘 뒤 일별 합. 0 가중(=진입 안 함)도 그대로 둔다."""
    w = np.asarray(w, float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return dict(bp_day=np.nan, sharpe=np.nan, mdd=np.nan, cv=np.nan, zero=int((w == 0).sum()))
    wn = w / w.mean()
    g = pd.DataFrame({"d": day, "x": bp * wn}).groupby("d").x.sum()
    cum = np.r_[0.0, g.cumsum().to_numpy()]     # 시작 0 을 봉우리에 포함(표준 MDD 관례)
    return dict(bp_day=float(g.mean()),
                sharpe=float(g.mean() / g.std(ddof=1)) if g.std(ddof=1) > 0 else np.nan,
                mdd=float((np.maximum.accumulate(cum) - cum).max()),
                cv=float(wn.std() / wn.mean()),
                zero=int((w == 0).sum()), daily=g)


def selftest() -> None:
    bp = np.array([10., -20., 30., 5.])
    d = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]))
    a, b = score(bp, np.ones(4), d), score(bp, np.full(4, 9.0), d)
    assert abs(a["bp_day"] - b["bp_day"]) < 1e-12 and abs(a["cv"]) < 1e-12   # 상수 가중 = 고정
    # 일별: (10-20)=-10, (30+5)=35 -> 평균 12.5, MDD 10
    assert abs(a["bp_day"] - 12.5) < 1e-9 and abs(a["mdd"] - 10.0) < 1e-9
    # 가중이 손실에 크면 더 나쁘다
    bad = score(bp, np.array([1., 3., 1., 1.]), d)
    assert bad["bp_day"] < a["bp_day"]
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest: selftest(); return 0
    selftest(); OUT.mkdir(parents=True, exist_ok=True)
    MQ = _mod("scripts/live_eth_mae_quantile_model_20260913.py", "MQ")
    PO = _mod("scripts/live_eth_risk_sizing_policy_20260913.py", "PO")
    TP = _mod("scripts/live_eth_trade_plan_20260913.py", "TP")
    SV = _mod("scripts/live_eth_sizing_vol_model_20260912.py", "SV")
    svm = MQ.svm

    kl = pd.read_csv(REPO / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
                     usecols=["timestamp", "open", "high", "low", "close", "volume",
                              "quote_volume", "trades"])
    kl["timestamp"] = pd.to_datetime(kl.timestamp)
    c, hi, lo = kl.close.to_numpy(float), kl.high.to_numpy(float), kl.low.to_numpy(float)
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy() / np.maximum(c, 1e-12)
    atr288 = pd.Series(tr).rolling(288, min_periods=144).mean().to_numpy() / np.maximum(c, 1e-12)
    base = svm.build_features(kl.timestamp, c, kl.quote_volume.to_numpy(float),
                              kl.trades.to_numpy(float), hi, lo).replace([np.inf, -np.inf], np.nan)
    bts = kl.timestamp.to_numpy()

    t = pd.DataFrame([json.loads(l) for l in open(REPO / "data/live/account_round_trips.jsonl")])
    t["entry_ts"] = pd.to_datetime(t.entry_time, unit="ms")
    t["exit_ts"] = pd.to_datetime(t.exit_time, unit="ms")
    t = t.sort_values("entry_ts").reset_index(drop=True)
    t["nom"] = t.entry_price * t.max_qty
    mv = np.where(t.side == "SHORT", -1, 1)
    t["bp"] = (t.exit_price - t.entry_price) / t.entry_price * mv * 1e4 - COST_BP
    day = t.entry_ts.dt.floor("D")
    i = np.clip(np.searchsorted(bts, t.entry_ts.to_numpy(), "right") - 1, 0, len(base) - 1)
    print(f"왕복 {len(t)} · 독립일 {day.nunique()} · {t.entry_ts.min():%m-%d} ~ {t.exit_ts.max():%m-%d}")

    # ---- 재료 ----
    HOLDS = list(TP.HOLD_CHOICES)
    sm = {}
    for hm in HOLDS:
        r = base.iloc[i].copy().reset_index(drop=True)
        r["log_h"] = np.log(float(hm)); r["side"] = np.where(t.side.to_numpy() == "SHORT", -1, 1)
        sm[hm] = MQ.safe_mae(models_ := MQ.load_model()["models"], r[MQ.FEATURES],
                             MQ.load_model()["mult"])
    eq = E0 + np.concatenate([[0.0], np.cumsum(t.net_pnl.to_numpy())[:-1]])
    nom = t.nom.to_numpy()
    ent, ext = t.entry_ts.to_numpy(), t.exit_ts.to_numpy()
    existing = np.array([nom[(ent < ent[k]) & (ext > ent[k])].sum() for k in range(len(t))])
    cap_led = np.array([LEDGER_MULT * float(np.median(nom[:k][-LEDGER_WIN:]))
                        if k >= LEDGER_MIN else np.nan for k in range(len(t))])
    cap_eq = eq * EQUITY_X
    # 배포 지평 선택
    hold, safe = np.zeros(len(t), int), np.zeros(len(t))
    for k in range(len(t)):
        pol = [v for v in (cap_led[k], cap_eq[k]) if np.isfinite(v)]
        px = min(pol) / eq[k] if pol and eq[k] > 0 else EQUITY_X
        rt = {str(h): {t.side[k]: {"safe_mae_pct": float(sm[h][k])}} for h in HOLDS}
        r = TP.recommend_hold(rt, t.side[k], px, float(atr[i][k]), acc=TP.PRESCRIBE_ACC)
        hold[k] = int(r["recommended_min"]) if r.get("available") else HOLD_FIXED_MIN
        safe[k] = sm[hold[k]][k]
    cap_mod = np.array([PO.entry_notional(eq[k], float(safe[k]))["total_notional"]
                        for k in range(len(t))])
    surv = np.array([PO.survival_leverage(float(safe[k])) * eq[k] for k in range(len(t))])
    pol = [PO.policy_leverage(float(safe[k])) for k in range(len(t))]
    grow = np.array([(p["growth_x"] or np.nan) * eq[k] for k, p in enumerate(pol)])
    # 배포 변동성 모델
    try:
        art = SV.load_model()
        Xv = SV.build_features(kl.timestamp, c, kl.quote_volume.to_numpy(float),
                               kl.trades.to_numpy(float), hi, lo)
        pv = SV.predict_vol(art["models"], Xv.iloc[i][art["features"]])
        volmodel_ok = np.isfinite(pv).all()
    except Exception as e:
        pv = np.full(len(t), np.nan); volmodel_ok = False
        print(f"  ⚠️배포 변동성 모델 로드 실패: {type(e).__name__}: {e}")

    def capmix(*caps):
        v = np.vstack([x for x in caps])
        m = np.nanmin(v, axis=0)
        return np.maximum(0.0, np.minimum(m, m - existing * 0) - 0.0)

    def capped(cap):
        return np.maximum(0.0, np.minimum(cap, cap - existing))

    ARMS: dict[str, np.ndarray] = {
        "①고정(기준)": np.ones(len(t)),
        "역ATR 1/atr14": 1.0 / np.maximum(atr[i], 1e-9),
        "역ATR 1/atr288": 1.0 / np.maximum(atr288[i], 1e-9),
        "역변동성(배포모델)": 1.0 / np.maximum(pv, 1e-9) if volmodel_ok else np.full(len(t), np.nan),
        "동일위험 1/safeMAE": 1.0 / np.maximum(safe, 1e-9),
        "생존레버(안전MAE)": surv,
        "성장레버(켈리)": grow,
        "모델상한만(D)": cap_mod,
        "원장상한만(중앙×2)": np.where(np.isfinite(cap_led), cap_led, np.nanmedian(cap_led)),
        "순자산상한만(×6)": cap_eq,
        "상한2개 min(원장,순자산)": np.nanmin(np.vstack([cap_led, cap_eq]), axis=0),
        "배포 그대로 min(3개)": np.minimum(cap_mod, np.nanmin(np.vstack([cap_led, cap_eq, cap_mod]), axis=0)),
        "사용자 실제 명목": nom,
    }
    # 기존 포지션 차감은 상한류에만 적용(배포 규약)
    for k in ("모델상한만(D)", "원장상한만(중앙×2)", "순자산상한만(×6)",
              "상한2개 min(원장,순자산)", "배포 그대로 min(3개)"):
        ARMS[k] = np.maximum(0.0, ARMS[k] - existing)
    for k in list(ARMS):
        v = ARMS[k]
        if not np.isfinite(v).all():
            v = np.where(np.isfinite(v), v, np.nanmedian(v))
            ARMS[k] = v

    base_daily = score(t.bp.to_numpy(), ARMS["①고정(기준)"], day)["daily"]
    rows = []
    for name, w in ARMS.items():
        s = score(t.bp.to_numpy(), w, day)
        if not np.isfinite(s["bp_day"]):
            rows.append(dict(arm=name, **{k: np.nan for k in
                                          ("bp_day", "sharpe", "mdd", "cv", "d_bp", "ci_lo", "ci_hi")},
                             zero=s["zero"])); continue
        d = (s["daily"] - base_daily).dropna()
        u = d.index.to_numpy()
        bs = np.array([d.loc[RNG.choice(u, len(u))].mean() for _ in range(4000)])
        rows.append(dict(arm=name, bp_day=s["bp_day"], sharpe=s["sharpe"], mdd=s["mdd"],
                         cv=s["cv"], zero=s["zero"], d_bp=float(d.mean()),
                         ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5))))
    D = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    D.round(4).to_csv(OUT / "horserace.csv", index=False)

    print(f"\n{'='*126}")
    print("■ 사이징 규칙 경주 — 실계좌 72왕복 · 진입/청산 고정 · 평균명목 동일 · 일 샤프 내림차순")
    print(f"{'#':>3}{'규칙':<24}{'bp/일':>9}{'일샤프':>8}{'MDD':>7}{'가중CV':>8}{'0수량':>6}"
          f"{'Δ vs 고정':>10}{'Δ CI95':>22}")
    for k, r in D.iterrows():
        star = " ★" if np.isfinite(r.ci_lo) and (r.ci_lo > 0) else ""
        print(f"{k+1:>3}{r.arm:<24}{r.bp_day:>+9.2f}{r.sharpe:>8.3f}{r.mdd:>7.0f}{r.cv:>8.2f}"
              f"{int(r.zero):>6}{r.d_bp:>+10.2f}"
              f"{f'[{r.ci_lo:+.2f},{r.ci_hi:+.2f}]':>22}{star}")
    print(f"\n  ★ = 고정 대비 Δ 의 일 부트 CI95 가 0 을 배제")
    print(f"  🔴독립일 {day.nunique()} · 왕복 {len(t)}. MDD 는 한 왕복이 지배하므로 2차 지표로만 읽는다.")
    # ---- 왜 고정이 이기는가: 이 표본에서 «변동성 ↔ 수익» 이 어떤 관계인가 ----
    from scipy.stats import spearmanr
    print(f"\n■ ⭐왜 고정이 이기나 — 이 원장에서 «작게 걸라는 신호」와 실현손익의 관계")
    for nm, v in (("atr14(진입봉)", atr[i]), ("atr288", atr288[i]), ("안전MAE", safe),
                  ("예측변동성", pv if volmodel_ok else np.full(len(t), np.nan))):
        if not np.isfinite(v).all(): continue
        r = spearmanr(v, t.bp.to_numpy())
        print(f"  ρ({nm:<12}, 실현bp) = {r.statistic:+.3f}  p {r.pvalue:.4f}"
              f"   ⇒ 양수면 «변동성 큰 자리가 더 벌었다» = 축소 규칙이 손해")
    q = pd.qcut(pd.Series(atr288[i]), 3, labels=["저변동", "중", "고변동"])
    print("  ATR288 3분위 실현bp: " + " · ".join(
        f"{k} {t.bp[q.to_numpy()==k].mean():+.1f}bp(n={int((q.to_numpy()==k).sum())})"
        for k in ["저변동", "중", "고변동"]))
    print(f"\n■ 각 규칙의 가중치 ↔ 실현손익 상관 (양수여야 좋은 규칙)")
    for name, w in ARMS.items():
        if name == "①고정(기준)": continue
        wn = w / w.mean()
        print(f"  {name:<24} ρ(w, 실현bp) {spearmanr(wn, t.bp.to_numpy()).statistic:+.3f}")
    print("=" * 126)
    print(json.dumps({"arms": len(D), "days": int(day.nunique()),
                      "n_sig": int((D.ci_lo > 0).sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
