#!/usr/bin/env python3
"""**전망된 변동성이 방향을 가르는가** — 크기 축의 유일한 수익 경로 검정 (2026-09-11). 모델 신규학습 없음.

사용자: *"다음 단계 진행해줘"* → 남은 레버 2 "크기 축을 제품으로".

## 왜 이 형태여야 하는가 (산술)
저장소 항등식 `순익 = (2a−1)·b − 비용`. 크기 축은 **b** 다. 그런데 **a ≈ 0.50** 이면
b 를 아무리 키워도 순익은 −비용이다. ⇒ **크기 축이 이익을 내는 경로는 "a 를 가르는 것" 하나뿐이다.**
무방향 배리어 거래로는 안 된다: 무드리프트 자산에서 변동성 지식은 **터치까지의 시간**을 바꿀 뿐
**확률**을 바꾸지 않는다.

## 왜 아직 안 닫혔는가
방향축 3,500셀과 레짐 조건화는 **전부 과거만 보는 변수**를 썼다(ATR·atr_percentile·실현변동성·S12_K3).
배포된 24시간 변동성 전망은 [[eth_dvol_vol_forecast_new_signal_20260910]] 기준 **화면 최초의 미래지향 지표**다.
미래지향 레짐으로 방향을 갈라 본 적은 없다.

## 사전등록 (결과 보기 전 고정)
방향  **음(−)**: 변동성 확대 전망 → 하락. 암호화폐 레버리지 효과(음의 변동성-수익 상관)에 근거.
      ⚠️**사후 반전 금지.** 양측 모두 보고하되 헤드라인은 이 부호다.
등급  배포 컷 그대로 — 위험 p≥0.5387 · 주의 p≥0.2836 · 안전 p<0.2836 (**새로 고르지 않는다**)
지평  1h · 4h · 24h (전망 지평이 24h 이므로 그 이하)
격자  3등급 × 3지평 = **9셀**
분할  아티팩트 자신의 분할 — TRAIN ≤2026-03-31 / OOS ~2026-08-04 / HOLDOUT ~2026-09-08.
      모델은 TRAIN 에만 적합됐으므로 OOS·HOLDOUT 이 정직하다.
비용  10bp 테이커 왕복
1차   **일군집 부트 CI 가 0 배제** — OOS 와 HOLDOUT **둘 다**. 🔴순환이동 p 만 믿지 않는다:
      09-10 VRP 기각에서 *"창 안에 지배적 에피소드가 하나면 순환이동 귀무가 0 근처로 좁아져
      '그 한 번을 맞혔다'가 유의로 보인다"* 를 실제로 밟았다. 순환이동은 보조로만 병기.
출력  tmp/volfc_dir_20260911/report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_vol_forecast_20260910 as VF  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/volfc_dir_20260911"
DVOL = ROOT / "data/derivatives/deribit_dvol/ETH_dvol_hourly.csv"
ART = ROOT / "data/live/eth_vol_forecast_artifact"
HORIZ = (1, 4, 24)
COST_BP = 10.0
DIRECTION = -1.0                 # 사전등록: 확대 전망 → 하락
B_BOOT, B_CYC = 4000, 600
SEED = 20260911


def log(m):
    print(f"[vfd {time.strftime('%H:%M:%S')}] {m}", flush=True)


def day_cluster_ci(vals, days, b=B_BOOT, seed=SEED):
    """⭐**일 군집** 부트 — 같은 날 관측을 통째로 재표집한다.
    09-10 VRP 기각의 교훈: 순환이동 p 는 지배적 에피소드가 하나면 속인다. 이게 옳은 잣대."""
    u = np.unique(days)
    if len(u) < 15:
        return [float("nan")] * 2, len(u)
    idx = {d: np.flatnonzero(days == d) for d in u}
    rng = np.random.default_rng(seed); o = []
    for _ in range(b):
        pick = rng.choice(u, len(u), replace=True)
        v = np.concatenate([vals[idx[d]] for d in pick])
        if len(v):
            o.append(v.mean())
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))], len(u)


def cyc_null(full, idx, b=B_CYC, seed=SEED):
    n = len(full); rng = np.random.default_rng(seed); o = []
    for s in rng.integers(1, n, b):
        v = full[(idx + s) % n]
        v = v[np.isfinite(v)]
        if len(v):
            o.append(v.mean())
    return np.array(o)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ART / "meta.json").read_text())
    cuts = meta["cuts"]
    log(f"아티팩트 {meta['rule_id']} · 컷 {cuts}")

    kl = VF.load_klines() if hasattr(VF, "load_klines") else None
    if kl is None:                                   # 라이브 모듈 규약에 맞춰 직접 적재
        kl = pd.read_csv(ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
                         parse_dates=["timestamp"])
    # ⭐라이브 모듈 규약과 **같은 형태**로 넘긴다(`_fetch_dvol` 이 close→dvol 로 개명해 반환).
    # 자기 사본 공식을 쓰면 불일치를 못 잡는다 — 09-10 파리티 함정의 교훈.
    dv = pd.read_csv(DVOL, parse_dates=["timestamp"])[["timestamp", "close"]] \
        .rename(columns={"close": "dvol"}).sort_values("timestamp")
    # 로컬 CSV 는 2026-08-04 까지라 **홀드아웃 구간이 없다**. 라이브 모듈의 공개 API 로 연장한다
    # (09-10 에도 같은 방식으로 홀드아웃 36일을 만들었다).
    ext = VF._fetch_dvol(hours=24 * 420)
    if ext is not None and len(ext):
        dv = pd.concat([dv, ext]).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        log(f"DVOL 연장: 로컬 끝 → {dv.timestamp.max()} (API {len(ext)}행 병합)")
    else:
        log("⚠️DVOL API 연장 실패 — 홀드아웃 구간 없이 진행")
    d = VF.build_features(kl, dv)
    if d is None or not len(d):
        log("🔴 build_features 실패"); return 1
    log(f"피쳐 프레임 {d.shape} · {d.index[0]} → {d.index[-1]}")

    # ⭐예측은 **라이브와 같은 경로**로: 표준화 후 clf. 자기 사본을 만들면 불일치를 못 잡는다.
    import joblib
    M = joblib.load(ART / "model.joblib")
    X = d[VF.FEATS].to_numpy(float)
    ok = np.isfinite(X).all(1)
    p = np.full(len(d), np.nan)
    p[ok] = M["clf"].predict_proba((X[ok] - M["mu"]) / M["sd"])[:, 1]
    d = d.assign(p=p)

    px = pd.Series(kl["close"].to_numpy(float),
                   index=pd.DatetimeIndex(kl["timestamp"])).resample("1h").last()
    px = px.reindex(d.index)
    rep = {"prereg": {"direction": "음(−) 확대전망→하락, 사후반전 금지", "cuts": cuts,
                      "horizons": HORIZ, "cost_bp": COST_BP,
                      "criterion": "일군집 부트 CI 0배제 — OOS·HOLDOUT 둘 다"},
           "meta_auc": meta["auc"], "cells": {}}
    tr_end = pd.Timestamp(meta["train_span"][1]); oos_end = pd.Timestamp(meta["oos_span"][1])
    seg = np.where(d.index <= tr_end, "TRAIN",
                   np.where(d.index <= oos_end, "OOS", "HOLDOUT"))
    grade = np.where(d.p >= cuts["위험"], "위험",
                     np.where(d.p >= cuts["주의"], "주의", "안정"))
    log(f"등급 분포: " + " ".join(f"{g}={int((grade==g).sum())}" for g in ("위험", "주의", "안정")))

    days = d.index.floor("D").astype("int64").to_numpy()
    for H in HORIZ:
        fwd = (px.shift(-H) / px - 1.0).to_numpy() * 1e4      # bp
        signed = DIRECTION * fwd                              # 사전등록 부호 적용
        for g in ("위험", "주의", "안정"):
            for s in ("OOS", "HOLDOUT", "TRAIN"):
                m = (grade == g) & (seg == s) & np.isfinite(signed)
                idx = np.flatnonzero(m)
                if len(idx) < 40:
                    continue
                v = signed[idx]
                ci, nday = day_cluster_ci(v, days[idx])
                nl = cyc_null(np.where(np.isfinite(signed), signed, np.nan), idx)
                rep["cells"][f"H{H}|{g}|{s}"] = {
                    "n": int(len(idx)), "n_days": nday, "mean_bp": float(v.mean()),
                    "acc": float((v > 0).mean()), "net_bp": float(v.mean() - COST_BP),
                    "day_ci95": ci, "ci_excludes_zero": bool(np.isfinite(ci[0]) and ci[0] > 0),
                    "cyc_null_p97.5": float(np.percentile(nl, 97.5)) if len(nl) else None,
                    "beats_cyc": bool(len(nl) and v.mean() > np.percentile(nl, 97.5))}
    log("=" * 100)
    log(f"{'셀':>18} {'n':>6} {'독립일':>6} {'평균bp':>9} {'정확도':>7} {'순bp':>9} "
        f"{'일군집 CI95':>22} {'순환':>5}")
    for k, c in rep["cells"].items():
        log(f"{k:>18} {c['n']:>6} {c['n_days']:>6} {c['mean_bp']:>+9.1f} {c['acc']:>7.3f} "
            f"{c['net_bp']:>+9.1f} [{c['day_ci95'][0]:>+8.1f},{c['day_ci95'][1]:>+8.1f}] "
            f"{'통과' if c['beats_cyc'] else '—':>5}"
            + ("  ⭐CI 0배제" if c["ci_excludes_zero"] else ""))
    oo = [k for k, c in rep["cells"].items() if k.endswith("|OOS") and c["ci_excludes_zero"]]
    ho = [k for k, c in rep["cells"].items() if k.endswith("|HOLDOUT") and c["ci_excludes_zero"]]
    both = [k.rsplit("|", 1)[0] for k in oo
            if k.rsplit("|", 1)[0] + "|HOLDOUT" in [x for x in ho]]
    rep["summary"] = {"oos_pass": oo, "holdout_pass": ho, "both_pass": both}
    log("=" * 100)
    log(f"⭐1차 기준 — OOS CI 0배제 {len(oo)} · HOLDOUT {len(ho)} · **둘 다 {len(both)}**")
    if both:
        log(f"    통과: {both}")
    # ── 2팔: **비용 축** — 전망으로 싼 구간을 고를 수 있는가 ────────────────────
    # 방향이 안 갈리면 크기 축의 남은 경로는 "비용을 낮춘다" 하나다.
    # 저장소 구속조건이 비용(엣지 1~5bp vs 비용 5.5~12bp)이므로 이건 별개 질문이다.
    # 대리변수: 시간봉 내 고저 폭(체결 시 불리하게 받는 폭의 상한) · 5분봉 VWAP 편차.
    k5 = pd.DataFrame({"t": pd.DatetimeIndex(kl["timestamp"]),
                       "h": kl["high"].to_numpy(float), "l": kl["low"].to_numpy(float),
                       "c": kl["close"].to_numpy(float), "v": kl["volume"].to_numpy(float),
                       "q": kl["quote_volume"].to_numpy(float)}).set_index("t")
    # 🔴**전방** 비용이라야 한다. 모델은 "지금 조용하나 앞으로 확대"를 예측하므로
    # 동시점 변동폭을 재면 위험 등급이 오히려 낮게 나와 정반대로 읽힌다(초판에서 실제로 그랬다).
    base = pd.DataFrame({
        "rng_bp": (k5["h"].resample("1h").max() / k5["l"].resample("1h").min() - 1) * 1e4,
        "vwap_dev_bp": ((k5["q"].resample("1h").sum() / k5["v"].resample("1h").sum())
                        / k5["c"].resample("1h").last() - 1).abs() * 1e4,
        "dollar_vol": k5["q"].resample("1h").sum()})
    FW = 24                                   # 전망 지평과 동일한 24시간 전방 창
    def fwd_mean(x):
        """t 시점 값 = t+1 .. t+FW 평균. 수치 검산으로 확인함(t=10 → 22.5)."""
        return x.shift(-1).rolling(FW).mean().shift(-(FW - 1))
    hr = pd.DataFrame({c: fwd_mean(base[c]) for c in base.columns}).reindex(d.index)
    rep["cost_arm"] = {}
    log("=" * 100)
    log("⭐2팔 — 등급별 **전방 24시간** 체결비용 대리변수 (낮을수록 싸다)")
    log(f"{'등급':>6}{'구간':>9} {'n':>6} {'전방 고저폭bp':>14} {'전방VWAP편차':>13} {'전방달러량':>12}")
    for g in ("위험", "주의", "안정"):
        for s_ in ("OOS", "HOLDOUT", "TRAIN"):
            m = (grade == g) & (seg == s_) & np.isfinite(hr["rng_bp"].to_numpy())
            if m.sum() < 40:
                continue
            sub = hr[m]
            cell = {"n": int(m.sum()), "range_bp_median": float(sub["rng_bp"].median()),
                    "vwap_dev_bp_median": float(sub["vwap_dev_bp"].median()),
                    "dollar_vol_median": float(sub["dollar_vol"].median())}
            rep["cost_arm"][f"{g}|{s_}"] = cell
            log(f"{g:>6}{s_:>9} {cell['n']:>6} {cell['range_bp_median']:>15.1f} "
                f"{cell['vwap_dev_bp_median']:>12.2f} ${cell['dollar_vol_median']/1e6:>10.1f}M")
    a = rep["cost_arm"].get("안정|OOS"); b = rep["cost_arm"].get("위험|OOS")
    if a and b:
        log(f"⭐안정 대비 위험 — 고저폭 {b['range_bp_median']/a['range_bp_median']:.2f}배 · "
            f"VWAP편차 {b['vwap_dev_bp_median']/a['vwap_dev_bp_median']:.2f}배 (OOS)")
        log("   ⚠️고저폭은 **변동성이지 스프레드가 아니다**. 싼 구간을 고를 수 있다는 증거로 쓰려면"
            " 실제 체결 원장이 필요하다(peg-maker 섀도우).")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
