"""박스 유지 확률 (2026-10-16 사용자 «①부터 재검증하고 시안» → 같은 날 «HGB로 배포») -- 지금가 ± 0.5×(지난 W분 고저폭) 띠를 다음 W분 안 안 벗어날 확률.

모델(dashboard/box_hold_hgb_20261016.joblib, 2022~24 학습 · 재현 python scripts/research_eth_box_hold_improve_20261016.py --export):
  30분·1h = HGB 27칸 -- 옛 6칸(기하 d_σ·σ1h/σ24·창폭·시각·주말) + 변동성 기간구조(5·15·60·240분 σ/σ24·파킨슨·1h 최대 점프) ·
            박스 모양(창 안 위치·추세성·압축·24h 대비 창폭) · 활동(5분/창 거래량·|테이커 불균형|) · 요일×시각 계절(지금·다음 W분) ·
            예정 발표(다음 W분 안 CPI·고용·PPI·PCE·GDP·소매판매·FOMC · 실업수당 · 직전 60분).
  4h = HGB 옛 6칸(넓힌 칸은 보정이 나빠져 판정 탈락).
근거 scripts/research_eth_box_hold_improve_20261016.py: 2025·2026 AUC 30분 .770/.770→.787/.789 · 1h .769/.761→.790/.786 · 4h .775/.770→.783/.787,
  보정 최대오차 30분 5.0/3.3→2.2/1.0pp · 1h 4.7/5.1→3.4/4.1 · 4h 7.2/3.3→6.2/3.3. 🔴방향 정보 아님(어느 쪽으로 깨질지는 모른다).
입력: 닫힌 1분봉(오름차순) timestamp(봉 시작, UTC)·high·low·close·volume·taker_buy_base 1499개 이상 + 거시 일정 events
  (scripts/live_macro_calendar_20260826 형식 {"time_utc","category","detail"} -- 없으면 발표 칸 = 없음).
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

B = joblib.load(Path(__file__).with_name("box_hold_hgb_20261016.joblib"))
PROF = np.asarray(B["prof"], float)            # 요일×시각(월 0시 = 0) 평균 1분 수익² / 전체 평균, 학습 구간
WIN = {"30m": 30, "1h": 60, "4h": 240}
NEED = 1499                                    # 1h 박스 24h 창폭 중앙값 = 1440 + 59 · σ24 = 1441
MAJOR_IDS = {10, 50, 53, 54, 46, 9}            # FRED release_id: CPI·고용·GDP·PCE·PPI·소매판매 (+ category fomc)
CLAIMS_ID = 180                                 # 실업수당(주간)


def event_times(events: list | None) -> tuple[np.ndarray, np.ndarray]:
    """거시 일정 → (주요 발표 ms, 실업수당 ms) 정렬 배열."""
    major, claims = [], []
    for e in events or []:
        try:
            t = int(pd.Timestamp(e["time_utc"]).value // 10 ** 6)
        except Exception:  # noqa: BLE001
            continue
        m = re.search(r"release_id=(\d+)", str(e.get("detail", "")))
        rid = int(m.group(1)) if m else None
        if e.get("category") == "fomc" or rid in MAJOR_IDS:
            major.append(t)
        elif rid == CLAIMS_ID:
            claims.append(t)
    return np.array(sorted(major), np.int64), np.array(sorted(claims), np.int64)


def _next_min(tc: int, ev: np.ndarray) -> float:
    j = np.searchsorted(ev, tc, side="right")
    return (ev[j] - tc) / 60000.0 if j < len(ev) else 1e12


def _prev_min(tc: int, ev: np.ndarray) -> float:
    j = np.searchsorted(ev, tc, side="right") - 1
    return (tc - ev[j]) / 60000.0 if j >= 0 else 1e12


def features(k1m: pd.DataFrame, events: list | None = None) -> dict[str, dict[str, float]]:
    """마지막 닫힌 1분봉 마감 시점의 척도별 특징(연구 features() 와 같은 식 -- __main__ 패리티 점검)."""
    h, l, c = (k1m[x].to_numpy(float) for x in ("high", "low", "close"))
    v, tb = k1m["volume"].to_numpy(float), k1m["taker_buy_base"].to_numpy(float)
    r = np.diff(np.log(c))
    r2 = r ** 2
    s24 = float(np.sqrt(np.mean(r2[-1440:])))
    pk = np.log(h / l) ** 2 / (4 * np.log(2))
    t0 = pd.Timestamp(k1m["timestamp"].iloc[-1])
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
    t_ms = int(t0.value // 10 ** 6)
    tc = t_ms + 60_000
    hr = t0.hour + t0.minute / 60
    how = lambda ms: (pd.Timestamp(ms, unit="ms").dayofweek * 24 + pd.Timestamp(ms, unit="ms").hour)  # noqa: E731
    lsig = lambda w: float(np.log(np.sqrt(np.mean(r2[-w:])) + 1e-9) - np.log(s24))  # noqa: E731
    base = {"V_lsig5m": lsig(5), "V_lsig15m": lsig(15), "V_lsig1h": lsig(60), "V_lsig4h": lsig(240), "V_lsig24": float(np.log(s24)),
            "V_park1h": float(np.log(np.sqrt(np.mean(pk[-60:])) + 1e-9) - np.log(s24)),
            "V_park24": float(np.log(np.sqrt(np.mean(pk[-1440:])) + 1e-9) - np.log(s24)),
            "V_jump1h": float(np.max(np.abs(r[-60:])) / s24),
            "Q_imb1h": float(abs((2 * tb[-60:] - v[-60:]).sum() / v[-60:].sum())),
            "S_season_now": float(np.log(np.mean([PROF[how(t_ms - k * 60_000)] for k in range(60)])))}
    major, claims = event_times(events)
    out = {}
    for name, W in WIN.items():
        rngs = pd.Series(h).rolling(W).max().to_numpy() - pd.Series(l).rolling(W).min().to_numpy()
        rng, prng = float(rngs[-1]), float(rngs[-1 - W])
        whi, wlo = float(h[-W:].max()), float(l[-W:].min())
        p = (c[-1] - wlo) / rng
        nm = _next_min(tc, major)
        f = dict(base)
        f.update({"log_ds": float(np.log(0.5 * rng / c[-1] / (s24 * np.sqrt(W)))),
                  "log_vr": float(np.log(max(np.sqrt(np.mean(r2[-60:])), 1e-6) / s24)),
                  "log_rng": float(np.log(rng / c[-1] * 1e4)),
                  "hs": float(np.sin(2 * np.pi * hr / 24)), "hc": float(np.cos(2 * np.pi * hr / 24)), "wkd": float(t0.dayofweek >= 5),
                  "R_edge": float(min(p, 1 - p)), "R_trend": float(abs(c[-1] - c[-1 - W]) / rng),
                  "R_compress": float(np.log(rng / prng)), "R_squeeze": float(np.log(rng / np.nanmedian(rngs[-1440:]))),
                  "Q_vol5_vs_W": float(np.log(v[-5:].mean() / v[-W:].mean())),
                  "Q_imbW": float(abs((2 * tb[-W:] - v[-W:]).sum() / v[-W:].sum())),
                  "S_season_fut": float(np.log(np.mean([PROF[how(t_ms + k * 60_000)] for k in range(1, W + 1)]))),
                  "M_major_in": float(nm <= W), "M_major_pos": nm / W if nm <= W else 1.0,
                  "M_major_recent": float(_prev_min(tc, major) <= 60), "M_claims_in": float(_next_min(tc, claims) <= W),
                  "_rng": rng})
        out[name] = f
    return out


def payload(k1m: pd.DataFrame, events: list | None = None) -> dict:
    """k1m: 닫힌 1분봉(오름차순). 마지막 봉 종가 시점의 확률."""
    if k1m is None or len(k1m) < NEED:
        return {"ok": False, "reason": f"1분봉 {0 if k1m is None else len(k1m)}개 < {NEED}"}
    k1m = k1m.tail(NEED + 60).reset_index(drop=True)
    for x in ("high", "low", "close", "volume", "taker_buy_base"):
        k1m[x] = k1m[x].astype(float)
    F = features(k1m, events)
    px = float(k1m["close"].iloc[-1])
    t0 = pd.Timestamp(k1m["timestamp"].iloc[-1])
    asof = int((t0.tz_localize("UTC") if t0.tzinfo is None else t0).value // 10 ** 6) + 60_000
    out = {"ok": True, "asof_ms": asof, "px": px, "model": "hgb-20261016", "events_n": len(events or []), "scales": {}}
    for name, W in WIN.items():
        f, cf = F[name], B["scales"][name]
        x = pd.DataFrame([[f[k] for k in cf["cols"]]], columns=cf["cols"])   # 학습 때 열 이름 그대로(이름 없으면 sklearn 경고가 매 호출)
        if not (f["_rng"] > 0 and np.isfinite(x.to_numpy()).all()):
            out["scales"][name] = {"ok": False}
            continue
        rng = f["_rng"]
        out["scales"][name] = {"ok": True, "p": round(float(cf["model"].predict_proba(x)[0, 1]), 4), "base": round(cf["train_base"], 4),
                               "q20": round(cf["q20"], 4), "q80": round(cf["q80"], 4),   # 화면 «깨질 쪽»(< q20) · «머물 쪽»(> q80) = 학습 예측 분위
                               "up": round(px + 0.5 * rng, 2), "dn": round(px - 0.5 * rng, 2), "half_pct": round(0.5 * rng / px * 100, 3),
                               "until_ms": asof + W * 60_000}
    return out


if __name__ == "__main__":
    # 자체점검 ① 패리티: 연구 산출(decisions.parquet·features.parquet)의 실제 행 40개 = 이 모듈이 같은 1분봉 1499개로 낸 특징·확률.
    #   (개발 PC 전용 -- 연구 산출이 없으면 건너뛴다) ② 짧은 입력 거부 ③ 발표 일정 해석.
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("imp", root / "scripts/research_eth_box_hold_improve_20261016.py")
    imp = importlib.util.module_from_spec(spec); spec.loader.exec_module(imp)
    maj, cl = event_times([{"time_utc": "2026-10-15T12:30:00+00:00", "category": "econ", "detail": "x (FRED, release_id=10)"},
                           {"time_utc": "2026-10-15T18:00:00+00:00", "category": "fomc"},
                           {"time_utc": "2026-10-16T12:30:00+00:00", "category": "econ", "detail": "release_id=180"},
                           {"time_utc": "2026-10-16T13:15:00+00:00", "category": "econ", "detail": "release_id=13"},   # 산업생산 = 주요 아님
                           {"time_utc": "bad", "category": "fomc"}])
    assert len(maj) == 2 and len(cl) == 1, (maj, cl)
    assert payload(pd.DataFrame({"timestamp": [], "high": [], "low": [], "close": [], "volume": [], "taker_buy_base": []}))["ok"] is False
    if imp.DEC.exists() and (imp.OUT / "features.parquet").exists() and (imp.OUT / "events.json").exists():
        D = pd.read_parquet(imp.DEC)
        X = pd.read_parquet(imp.OUT / "features.parquet")
        K = imp.load_1m()
        ev = imp.events()
        evl = [{"time_utc": pd.Timestamp(int(t), unit="ms", tz="UTC").isoformat(), "category": "fomc"} for t in ev["major"]] + \
              [{"time_utc": pd.Timestamp(int(t), unit="ms", tz="UTC").isoformat(), "category": "econ", "detail": "release_id=180"} for t in ev["claims"]]
        A = pd.concat([D, X], axis=1)
        A = A[A.scale.isin(list(WIN)) & (A.ts >= "2025-01-01")].replace([np.inf, -np.inf], np.nan).dropna(subset=B["scales"]["1h"]["cols"])
        pick = pd.concat([A[A.M_major_in == 1].sample(10, random_state=1), A.sample(30, random_state=2)])   # 발표 낀 행을 꼭 넣는다
        pos = pd.Series(np.arange(len(K)), index=K.t)
        worst = 0.0
        for _, row in pick.iterrows():
            i = int(pos[int(row.ts.value // 10 ** 6)])
            k = K.iloc[i - NEED + 1:i + 1]
            k1m = pd.DataFrame({"timestamp": pd.to_datetime(k.t, unit="ms", utc=True), "high": k.h, "low": k.l, "close": k.c,
                                "volume": k.v, "taker_buy_base": k.tb})
            f = features(k1m.reset_index(drop=True), evl)[row.scale]
            cols = B["scales"][row.scale]["cols"]
            a, b = np.array([f[c] for c in cols]), row[cols].to_numpy(float)
            worst = max(worst, float(np.max(np.abs(a - b))))
            assert np.allclose(a, b, rtol=1e-6, atol=1e-8), (row.scale, row.ts, {c: (x, y) for c, x, y in zip(cols, a, b) if not np.isclose(x, y)})
            m = B["scales"][row.scale]["model"]
            pp = payload(k1m, evl)["scales"][row.scale]["p"]
            assert abs(pp - m.predict_proba(row[cols].to_frame().T.astype(float))[0, 1]) < 1e-4
        print(f"패리티 OK -- 연구 행 {len(pick)}개(발표 낀 행 10 포함) 특징 최대 차 {worst:.2e} · 확률 같음")
    else:
        print("패리티 건너뜀(연구 산출 없음)")
    print("selftest OK")
