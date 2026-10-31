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
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# 코인 -> HGB 번들 파일. 2026-10-20 XRP 전용(같은 식 · 2022~24 학습 · 2025/2026 AUC 30m .777/.779 · 1h .776/.776 · 보정 ≤5.7pp 통과 · 4h 2025 보정 6.3pp 불통과 → deployable False)
#   재현 python scripts/research_xrp_box_hold_20261020.py · 결과 tmp/xrp_parity_20261020/box_hold/report.md
#   2026-10-31 BTC·SOL 전용(같은 식·같은 관문 · 30분·1h 통과 · 4h 2026 보정 8.4/9.8pp 불통과 → deployable False)
#   재현 python scripts/research_btc_sol_box_hold_20261031.py --run · 결과 docs/experiments/btc_sol_box_hold_20261031.md
HGB = {"eth": "box_hold_hgb_20261016.joblib", "xrp": "box_hold_hgb_xrp_20261020.joblib",
       "btc": "box_hold_hgb_btc_20261031.joblib", "sol": "box_hold_hgb_sol_20261031.joblib"}
BUNDLES = {a: joblib.load(Path(__file__).with_name(f)) for a, f in HGB.items()}
B = BUNDLES["eth"]
PROF = np.asarray(B["prof"], float)            # 요일×시각(월 0시 = 0) 평균 1분 수익² / 전체 평균, 학습 구간
WIN = {"30m": 30, "1h": 60, "4h": 240}
NEED = 1499                                    # 1h 박스 24h 창폭 중앙값 = 1440 + 59 · σ24 = 1441
MAJOR_IDS = {10, 50, 53, 54, 46, 9}            # FRED release_id: CPI·고용·GDP·PCE·PPI·소매판매 (+ FOMC 금리결정)
CLAIMS_ID = 180                                 # 실업수당(주간)
# 2026-10-26 ETH 1h 굴림 Platt90(docs/eth_box_hold_calibration_drift_prereg_20261025.md 8라운드 · 2026 ECE 1.45→0.67pp) --
#   scripts/fit_box_hold_platt90_20261026.py 가 매월 1일 쓴다. 없거나 40일 넘었거나 다른 모델용이면 보정 없음(원래 p).
#   2026-10-31 XRP 1h 도(사전등록 docs/prereg/xrp_box_hold_platt90_qx4_20261031.md · ΔBrier −.00079 단측 90% 통과 · ECE 2.31→0.40pp) -- 같은 적합 스크립트 --asset xrp
#   2026-10-31 BTC 1h 도(docs/prereg/btc_sol_box_hold_20261031.md · ΔBrier −.00022 단측 90% 통과) · SOL 은 2025 창 ΔBrier +.000003(V5 ✗) 불통과 → 표에 없음 = 보정 없음
PLATT90 = {a: Path(__file__).with_name(f"box_hold_platt90_{a}_1h.json") for a in ("eth", "xrp", "btc")}
# 2026-10-29 ETH 4h = QX4I σ(옵션 IV 보정 4h 변동성) → 두 벽 무접촉 확률(반사 원리) → 2024·25 등조 지도. 박스 HGB 보다 시험 26-06~10 Brier .1597→.1524
#   (ΔBrier −.00732 단측 90% 통과 · 사전등록 docs/prereg/box_hold_vs_qx4i_20261029.md · 결과 docs/experiments/box_hold_vs_qx4i_20261029.md · 사용자 «둘 다 진행»).
#   지도 = scripts/export_box_hold_qx4i_iso_20261029.py. IV 가 없으면 QX4 눈금 · 변동성 예측이 15분 넘게 묵었으면 HGB 그대로.
QX4I_ISO = json.loads(Path(__file__).with_name("box_hold_qx4i_iso_20261029.json").read_text())
VOL_STALE_MS = 15 * 60_000


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt90(asset: str = "eth") -> dict | None:
    try:
        c = json.loads(PLATT90[asset].read_text())
        fresh = pd.Timestamp.now(tz="UTC") - pd.Timestamp(c["created_utc"]) <= pd.Timedelta(days=40)
        ok = np.isfinite([float(c["a"]), float(c["b"])]).all() and str(c["fit_start"]) and str(c["fit_end"])   # 10-26 리뷰: 키가 빠지면 payload 전체(30분·4h 도)가 죽었다
        return c if fresh and ok and c["model_file"] == HGB[asset] else None
    except Exception:  # noqa: BLE001 -- 파일 없음·깨짐·보정 없는 코인 = 보정 없음
        return None


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
        # 10-19 리뷰: 일정의 category fomc 에는 기자회견(결정 30분 뒤)·의사록도 있다 -- 학습 major 는 금리결정(14:00 ET)만
        if (e.get("category") == "fomc" and "금리결정" in str(e.get("title_ko", ""))) or rid in MAJOR_IDS:
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


def features(k1m: pd.DataFrame, events: list | None = None, prof: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """마지막 닫힌 1분봉 마감 시점의 척도별 특징(연구 features() 와 같은 식 -- __main__ 패리티 점검)."""
    PROF = globals()["PROF"] if prof is None else prof   # noqa: N806 -- 코인별 요일×시각 표(아래 식은 그대로)
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
                  # 2026-10-24: prng = 0(직전 창이 평평) 이면 파이썬 float 나눗셈이 ZeroDivisionError 로 payload 전체를 죽였다 -- numpy 나눗셈은
                  #   연구 피쳐(np 배열)와 같은 inf 를 내고, payload 의 isfinite 검사가 그 척도만 ok False 로 뺀다
                  "R_compress": float(np.log(np.divide(rng, prng))), "R_squeeze": float(np.log(rng / np.nanmedian(rngs[-1440:]))),
                  "Q_vol5_vs_W": float(np.log(v[-5:].mean() / v[-W:].mean())),
                  "Q_imbW": float(abs((2 * tb[-W:] - v[-W:]).sum() / v[-W:].sum())),
                  "S_season_fut": float(np.log(np.mean([PROF[how(t_ms + k * 60_000)] for k in range(1, W + 1)]))),
                  "M_major_in": float(nm <= W), "M_major_pos": nm / W if nm <= W else 1.0,
                  "M_major_recent": float(_prev_min(tc, major) <= 60), "M_claims_in": float(_next_min(tc, claims) <= W),
                  "_rng": rng})
        out[name] = f
    return out


def payload(k1m: pd.DataFrame, events: list | None = None, asset: str = "eth") -> dict:
    """k1m: 닫힌 1분봉(오름차순). 마지막 봉 종가 시점의 확률."""
    if k1m is None or len(k1m) < NEED:
        return {"ok": False, "reason": f"1분봉 {0 if k1m is None else len(k1m)}개 < {NEED}"}
    k1m = k1m.tail(NEED + 60).reset_index(drop=True)
    for x in ("high", "low", "close", "volume", "taker_buy_base"):
        k1m[x] = k1m[x].astype(float)
    bb = BUNDLES[asset]
    F = features(k1m, events, np.asarray(bb["prof"], float))
    px = float(k1m["close"].iloc[-1])
    dp = 2 if px >= 10 else 6   # XRP $1.3 -- 2자리면 박스 위·아래가 1.30/1.29
    t0 = pd.Timestamp(k1m["timestamp"].iloc[-1])
    asof = int((t0.tz_localize("UTC") if t0.tzinfo is None else t0).value // 10 ** 6) + 60_000
    out = {"ok": True, "asof_ms": asof, "px": px, "model": bb.get("model", "hgb-20261016"), "asset": asset, "events_n": len(events or []), "scales": {}}
    for name, W in WIN.items():
        f, cf = F[name], bb["scales"][name]
        if not cf.get("deployable", True):
            out["scales"][name] = {"ok": False, "reason": "검정 불통과(보정)"}
            continue
        x = pd.DataFrame([[f[k] for k in cf["cols"]]], columns=cf["cols"])   # 학습 때 열 이름 그대로(이름 없으면 sklearn 경고가 매 호출)
        if not (f["_rng"] > 0 and np.isfinite(x.to_numpy()).all()):
            out["scales"][name] = {"ok": False}
            continue
        rng = f["_rng"]
        p = float(cf["model"].predict_proba(x)[0, 1])
        cal = platt90(asset) if name == "1h" else None
        if cal:
            p = float(1 / (1 + np.exp(-(cal["a"] * logit(p) + cal["b"]))))
        out["scales"][name] = {"ok": True, "p": round(p, 4), "base": round(cf["train_base"], 4),
                               "q20": round(cf["q20"], 4), "q80": round(cf["q80"], 4),   # 화면 «깨질 쪽»(< q20) · «머물 쪽»(> q80) = 학습 예측 분위
                               "up": round(px + 0.5 * rng, dp), "dn": round(px - 0.5 * rng, dp), "half_pct": round(0.5 * rng / px * 100, 3), "half_frac": 0.5 * rng / px,   # half_frac = 반올림 전(QX4I 등조가 쓴다 -- 반올림하면 등조 계단에서 최대 5pp 튐)
                               "until_ms": asof + W * 60_000}
        if cal:   # 화면 q20·q80 은 그대로(연구 채점도 원 분위 · 라벨 바뀜 2.6%)
            out["scales"][name]["calib"] = f"platt90 {cal['fit_start'][:10]}~{cal['fit_end'][:10]}"
    return out


def p_stay(s: float) -> float:
    """구간 [−a, a] 가운데 출발 · 표류 0 브라운 운동이 시간 T 동안 두 벽에 안 닿을 확률, s = σ√T / a (연구 p_stay 와 같은 급수)."""
    if s < 0.02:
        return 1.0
    k = np.arange(400)
    return float(np.clip(4 / np.pi * np.sum((-1.0) ** k / (2 * k + 1) * np.exp(-(2 * k + 1) ** 2 * np.pi ** 2 * s * s / 8)), 0, 1))


def with_qx4i(out: dict, vol: dict | None) -> dict:
    """박스 payload 의 4h 유지 확률을 QX4I 등조로 바꾼 사본(원 HGB 값 = p_hgb). vol = 서버 QX4 결과(log_rv · log_rv_iv · asof_ms) -- 없거나 묵으면 그대로."""
    s4 = (out.get("scales") or {}).get("4h") or {}
    if not (out.get("ok") and s4.get("ok") and vol and vol.get("ok") and abs(out["asof_ms"] - vol.get("asof_ms", 0)) <= VOL_STALE_MS):
        return out
    iv = vol.get("log_rv_iv") is not None
    lq = (vol["log_rv_iv"] + QX4I_ISO["c_qx4i"]) if iv else (vol["log_rv"] + QX4I_ISO["c_qx4"])   # 연구 = 평균 RV 눈금(서빙 표시 σ 는 중앙값형이라 c 를 더한다)
    p = float(np.interp(p_stay(np.sqrt(np.exp(lq)) / s4.get("half_frac", s4["half_pct"] / 100)), QX4I_ISO["x"], QX4I_ISO["y"]))
    return {**out, "scales": {**out["scales"], "4h": {**s4, "p": round(p, 4), "p_hgb": s4["p"], "src": "qx4i" if iv else "qx4"}}}


if __name__ == "__main__":
    # ⑤ QX4I 4h: 급수 경계·단조 · 지도 = 등조 표(내보낼 때 적은 값) · 묵은 변동성·IV 없음
    assert p_stay(0.0) == 1.0 and p_stay(5.0) < 1e-6 and p_stay(0.5) > p_stay(1.0) > p_stay(1.5)
    assert all(abs(np.interp(g, QX4I_ISO["x"], QX4I_ISO["y"]) - v) < 1e-12 for g, v in QX4I_ISO["parity"])
    bx = {"ok": True, "asof_ms": 10**12, "scales": {"4h": {"ok": True, "p": 0.3, "half_pct": 0.9}}}
    v_ = {"ok": True, "asof_ms": 10**12 - 60_000, "log_rv": np.log(0.006 ** 2), "log_rv_iv": np.log(0.007 ** 2)}
    q = with_qx4i(bx, v_)["scales"]["4h"]
    assert q["src"] == "qx4i" and q["p_hgb"] == 0.3 and 0 < q["p"] < 1 and bx["scales"]["4h"]["p"] == 0.3   # 원본은 안 바뀐다
    assert with_qx4i(bx, {**v_, "log_rv_iv": np.log(0.02 ** 2)})["scales"]["4h"]["p"] < q["p"]                # σ 클수록 유지 확률 낮다
    assert with_qx4i(bx, {**v_, "log_rv_iv": None})["scales"]["4h"]["src"] == "qx4"
    assert with_qx4i(bx, {**v_, "asof_ms": 10**12 - 20 * 60_000}) is bx and with_qx4i(bx, None) is bx

    # 자체점검 ① 패리티: 연구 산출(decisions.parquet·features.parquet)의 실제 행 40개 = 이 모듈이 같은 1분봉 1499개로 낸 특징·확률.
    #   (개발 PC 전용 -- 연구 산출이 없으면 건너뛴다) ② 짧은 입력 거부 ③ 발표 일정 해석.
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("imp", root / "scripts/research_eth_box_hold_improve_20261016.py")
    imp = importlib.util.module_from_spec(spec); spec.loader.exec_module(imp)
    maj, cl = event_times([{"time_utc": "2026-10-15T12:30:00+00:00", "category": "econ", "detail": "x (FRED, release_id=10)"},
                           {"time_utc": "2026-10-15T18:00:00+00:00", "category": "fomc", "title_ko": "FOMC 금리결정 발표"},
                           {"time_utc": "2026-10-15T18:30:00+00:00", "category": "fomc", "title_ko": "FOMC 기자회견"},   # 실제 일정 모양 -- 주요 아님
                           {"time_utc": "2026-10-21T18:00:00+00:00", "category": "fomc", "title_ko": "FOMC 의사록"},
                           {"time_utc": "2026-10-16T12:30:00+00:00", "category": "econ", "detail": "release_id=180"},
                           {"time_utc": "2026-10-16T13:15:00+00:00", "category": "econ", "detail": "release_id=13"},   # 산업생산 = 주요 아님
                           {"time_utc": "bad", "category": "fomc", "title_ko": "FOMC 금리결정 발표"}])
    assert len(maj) == 2 and len(cl) == 1, (maj, cl)
    assert payload(pd.DataFrame({"timestamp": [], "high": [], "low": [], "close": [], "volume": [], "taker_buy_base": []}))["ok"] is False
    # ④ 직전 1h 창이 평평(prng = 0)해도 예외 없이 그 척도만 ok False -- R_compress 를 안 쓰는 척도는 그대로 나온다
    n_ = 1560
    c_ = 2500 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 5e-4, n_)))
    c_[-120:-60] = c_[-121]
    h_, l_ = c_ * (1 + 1e-4), c_ * (1 - 1e-4)
    h_[-120:-60] = l_[-120:-60] = c_[-120:-60]
    flat = payload(pd.DataFrame({"timestamp": pd.date_range("2026-10-01", periods=n_, freq="1min"), "high": h_, "low": l_, "close": c_,
                                 "volume": np.ones(n_), "taker_buy_base": np.full(n_, 0.5)}))
    assert flat["ok"] and flat["scales"]["1h"]["ok"] is False, flat["scales"]["1h"]
    assert all(flat["scales"][s]["ok"] for s in WIN if "R_compress" not in B["scales"][s]["cols"]), flat["scales"]
    if imp.DEC.exists() and (imp.OUT / "features.parquet").exists() and (imp.OUT / "events.json").exists():
        D = pd.read_parquet(imp.DEC)
        X = pd.read_parquet(imp.OUT / "features.parquet")
        K = imp.load_1m()
        ev = imp.events()
        evl = [{"time_utc": pd.Timestamp(int(t), unit="ms", tz="UTC").isoformat(), "category": "fomc", "title_ko": "FOMC 금리결정 발표"} for t in ev["major"]] + \
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
