#!/usr/bin/env python3
"""전진 판정 실행 도구 (2026-10-25) — 사전 등록 4건의 «전진 판정» 절을 판정일에 그대로 돌린다.

    python3 scripts/forward_verdicts_20261025.py --status                       # 날짜 셈만(수익·지표 없음)
    python3 scripts/forward_verdicts_20261025.py vol        # 2026-12-23 부터 · 4h 변동성 QX4 vs M3·M0
    python3 scripts/forward_verdicts_20261025.py box        # 2026-12-23 부터 · 박스 유지 30분·1h R2a · 4h R2c(Platt180)·VA180 vs 배포 C0
    python3 scripts/forward_verdicts_20261025.py trend90    # 2027-01-22 부터 · 추세 (b) 1차 점검(90일)
    python3 scripts/forward_verdicts_20261025.py trend180   # 2027-04-22 부터 · 추세 (b) 최종(180일)
    python3 scripts/forward_verdicts_20261025.py rl         # 2027-01-22 부터 · RL v3.1 동결 앙상블(기록용)
    python3 scripts/forward_verdicts_20261025.py vol --dry-run-past 2026-09-20 2026-10-05   # 시험용: 2026-10-24 전 창만, «판정 아님»

🔴 전진 창 = 2026-10-24 00:00 UTC ~ 판정일 전날 23:59. 판정일 전에는 거절하고(오늘 UTC 날짜 < 판정일), 가져오기 상한 LIMIT_MS 가
   창 끝 다음 날 00:00 이라 그 뒤 자료는 요청 자체가 assert 로 막힌다. 판정 전 상한 = 2026-10-24 00:00.
데이터: 2026-10-23 까지 로컬 캐시(10-24 이후 행은 잘라 버림) + 공개 바이낸스 REST GET(418/429 는 Retry-After 만큼 쉼).
판정 파일은 tmp/forward_verdicts_20261025/<후보>_verdict_<시작>_<끝>.json, 한 번만 쓴다(있으면 거절).
ML 크기 2배(2026-12-23)는 자체 `scripts/research_ml_size_stop_20261025.py --forward 2026-12-23` — 여기선 --status 에만 적는다.
실주문 없음.
"""
import argparse, datetime as dt, glob, hashlib, json, sys, time, urllib.error, urllib.parse, urllib.request, zipfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "tmp" / "forward_verdicts_20261025"
FWD = pd.Timestamp("2026-10-24"); FWD_MS = int(FWD.value // 10**6); DAY = pd.Timedelta("1D")
VERDICTS = {   # 후보: (판정일, 설명, 사전 등록 문서)
    "vol": ("2026-12-23", "4h 변동성 QX4 vs M3·M0", "docs/prereg/vol4h_forecast_horserace_20261025.md"),
    "box": ("2026-12-23", "박스 유지 R2a·R2c·VA180 vs C0", "docs/eth_box_hold_calibration_drift_prereg_20261025.md"),
    "trend90": ("2027-01-22", "추세 (b) 현재 규칙+pv 1차(90일)", "docs/prereg/trend_portfolio_lit_20261025.md"),
    "trend180": ("2027-04-22", "추세 (b) 현재 규칙+pv 최종(180일)", "docs/prereg/trend_portfolio_lit_20261025.md"),
    "rl": ("2027-01-22", "RL v3.1 동결 앙상블(기록용)", "docs/prereg/rl_v31_tcn_ensemble_20261025.md"),
    "mlsize": ("2026-12-23", "ML 크기 2배 — 자체 --forward", "docs/prereg/ml_size_fwd_2x_20261025.md"),
    "voliv": ("2027-01-22", "4h 변동성 QX4I(QX4+옵션 IV 보정, 12라운드 계수 고정) vs QX4", "docs/prereg/vol4h_round12_iv_forward_20261025.md"),
    "optsell": ("2027-03-24", "옵션 순 델타 매도 꼬리 z<−1 — 자체 research_deribit_option_flow_direction_20261025.py --forward", "docs/prereg/deribit_option_sell_tail_forward_20261025.md"),
}
QX4_ART, QX4_MD5 = ROOT / "tmp/vol4h_qx4/qx4_2026-10-23.joblib", "2fce43f5e5ece42048ccdd389bff6109"   # 10-25 동결
RL_MD5 = {"162695": "bd1260640a79efcd52a80df752f92804", "946401": "6b182ff9d5f861843ace40e42e431362", "115181": "4e380d6a77171c9ae0e2d0c9af08a73e",
          "684220": "d0c81c426afe90d25cb4a086fcecd164", "577681": "c9e0fe5f0458d624535b40f3925ae345"}   # report_v31.json frozen_md5
TREND_TAU = {"a_best": 0.15, "b_current_pv": 0.20, "c_current": 0.30}   # 4회차 학습 구간 τ(round4_report.json)
FRESH = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
LIMIT_MS = FWD_MS   # 가져오기 상한(배타). window() 만 올린다.


# ---------- 날짜 가드 ----------
def window(cand, today, past=None):
    """(시작, 끝(포함하는 날), 판정 여부). 판정일 전이면 SystemExit. past=(시작, 끝) 는 2026-10-24 전 창만 — 시험용."""
    global LIMIT_MS
    if past:
        s, e = (pd.Timestamp(x) for x in past)
        if not s <= e < FWD:
            raise SystemExit(f"--dry-run-past 는 {FWD.date()} 전 창만 ({s.date()}~{e.date()} 거절)")
        start, end, real = s, e, False
    else:
        v = pd.Timestamp(VERDICTS[cand][0])
        if today < v.date():
            raise SystemExit(f"{cand}: 판정일 {v.date()} 전(오늘 {today}) — 전진 자료를 보지 않는다. --status 만")
        start, end, real = FWD, v - DAY, True
    LIMIT_MS = int((end + DAY).value // 10**6)
    return start, end, real


def status(today):
    print(f"오늘 {today} UTC · 전진 시작 {FWD.date()} (날짜 셈만 — 수익·지표 없음)")
    print(f"{'후보':10}{'판정일':>12}{'경과일':>8}{'필요일':>8}  준비  설명")
    for k, (d, desc, _) in VERDICTS.items():
        dd = pd.Timestamp(d).date()
        print(f"{k:10}{d:>12}{max(0, (today - FWD.date()).days):>8}{(dd - FWD.date()).days:>8}  {'yes' if today >= dd else 'no ':4}  {desc}")


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def check_md5(p, want):
    got = md5(p)
    if got != want:
        raise SystemExit(f"🔴 동결 파일 md5 불일치 {p}: {got} ≠ {want} — 판정 중단")
    return got


# ---------- 공개 REST (상한 assert) ----------
def get(path, **q):
    url = "https://fapi.binance.com" + path + "?" + urllib.parse.urlencode(q)
    for i in range(10):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):   # 차단·한도: 서버가 말한 만큼 쉰다
                time.sleep(int(e.headers.get("Retry-After") or 60)); continue
            if i >= 4:
                raise
            time.sleep(2 * (i + 1))
        except (urllib.error.URLError, TimeoutError):
            if i >= 4:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def klines(sym, interval, t0, t1):
    """완결봉 원 행 [t,o,h,l,c,v,ct,qv,n,tbv,…], t0 ≤ t < t1."""
    assert t1 <= LIMIT_MS, f"가져오기 상한 {pd.Timestamp(LIMIT_MS, unit='ms')} 초과({pd.Timestamp(t1, unit='ms')}) — 전진 자료 보호"
    step, out, t = {"1m": 60_000, "5m": 300_000}[interval], [], t0
    while t < t1:
        part = get("/fapi/v1/klines", symbol=sym, interval=interval, startTime=t, endTime=t1 - 1, limit=1500)
        if not part:
            break
        out += part; t = part[-1][0] + step; time.sleep(0.3)
    now = time.time() * 1000
    return [r for r in out if r[0] < t1 and r[6] < now]


def funding(sym, t0, t1):
    assert t1 <= LIMIT_MS, "가져오기 상한 초과 — 전진 자료 보호"
    out, t = [], t0
    while t < t1:
        part = get("/fapi/v1/fundingRate", symbol=sym, startTime=t, endTime=t1 - 1, limit=1000)
        if not part:
            break
        out += part; t = part[-1]["fundingTime"] + 1; time.sleep(0.3)
    return pd.DataFrame([(r["fundingTime"], float(r["fundingRate"])) for r in out if r["fundingTime"] < t1], columns=["t", "rate"])


def extend_sys(S, end):
    """research_system_trend_ruleb_20261025 모듈 S 의 bars5·load_all 이 end 23:55 까지 보게: 캐시(10-24 이후 행 버림) + REST 이어 붙임."""
    k0, f0 = S.rest_klines, S.rest_funding; e1 = int((end + DAY).value // 10**6); cut = min(FWD_MS, e1)
    t22 = int(pd.Timestamp("2022-01-01").value // 10**6)

    def rk(sym, t0, t1):
        x = k0(sym, t0, FWD_MS); x = x[x.t < cut]
        if e1 > x.t.max() + 300_000:
            r = klines(f"{sym}USDT", "5m", int(x.t.max()) + 300_000, e1)
            x = pd.concat([x, pd.DataFrame([(a[0], float(a[2]), float(a[3]), float(a[4]), a[6]) for a in r], columns=x.columns)])
        return x.reset_index(drop=True)

    def rf(sym, t0, t1):
        x = f0(sym, t22, FWD_MS); x = x[x.t < cut]
        return pd.concat([x, funding(f"{sym}USDT", cut, e1)], ignore_index=True) if e1 > cut else x.reset_index(drop=True)
    S.rest_klines, S.rest_funding, S.END = rk, rf, str(end.date())


# ---------- 1. 4h 변동성 ----------
def vol_verdict(start, end, rows=None, extra=None):
    """docs/prereg/vol4h_forecast_horserace_20261025.md «전진 판정»:
    > - 데이터: 2026-10-24 00:00 UTC 이후 정시 결정(라벨 끝 ≤ 판정일 전날) · 1차 판정 2026-12-23(약 60일 · 코인당 ~1,400 결정).
    > - 관문: ETH 에서 QG − M3, QG − M0 QLIKE CI 상한 < 0 · 복제 ≥ 2/3 · lvl 기울기 CI 가 1 을 포함.
    + 5라운드 «배포 후보는 더 단순한 QX4 … 2026-12-23 전진 판정은 그대로 진행» · 7라운드 «전진 판정 2026-12-23 은 QX4 로만».
    해석(가장 문자 그대로): QG → QX4 · QX4 = 동결 서빙 아티팩트(qx4_2026-10-23, 재학습 없음, md5 기록) · M3 = 그 아티팩트의 M3 구성원
    (배포 레시피를 같은 학습 구간 2024-05-23~2026-05-23 으로 적합) · M0 HAR = 같은 학습 구간 OLS · 눈금 c = 아티팩트 눈금 구간
    (2026-05-23~10-23) 에서 모델마다. CI = 일 평균 손실 차 5일 블록 4,000회(1~5라운드와 같음). lvl 기울기는 ETH QX4 로.
    피쳐 = 서빙 경로(frames_from_klines, 공개 REST 5분 klines 4코인)."""
    import live_vol4h_qx4_20261025 as L
    R = L.R
    rep = {"artifact": str(QX4_ART), "artifact_md5": check_md5(QX4_ART, QX4_MD5)}
    art = L.load_qx4(QX4_ART); assert art is not None, "아티팩트 계약 불일치"
    cend = pd.Timestamp(art["calib_end"]) + pd.Timedelta("23:55:00"); ca0 = cend - pd.DateOffset(months=5); tr0 = pd.Timestamp(art["train_start"])
    assert tr0 == ca0 - pd.DateOffset(years=2) and str(ca0.date()) == art["train_end"], "아티팩트 학습 구간 해석 불일치"
    fr, oks = L.load_frames(); M = L.masks(oks, fr, tr0, ca0, ca0, cend); B0, C = {}, {}   # 로컬 연구 프레임(≤ 2026-10-23)
    for s in L.COINS:
        F, y = fr[s]; tr, ca = M[s]["tr"], M[s]["ca"]; a = art["coins"][s]
        B0[s] = np.linalg.lstsq(R.design(F[tr], "M0"), y[tr], rcond=None)[0]
        p = {"QX4": L.predict_log(art, s, F[ca]), "M3": L._pred(a["M3"], F[ca]), "M0": R.design(F[ca], "M0") @ B0[s]}
        if extra:                                                   # voliv 같은 추가 모형(기존 관문은 그대로)
            p.update(extra(s, F[ca].index, p["QX4"]))
        C[s] = {m: float(np.log(np.mean(np.exp(y[ca] - v)))) for m, v in p.items()}
        assert abs(C[s]["QX4"] - a["c"]) < 1e-6, (s, C[s]["QX4"], a["c"])   # 같은 자료·같은 아티팩트 확인
    del fr, oks
    t1 = int((end + DAY).value // 10**6)
    if rows is None:
        t0 = int((start - pd.Timedelta(days=75)).value // 10**6)   # 예열 30일 + lvl 기준 720 정시 + 여유
        rows = {s: klines(L.SYMS[s], "5m", t0, t1) for s in L.COINS}
    Fs, xs, filled = L.frames_from_klines(rows, t1); rep["filled_bars_60d"] = filled; out = {}
    for s in L.COINS:
        x = xs[s].copy(); x["filled"] = ((x.h == x.l) & (x.h == x.c) & (x.c == x.c.shift()) & (x.n == 0)).astype(float)
        y, nf = R.target(x); F, y, nf = Fs[s].iloc[L.WARM:], y[L.WARM:], nf[L.WARM:]; ts = F.index
        hr = (ts.minute == 55) & np.isfinite(F[R.SETS["M4"] + R.MKT].to_numpy(float)).all(1)
        fw = hr & (ts >= start - pd.Timedelta("5min")) & (ts + pd.Timedelta(minutes=5 * R.H) <= end + pd.Timedelta("23:55:00")) & np.isfinite(y) & (nf < 5)
        Fh, a = F[hr], art["coins"][s]
        p = {"QX4": L.predict_log(art, s, Fh), "M3": L._pred(a["M3"], Fh), "M0": R.design(Fh, "M0") @ B0[s]}
        if extra:
            p.update(extra(s, Fh.index, p["QX4"]))
        k = fw[hr]; yf, tf = y[fw], ts[fw]; Lq = {m: R.qlike(yf, v[k] + C[s][m]) for m, v in p.items()}
        sig = pd.Series(np.exp((p["QX4"] + C[s]["QX4"]) / 2), index=ts[hr]); ref = sig.rolling(720, min_periods=600).median().values[k]
        assert np.isfinite(ref).all(), "lvl 기준(직전 720 정시) 부족 — klines 를 더 길게"
        out[s] = {"n": int(k.sum()), "days": int(len(np.unique(tf.normalize()))), "c": C[s], "qlike": {m: float(v.mean()) for m, v in Lq.items()},
                  "dq_QX4_vs_M3": R.day_ci(tf, Lq["QX4"] - Lq["M3"]), "dq_QX4_vs_M0": R.day_ci(tf, Lq["QX4"] - Lq["M0"]),
                  "lvl_mz_QX4": R.mz_slope(tf, np.log(np.exp(yf / 2) / ref), np.log(sig.values[k] / ref))}
        out[s].update({f"dq_{m}_vs_QX4": R.day_ci(tf, Lq[m] - Lq["QX4"]) for m in p if m not in ("QX4", "M3", "M0")})
        print(s, json.dumps(out[s], ensure_ascii=False), flush=True)
    e, rep["coins"] = out["ETH"], out
    rep["gates"] = {"V1_ETH_QX4_vs_M3_ci_hi_lt0": e["dq_QX4_vs_M3"][2] < 0, "V2_ETH_QX4_vs_M0_ci_hi_lt0": e["dq_QX4_vs_M0"][2] < 0,
                    "V3_replication_ge2": sum(out[s]["dq_QX4_vs_M3"][2] < 0 for s in ("XRP", "BTC", "SOL")) >= 2,
                    "V4_ETH_lvl_slope_ci_has_1": e["lvl_mz_QX4"][1] <= 1 <= e["lvl_mz_QX4"][2]}
    return rep


# ---------- 2. 박스 유지 확률 ----------
def eth_1m(B, end):
    """ETHUSDT 1분봉: 로컬(binance_vision 월·일 파일 + vision zip, 10-24 전만) + 공개 REST 1분으로 end 23:59 까지."""
    parts = [pd.read_parquet(f)[["t", "h", "l", "c", "v", "tb"]] for f in sorted(glob.glob(f"{B.BV}/klines1m/ETHUSDT-1m-*.parquet")) +
             sorted(glob.glob(f"{B.BV}/klines1m_daily/ETHUSDT-1m-*.parquet"))]
    for z in sorted(glob.glob(str(B.VIS / "*.zip"))):
        with zipfile.ZipFile(z) as zf:
            x = pd.read_csv(zf.open(zf.namelist()[0]))
        parts.append(pd.DataFrame({"t": x.open_time, "h": x.high, "l": x.low, "c": x.close, "v": x.volume, "tb": x.taker_buy_volume}))
    e1 = int((end + DAY).value // 10**6)
    K = pd.concat(parts).drop_duplicates("t").sort_values("t"); K = K[K.t < min(FWD_MS, e1)].reset_index(drop=True)
    gap = np.flatnonzero(np.diff(K.t) != 60_000); K = K.iloc[:gap[0] + 1] if len(gap) else K   # 로컬 첫 빈칸부터는 REST 로
    if K.t.iloc[-1] + 60_000 < e1:
        r = klines("ETHUSDT", "1m", int(K.t.iloc[-1]) + 60_000, e1)
        K = pd.concat([K, pd.DataFrame({"t": [a[0] for a in r], **{k: [float(a[j]) for a in r] for k, j in (("h", 2), ("l", 3), ("c", 4), ("v", 5), ("tb", 9))}})], ignore_index=True)
    assert (np.diff(K.t) == 60_000).all() and K.t.iloc[-1] == e1 - 60_000, "1분봉 빈칸 또는 끝 불일치"
    return K


def box_verdict(start, end):
    """docs/eth_box_hold_calibration_drift_prereg_20261025.md «전진 판정»:
    > - 대상(배포 후보): 30분·1h = R2a(배포 칸·하이퍼파라미터 HGB, 2022-01~2025-12 학습, 시드 0) · 4h = R2c(27칸 HGB ≤2024 학습 + 굴림 Platt 180일,
    >   2025-01-01 부터 표본 밖 예측으로 매월 1일 인과적 재적합). 기준선 = 배포 C0.
    > - 데이터: 2026-10-24 00:00 UTC 이후 결정(라벨 창 끝 ≤ 판정일 전날) · 1차 판정 2026-12-23(약 60일).
    > - 관문: 1~3라운드의 W1~W4 그대로(ΔBrier 일 블록 CI 상한 < 0 · 로그손실 ≤ · ECE ≤ · AUC ≥ C0 − .002), 척도별 · 4h 치우침 전후 보고.
    + 7라운드 «전진 판정(2026-12-23)에서 R2c(27칸+Platt180)와 함께 VA180 을 4h 후보로 추가 채점한다».
    해석: 기준선은 문서대로 배포 C0(= 6칸 배포 레시피 2022~24 학습 — 5~7라운드의 A 레시피를 R2c 와 같은 학습 구간에 둔 것).
    VA180 = R2c 와 같은 27칸 HGB(≤2024) + 굴림 벤–에이버스 180일(같은 t0·같은 인과 규칙). 4h 후보 2개에 본페로니는 문서에 없어 안 넣는다.
    행·라벨·피쳐 = 연구 build() 그대로(관문 L1~L4·배포 행 수 재현 포함), 1분봉만 end 까지 늘림."""
    import research_eth_box_hold_calibration_drift_20261025 as B
    from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
    ev = ROOT / "tmp/box_improve_20261016/events.json"
    assert ev.exists(), f"{ev} 없음 — 연구 피쳐의 발표 일정 캐시(FRED, research_eth_box_hold_improve_20261016.events) 필요"
    assert max(json.loads(ev.read_text())["major"]) > int((end + DAY).value // 10**6), "발표 일정 캐시가 창 끝을 못 덮음 — events() 다시"
    K = eth_1m(B, end); fdir = OUT / f"box_rows_{start.date()}_{end.date()}"
    B.OUT, B.load_1m = fdir, (lambda: K)
    B.build(); A = pd.read_parquet(fdir / "rows.parquet"); rep = {"rows": str(fdir / "rows.parquet")}
    for sc, W in B.WIN.items():
        cf = B.BH["scales"][sc]; g = A[A.scale == sc]; need = list(dict.fromkeys(cf["cols"] + B.LEAN)); w1 = pd.Timedelta(minutes=W + 1)
        e0, s0 = end + DAY, start - pd.Timedelta("1min")
        if sc == "4h":
            tr = g[g.ts < "2025-01-01"].dropna(subset=B.LEAN)
            g25 = g[(g.ts >= "2025-01-01") & (g.ts + w1 <= e0)].dropna(subset=need).reset_index(drop=True)
            t25, y25 = g25.ts.to_numpy(), g25.y.to_numpy().astype(float); q = HGBC(**B.HGB).fit(tr[B.LEAN], tr.y).predict_proba(g25[B.LEAN])[:, 1]
            m = t25 >= np.datetime64(s0); te = g25[m]
            P = {"R2c 27칸+굴림Platt180": B.rolling("platt", t25, q, y25, W, 180)[m], "VA180 27칸+굴림벤에이버스180": B.rolling("va", t25, q, y25, W, 180)[m]}
        else:
            tr = g[g.ts + w1 <= pd.Timestamp("2026-01-01")].dropna(subset=B.LEAN)
            te = g[(g.ts >= s0) & (g.ts + w1 <= e0)].dropna(subset=need)
            P = {"R2a 재학습~2025": HGBC(**B.HGB).fit(tr[cf["cols"]], tr.y).predict_proba(te[cf["cols"]])[:, 1]}
        y, d = te.y.to_numpy().astype(float), pd.DatetimeIndex(te.ts).normalize().to_numpy(); p0 = cf["model"].predict_proba(te[cf["cols"]])[:, 1]
        base = B.metrics(y, p0, cf["q20"], cf["q80"]); base["bias_ci_pp"] = [v * 100 for v in B.day_boot(p0 - y, d)]
        r = {"n": int(len(y)), "days": int(len(np.unique(d))), "hold_rate": float(y.mean()), "C0": base, "cands": {}}
        for k, p in P.items():
            mt = B.metrics(y, p, cf["q20"], cf["q80"]); mt["dbrier"] = mt["brier"] - base["brier"]
            mt["dbrier_ci"] = B.day_boot((p - y) ** 2 - (p0 - y) ** 2, d); mt["dauc"] = mt["auc"] - base["auc"]; mt["dauc_ci"] = B.auc_boot(y, p, p0, d)
            mt["bias_ci_pp"] = [v * 100 for v in B.day_boot(p - y, d)]
            mt["gates"] = {"W1": mt["dbrier_ci"][1] < 0, "W2": mt["logloss"] <= base["logloss"], "W3": mt["ece_pp"] <= base["ece_pp"], "W4": mt["auc"] >= base["auc"] - 0.002}
            mt["win"] = all(mt["gates"].values()); r["cands"][k] = mt
        print(sc, json.dumps({k: (v["dbrier"], v["dbrier_ci"], v["gates"]) for k, v in r["cands"].items()}, ensure_ascii=False, default=float), flush=True)
        rep[sc] = r
    rep["gates"] = {f"{sc} {k}": v["win"] for sc in B.WIN for k, v in rep[sc]["cands"].items()}   # 척도·후보별 W1~W4 전부(교체 제안은 통과한 척도만)
    return rep


# ---------- 3. 추세 포트 ----------
def trend_verdict(start, end):
    """docs/prereg/trend_portfolio_lit_20261025.md «전진 판정»:
    > - 대상: (b) 현재 규칙 + pv(τ = 4회차 학습 구간 값 그대로) · 보고용으로 (a)·(c)·보유도 같은 창에서.
    > - 데이터: 2026-10-24 이후 일봉 · 1차 점검 2027-01-22(90일, 건수만이 아니라 관문 판정 1회) · 최종 2027-04-22(180일).
    > - 관문: T1 샤프 > 같은 창 4코인 보유 · T2 MDD ≥ −20% · T3 무작위 부호 대조 200시드 95분위 초과 — 180일 최종 판정이 주(90일은 조기 경보용).
    > - 판정 코드는 이 스크립트의 holdout4 를 창만 바꿔 쓴다(5분봉·펀딩은 판정일에 REST 로 2026-10-24~판정일 전날까지).
    holdout4 와 같은 계산(run · null_targets 200시드 · τ 고정), 창만 [시작, 끝]."""
    import research_trend_portfolio_lit_20261025 as T
    r4 = T.OUT / "round4_report.json"
    if r4.exists():
        assert {k: v["tau"] for k, v in json.loads(r4.read_text())["configs"].items()} == TREND_TAU, "τ 가 4회차 보고와 다름"
    extend_sys(T.SYS, end); F = {s: T.daily_frame(s)[0] for s in T.COINS}
    te = pd.date_range(start, end, freq="D"); assert all(F[s].index[-1] >= end for s in T.COINS)
    h = T.hold_ret(F, te); out = {"days": len(te), "hold_sharpe": T.sharpe(h), "hold_ret": float((1 + h).prod() - 1), "hold_mdd": T.OE.mdd(h), "configs": {}}
    for k, tau in TREND_TAU.items():
        Wt = T.r4_weights(F, k, tau); x = T.run(F, Wt, te, band=T.band4(k))
        nulls = [T.sharpe(T.run(F, T.null_targets(F, None, None, te, sd, W=Wt), te, band=T.band4(k))) for sd in range(200)]
        out["configs"][k] = {"tau": tau, "sharpe": T.sharpe(x), "ret": float((1 + x).prod() - 1), "mdd": T.OE.mdd(x),
                             "null_p95": float(np.percentile(nulls, 95)), "null_median": float(np.median(nulls))}
        print(k, out["configs"][k], flush=True)
    c = out["configs"]["b_current_pv"]
    out["gates"] = {"T1_gt_hold": c["sharpe"] > out["hold_sharpe"], "T2_mdd": c["mdd"] >= -0.20, "T3_null_p95": c["sharpe"] > c["null_p95"]}
    return {**out, **FRESH}


# ---------- 4. RL v3.1 ----------
def rl_verdict(start, end):
    """docs/prereg/rl_v31_tcn_ensemble_20261025.md «전진 판정»:
    > - 대상: 위 동결 앙상블 그대로(재학습 없음) · 환경·가드·체결·펀딩 동일 · 피쳐 정규화 평균·표준편차는 학습 구간 값 그대로.
    > - 데이터: 2026-10-24 00:00 UTC 이후 · 최소 90일 → 1차 판정 2027-01-22(그 전엔 건수만 본다).
    > - 관문: 같은 G1~G4(G3 는 앙상블의 2026-01~05 검증 행동열 전이로 만든 무작위 400 시드, 판정 창 길이 기준 95분위 — 후보 1개라 Bonferroni 없음).
    > - 판정 코드는 판정일에 이 절을 그대로 구현한다(5분봉 REST 로 2026-10-24~판정일 전날까지 · 동결 md5 확인부터).
    G1 샤프 > 0 · MDD ≥ −20% · G2 앙상블 − 교사 일 차 20일 블록 CI 하한 > 0 · G4 개별 5시드 중 샤프 > 0 이 4개 이상.
    rl_v31_rolling_origin_20261025.md 결론대로 «기록용»(RL 파이프라인은 닫힘)."""
    import research_rl_v3_reward_arch_20261025 as V3
    V2, S = V3.V2, V3.SYS
    frozen = json.loads((V3.OUT / "report_v31.json").read_text())["frozen_md5"]
    assert frozen == RL_MD5, "report_v31.json 동결 md5 가 등록값과 다름"
    rep = {"frozen_md5": {sd: check_md5(V3.OUT / f"c11_s{sd}" / "250000.zip", h) for sd, h in RL_MD5.items()}, "record_only": True}
    from stable_baselines3 import PPO
    extend_sys(S, end); Wd = V3.World(); va, fw = Wd.span(V3.VAL), Wd.span((str(start.date()), str(end.date())))
    ms = [PPO.load(V3.OUT / f"c11_s{sd}" / "250000.zip", device="cpu") for sd in RL_MD5]; pk = V3.ens_picker(Wd, ms, "tcn")
    _, _, lg_va = V2.run(Wd, va, pk); r_e, rc, lg_e = V2.run(Wd, fw, pk); r_t, _, lg_t = V2.run(Wd, fw, V2.teacher(Wd))
    days = Wd.days[fw[0]:fw[1]]; hold = pd.concat([S.daily_close(Wd.data[s]).pct_change() for s in V2.COINS], axis=1).mean(1).loc[days]
    nulls = [S.sharpe(V2.run(Wd, fw, V2.markov(lg_va, sd))[0]) for sd in range(400)]
    seeds = {sd: S.sharpe(V2.run(Wd, fw, V3.picker(Wd, m, "tcn"))[0]) for sd, m in zip(RL_MD5, ms)}; diff = r_e - r_t
    rep.update(days=len(days), ensemble=V2.summarize(r_e, lg_e), teacher=V2.summarize(r_t, lg_t), per_coin={s: S.sharpe(v) for s, v in rc.items()},
               hold_1x={"sharpe": S.sharpe(hold), "ret": float((1 + hold).prod() - 1), "mdd": S.mdd(hold)}, seed_sharpe=seeds,
               ens_minus_teacher_ci=S.block_ci(diff), ens_minus_teacher_mean_bp=float(diff.mean() * 1e4),
               random_p95=float(np.percentile(nulls, 95)), random_median=float(np.median(nulls)))
    E = rep["ensemble"]
    rep["gates"] = {"G1": E["sharpe"] > 0 and E["mdd"] >= -0.20, "G2": rep["ens_minus_teacher_ci"][0] > 0, "G3": E["sharpe"] > rep["random_p95"],
                    "G4": sum(v > 0 for v in seeds.values()) >= 4}
    return {**rep, **FRESH}


def voliv_verdict(start, end):
    """docs/prereg/vol4h_round12_iv_forward_20261025.md — 동결 QX4 아티팩트 + 12라운드 계수 고정 IV 보정(ETH) vs QX4."""
    import research_vol4h_round12_option_iv_20261025 as R12, research_deribit_option_flow_direction_20261025 as O
    for d in pd.date_range(FWD, end, freq="D"):
        O.fetch_day(d)                                              # history.deribit.com · 창 끝 날까지만
    iv = R12.iv_hourly(int((end + DAY).value // 10**6))
    rep = vol_verdict(start, end, extra=lambda s, ts, q: {"QX4I": R12.qx4i(ts, q, iv)} if s == "ETH" else {})
    rep["gates"] = {"I1_ETH_QX4I_vs_QX4_ci_hi_lt0": rep["coins"]["ETH"]["dq_QX4I_vs_QX4"][2] < 0}
    return rep


FN = {"vol": vol_verdict, "box": box_verdict, "trend90": trend_verdict, "trend180": trend_verdict, "rl": rl_verdict, "voliv": voliv_verdict}


def main(argv=None, today=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cand", nargs="?", choices=list(FN)); ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run-past", nargs=2, metavar=("START", "END"))
    a = ap.parse_args(argv); today = today or dt.datetime.now(dt.timezone.utc).date()
    if a.status or not a.cand:
        return status(today)
    start, end, real = window(a.cand, today, a.dry_run_past)
    OUT.mkdir(parents=True, exist_ok=True); f = OUT / f"{a.cand}_{'verdict' if real else 'dryrun'}_{start.date()}_{end.date()}.json"
    if real and f.exists():
        raise SystemExit(f"{f} 이미 있음 — 전진 판정은 1회만")
    print(("🔴 판정" if real else "DRY RUN — 판정 아님(과거 창 코드 점검)") + f" {a.cand} {start.date()}~{end.date()}", flush=True)
    rep = FN[a.cand](start, end)
    g = rep.get("gates", {})
    rep = {"candidate": a.cand, "is_verdict": real, "label": "전진 판정" if real else "DRY RUN — 판정 아님", "prereg": VERDICTS[a.cand][2],
           "window": [str(start.date()), str(end.date())], "data_until": str((end + DAY).date()) + " 00:00 UTC 배타", "run_utc": str(pd.Timestamp.now(tz="UTC")),
           **rep, "pass": bool(g) and all(g.values())}
    f.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: rep[k] for k in ("candidate", "label", "window", "gates", "pass") if k in rep}, ensure_ascii=False, default=float), "\n→", f)
    return rep


if __name__ == "__main__":
    main()
