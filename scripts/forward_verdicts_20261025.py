#!/usr/bin/env python3
"""전진 판정 실행 도구 (2026-10-25) — 사전 등록 4건의 «전진 판정» 절을 판정일에 그대로 돌린다.

    python3 scripts/forward_verdicts_20261025.py --status                       # 날짜 셈만(수익·지표 없음)
    python3 scripts/forward_verdicts_20261025.py vol        # 2026-12-23 부터 · 4h 변동성 QX4 vs M3·M0
    python3 scripts/forward_verdicts_20261025.py box        # 2026-12-23 부터 · 박스 유지 30분·1h R2a · 4h R2c(Platt180)·VA180 vs 배포 C0
    python3 scripts/forward_verdicts_20261025.py trend90    # 2027-01-22 부터 · 추세 (b) 1차 점검(90일)
    python3 scripts/forward_verdicts_20261025.py trend180   # 2027-04-22 부터 · 추세 (b) 최종(180일)
    python3 scripts/forward_verdicts_20261025.py rl         # 2027-01-22 부터 · RL v3.1 동결 앙상블(기록용)
    python3 scripts/forward_verdicts_20261025.py tsmom      # 2027-04-24 부터 · ETH 변동성 크기 30일 TSMOM(창 2026-10-26~)
    python3 scripts/forward_verdicts_20261025.py rlv19 --dir4h <서버에서 받은 eth_dir4h_model.joblib|parquet>   # 2027-04-24 부터 · RL v19 앙상블(창 2026-10-26~)
    python3 scripts/forward_verdicts_20261025.py xsec       # 2027-04-24 부터 · 20코인 4h 상대 수익 DL(R3 K1 5시드·EWMA3·띠) vs 리지(창 2026-10-26~)
    python3 scripts/forward_verdicts_20261025.py vol --dry-run-past 2026-09-20 2026-10-05   # 시험용: 2026-10-24 전 창만, «판정 아님»

🔴 전진 창 = 2026-10-24 00:00 UTC ~ 판정일 전날 23:59. 판정일 전에는 거절하고(오늘 UTC 날짜 < 판정일), 가져오기 상한 LIMIT_MS 가
   창 끝 다음 날 00:00 이라 그 뒤 자료는 요청 자체가 assert 로 막힌다. 판정 전 상한 = 2026-10-24 00:00.
데이터: 2026-10-23 까지 로컬 캐시(10-24 이후 행은 잘라 버림) + 공개 바이낸스 REST GET(418/429 는 Retry-After 만큼 쉼).
판정 파일은 tmp/forward_verdicts_20261025/<후보>_verdict_<시작>_<끝>.json, 한 번만 쓴다(있으면 거절).
ML 크기 2배(2026-12-23)는 자체 `scripts/research_ml_size_stop_20261025.py --forward 2026-12-23` — 여기선 --status 에만 적는다.
실주문 없음.
"""
import argparse, datetime as dt, glob, hashlib, io, json, sys, time, urllib.error, urllib.parse, urllib.request, zipfile
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
    "tsmom": ("2027-04-24", "ETH 변동성 크기 30일 TSMOM vs 항상 롱 1배", "docs/prereg/tsmom_volsized_forward_20261026.md"),
    "rlv19": ("2027-04-24", "RL v19 10시드 앙상블 vs T3 TSMOM", "docs/prereg/rl_v19_forward_20261026.md"),
    "xsec": ("2027-04-24", "20코인 4h 상대 수익 DL(R3 K1·EWMA3·띠) vs 리지 B3", "docs/prereg/xsec_dl_forward_20261026.md"),
}
START = {"tsmom": pd.Timestamp("2026-10-26"), "rlv19": pd.Timestamp("2026-10-26"), "xsec": pd.Timestamp("2026-10-26")}   # 10-24·25 가격은 급락 분석에서 봄 → 창을 10-26 부터(문서)
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
        start, end, real = START.get(cand, FWD), v - DAY, True
    LIMIT_MS = int((end + DAY).value // 10**6)
    return start, end, real


def status(today):
    print(f"오늘 {today} UTC · 전진 시작 {FWD.date()} (날짜 셈만 — 수익·지표 없음)")
    print(f"{'후보':10}{'판정일':>12}{'경과일':>8}{'필요일':>8}  준비  설명")
    for k, (d, desc, _) in VERDICTS.items():
        dd, s0 = pd.Timestamp(d).date(), START.get(k, FWD).date()
        print(f"{k:10}{d:>12}{max(0, (today - s0).days):>8}{(dd - s0).days:>8}  {'yes' if today >= dd else 'no ':4}  {desc}")


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


# ---------- 5·6. TSMOM · RL v19 (2026-10-26 등록, 창 10-26 부터) ----------
TSMOM_SSTAR = 0.00613139826386366   # tsmom 문서 σ*(2024~25 중앙값, frame.parquet)
RL19_NORM = {"sstar": 0.006131411595098756, "pred_std": 50.12054234740778, "fund_std": 9.59100922819097e-05}   # R3 학습 2024~25: research load(frame_v18) 값
RL19_MD5 = {"1973391873": "74f7a5dc2302da8c68b9a77b889fb252", "1051569749": "1fce18f37f51914d27ef55a61f40e46b", "679622530": "2115f7fd7968db6abca372c1444632c9",
            "1296968475": "11616ef811571897ac720f059cc0ab06", "1703445932": "c291523b32d640c5fc6c2b3d0f76e980", "1181398192": "69c89a3469265bf58e042e26b91b94b6",
            "29407306": "c620408186b011669bd037b2cb8f61fc", "1898130994": "624d645e02e32e53c44d454d75cdc84c", "455516995": "583691b9e41f3b817f25b5d43e390902",
            "937460950": "403b1707f41dbc06476e2d3cfcac992f"}   # tmp/rl_v14_20261026/v19_md5.txt · 커밋 52d698b2
DIR4H = ROOT / "data/live/eth_dir4h_oos_seed.parquet"   # 기본 = 09-14 까지 seed(dry-run 용). 판정 땐 --dir4h 로 서버 joblib(art["oos"])


def k5(sym, t0, t1):
    r = klines(f"{sym}USDT", "5m", int(t0.value // 10**6), int(t1.value // 10**6))
    x = pd.DataFrame([(a[4], a[5], a[9]) for a in r], index=pd.to_datetime([a[0] for a in r], unit="ms"), columns=["close", "volume", "taker_buy_base"]).astype(float)
    return x[~x.index.duplicated()].sort_index()


def rl14_frame(start, end, dir4h=None):
    """research_rl_v14_vol_sizing_20261026.build(extra=dir4h 있음) 와 같은 식 · 결정 정시 t ∈ [start, end 23:00].
    원천만 전진용: ETH·BTC 5분 = 공개 REST klines(테이커 매수 = 열 9) · OI·상위 롱숏 = data.binance.vision daily metrics
    (연구 패널 = 같은 파일을 봉 시작 = create_time 으로 붙인 것) · 펀딩 = REST · 방향 예측 = dir4h(ts, pred) 또는 라이브 joblib 의 art["oos"]."""
    t0, t1 = start - pd.Timedelta(days=40), end + DAY                  # 예열 30일(m30d) + 7일(ls7d·oiz·lsz) + 여유
    g = pd.date_range(t0, t1 - pd.Timedelta("5min"), freq="5min"); e = k5("ETH", t0, t1)
    assert e.index[0] == t0 and e.index[-1] == g[-1], f"ETH 5분봉 범위 {e.index[0]}~{e.index[-1]} ≠ {t0}~{g[-1]}"
    p = e.close.reindex(g).ffill(); r2 = np.log(p).diff() ** 2
    t = pd.date_range(start, end + pd.Timedelta("23:00:00"), freq="1h"); te = t - pd.Timedelta("5min")
    sig = lambda n: np.sqrt(r2.rolling(n).sum() / (n / 12)).reindex(te).to_numpy()   # noqa: E731
    c = p.reindex(te).to_numpy(); s24 = sig(288); F = pd.DataFrame(index=t); info = {"eth_5m_missing": int(len(g) - len(e))}
    for k, n in (("ls1h", 12), ("ls4h", 48), ("ls24h", 288), ("ls7d", 2016)):
        F[k] = sig(n)
    for k, h in (("m1d", 24), ("m7d", 168), ("m30d", 720), ("m4h", 4)):
        F[k] = np.log(c / p.reindex(te - pd.Timedelta(hours=h)).to_numpy()) / (s24 * np.sqrt(h))
    F["sig"], F["ret"] = s24, np.log(p.reindex(te + pd.Timedelta("1h")).to_numpy() / c)
    if dir4h is None:
        return F, info
    import build_binance_vision_panel_20260915 as P
    d = (__import__("joblib").load(dir4h)["oos"] if str(dir4h).endswith(".joblib") else pd.read_parquet(dir4h)).set_index("ts").pred.astype(float)
    F["pred"] = d.reindex(te).to_numpy()
    ms = []
    for dd in pd.date_range(t0, end, freq="D"):
        assert int((dd + DAY).value // 10**6) <= LIMIT_MS, "metrics 가져오기 상한 초과 — 전진 자료 보호"
        x = P.fetch(f"{P.BASE}/metrics/ETHUSDT/ETHUSDT-metrics-{dd.date()}.zip")   # 없거나 실패면 None(연구 패널과 같은 규칙) → 앞채움
        ms += [] if x is None else [pd.read_csv(io.StringIO(x))]
    m = pd.concat(ms); m.index = pd.to_datetime(m.create_time); m = m[~m.index.duplicated()]
    info["metrics_days_missing"] = int(len(pd.date_range(t0, end, freq="D")) - len(ms))
    q = pd.concat([e[["taker_buy_base", "volume"]], m[["sum_open_interest", "sum_toptrader_long_short_ratio"]].reindex(e.index)], axis=1).reindex(g).ffill()
    oi = q.sum_open_interest
    F["oi4h"] = np.log(oi / oi.shift(48)).reindex(te).to_numpy(); F["oi24h"] = np.log(oi / oi.shift(288)).reindex(te).to_numpy()
    F["oiz"] = ((oi - oi.rolling(2016).mean()) / oi.rolling(2016).std()).reindex(te).to_numpy()
    for k, n in (("tb1h", 12), ("tb4h", 48)):
        F[k] = (2 * q.taker_buy_base.rolling(n).sum() / q.volume.rolling(n).sum() - 1).reindex(te).to_numpy()
    ls = np.log(q.sum_toptrader_long_short_ratio)
    F["lsz"] = ((ls - ls.rolling(2016).mean()) / ls.rolling(2016).std()).reindex(te).to_numpy()
    fu = funding("ETHUSDT", int(t0.value // 10**6), int(t1.value // 10**6)); fs = pd.Series(fu.rate.to_numpy(), index=pd.to_datetime(fu.t, unit="ms")).sort_index()
    F["fund"] = fs.reindex(fs.index.union(t)).ffill().reindex(t).to_numpy()
    b = k5("BTC", t0, t1).close.reindex(g).ffill(); bs = np.sqrt((np.log(b).diff() ** 2).rolling(288).sum() / 24).reindex(te).to_numpy(); bc = b.reindex(te).to_numpy()
    for k, h in (("b4h", 4), ("b24h", 24)):
        F[k] = np.log(bc / b.reindex(te - pd.Timedelta(hours=h)).to_numpy()) / (bs * np.sqrt(h))
    F["bvr"] = np.log(bs / s24)
    return F, info


def rl14_X(F):
    """research load() 의 상태 열(순서 그대로) · 정규화 = R3 학습 구간 상수."""
    N, X = RL19_NORM, pd.DataFrame(index=F.index)
    X["pred_z"] = F.pred / N["pred_std"]
    for k in ("ls1h", "ls4h", "ls24h", "ls7d"):
        X[k] = np.log(F[k] / N["sstar"])
    for k in ("m1d", "m7d", "m30d", "m4h", "oi4h", "oi24h", "oiz", "tb1h", "tb4h", "lsz", "b4h", "b24h", "bvr"):
        X[k] = F[k]
    X["fund"] = F.fund / N["fund_std"]
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5).astype(np.float32)


def tsmom_verdict(start, end):
    """docs/prereg/tsmom_volsized_forward_20261026.md:
    > - 매 정시 t: 명목 = sign(ln(close_t / close_{t−720h})) × min(σ*/σ̂_t, 2). σ̂_t = 직전 24h(288개) 5분 로그수익 제곱합 → 1시간 환산 σ. σ* = 0.00613139826386366
    > - 1시간 보유 · 비용 4bp × |명목 변화| · ETHUSDT 선물 5분봉.
    > - 관문: F1 샤프(시간 손익 연환산) > 같은 창 «항상 롱 1배» 샤프 · F2 명목 수열 원형이동(7일 이상, 1,000회) 귀무 95분위 초과 · 보고: 총·MDD·회전·월별.
    pnl·stats·원형이동 = research_rl_v14_vol_sizing_20261026 의 pnl·stats·test() 그대로(rng 1, 168h ~ n−168h)."""
    import research_rl_v14_vol_sizing_20261026 as V
    F, rep = rl14_frame(start, end); r = F.ret.to_numpy()
    n = np.sign(F.m30d.to_numpy()) * np.minimum(TSMOM_SSTAR / F.sig.to_numpy(), V.CAP)
    assert np.isfinite(n).all() and np.isfinite(r).all(), "프레임 결측"
    x = V.pnl(n, r); null = np.array([V.pnl(np.roll(n, int(k)), r).sum() for k in np.random.default_rng(1).integers(168, len(n) - 168, 1000)])
    rep.update(hours=len(r), sstar=TSMOM_SSTAR, tsmom=V.stats(x), long_1x=V.stats(V.pnl(np.ones(len(r)), r)),
               long_volsized=V.stats(V.pnl(np.minimum(TSMOM_SSTAR / F.sig.to_numpy(), V.CAP), r)), mean_abs_notional=float(np.abs(n).mean()),
               long_share=float((n > 0).mean()), turnover_per_day=float(np.abs(np.diff(n)).sum() / (len(n) / 24)), shift_pct=float((null < x.sum()).mean() * 100),
               monthly_bp={str(k): float(v) for k, v in (pd.Series(x, index=F.index).groupby(F.index.to_period("M")).sum() * 1e4).items()})
    rep["gates"] = {"F1_sharpe_gt_long1x": rep["tsmom"]["sharpe"] > rep["long_1x"]["sharpe"], "F2_shift_gt95": rep["shift_pct"] > 95}
    return {**rep, **FRESH}


def rlv19_verdict(start, end, dir4h=DIR4H):
    """docs/prereg/rl_v19_forward_20261026.md:
    > - 모델 = tmp/rl_v14_20261026/v19_R3_{시드}/300000.zip 10개(R3 = 학습 2024~25) … 정규화(σ*·pred 표준편차·펀딩 표준편차·T2 문턱)는 R3 학습 구간 값 그대로.
    > - 상태 = v18 프레임 정의(build(extra=True))를 전진 자료로 다시 계산 … 명목 = TSMOM 부호 + RL 조정, 상한 2배, 1h, 4bp.
    > - 관문: W1 RL 샤프 > 같은 창 T3 TSMOM 샤프(tsmom 규칙 그대로) · W2 조정(RL − T3) 원형이동 1,000회 > 90분위 · 보고: 롱·MDD·회전.
    rollout·앙상블(10시드 명목 평균)·원형이동 = research test_v17 그대로. T2 는 관문·보고에 없어 계산 안 함(문턱 상수 불필요).
    결측 행은 연구처럼 dropna 하지 않고 멈춘다(전진 창에서 시간을 조용히 빼지 않게)."""
    import research_rl_v14_vol_sizing_20261026 as V
    from stable_baselines3 import PPO
    txt = dict(ln.split() for ln in (V.OUT / "v19_md5.txt").read_text().splitlines() if ln.strip())
    assert txt == RL19_MD5 and [int(s) for s in RL19_MD5] == V.SEEDS19, "v19_md5.txt·SEEDS19 가 등록값과 다름"
    zips = {sd: V.OUT / f"v19_R3_{sd}" / f"{V.TOTAL}.zip" for sd in RL19_MD5}
    rep = {"frozen_md5": {sd: check_md5(z, RL19_MD5[sd]) for sd, z in zips.items()}, "dir4h": str(dir4h), "dir4h_md5": md5(dir4h), "norm": RL19_NORM}
    F, info = rl14_frame(start, end, dir4h); rep.update(info)
    bad = {k: int(v) for k, v in F.isna().sum().items() if v}
    if bad:
        raise SystemExit(f"🔴 프레임 결측 {bad} — 방향 예측(--dir4h)이 창 끝까지 덮는지 확인")
    X, r, base = rl14_X(F).to_numpy(), F.ret.to_numpy(), np.sign(F.m30d.to_numpy())
    size, m = np.minimum(RL19_NORM["sstar"] / F.sig.to_numpy(), V.CAP), np.ones(len(r), bool)
    seeds = {sd: V.rollout(PPO.load(z, device="cpu"), X, size, m, base) for sd, z in zips.items()}; ens = np.mean(list(seeds.values()), 0)
    t3 = base * np.minimum(TSMOM_SSTAR / F.sig.to_numpy(), V.CAP); a = ens - t3; rng = np.random.default_rng(1)
    null = np.array([V.pnl(t3 + np.roll(a, int(rng.integers(168, len(a) - 168))), r).sum() for _ in range(1000)])
    rep.update(hours=len(r), RL=V.stats(V.pnl(ens, r)), T3_tsmom=V.stats(V.pnl(t3, r)), long_1x=V.stats(V.pnl(np.ones(len(r)), r)),
               long_volsized=V.stats(V.pnl(size, r)), per_seed={sd: V.stats(V.pnl(v, r))["sharpe"] for sd, v in seeds.items()},
               adj_mean_abs=float(np.abs(a).mean()), mean_abs_notional=float(np.abs(ens).mean()), long_share=float((ens > 0).mean()),
               turnover_per_day=float(np.abs(np.diff(ens)).sum() / (len(r) / 24)), adj_shift_pct=float((null < V.pnl(ens, r).sum()).mean() * 100))
    rep["gates"] = {"W1_RL_gt_T3": rep["RL"]["sharpe"] > rep["T3_tsmom"]["sharpe"], "W2_adj_shift_gt90": rep["adj_shift_pct"] > 90}
    return {**rep, **FRESH}


# ---------- xsec: 20코인 4h 상대 수익 DL ----------
XS_NPZ_MD5 = {"1961169912": "1de6dfe8788a3df1ae3e82286af62222", "2067129892": "1f0a12c6aa9b139e45b842c05c1c7975",
              "418362838": "877bcfc07e234cde6b3755761445d484", "436606973": "5f7dd9ecc5b58c9dbbbbfa7148b3f295",
              "501218739": "e41d6dbb7c891af377bd820f550d07ea"}   # 사전 등록 동결 예측(xsec_forward_md5.txt)
XS_W_MD5 = {"R3_K1_1961169912.pt": "80b6f36513d041b6591214bd13a3e361", "R3_K1_2067129892.pt": "a39acb8cfc526e88c5a0b34af4578f74",
            "R3_K1_418362838.pt": "6af8e5c88e8578af34b9f7f2edf41fbf", "R3_K1_436606973.pt": "595d5b698f2b796fef6fff58eafa8812",
            "R3_K1_501218739.pt": "d793c08a74f5be76e73e5019abf09ff2", "ridge_R3.npz": "b257e14f68e23e7aaedc80752d3a6abf"}
# ↑ tmp/xsec_dl_20261026/frozen_r3/frozen_r3_md5.txt — 1단계 동결 npz 는 예측만 담아 가중치가 없었다 → 같은 시드·설정 재학습(research --freeze_r3)이
#   동결 예측(va·te)을 최대 차 0.0 으로 재현한 가중치. 리지 = α 1000 재적합(base_R3 B3_te 차 0.0).
XS_HL, XS_COST = 3, 4e-4   # 사전 등록: EWMA 반감기 3 결정 · 문턱 띠 · 4bp


def xsec_check_frozen():
    """사전 등록 md5(예측 npz) + 가중치·리지 md5 검사 → 가중치로 R3 시험 예측을 다시 내어 동결 예측과 1e-6 안에서 같은지(가중치 = 동결 모형 증명)."""
    import research_xsec_dl_20coin_20261026 as XS
    txt = dict(ln.split() for ln in (XS.OUT / "xsec_forward_md5.txt").read_text().splitlines() if ln.strip())
    assert txt == {f"R3_K1_{k}.npz": v for k, v in XS_NPZ_MD5.items()}, "xsec_forward_md5.txt 가 등록값과 다름"
    wd = XS.OUT / "frozen_r3"
    rep = {"npz_md5": {sd: check_md5(XS.OUT / "dl" / f"R3_K1_{sd}.npz", v) for sd, v in XS_NPZ_MD5.items()},
           "weights_md5": {f: check_md5(wd / f, v) for f, v in XS_W_MD5.items()}}
    assert len(XS_W_MD5) == 6, "가중치 md5 미등록"
    D = np.load(XS.OUT / "data.npz")
    w = XS.WINDOWS["R3"]; te = XS.split_idx(D["ts"], w[4], w[5])
    nets = xsec_load_nets()
    sub = {"X": D["X"][te], "seq": D["seq"][te], "valid": D["valid"][te]}
    rep["frozen_pred_maxdiff"] = {sd: float(np.nanmax(np.abs(xsec_predict(nets[sd], sub) - np.load(XS.OUT / "dl" / f"R3_K1_{sd}.npz")["te"]))) for sd in nets}
    assert max(rep["frozen_pred_maxdiff"].values()) < 1e-6, f"가중치가 동결 예측을 재현 못 함 {rep['frozen_pred_maxdiff']}"
    return rep


def xsec_load_nets():
    import torch, research_xsec_dl_20coin_20261026 as XS
    out = {}
    for sd in XS_NPZ_MD5:
        ck = torch.load(XS.OUT / "frozen_r3" / f"R3_K1_{sd}.pt", weights_only=False)
        net = XS.make_net(ck["h"], ck["F"]); net.load_state_dict(ck["state"]); net.eval(); out[sd] = (net, ck)
    return out


def xsec_predict(nc, D):
    import torch, research_xsec_dl_20coin_20261026 as XS
    net, ck = nc
    Xn, Sn = XS.apply_norm(D["X"], ck["mu"], ck["sd"]), XS.apply_norm(D["seq"], ck["smu"], ck["ssd"])
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))   # noqa: E731
    with torch.no_grad():
        return np.concatenate([net(T(Sn[k:k + 256]), T(Xn[k:k + 256]), T(D["valid"][k:k + 256])).numpy() for k in range(0, len(Xn), 256)])


def xsec_frame(start, end):
    """연구 build() 와 같은 식(scripts/research_xsec_dl_20coin_20261026.py) · 원천만 전진용:
    20코인 5분 = 공개 REST klines(연구 패널 = 같은 열의 vision klines) · OI·롱숏 = vision daily metrics 를 create_time = 봉 시작에 붙임(패널 빌더와 같은 규칙).
    결정 t ∈ [start, end 20:00] 4h. 예열 40일(720h 창 + 48h 시퀀스)."""
    import research_xsec_dl_20coin_20261026 as XS, build_binance_vision_panel_20260915 as P
    from concurrent.futures import ThreadPoolExecutor
    t0, t1 = start - pd.Timedelta(days=40), end + DAY
    days = pd.date_range(t0, end, freq="D")
    assert int((days[-1] + DAY).value // 10**6) <= LIMIT_MS, "metrics 가져오기 상한 초과 — 전진 자료 보호"
    src, info = {}, {"klines_missing": {}, "metrics_days_missing": {}}
    n5 = len(pd.date_range(t0, t1 - pd.Timedelta("5min"), freq="5min"))
    for sym in XS.COINS:
        r = klines(f"{sym}USDT", "5m", int(t0.value // 10**6), int(t1.value // 10**6))
        k = pd.DataFrame([(a[2], a[3], a[4], a[5], a[8], a[9]) for a in r], index=pd.to_datetime([a[0] for a in r], unit="ms"),
                         columns=["high", "low", "close", "volume", "trades", "taker_buy_base"]).astype(float)
        k = k[~k.index.duplicated()].sort_index(); k["trades"] = k.trades.astype(int)
        with ThreadPoolExecutor(8) as ex:
            txt = list(ex.map(lambda d: P.fetch(f"{P.BASE}/metrics/{sym}USDT/{sym}USDT-metrics-{d.date()}.zip"), days))
        ms = [m for m in (pd.read_csv(io.StringIO(x)) for x in txt if x is not None) if "create_time" in m.columns]
        m = pd.concat(ms); m.index = pd.to_datetime(m.create_time); m = m[~m.index.duplicated()]
        for c in P.MCOLS:
            k[c] = m[c].reindex(k.index) if c in m.columns else np.nan
        src[sym] = k
        info["klines_missing"][sym] = int(n5 - len(k)); info["metrics_days_missing"][sym] = int(len(days) - len(ms))
        print(f"  {sym}: 5분봉 {len(k)}/{n5} · metrics {len(ms)}/{len(days)}일", flush=True)
    D = XS.build(src=src, idx=pd.date_range(t0, end + pd.Timedelta("23:00:00"), freq="1h"), first=start)
    want = pd.date_range(start, end + pd.Timedelta("20:00:00"), freq="4h")
    got = pd.to_datetime(D["ts"])
    if len(got) != len(want) or (got != want).any() or not D["valid"].all():
        raise SystemExit(f"🔴 결정 격자·코인 결측: 시점 {len(got)}/{len(want)} · 무효 코인·시점 {int((~D['valid']).sum())} — 조용히 빼지 않고 멈춤")
    return D, info


def xsec_verdict(start, end):
    """docs/prereg/xsec_dl_forward_20261026.md:
    > - 모형 = R3 검증 선택 K1 시드 5개 · 포트폴리오 = 점수 EWMA 반감기 3 결정(12h) · 문턱 띠(20% 들어감 / 30% 밖이면 나감) · 동일가중 롱숏 · 4h · 비용 4bp×회전.
    > - 비교 = 리지 B3 R3(같은 결정적 재적합, 반감기 3·띠) · 20코인·피쳐·정규화는 1단계 빌드 정의(R3 학습 구간 통계).
    > - 관문: Z1 DL 4bp 순 샤프 > 0 · Z2 DL 순 샤프 > 리지 순 샤프 · Z3 시점별 횡단면 IC 평균 > 0(5일 블록 CI 보고). 통과 = 셋 다.
    평활은 창 시작에서 새로 시작(연구 2단계와 같음). 점수·구성·손익·샤프·IC = 연구 함수 그대로."""
    import research_xsec_dl_20coin_20261026 as XS
    rep = xsec_check_frozen()
    D, info = xsec_frame(start, end); rep.update(info)
    nets = xsec_load_nets()
    per = {sd: xsec_predict(nc, D) for sd, nc in nets.items()}
    rz = np.load(XS.OUT / "frozen_r3" / "ridge_R3.npz")
    P = {"DL": np.mean(list(per.values()), 0),
         "ridge": XS.apply_norm(D["X"], rz["mu"], rz["sd"]).astype(np.float64) @ rz["coef"] + float(rz["intercept"])}
    valid, lab, y = D["valid"], D["lab"], D["y"]
    for k, p in P.items():
        g, tu = XS.pnl2(XS.weights2(XS.smooth_rank(p, valid, XS_HL), valid, True), lab)
        ic = XS.ic_series(p, y, valid)
        rep[k] = {"net4_sharpe": XS.sharpe(g - XS_COST * tu), "net2_sharpe": XS.sharpe(g - 2e-4 * tu), "gross_sharpe": XS.sharpe(g),
                  "gross_bp_4h": float(g.mean() * 1e4), "turnover_4h": float(tu.mean()), "ic_mean_ci": XS.block_ci(ic),
                  **{f"net4_{a}": b for a, b in XS.ls_stats(g - XS_COST * tu).items()}}
    rep["per_seed_ic"] = {sd: float(np.nanmean(XS.ic_series(p, y, valid))) for sd, p in per.items()}
    rep["decisions"] = int(len(D["ts"]))
    rep["gates"] = {"Z1_DL_net4_sharpe_gt0": rep["DL"]["net4_sharpe"] > 0, "Z2_DL_gt_ridge": rep["DL"]["net4_sharpe"] > rep["ridge"]["net4_sharpe"],
                    "Z3_DL_ic_gt0": rep["DL"]["ic_mean_ci"][0] > 0}
    if pd.Timestamp(end) < FWD:   # dry-run: 연구 패널(data.npz) 과 같은 결정의 피쳐·라벨 차(원천 패리티)
        A = np.load(XS.OUT / "data.npz"); ii = np.searchsorted(A["ts"], D["ts"]); assert (A["ts"][ii] == D["ts"]).all()
        rep["parity_vs_panel"] = {"X_maxabs": float(np.nanmax(np.abs(A["X"][ii] - D["X"]))), "seq_maxabs": float(np.nanmax(np.abs(A["seq"][ii] - D["seq"]))),
                                  "lab_maxabs": float(np.nanmax(np.abs(A["lab"][ii] - lab))),
                                  "X_nan_mismatch": int((np.isnan(A["X"][ii]) != np.isnan(D["X"])).sum())}
    return {**rep, **FRESH}


FN = {"vol": vol_verdict, "box": box_verdict, "trend90": trend_verdict, "trend180": trend_verdict, "rl": rl_verdict, "voliv": voliv_verdict,
      "tsmom": tsmom_verdict, "rlv19": rlv19_verdict, "xsec": xsec_verdict}


def main(argv=None, today=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cand", nargs="?", choices=list(FN)); ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run-past", nargs=2, metavar=("START", "END"))
    ap.add_argument("--dir4h", help="rlv19: 방향 예측 (ts, pred) parquet 또는 라이브 eth_dir4h_model.joblib (기본 = 09-14 까지 seed)")
    a = ap.parse_args(argv); today = today or dt.datetime.now(dt.timezone.utc).date()
    if a.status or not a.cand:
        return status(today)
    start, end, real = window(a.cand, today, a.dry_run_past)
    OUT.mkdir(parents=True, exist_ok=True); f = OUT / f"{a.cand}_{'verdict' if real else 'dryrun'}_{start.date()}_{end.date()}.json"
    if real and f.exists():
        raise SystemExit(f"{f} 이미 있음 — 전진 판정은 1회만")
    print(("🔴 판정" if real else "DRY RUN — 판정 아님(과거 창 코드 점검)") + f" {a.cand} {start.date()}~{end.date()}", flush=True)
    rep = FN[a.cand](start, end, **({"dir4h": Path(a.dir4h)} if a.dir4h else {}))
    g = rep.get("gates", {})
    rep = {"candidate": a.cand, "is_verdict": real, "label": "전진 판정" if real else "DRY RUN — 판정 아님", "prereg": VERDICTS[a.cand][2],
           "window": [str(start.date()), str(end.date())], "data_until": str((end + DAY).date()) + " 00:00 UTC 배타", "run_utc": str(pd.Timestamp.now(tz="UTC")),
           **rep, "pass": bool(g) and all(g.values())}
    f.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: rep[k] for k in ("candidate", "label", "window", "gates", "pass") if k in rep}, ensure_ascii=False, default=float), "\n→", f)
    return rep


if __name__ == "__main__":
    main()
