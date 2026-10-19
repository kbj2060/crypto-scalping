"""박스 유지 확률(dashboard/box_hold.py · box_hold_hgb_20261016.joblib) 재검증 (2026-10-19, 리뷰 «2025~ AUC 는 재채점·새 증거는 전진 5일뿐» 후속).

과제
  1. 2025-01-01~2026-10-14 (재채점 -- 모델 변형을 2025·26 AUC 를 보고 골랐으니 새 증거 아님): (a) 학습 기저율 상수 (b) 기하 로지스틱 6칸
     (log_ds·log_vr·log_rng·hs·hc·wkd, 2022~24 로만 다시 적합 = 옛 배포 F) (c) 배포 HGB. AUC · Brier · 10분위 보정 최대 차.
     HGB − 로지스틱 ΔAUC·ΔBrier = 같은 행 짝지은 일 블록 부트스트랩 95% CI.
  2. 전진: 배포(0929bb79, 2026-10-16 14:06 UTC) 이후 ~ data.binance.vision 1분봉 마지막 완결일(10-18). 닫힌 1분봉으로 box_hold.payload 그대로,
     결정 = 학습 격자(30분·1h 분%5==4 · 4h 매시 59분) 와 매분(서빙은 아무 분에나 계산) 둘 다. 라벨 = 연구 정의(다음 W분 고가 < 종가+½창폭 · 저가 > 종가−½창폭).
     참고 창: 연구 데이터 끝(10-14) 다음 날 10-15 00:00~ (모델은 10-14 까지 데이터로 골랐으니 이것도 안 본 데이터).
     거시 일정: 서버 상태 파일 사본(backups/…/macro_calendar_state.json)은 10-20 이후 일정만 있음 → 연구 events.json(FRED 공식 일정, 10-16 받음)을
     payload 형식으로 바꿔 넣는다. 배포 후 창 안 발표 칸 켜짐 0건(다음 주요 10-28 · 실업수당 10-22) → «없음» 재생과 같다.
     (참고 창 10-15 는 12:30 UTC 주요 발표·실업수당이 끼어 있어 발표 칸 330행 켜짐.)
  3. «4h 2026 과소 예측 3~7pp» = (예측 평균 − 실제 유지율) 일 블록 CI · 2026 · 최근 90일 · 10월(1~14일) · 전진.

판정(계산 전에 정함)
  R1 «AUC .78 유지»(재채점)  2025~ HGB AUC 95% CI 하한 ≥ .77 이 세 척도 모두. 통과해도 «새 증거 아님» 꼬리표.
  R2 «HGB 가 기하를 넘는다»  척도별 ΔAUC(HGB−로지스틱) CI 하한 > 0 이고 ΔBrier CI 상한 < 0. 크기 ΔAUC < .01 이면 «통계적으로만».
     (변형 선택에 2025·26 을 봤으므로 이 Δ 는 위쪽으로 치우친 값 -- 상한으로 읽는다.)
  R3 «전진 증거»  일 블록 ≥ 10 일 때만 판정: AUC CI 하한 > .70 이고 보정 최대 차 ≤ 10pp. 일 블록 < 10 이면 «판정 불가»(점추정만 보고).
  R4 «4h 과소 예측»  2026 (예측−실제) CI 상한 < 0 이고 점추정 ≤ −3pp → «확인» · CI 상한 < 0 이지만 > −3pp → «있지만 작음» · CI 가 0 포함 → «불확인».

결과(10-19 실행 — 아래 숫자는 tmp/box_hold_reverify_20261019/report.json)
  1 (재채점) AUC 상수 .5 · 로지스틱 6칸 30분 .769 / 1h .765 / 4h .771 · HGB .788[.785,.791] / .789[.784,.793] / .785[.777,.795]
    ΔAUC HGB−로지 +.019[.018,.021] / +.024[.022,.026] / +.014[.010,.017] · ΔBrier −.0058 / −.0068 / −.0032 (CI 전부 < 0)
    Brier skill(상수 대비) 로지 .17/.17/.19 → HGB .20/.21/.21 = skill 의 85~93% 가 기하 · 보정 최대 상수 .6/.8/.3 · 로지 4.1/4.7/4.8 · HGB 1.6/3.4/2.8pp
    → R1 통과(재채점) · R2 통과(크기 +.014~.024, 상한으로 읽음; 4h 2025 단독 ΔBrier CI 0 포함)
  2 전진 배포 후(10-16 14:06~10-18, 일 블록 3개) 격자 n 689/683/54 · HGB AUC .758/.756/.812 · 로지 .745/.753/.797 · 보정 최대 8.0/12.3/33.9pp
    매분 n 3,444/3,414/3,234 · AUC .758/.746/.839 · 참고 10-15~(일 4개) 격자 .787/.766/.870 → R3 판정 불가(일 블록 < 10)
  3 4h (예측−실제) 2025 +1.1[−0.2,+2.4] · 2026 −1.7[−3.1,−0.4] · 최근 90일 −3.0[−5.2,−1.0] · 10월 1~14일 −6.7[−12.5,−0.9] · 로지스틱도 같은 크기
    · 전진 +4.4[−23,+13] → R4 «있지만 작음»(2026 전체 −1.7pp; 3~7pp 는 최근 창을 골라 본 값) · 모델 탓 아니라 국면(기하 로지스틱도 같음)

    nice -n 15 python scripts/research_eth_box_hold_reverify_20261019.py            # 전부
    nice -n 15 python scripts/research_eth_box_hold_reverify_20261019.py --selftest
"""
from __future__ import annotations

import glob
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT))
from dashboard.box_hold import B as BH, NEED, WIN, features, payload  # noqa: E402

DEC = MAIN / ".claude/worktrees/pingpong-long-short-hedge-de2ac6/tmp/range_hold_20261016/decisions.parquet"
BOXD = MAIN / ".claude/worktrees/futures-discount-basis-c5126d/tmp/box_improve_20261016"
OUT = ROOT / "tmp/box_hold_reverify_20261019"
VIS = OUT / "vision_1m"                       # data.binance.vision ETHUSDT 1m 일 파일 10-13~10-18 (CHECKSUM 대조함)
F_BASE = ["log_ds", "log_vr", "log_rng", "hs", "hc", "wkd"]
LEAN = BH["scales"]["1h"]["cols"]             # 배포 학습 행 = dropna(lean) (export 와 같음)
DEPLOY = pd.Timestamp("2026-10-16 14:06", tz="UTC")
NB = 400
log = lambda m: print(m, flush=True)  # noqa: E731


def blocks(lab: np.ndarray) -> list[np.ndarray]:
    codes, uniq = pd.factorize(lab)
    order = np.argsort(codes, kind="stable")
    st = np.searchsorted(codes[order], np.arange(len(uniq)))
    return [order[a:b] for a, b in zip(st, np.r_[st[1:], len(codes)])]


def boot(fn, lab, B=NB, seed=0) -> list[float]:
    """블록 부트스트랩 95% CI (fn(ii) → 값)."""
    bl = blocks(lab)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        ii = np.concatenate([bl[k] for k in rng.integers(0, len(bl), len(bl))])
        v = fn(ii)
        if np.isfinite(v):
            out.append(v)
    return [round(float(np.quantile(out, .025)), 4), round(float(np.quantile(out, .975)), 4)]


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


def calib_gap(y, p) -> float:
    if np.ptp(p) == 0:                        # 상수 예측 = 칸 하나
        return float(abs(p.mean() - y.mean()))
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").mean()
    return float((g.p - g.y).abs().max())


def basic(y, p) -> dict:
    return {"auc": round(float(auc(y, p)), 4) if np.ptp(p) > 0 else 0.5, "brier": round(float(np.mean((p - y) ** 2)), 5),
            "calib_max_pp": round(calib_gap(y, p) * 100, 1), "bias_pp": round(float((p.mean() - y.mean()) * 100), 2)}


def compare(y, ph, pl, base, days) -> dict:
    """상수·로지스틱·HGB 지표 + HGB−로지스틱 짝지은 블록 CI."""
    return {"n": int(len(y)), "pos": int(y.sum()), "days": int(len(np.unique(days))), "hold_rate": round(float(y.mean()), 4),
            "const": basic(y, np.full(len(y), base)), "logit6": basic(y, pl), "hgb": basic(y, ph),
            "hgb_auc_ci": boot(lambda ii: auc(y[ii], ph[ii]), days, seed=1),
            "logit_auc_ci": boot(lambda ii: auc(y[ii], pl[ii]), days, seed=1),
            "dauc": round(float(auc(y, ph) - auc(y, pl)), 4),
            "dauc_ci": boot(lambda ii: auc(y[ii], ph[ii]) - auc(y[ii], pl[ii]), days, seed=2),
            "dbrier": round(float(np.mean((ph - y) ** 2) - np.mean((pl - y) ** 2)), 5),
            "dbrier_ci": boot(lambda ii: np.mean((ph[ii] - y[ii]) ** 2) - np.mean((pl[ii] - y[ii]) ** 2), days, seed=3)}


def fit_logit(tr: pd.DataFrame):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(tr[F_BASE], tr.hold)


def part1() -> tuple[dict, dict, dict]:
    D = pd.read_parquet(DEC, columns=["scale", "ts", "hold"] + F_BASE)
    D = D[D.scale.isin(list(WIN))].reset_index(drop=True)          # features.parquet 와 같은 행 순서
    X = pd.read_parquet(BOXD / "features.parquet", columns=[c for c in LEAN if c not in F_BASE])
    A = pd.concat([D, X], axis=1).replace([np.inf, -np.inf], np.nan)
    rep, models, keep = {}, {}, {}
    for sc in WIN:
        cf = BH["scales"][sc]
        g = A[A.scale == sc]
        tr = g[g.ts < "2025-01-01"].dropna(subset=LEAN)
        assert len(tr) == cf["n"], (sc, len(tr), cf["n"])           # 배포 학습 행과 같은 행
        lr = models[sc] = fit_logit(tr)
        te = g[g.ts >= "2025-01-01"].dropna(subset=cf["cols"])
        y = te.hold.to_numpy()
        ph = cf["model"].predict_proba(te[cf["cols"]])[:, 1]
        pl = lr.predict_proba(te[F_BASE])[:, 1]
        ts = pd.DatetimeIndex(te.ts)
        days = ts.normalize().to_numpy()
        rep[sc] = {}
        for lab, m in (("2025~", np.ones(len(y), bool)), ("2025", ts.year == 2025), ("2026", ts.year == 2026)):
            rep[sc][lab] = compare(y[m], ph[m], pl[m], cf["train_base"], days[m])
            r = rep[sc][lab]
            log(f"[{sc} {lab}] n {r['n']:,} 유지 {r['hold_rate']:.3f} · AUC 상수 .5 로지스틱 {r['logit6']['auc']:.4f} HGB {r['hgb']['auc']:.4f}{r['hgb_auc_ci']} "
                f"ΔAUC {r['dauc']:+.4f}{r['dauc_ci']} · Brier 상수 {r['const']['brier']:.4f} 로지 {r['logit6']['brier']:.4f} HGB {r['hgb']['brier']:.4f} "
                f"ΔBrier {r['dbrier']:+.5f}{r['dbrier_ci']} · 보정최대 {r['const']['calib_max_pp']}/{r['logit6']['calib_max_pp']}/{r['hgb']['calib_max_pp']}pp")
        keep[sc] = pd.DataFrame({"ts": ts, "y": y, "ph": ph, "pl": pl})
    return rep, models, keep


def bias_ci(y, p, days) -> dict:
    return {"n": int(len(y)), "days": int(len(np.unique(days))), "pred": round(float(p.mean()), 4), "real": round(float(y.mean()), 4),
            "bias_pp": round(float((p.mean() - y.mean()) * 100), 2),
            "ci_pp": [round(v * 100, 2) for v in boot(lambda ii: p[ii].mean() - y[ii].mean(), days, seed=4)]}


def part3(k4: pd.DataFrame, fwd4: pd.DataFrame | None) -> dict:
    ts = pd.DatetimeIndex(k4.ts)
    days = ts.normalize().to_numpy()
    out = {}
    for lab, m in (("2025", ts.year == 2025), ("2026", ts.year == 2026), ("최근90일", ts >= ts.max() - pd.Timedelta("90D")),
                   ("2026-10(1~14일)", ts >= "2026-10-01")):
        out[lab] = {"hgb": bias_ci(k4.y.to_numpy()[m], k4.ph.to_numpy()[m], days[m]), "logit6": bias_ci(k4.y.to_numpy()[m], k4.pl.to_numpy()[m], days[m])}
    g = k4[ts.year == 2026].assign(m=ts[ts.year == 2026].strftime("%Y-%m")).groupby("m")
    out["2026 월별 (예측−실제)pp HGB"] = {k: round(float((v.ph.mean() - v.y.mean()) * 100), 1) for k, v in g}
    if fwd4 is not None and len(fwd4):
        d = pd.DatetimeIndex(fwd4.ts).normalize().to_numpy()
        out["전진 격자"] = {"hgb": bias_ci(fwd4.y.to_numpy(), fwd4.ph.to_numpy(), d), "logit6": bias_ci(fwd4.y.to_numpy(), fwd4.pl.to_numpy(), d)}
    return out


def load_k() -> pd.DataFrame:
    parts = []
    for z in sorted(glob.glob(str(VIS / "ETHUSDT-1m-*.zip"))):
        with zipfile.ZipFile(z) as zf:
            x = pd.read_csv(zf.open(zf.namelist()[0]))
        parts.append(x)
    x = pd.concat(parts).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    assert (np.diff(x.open_time) == 60000).all(), "1분봉 빈칸"
    return pd.DataFrame({"timestamp": pd.to_datetime(x.open_time, unit="ms", utc=True), "high": x.high.astype(float), "low": x.low.astype(float),
                         "close": x.close.astype(float), "volume": x.volume.astype(float), "taker_buy_base": x.taker_buy_volume.astype(float)})


def hold_labels(h, l, c, W) -> np.ndarray:
    """연구 정의: y[i] = max h[i+1..i+W] < c[i]+½창폭 & min l[i+1..i+W] > c[i]−½창폭 (창폭 = h[i−W+1..i] 최대 − 최소). 미완결 = −1."""
    rng = pd.Series(h).rolling(W).max().to_numpy() - pd.Series(l).rolling(W).min().to_numpy()
    fmx = np.r_[pd.Series(h[::-1]).rolling(W).max().to_numpy()[::-1][1:], np.nan]
    fmn = np.r_[pd.Series(l[::-1]).rolling(W).min().to_numpy()[::-1][1:], np.nan]
    y = ((fmx < c + 0.5 * rng) & (fmn > c - 0.5 * rng)).astype(int)
    return np.where(np.isfinite(fmx) & np.isfinite(rng), y, -1)


def research_events() -> list[dict]:
    ev = json.loads((BOXD / "events.json").read_text())
    iso = lambda t: pd.Timestamp(int(t), unit="ms", tz="UTC").isoformat()  # noqa: E731
    return ([{"time_utc": iso(t), "category": "fomc", "title_ko": "FOMC 금리결정 발표"} for t in ev["major"]] +
            [{"time_utc": iso(t), "category": "econ", "detail": "release_id=180"} for t in ev["claims"]])


def part2(models: dict) -> tuple[dict, pd.DataFrame]:
    K = load_k()
    events = research_events()
    h, l, c = K.high.to_numpy(), K.low.to_numpy(), K.close.to_numpy()
    Y = {sc: hold_labels(h, l, c, W) for sc, W in WIN.items()}
    mins = (K.timestamp.astype("int64") // 60_000_000_000).to_numpy()
    start = pd.Timestamp("2026-10-15", tz="UTC")
    rows, mstat = [], {"major_in": 0, "claims_in": 0, "major_recent": 0}
    for j in range(NEED + 60, len(K)):
        if K.timestamp.iloc[j] < start or Y["30m"][j] < 0:
            continue
        k = K.iloc[j - NEED - 59:j + 1]
        pl_ = payload(k, events)
        F = features(k.reset_index(drop=True), events)
        for sc, W in WIN.items():
            s = pl_["scales"][sc]
            if Y[sc][j] < 0 or not s.get("ok"):
                continue
            f = F[sc]
            mstat["major_in"] += f["M_major_in"]; mstat["claims_in"] += f["M_claims_in"]; mstat["major_recent"] += f["M_major_recent"]
            grid = (mins[j] % 60 == 59) if sc == "4h" else (mins[j] % 5 == 4)
            pl = float(models[sc].predict_proba(pd.DataFrame([[f[x] for x in F_BASE]], columns=F_BASE))[0, 1])
            rows.append((K.timestamp.iloc[j], sc, grid, s["p"], pl, int(Y[sc][j])))
    R = pd.DataFrame(rows, columns=["ts", "scale", "grid", "ph", "pl", "y"])
    R.to_parquet(OUT / "forward_rows.parquet")
    log(f"전진 행 {len(R):,} ({R.ts.min()} ~ {R.ts.max()}) · 일정 칸 켜진 행 {mstat}")
    rep = {"events": f"연구 events.json(FRED) {len(events)}건 · 전진 창 안 켜진 발표 칸 {mstat}", "data_last_bar": str(K.timestamp.iloc[-1])}
    for wlab, w0 in (("배포후(10-16 14:06 UTC~)", DEPLOY), ("참고 10-15~", start)):
        for glab, gm in (("격자", True), ("매분", False)):
            for sc in WIN:
                g = R[(R.scale == sc) & (R.ts >= w0) & (R.grid if gm else True)]
                y, ph, pl = g.y.to_numpy(), g.ph.to_numpy(), g.pl.to_numpy()
                ts = pd.DatetimeIndex(g.ts)
                cf = BH["scales"][sc]
                r = compare(y, ph, pl, cf["train_base"], ts.normalize().to_numpy())
                r["hgb_auc_ci_4h_blocks"] = boot(lambda ii: auc(y[ii], ph[ii]), ts.floor("4h").to_numpy(), seed=5)
                r["labels"] = {"깨질 쪽(<q20)": [int((ph < cf["q20"]).sum()), round(float(y[ph < cf["q20"]].mean()), 3) if (ph < cf["q20"]).any() else None],
                               "머물 쪽(>q80)": [int((ph > cf["q80"]).sum()), round(float(y[ph > cf["q80"]].mean()), 3) if (ph > cf["q80"]).any() else None]}
                r["range"] = [str(ts.min()), str(ts.max())]
                rep.setdefault(wlab, {}).setdefault(glab, {})[sc] = r
                log(f"[전진 {wlab} {glab} {sc}] n {r['n']} 유지 {r['pos']} 일 {r['days']} · HGB AUC {r['hgb']['auc']:.3f} 일블록{r['hgb_auc_ci']} 4h블록{r['hgb_auc_ci_4h_blocks']} "
                    f"로지 {r['logit6']['auc']:.3f} Δ {r['dauc']:+.3f} · Brier 상수 {r['const']['brier']:.4f} 로지 {r['logit6']['brier']:.4f} HGB {r['hgb']['brier']:.4f} · "
                    f"보정최대 HGB {r['hgb']['calib_max_pp']}pp 치우침 {r['hgb']['bias_pp']:+.1f}pp · {r['labels']}")
    return rep, R[(R.scale == "4h") & R.grid & (R.ts >= DEPLOY)]


def selftest():
    rs = np.random.default_rng(0)
    c = 100 + rs.normal(size=400).cumsum(); h = c + rs.random(400); l = c - rs.random(400)
    for W in (3, 7):
        y = hold_labels(h, l, c, W)
        for i in range(W, 400 - W - 1):
            rg = h[i - W + 1:i + 1].max() - l[i - W + 1:i + 1].min()
            assert y[i] == int(h[i + 1:i + 1 + W].max() < c[i] + rg / 2 and l[i + 1:i + 1 + W].min() > c[i] - rg / 2), (W, i)
        assert (y[-W:] == -1).all()
    p = rs.random(20000); yy = (rs.random(20000) < p).astype(int)
    assert calib_gap(yy, p) < 0.03 and calib_gap(yy, np.clip(p + 0.2, 0, 1)) > 0.1 and abs(calib_gap(yy, np.full(20000, .3)) - abs(.3 - yy.mean())) < 1e-12
    lo, hi = boot(lambda ii: yy[ii].mean(), np.arange(20000) // 100, B=200)
    assert lo < yy.mean() < hi
    from dashboard.box_hold import event_times
    if (BOXD / "events.json").exists():
        ev = json.loads((BOXD / "events.json").read_text())
        maj, cl = event_times(research_events())
        assert len(maj) == len(set(ev["major"])) and len(cl) == len(set(ev["claims"]))
    log("selftest OK")


if __name__ == "__main__":
    selftest()
    if "--selftest" in sys.argv:
        sys.exit()
    OUT.mkdir(parents=True, exist_ok=True)
    rep1, models, keep = part1()
    rep2, fwd4 = part2(models)
    rep3 = part3(keep["4h"], fwd4)
    for k, v in rep3.items():
        log(f"[4h 치우침 {k}] {v}")
    (OUT / "report.json").write_text(json.dumps({"1 기하 대비 2025~(재채점)": rep1, "2 전진": rep2, "3 4h 치우침": rep3}, ensure_ascii=False, indent=1))
    log(f"→ {OUT / 'report.json'}")
