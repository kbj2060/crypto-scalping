"""리스크 딥러닝 1차 시험 — 보유시간 조건부 MAE 분위를 딥러닝이 기존 GBM 보다 «더 날카롭게, 같은 안전도로» 맞히나 (2026-10-12).

사용자: «이제 리스크 관리 모델을 만들어야 해. 리스크 관리 딥러닝 모델은 어떻게 만들 수 있는지 연구해줘.»

목적(리스크 모델이 하는 일): 지금 진입하면 보유 h 동안 최대 역행폭(MAE)이 얼마까지 갈 수 있나 → 크기 = 안전계수 / MAE_q.
목적 지표(수익 아님):
  1) 보정 후 표본외 초과율이 목표(5%·1%)에 맞나 — 전체 · 변동성 상위 10% 봉(조건부 커버리지) · 보유시간별.
  2) 같은 초과율에서 예측 MAE 가 얼마나 작나(날카로움) = 허용 레버리지 평균 1/MAE 가 GBM 대비 몇 % 큰가.
  3) q0.9 핀볼 손실(보정 전).
모델:
  B0 HAR 배수   exp(rv48)·√h 에 보유시간별 배수(보정구간에서) — 변동성 하나만.
  B1 GBM        배포했던 그대로(22피쳐 + log h + 방향, HistGBR quantile 0.9).
  D  MLP        같은 22피쳐 → 10 목표(5 보유 × 2 방향) × 분위 3(.5/.9/.99), log MAE 핀볼.
  D  TCN        최근 288봉 원시 시퀀스(로그수익·로그 고저폭·거래대금·체결수 상대값) + 22피쳐, 같은 출력.
분할(배포 모델과 같다): 학습 2022-01~2025-06(뒤 10% 조기종료용) · 보정 2025-07~12 · 검증 2026-01~09-15.
보정 = 각 모델의 q0.9 예측에 보유시간·방향별 배수 m 을 곱해 보정구간 초과율을 목표에 맞춘다(배포 운영 규칙과 같은 방식).
딥러닝 시드 3개 평균(연구용, 승격 주장 아님). CPU 전용(로컬 GPU 없음, 서버 GPU 금지).

실행: python scripts/research_eth_risk_dl_mae_quantile_20261012.py [--selftest] [--epochs 12]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = pathlib.Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(MAIN / "scripts"))
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402

KLINES = MAIN / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
HB = [12, 24, 48, 96, 288]
QS = [0.5, 0.9, 0.99]
TRAIN_END, CALIB_END = "2025-06-30T23:55", "2025-12-31T23:55"
TARGETS = (0.05, 0.01)
SEQ = 288
SEEDS = (11, 23, 37)


# ---------------------------------------------------------------- 데이터
def load():
    df = pd.read_csv(KLINES)
    df["ts"] = pd.to_datetime(df.timestamp)
    df = df[df.ts >= "2021-12-01"].reset_index(drop=True)
    c, hi, lo = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    F = svm.build_features(df.ts, c, df.quote_volume.to_numpy(float), df.trades.to_numpy(float), hi, lo)
    Y = np.full((len(df), len(HB), 2), np.nan)          # [봉, 보유, 방향(0 롱 · 1 숏)] MAE %
    for k, h in enumerate(HB):
        fmin = pd.Series(lo).iloc[::-1].rolling(h, min_periods=h).min().iloc[::-1].shift(-1).to_numpy()
        fmax = pd.Series(hi).iloc[::-1].rolling(h, min_periods=h).max().iloc[::-1].shift(-1).to_numpy()
        Y[:, k, 0] = (c - fmin) / c * 100
        Y[:, k, 1] = (fmax - c) / c * 100
    Y = np.where(np.isfinite(Y), np.maximum(Y, 1e-3), np.nan)   # 역행 0(유리하게만 감) → 바닥값
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    rng = np.log(np.maximum((hi - lo) / c, 1e-6))
    qv = df.quote_volume.to_numpy(float); nt = df.trades.to_numpy(float)
    rel = lambda x: np.log(np.maximum(x, 1) / pd.Series(np.maximum(x, 1)).rolling(288, min_periods=50).mean().shift(1).bfill().to_numpy())
    S = np.stack([lr, rng, rel(qv), rel(nt)], 1).astype(np.float32)
    return df.ts, F.to_numpy(np.float32), Y, S


# ---------------------------------------------------------------- 평가
def calib_mult(pred, y, target):
    """보유·방향별 배수 m: 보정구간에서 y > m·pred 비율 = target (= y/pred 의 1−target 분위)."""
    return np.nanquantile(y / pred, 1 - target, axis=0)


def evaluate(name, p9_cal, y_cal, p9_te, y_te, vol_te, out):
    r = {}
    top = vol_te >= np.nanquantile(vol_te, 0.9)
    for t in TARGETS:
        safe = p9_te * calib_mult(p9_cal, y_cal, t)
        ex = y_te > safe
        r[f"목표{t:.0%}"] = {
            "초과율 전체": round(float(np.mean(ex)), 4),
            "초과율 변동성 상위10%": round(float(np.mean(ex[top])), 4),
            "보유별 초과율": [round(float(np.mean(ex[:, k])), 4) for k in range(len(HB))],
            "평균 1/MAE": float(np.mean(1 / safe)),
            "안전MAE 중앙 %": [round(float(np.median(safe[:, k])), 3) for k in range(len(HB))]}
    out[name] = r


def pinball(pred, y, q):
    d = y - pred
    return float(np.nanmean(np.maximum(q * d, (q - 1) * d)))


# ---------------------------------------------------------------- 모델
def fit_b0(F, Y):
    sig = np.exp(F[:, svm.FEATURES.index("rv48")])[:, None, None] * np.sqrt(np.array(HB, float))[None, :, None] * 100
    return np.broadcast_to(sig, Y.shape).copy()


def fit_gbm(F, Y, itr):
    from sklearn.ensemble import HistGradientBoostingRegressor
    sub = itr[::3]                                        # 겹치는 라벨이라 3봉 간격으로 충분
    side_h = [(k, h, s) for k, h in enumerate(HB) for s in (0, 1)]
    X = np.vstack([np.c_[F[sub], np.full(len(sub), np.log(h * 5.0)), np.full(len(sub), 1 - 2 * s)] for k, h, s in side_h])
    y = np.concatenate([Y[sub, k, s] for k, h, s in side_h])
    ok = np.isfinite(y)
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, max_iter=300, learning_rate=0.06,
                                      max_depth=6, min_samples_leaf=200, random_state=20260913)
    m.fit(X[ok], y[ok])
    out = np.full(Y.shape, np.nan)
    for k, h, s in side_h:
        out[:, k, s] = m.predict(np.c_[F, np.full(len(F), np.log(h * 5.0)), np.full(len(F), 1 - 2 * s)])
    return np.maximum(out, 1e-3)


def fit_dl(kind, F, Y, S, itr, iva, seed, epochs, log):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(6)
    mu, sd = np.nanmean(F[itr], 0), np.nanstd(F[itr], 0) + 1e-6
    Fz = np.nan_to_num((F - mu) / sd).astype(np.float32).clip(-8, 8)
    smu, ssd = np.nanmean(S[itr], 0), np.nanstd(S[itr], 0) + 1e-6
    Sz = np.nan_to_num((S - smu) / ssd).astype(np.float32).clip(-8, 8)
    LY = np.log(Y).astype(np.float32)                     # log MAE (분위는 단조변환에 불변)
    nq, no = len(QS), len(HB) * 2
    qv = torch.tensor(QS)

    class Head(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.base = nn.Linear(d, no)
            self.gap = nn.Linear(d, no * (nq - 1))        # 분위 교차 방지: 위 분위 = 아래 + softplus 누적
        def forward(self, z):
            b = self.base(z)[..., None]
            g = nn.functional.softplus(self.gap(z)).view(-1, no, nq - 1)
            return torch.cat([b, b + torch.cumsum(g, -1)], -1)   # [B, 10, 3]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(F.shape[1], 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.GELU())
            self.h = Head(64)
        def forward(self, xs, xf):
            return self.h(self.f(xf))

    class TCN(nn.Module):
        def __init__(self, ch=24):
            super().__init__()
            convs, cin = [], S.shape[1]
            for d in (1, 2, 4, 8, 16, 32, 64):           # 수용 영역 255봉
                convs.append(nn.Conv1d(cin, ch, 3, dilation=d, padding=2 * d)); cin = ch
            self.convs = nn.ModuleList(convs)
            self.f = nn.Sequential(nn.Linear(ch + F.shape[1], 64), nn.GELU(), nn.Dropout(0.1))
            self.h = Head(64)
        def forward(self, xs, xf):
            z = xs.transpose(1, 2); L = z.shape[-1]
            for c in self.convs:
                z = nn.functional.gelu(c(z)[..., :L])    # 인과: 왼쪽만 패딩한 효과(오른쪽 잘라냄)
            return self.h(self.f(torch.cat([z[..., -1], xf], 1)))

    net = MLP() if kind == "MLP" else TCN()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    win = np.arange(-SEQ + 1, 1)

    def batch(ix):
        xs = torch.from_numpy(Sz[ix[:, None] + win]) if kind == "TCN" else None
        return xs, torch.from_numpy(Fz[ix]), torch.from_numpy(LY[ix].reshape(len(ix), no))

    def predict(ix_all, bs=4096):
        net.eval(); out = []
        with torch.no_grad():
            for a in range(0, len(ix_all), bs):
                xs, xf, _ = batch(ix_all[a:a + bs])
                out.append(net(xs, xf).numpy())
        net.train()
        return np.concatenate(out)

    tr = itr[::2]
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        t0 = time.time(); perm = np.random.permutation(tr); tot = 0.0
        for a in range(0, len(perm), 512):
            xs, xf, y = batch(perm[a:a + 512])
            d = y[..., None] - net(xs, xf)
            l = torch.maximum(qv * d, (qv - 1) * d).mean()
            opt.zero_grad(); l.backward(); opt.step(); tot += float(l) * len(xf)
        pv = predict(iva[::3]); dv = LY[iva[::3]].reshape(-1, no)[..., None] - pv
        vl = float(np.mean(np.maximum(np.array(QS) * dv, (np.array(QS) - 1) * dv)))
        log(f"  {kind} seed{seed} ep{ep} train {tot / len(perm):.4f} val {vl:.4f} ({time.time() - t0:.0f}s)")
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(best_state)
    return predict


# ---------------------------------------------------------------- 실행
def run(out_dir: pathlib.Path, epochs: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "run.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    ts, F, Y, S = load()
    tsv = ts.to_numpy()
    valid = np.isfinite(F).all(1) & np.isfinite(Y).all((1, 2)) & (np.arange(len(F)) >= 600)
    itr_all = np.flatnonzero(valid & (tsv >= np.datetime64("2022-01-01")) & (tsv <= np.datetime64(TRAIN_END)))
    cut = itr_all[int(len(itr_all) * 0.9)]
    itr, iva = itr_all[itr_all < cut - 288], itr_all[itr_all >= cut]     # 조기종료 구간은 288봉 띄움
    ica = np.flatnonzero(valid & (tsv > np.datetime64(TRAIN_END)) & (tsv <= np.datetime64(CALIB_END)))
    ite = np.flatnonzero(valid & (tsv > np.datetime64(CALIB_END)))
    log(f"학습 {len(itr):,} · 조기종료 {len(iva):,} · 보정 {len(ica):,} · 검증 {len(ite):,} (봉)")
    vol = np.exp(F[:, svm.FEATURES.index("rv48")])
    res, pin = {}, {}
    p = fit_b0(F, Y)
    evaluate("B0 HAR 배수(rv48·√h)", p[ica], Y[ica], p[ite], Y[ite], vol[ite], res)
    pin["B0"] = pinball(p[ite], Y[ite], 0.9)       # B0 는 보정 전 수준이 의미 없음(참고용)
    log("B0 done")
    p = fit_gbm(F, Y, itr_all)
    evaluate("B1 GBM(배포판)", p[ica], Y[ica], p[ite], Y[ite], vol[ite], res)
    pin["B1 GBM"] = pinball(p[ite], Y[ite], 0.9)
    log(f"B1 done {json.dumps(res['B1 GBM(배포판)']['목표1%'], ensure_ascii=False)}")
    for kind in ("MLP", "TCN"):
        preds = []
        for sd in SEEDS:
            pr = fit_dl(kind, F, Y, S, itr, iva, sd, epochs, log)
            preds.append(pr(np.r_[ica, ite]))
        P = np.exp(np.mean(preds, 0)).reshape(-1, len(HB), 2, len(QS))
        p9 = P[..., QS.index(0.9)]
        evaluate(f"D {kind}", p9[:len(ica)], Y[ica], p9[len(ica):], Y[ite], vol[ite], res)
        pin[f"D {kind}"] = pinball(p9[len(ica):], Y[ite], 0.9)
        res[f"D {kind}"]["보정 전 q.9 초과율"] = round(float(np.mean(Y[ite] > p9[len(ica):])), 4)
        res[f"D {kind}"]["보정 전 q.99 초과율"] = round(float(np.mean(Y[ite] > P[len(ica):, ..., 2])), 4)
        log(f"{kind} done {json.dumps(res[f'D {kind}']['목표1%'], ensure_ascii=False)}")
    base = res["B1 GBM(배포판)"]
    for k in list(res):
        for t in TARGETS:
            key = f"목표{t:.0%}"
            res[k][key]["1/MAE 대비 GBM %"] = round(100 * (res[k][key]["평균 1/MAE"] / base[key]["평균 1/MAE"] - 1), 2)
    res["q0.9 핀볼(검증, MAE %)"] = {k: round(v, 5) for k, v in pin.items()}
    (out_dir / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(json.dumps(res, ensure_ascii=False, indent=1))


def aci_path(r, lag, m0, tau, gamma):
    """ACI(Gibbs·Candès) 지연 피드백판. r: 시간순 비율 y/p (한 보유·방향). 봉 s 의 결과는 s+lag 봉에 확정되므로
    봉 t 에선 s ≤ t−lag 만 반영한다. log m ← log m + γ(초과 − τ). 반환: 각 봉에 쓴 m."""
    lm_hist = np.empty(len(r)); lm = np.log(m0)
    for t in range(len(r)):
        s = t - lag
        if s >= 0:
            lm += gamma * (float(r[s] > np.exp(lm_hist[s])) - tau)
        lm_hist[t] = lm
    return np.exp(lm_hist)


def tail(out_dir: pathlib.Path):
    """2차(문헌 처방): GBM q0.9 위에 ① 정적 배수(현행) ② EVT(GPD) 꼬리 접합 ③ ACI 온라인 보정 ④ 월별 재보정.
    목적 = 0.1%·1% 꼬리 초과율이 목표에 맞고 시간적으로 안정한가(월별) + 날카로움(평균 1/안전MAE)."""
    from scipy.stats import genpareto
    out_dir.mkdir(parents=True, exist_ok=True)
    ts, F, Y, S = load()
    tsv = ts.to_numpy()
    valid = np.isfinite(F).all(1) & np.isfinite(Y).all((1, 2)) & (np.arange(len(F)) >= 600)
    itr = np.flatnonzero(valid & (tsv >= np.datetime64("2022-01-01")) & (tsv <= np.datetime64(TRAIN_END)))
    ica = np.flatnonzero(valid & (tsv > np.datetime64(TRAIN_END)) & (tsv <= np.datetime64(CALIB_END)))
    ite = np.flatnonzero(valid & (tsv > np.datetime64(CALIB_END)))
    p = fit_gbm(F, Y, itr)
    rc, rt = Y[ica] / p[ica], Y[ite] / p[ite]
    vol = np.exp(F[ite, svm.FEATURES.index("rv48")]); top = vol >= np.quantile(vol, 0.9)
    month = pd.DatetimeIndex(tsv[ite]).to_period("M").astype(str).to_numpy()
    res = {"검증 기간": f"{str(tsv[ite[0]])[:10]}~{str(tsv[ite[-1]])[:10]}", "n": len(ite)}
    for tau in (0.01, 0.001):
        M = {}
        M["① 정적 경험분위(현행)"] = np.broadcast_to(np.quantile(rc, 1 - tau, axis=0), rt.shape)
        u = np.quantile(rc, 0.95, axis=0); mg = np.empty_like(u)
        for k in range(len(HB)):
            for s in (0, 1):
                e = rc[:, k, s][rc[:, k, s] > u[k, s]] - u[k, s]
                c, _, sc = genpareto.fit(e, floc=0)
                mg[k, s] = u[k, s] + genpareto.ppf(1 - tau / 0.05, c, 0, sc)
        M["② EVT 꼬리 접합"] = np.broadcast_to(mg, rt.shape)
        for g in (0.002, 0.01):
            mm = np.empty_like(rt)
            for k, h in enumerate(HB):
                for s in (0, 1):
                    mm[:, k, s] = aci_path(rt[:, k, s], h, M["① 정적 경험분위(현행)"][0, k, s], tau, g)
            M[f"③ ACI γ={g}"] = mm
        # ④ 월별 재보정: 매달 초, 결과가 확정된 직전 6개월(보정+검증 누적)의 경험분위
        allr = np.concatenate([rc, rt]); allt = np.r_[tsv[ica], tsv[ite]]
        mm = np.empty_like(rt)
        for mo in np.unique(month):
            start = np.datetime64(pd.Period(mo).start_time)
            for k, h in enumerate(HB):
                known = (allt < start - np.timedelta64(5 * h, "m")) & (allt >= start - np.timedelta64(183, "D"))
                for s in (0, 1):
                    mm[month == mo, k, s] = np.quantile(allr[known, k, s], 1 - tau)
        M["④ 월별 재보정(6개월)"] = mm
        ref = None
        for name, m in M.items():
            ex = rt > m
            mon = pd.Series(ex.mean((1, 2))).groupby(month).mean()
            sharp = float(np.mean(1 / (p[ite] * m)))
            ref = ref or sharp
            res[f"τ={tau:.1%} {name}"] = {
                "초과율": round(float(ex.mean()), 5), "변동성 상위10%": round(float(ex[top].mean()), 5),
                "보유별": [round(float(ex[:, k].mean()), 5) for k in range(len(HB))],
                "월별 최대": round(float(mon.max()), 5), "월 중 목표 2배 초과 비율": round(float((mon > 2 * tau).mean()), 3),
                "1/MAE 대비 ① %": round(100 * (sharp / ref - 1), 2)}
    (out_dir / "tail.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


def selftest():
    y = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]])   # [n=4, 보유 1, 방향 2]
    m = calib_mult(np.ones_like(y), y, 0.25)
    assert np.all((y > m).mean(0) <= 0.25 + 1e-9), m
    assert abs(pinball(np.array([1.0]), np.array([2.0]), 0.9) - 0.9) < 1e-12
    assert abs(pinball(np.array([3.0]), np.array([2.0]), 0.9) - 0.1) < 1e-12
    # ACI: 초과가 계속되면 배수가 커지고, 지연 lag 전 결과는 쓰지 않는다
    r = np.full(200, 5.0)
    m = aci_path(r, 10, 1.0, 0.01, 0.05)
    assert m[9] == 1.0 and m[10] > 1.0 and m[-1] > 5.0
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "tmp/risk_dl_20261012")
    ap.add_argument("--tail", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else (tail(a.out) if a.tail else run(a.out, a.epochs))
