#!/usr/bin/env python3
"""4h 변동성 8라운드 — DeepVol(Moreno-Pino·Zohren 2210.04797 표 1) 구성원 → QX5 vs QX4. 사전 등록 docs/prereg/vol4h_forecast_horserace_20261025.md 8라운드.

    python3 scripts/research_vol4h_deepvol_20261025.py --export                 # 로컬: 4코인 5분 로그수익·타깃·창별 행 색인 → tmp/vol4h_deepvol/dv_data.npz
    python3 scripts/research_vol4h_deepvol_20261025.py --selfcheck              # 같은 창 같은 출력 · 입력 창 인과성 · QLIKE 단위
    python3 scripts/research_vol4h_deepvol_20261025.py --train W1 926908        # 서버 GPU: 창(쉼표)·시드(쉼표) 학습 → dv_pred_<창>_<시드>.npz
    python3 scripts/research_vol4h_deepvol_20261025.py --eval                   # 로컬: QX5 vs QX4 3창 판정 → report_r8.json

--train 은 numpy·torch 만 쓴다(서버에 연구 캐시 없이 dv_data.npz 하나로 돈다).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("DV_OUT", ROOT / "tmp" / "vol4h_deepvol"))   # 서버에선 tmp 복사본으로 돈다(서버 scripts/ 에 미추적 파일을 안 남김)
COINS = ("ETH", "XRP", "BTC", "SOL")
WINS = ("W1", "W2", "W3")
TAG = {"W1": "5W1", "W2": "5W2", "W3": "4"}         # 저장된 QX4 예측 파일 꼬리표
SEEDS = (926908, 874109, 562517, 640597, 141044)
L = 288                                             # 입력 = 직전 하루 5분 수익(논문: 하루 수용장)


# ---------- 모델(논문 표 1) ----------
def build_model():
    import torch, torch.nn as nn

    class Layer(nn.Module):
        def __init__(self, d):
            super().__init__(); self.d = d
            self.f, self.g = nn.Conv1d(32, 64, 3, dilation=d), nn.Conv1d(32, 64, 3, dilation=d)
            self.res, self.skip = nn.Conv1d(64, 32, 1), nn.Conv1d(64, 128, 1)

        def forward(self, x):
            xp = nn.functional.pad(x, (2 * self.d, 0))                         # 인과 패딩(왼쪽만)
            z = torch.tanh(self.f(xp)) * torch.sigmoid(self.g(xp))
            return x + self.res(z), self.skip(z)

    class DeepVol(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(1, 32, 1)
            self.layers = nn.ModuleList([Layer(2 ** k) for _ in range(2) for k in range(6)])
            self.end = nn.Conv1d(128, 64, 1)
            self.att_w, self.att_v = nn.Linear(64, 64), nn.Linear(64, 1, bias=False)   # 덧셈형(Bahdanau) 시점 어텐션
            self.out = nn.Linear(64, 1)

        def forward(self, r):                                                   # r: (B, L)
            x, s = self.inp(r[:, None, :]), 0
            for lay in self.layers:
                x, k = lay(x); s = s + k
            h = torch.relu(self.end(torch.relu(s))).transpose(1, 2)            # (B, L, 64)
            a = torch.softmax(self.att_v(torch.tanh(self.att_w(h))).squeeze(-1), dim=1)
            return self.out((a[..., None] * h).sum(1)).squeeze(-1)

    return DeepVol()


def qlike_t(y, yh):
    e = y - yh
    return (e.exp() - e - 1).mean()


# ---------- 로컬: 내보내기 ----------
def export():
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    import research_vol4h_forecast_horserace_20261025 as R
    bounds = {"W1": R.ROLL["5W1"][:2], "W2": R.ROLL["5W2"][:2], "W3": (R.TR4_START, R.TR4_END)}
    fr, oks, xs = {}, {}, {}
    for s in COINS:
        x, F, y, ok, miss = R.frame(s); fr[s] = (F, y); oks[s] = ok; xs[s] = x
    R.add_mkt(fr); D = {}
    for s in COINS:
        x, (F, y) = xs[s], fr[s]; ts = x.index; ok = oks[s] & np.isfinite(F[R.MKT].to_numpy(float)).all(1)
        r = np.diff(np.log(x.c.values), prepend=np.log(x.c.values[0])); le = ts + pd.Timedelta(minutes=5 * R.H)
        D[f"{s}_r"], D[f"{s}_y"] = r.astype(np.float32), np.nan_to_num(y).astype(np.float32)
        pos = pd.Series(np.arange(len(ts)), index=ts)
        for w in WINS:
            tr0, tr1 = bounds[w]; tr = np.flatnonzero(ok & (ts >= tr0) & (le <= tr1)); tr = tr[tr >= L][::3]
            V = pd.read_parquet(R.OUT / f"pred_val_r{TAG[w]}_{s}.parquet"); T = pd.read_parquet(R.OUT / f"pred_test_r{TAG[w]}_{s}.parquet")
            D[f"{s}_{w}_tr"], D[f"{s}_{w}_va"], D[f"{s}_{w}_te"] = tr, pos[V.index].values, pos[T.index].values
            assert np.allclose(D[f"{s}_y"][D[f"{s}_{w}_va"]], V.y.values, atol=1e-5) and np.allclose(D[f"{s}_y"][D[f"{s}_{w}_te"]], T.y.values, atol=1e-5)
            print(s, w, "학습", len(tr), "검증", len(V), "시험", len(T), flush=True)
    OUT.mkdir(parents=True, exist_ok=True); np.savez_compressed(OUT / "dv_data.npz", **D); print(OUT / "dv_data.npz")


# ---------- 서버 GPU: 학습 ----------
def train(w, seed, Z, max_ep=1000, patience=50):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.25)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    sd = float(np.concatenate([Z[f"{s}_r"][Z[f"{s}_{w}_tr"]] for s in COINS]).std())   # 학습 행 수익 표준편차
    R = {s: torch.tensor(Z[f"{s}_r"] / sd, device=dev) for s in COINS}; Y = {s: torch.tensor(Z[f"{s}_y"], device=dev) for s in COINS}
    off = torch.arange(-L + 1, 1, device=dev)
    tr = np.concatenate([np.c_[np.full(len(Z[f"{s}_{w}_tr"]), j), Z[f"{s}_{w}_tr"]] for j, s in enumerate(COINS)])

    def batch(sel):
        X = torch.cat([R[s][torch.tensor(sel[sel[:, 0] == j, 1], device=dev)[:, None] + off] for j, s in enumerate(COINS)])
        y = torch.cat([Y[s][torch.tensor(sel[sel[:, 0] == j, 1], device=dev)] for j, s in enumerate(COINS)])
        return X, y

    def predict(m, k):
        m.eval(); out = {}
        with torch.no_grad():
            for s in COINS:
                ii = torch.tensor(Z[f"{s}_{w}_{k}"], device=dev)
                out[s] = torch.cat([m(R[s][ii[a:a + 2048, None] + off]) for a in range(0, len(ii), 2048)]).cpu().numpy()
        m.train(); return out

    m = build_model().to(dev)
    with torch.no_grad():
        m.out.bias.fill_(float(np.mean([Z[f"{s}_y"][Z[f"{s}_{w}_tr"]].mean() for s in COINS])))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    yva = {s: Y[s][torch.tensor(Z[f"{s}_{w}_va"], device=dev)] for s in COINS}
    best, best_ep, state, t0 = np.inf, 0, None, time.time()
    for ep in range(max_ep):
        sel = tr[rng.choice(len(tr), 25_600, replace=False)]
        for b in np.array_split(rng.permutation(len(sel)), 50):
            X, y = batch(sel[b]); opt.zero_grad(); loss = qlike_t(y, m(X)); loss.backward(); opt.step()
        pv = predict(m, "va"); vl = float(np.mean([qlike_t(yva[s], torch.tensor(pv[s], device=dev)).item() for s in COINS]))
        if vl < best - 1e-6:
            best, best_ep, state = vl, ep, {k: v.detach().clone() for k, v in m.state_dict().items()}
        if ep - best_ep >= patience:
            break
    m.load_state_dict(state); pv, pt = predict(m, "va"), predict(m, "te")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"dv_pred_{w}_{seed}.npz", **{f"{s}_va": pv[s] for s in COINS}, **{f"{s}_te": pt[s] for s in COINS}, best_ep=best_ep, best_val=best, sd=sd)
    print(f"{w} {seed} 최선 에폭 {best_ep} 검증 QLIKE {best:.4f} · {time.time() - t0:.0f}s · {dev}", flush=True)


def selfcheck():
    import torch
    torch.manual_seed(0); m = build_model().eval(); r = torch.randn(4, L); full = torch.randn(4, L + 50); full[:, :L] = r
    with torch.no_grad():
        assert torch.allclose(m(r), m(full[:, :L])), "같은 창 같은 출력"
        r2 = r.clone(); r2[:, :100] += 5.0                                      # 인과 합성곱: 앞쪽을 바꿔도 출력이 바뀌어야(수용장 안)
        assert not torch.allclose(m(r), m(r2))
    z = np.arange(1000, dtype=np.float32); i = 600; z2 = z.copy(); z2[i + 1:] = -1   # 입력 창 = r[i-L+1..i], i 뒤 값은 안 봄
    assert np.array_equal(z[i + np.arange(-L + 1, 1)], z2[i + np.arange(-L + 1, 1)])
    y = torch.tensor([0.0, 1.0]); assert abs(qlike_t(y, y).item()) < 1e-12 and qlike_t(y, y + 0.5).item() > 0
    print("selfcheck OK")


# ---------- 로컬: 판정 ----------
def evaluate():
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    import research_vol4h_forecast_horserace_20261025 as R
    out, eps = {}, {w: {} for w in WINS}
    for s in COINS:
        days, per, dv_days, seed_d = [], {}, [], {sd: [] for sd in SEEDS}
        for w in WINS:
            V, T = pd.read_parquet(R.OUT / f"pred_val_r{TAG[w]}_{s}.parquet"), pd.read_parquet(R.OUT / f"pred_test_r{TAG[w]}_{s}.parquet")
            P = {sd: np.load(OUT / f"dv_pred_{w}_{sd}.npz") for sd in SEEDS}
            for sd in SEEDS:
                eps[w][sd] = int(P[sd]["best_ep"])
            dvv, dvt = np.mean([P[sd][f"{s}_va"] for sd in SEEDS], 0), np.mean([P[sd][f"{s}_te"] for sd in SEEDS], 0)
            mem = lambda D, p, extra: np.mean([D[f"{p}_M3"], D[f"{p}_M6"], D[f"{p}_PX3"], D[f"{p}_PX6"]] + extra, 0)   # noqa: E731
            pv = {"QX4": V.v_QX4.values, "QX5": mem(V, "v", [dvv]), "DV": dvv}; pt = {"QX4": T.t_QX4.values, "QX5": mem(T, "t", [dvt]), "DV": dvt}
            for sd in SEEDS:
                pv[f"QX5_{sd}"], pt[f"QX5_{sd}"] = mem(V, "v", [P[sd][f"{s}_va"]]), mem(T, "t", [P[sd][f"{s}_te"]])
            Lq = {k: R.qlike(T.y.values, pt[k] + float(np.log(np.mean(np.exp(V.y.values - pv[k]))))) for k in pv}
            g = lambda a: pd.Series(a, index=T.index).groupby(T.index.normalize()).mean()   # noqa: E731
            d = g(Lq["QX5"] - Lq["QX4"]); days.append(d); per[w] = float(d.mean()); dv_days.append(g(Lq["DV"] - Lq["QX4"]))
            for sd in SEEDS:
                seed_d[sd].append(g(Lq[f"QX5_{sd}"] - Lq["QX4"]))
        ser = pd.concat(days).values; lo, hi = R.block_ci(ser, b=5, n=4000); dv = pd.concat(dv_days).values
        out[s] = {"mean": float(ser.mean()), "ci": [float(lo), float(hi)], "per_window": per, "DV_alone_minus_QX4": float(dv.mean()),
                  "seed_QX5_minus_QX4": {str(sd): float(pd.concat(v).mean()) for sd, v in seed_d.items()}}
        print(s, "QX5−QX4 %+.4f [%+.4f,%+.4f]" % (ser.mean(), lo, hi), {k: round(v, 4) for k, v in per.items()}, "DV단독−QX4 %+.4f" % dv.mean(), flush=True)
    e = out["ETH"]; out["best_epochs"] = eps
    out["verdict"] = {"ETH_win": e["ci"][1] < 0 and sum(v < 0 for v in e["per_window"].values()) >= 2,
                      "replication": sum(out[s]["ci"][1] < 0 for s in ("XRP", "BTC", "SOL")),
                      "seeds_improving_ETH": sum(v < 0 for v in e["seed_QX5_minus_QX4"].values())}
    (OUT / "report_r8.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps(out["verdict"]), json.dumps(eps))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--export" in a:
        export()
    elif "--selfcheck" in a:
        selfcheck()
    elif "--train" in a:
        i = a.index("--train"); Z = dict(np.load(OUT / "dv_data.npz"))
        for w in a[i + 1].split(","):
            for sd in a[i + 2].split(","):
                train(w, int(sd), Z)
    elif "--eval" in a:
        evaluate()
    else:
        print(__doc__)
