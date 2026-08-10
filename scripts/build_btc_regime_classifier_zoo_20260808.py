"""Four candidate HMM-replacement regime CLASSIFIERS for BTC 5m, built and charted (2026-08-08).

Every classifier: fit on TRAIN ONLY (<= 2025-08-31), CAUSAL per-bar decode over the full panel,
3 states named bear/chop/bull by train-mean TRAILING 24h log return.  Scored against the
retrospective 4% zigzag oracle (scoring reference only, never an input).

  A) jm        Statistical Jump Model, k=3, lambda=32 -- Nystrup et al. / Shu & Kolm 2024.
               EWM return + EWM downside-deviation at halflives {72,288,864}; coordinate-descent
               fit; causal online DP decode.  (Reused from the 2026-08-08 detector work.)
  B) dc        Directional-Change indicator HMM -- Chen & Tsang 2021.  From causal 4% DC events,
               per-bar features carried from the LAST CONFIRMED pivot: TMV (move/theta), log T
               (bars to complete), R (time-adjusted return), OSV (current overshoot past the last
               extreme, in theta units), and current DC direction.  Fed to the project's 3-state
               sticky Gaussian HMM (same class the retired price/vol HMM used) -> causal filter.
  C) cnn       Supervised 1D CNN -- crypto-regime-CNN family (arXiv 2605.00875).  That paper found
               price-only 128x128 candlestick images with a shallow 4-layer CNN beat GAF/ViT; the
               1D analog used here is a 4-block CNN over a 128-bar window of normalized OHLC (same
               information, ~100x cheaper than rasterizing 230k windows -- deviation logged).
               Target = zigzag wave direction; trained on TRAIN with a 2880-bar purge so no
               training label window reaches VAL.  Decode: p(bull)>0.6 bull, <0.4 bear, else chop.
  D) qcml      Geometric observables -- QCML family (arXiv 2605.17117).  Simplified faithful
               reimplementation of the OBSERVABLES, not the full quantum-cognitive embedding:
               per bar, a rolling-window local covariance of PCA'd multi-scale returns gives
               (i) spectral entropy of the eigenvalue (excitation-energy) weights, (ii) a complex
               state psi = (v1 + i v2)/sqrt2 whose consecutive overlap arg gives the Berry-phase
               (U(1) holonomy) rate, (iii) reduced-subsystem purity Tr(rho_A^2) from psi reshaped
               2x4, (iv) Hamiltonian sensitivity = variance of <psi|V_k|psi> over fixed random
               Hermitian perturbations.  Four observables -> k-means(k=3) fit on train, causal
               nearest-centroid assignment.  (The paper uses the observables as crisis detectors;
               clustering them into 3 states is this project's adaptation, logged as a deviation.)

Outputs: data/research/btc_regime_classifier_zoo_20260808.parquet (per-bar states, all four +
czz4 + oracle), one last-7-day chart per classifier, a combined strip chart, and a scorecard
(oracle agreement / coverage / median run / flips) per window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import RobustScaler, StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_clean_regime_hmm_20260517 import GaussianStateModel  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026_regimeline.csv"
STATES_PATH = ROOT / "data/research/btc_jm_regime_states_20260808.parquet"
OUT_PARQUET = ROOT / "data/research/btc_regime_classifier_zoo_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_classifier_zoo_20260808"
SEED = 903174
THETA = 0.04
CNN_WINDOW = 128
CNN_PURGE = 2880
QCML_WINDOW = 288
QCML_DIM = 8
KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
REGIME_COLORS = {0: C_BEAR, 1: C_CHOP, 2: C_BULL}
INK = "#1F2430"
LABELS = {
    "jm": "A. Jump Model (k3, λ32)",
    "dc": "B. Directional-Change 지표 HMM",
    "cnn": "C. 1D-CNN (zigzag 파동 지도학습)",
    "qcml": "D. Geometric/QCML observables",
}


def name_states(states: np.ndarray, r288: np.ndarray, train_mask: np.ndarray, k: int = 3) -> np.ndarray:
    means = [np.nanmean(r288[train_mask & (states == s)]) if (train_mask & (states == s)).any() else np.nan
             for s in range(k)]
    order = np.argsort(np.nan_to_num(means, nan=0.0))
    remap = {int(order[i]): i for i in range(k)}
    return np.array([remap.get(int(s), 1) for s in states], dtype=np.int8)


# ---------------------------------------------------------------- B) DC indicators
def dc_indicator_features(close: np.ndarray, theta: float = THETA) -> np.ndarray:
    """Per-bar [TMV, log T, R, OSV, dir] carried from the last CONFIRMED directional-change
    pivot (Chen & Tsang).  Strictly causal: a pivot is only booked once the theta reversal
    from the running extreme has actually happened."""
    n = len(close)
    out = np.zeros((n, 5), dtype=np.float64)
    hi_i = lo_i = 0
    up: bool | None = None
    ext_i = 0
    prev_ext_i = 0
    tmv = t_bars = r_val = 0.0
    for t in range(1, n):
        if close[t] > close[hi_i]:
            hi_i = t
        if close[t] < close[lo_i]:
            lo_i = t
        pivot = None
        if up is None:
            if close[t] >= close[lo_i] * (1 + theta):
                up, pivot = True, lo_i
            elif close[t] <= close[hi_i] * (1 - theta):
                up, pivot = False, hi_i
        elif up:
            if close[t] > close[ext_i]:
                ext_i = t
            elif close[t] <= close[ext_i] * (1 - theta):
                pivot = ext_i
                up = False
        else:
            if close[t] < close[ext_i]:
                ext_i = t
            elif close[t] >= close[ext_i] * (1 + theta):
                pivot = ext_i
                up = True
        if pivot is not None:
            # a wave from prev_ext_i to pivot just got confirmed
            tmv = abs(close[pivot] - close[prev_ext_i]) / max(close[prev_ext_i], 1e-12) / theta
            t_bars = float(max(pivot - prev_ext_i, 1))
            r_val = tmv / t_bars * theta
            prev_ext_i = pivot
            ext_i = t
        osv = 0.0 if up is None else (close[t] - close[ext_i]) / max(close[ext_i], 1e-12) / theta
        out[t] = [tmv, np.log1p(t_bars), r_val, osv, 0.0 if up is None else (1.0 if up else -1.0)]
    return out


def build_dc(close, r288, train_mask):
    feats = dc_indicator_features(close)
    valid = np.isfinite(feats).all(axis=1) & (np.arange(len(close)) > 0)
    sc = RobustScaler().fit(feats[train_mask & valid])
    z = np.zeros_like(feats)
    z[valid] = sc.transform(feats[valid])
    hmm = GaussianStateModel(n_states=3, n_iter=50, seed=SEED)
    hmm.fit(z[train_mask & valid])
    st = np.full(len(close), -1, dtype=int)
    st[valid] = np.nanargmax(hmm.filter_proba(z[valid]), axis=1)
    st[~valid] = st[valid][0] if valid.any() else 0
    return name_states(st, r288, train_mask)


# ---------------------------------------------------------------- C) CNN
class PriceCNN(nn.Module):
    def __init__(self, in_ch: int = 4):
        super().__init__()
        ch = [in_ch, 32, 64, 64, 64]
        blocks = []
        for i in range(4):
            blocks += [nn.Conv1d(ch[i], ch[i + 1], kernel_size=5, padding=2),
                       nn.BatchNorm1d(ch[i + 1]), nn.ReLU(), nn.MaxPool1d(2)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * (CNN_WINDOW // 16), 64),
                                  nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(self.body(x))


def cnn_windows(ohlc: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(len(idx), 4, W) windows ending AT each idx, per-window normalized by the window's own
    last close and its return scale -- no cross-window statistics, so no leakage."""
    out = np.empty((len(idx), 4, CNN_WINDOW), dtype=np.float32)
    for j, i in enumerate(idx):
        w = ohlc[i - CNN_WINDOW + 1: i + 1]
        base = w[-1, 3]
        rel = (w / max(base, 1e-12) - 1.0).T
        scale = max(float(np.abs(rel[3]).std()), 1e-6)
        out[j] = rel / scale
    return out


def build_cnn(panel, close, oracle_dir, r288, train_mask, device):
    ohlc = panel[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    n = len(close)
    all_idx = np.arange(CNN_WINDOW - 1, n)
    tr_end = int(np.flatnonzero(train_mask)[-1]) - CNN_PURGE
    tr_idx = all_idx[(all_idx <= tr_end) & (oracle_dir[all_idx] != 0)]
    rng = np.random.default_rng(SEED)
    rng.shuffle(tr_idx)
    y = (oracle_dir[tr_idx] == 1).astype(np.int64)
    print(json.dumps({"cnn_train_rows": int(len(tr_idx)), "bull_frac": round(float(y.mean()), 3)}), flush=True)

    torch.manual_seed(SEED)
    model = PriceCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    bs = 512
    model.train()
    for epoch in range(3):
        perm = rng.permutation(len(tr_idx))
        tot = 0.0
        for b in range(0, len(perm), bs):
            sel = tr_idx[perm[b: b + bs]]
            xb = torch.from_numpy(cnn_windows(ohlc, sel)).to(device)
            yb = torch.from_numpy(y[perm[b: b + bs]]).to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(sel)
        print(json.dumps({"cnn_epoch": epoch, "train_loss": round(tot / len(perm), 4)}), flush=True)

    model.eval()
    p_bull = np.full(n, 0.5, dtype=np.float64)
    with torch.no_grad():
        for b in range(0, len(all_idx), 4096):
            sel = all_idx[b: b + 4096]
            xb = torch.from_numpy(cnn_windows(ohlc, sel)).to(device)
            p_bull[sel] = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
    st = np.where(p_bull > 0.6, 2, np.where(p_bull < 0.4, 0, 1)).astype(int)
    named = name_states(st, r288, train_mask)
    return named, p_bull


# ---------------------------------------------------------------- D) QCML geometric observables
def qcml_observables(close: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    logc = np.log(close)
    lags = [3, 6, 12, 24, 48, 96, 192, 288, 576, 864]
    feats = np.column_stack([np.concatenate([np.full(L, np.nan), logc[L:] - logc[:-L]]) for L in lags])
    valid_f = np.isfinite(feats).all(axis=1)
    sc = StandardScaler().fit(feats[train_mask & valid_f])
    z = np.zeros_like(feats)
    z[valid_f] = sc.transform(feats[valid_f])
    pca = PCA(n_components=QCML_DIM, random_state=SEED).fit(z[train_mask & valid_f])
    y = np.zeros((len(close), QCML_DIM))
    y[valid_f] = pca.transform(z[valid_f])

    rng = np.random.default_rng(SEED)
    Vs = []
    for _ in range(16):
        a = rng.normal(size=(QCML_DIM, QCML_DIM)) + 1j * rng.normal(size=(QCML_DIM, QCML_DIM))
        Vs.append((a + a.conj().T) / 2.0)
    Vs = np.array(Vs)

    n = len(close)
    obs = np.full((n, 4), np.nan)
    psi_prev = None
    df = pd.DataFrame(y)
    # rolling covariance via rolling sums for speed
    for t in range(QCML_WINDOW, n):
        if not valid_f[t]:
            psi_prev = None
            continue
        w = y[t - QCML_WINDOW + 1: t + 1]
        c = np.cov(w, rowvar=False)
        vals, vecs = np.linalg.eigh(c)
        vals = np.clip(vals, 1e-12, None)
        p = vals / vals.sum()
        spec_entropy = float(-(p * np.log(p)).sum())
        v1, v2 = vecs[:, -1], vecs[:, -2]
        psi = (v1 + 1j * v2) / np.sqrt(2.0)
        psi = psi / np.linalg.norm(psi)
        berry = 0.0 if psi_prev is None else float(abs(np.angle(np.vdot(psi_prev, psi))))
        psi_prev = psi
        m = psi.reshape(2, 4)
        rho_a = m @ m.conj().T
        purity = float(np.real(np.trace(rho_a @ rho_a)))
        exp_v = np.array([np.real(np.vdot(psi, V @ psi)) for V in Vs])
        h_sens = float(exp_v.var())
        obs[t] = [berry, spec_entropy, purity, h_sens]
    return obs


def build_qcml(close, r288, train_mask):
    obs = qcml_observables(close, train_mask)
    valid = np.isfinite(obs).all(axis=1)
    sc = StandardScaler().fit(obs[train_mask & valid])
    z = np.zeros_like(obs)
    z[valid] = sc.transform(obs[valid])
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(z[train_mask & valid])
    st = np.full(len(close), -1, dtype=int)
    st[valid] = km.predict(z[valid])
    st[~valid] = 0
    return name_states(st, r288, train_mask), obs


# ---------------------------------------------------------------- scoring & charts
def score(named, oracle_dir, idx):
    det = np.where(named == 2, 1, np.where(named == 0, -1, 0))
    act = det[idx] != 0
    agree = float(np.mean(det[idx][act] == oracle_dir[idx][act])) * 100 if act.any() else np.nan
    runs = [e - s + 1 for s, e, _ in contiguous_runs(named[idx])]
    return {"oracle_agreement_pct": None if not np.isfinite(agree) else round(agree, 1),
            "coverage_pct": round(float(act.mean()) * 100, 1),
            "median_run_bars": float(np.median(runs)), "n_flips": len(runs) - 1}


def week_chart(tag, named, ts, close, idx, sub, out_path, extra_line=None):
    h_ts = ts.to_numpy()[idx]
    has_extra = extra_line is not None
    fig, axes = plt.subplots(2 + int(has_extra), 1, figsize=(15, 6.2 + 1.2 * has_extra), sharex=True,
                             gridspec_kw={"height_ratios": [10, 0.8] + ([2.4] if has_extra else []),
                                          "hspace": 0.08})
    ax = axes[0]
    for s, e, stt in contiguous_runs(named[idx]):
        seg = slice(s, min(e + 2, len(idx)))
        ax.plot(h_ts[seg], close[idx][seg], color=REGIME_COLORS[stt], linewidth=1.3)
    ax.set_title(f"{LABELS[tag]} — 최근 7일   "
                 f"[오라클 일치 {sub['oracle_agreement_pct']}%  커버리지 {sub['coverage_pct']}%  "
                 f"전환 {sub['n_flips']}회  중앙런 {int(sub['median_run_bars'])}bar]",
                 loc="left", fontsize=12, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                       (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper left", frameon=False, fontsize=9, ncol=3)
    strip = axes[1]
    for s, e, stt in contiguous_runs(named[idx]):
        strip.axvspan(h_ts[s], h_ts[min(e + 1, len(idx) - 1)], color=REGIME_COLORS[stt], linewidth=0)
    strip.set_yticks([])
    strip.set_ylabel("regime  ", rotation=0, ha="right", va="center", fontsize=9, color=INK)
    for side in ("top", "right", "left", "bottom"):
        strip.spines[side].set_visible(False)
    if has_extra:
        axl = axes[2]
        for name, series, color in extra_line:
            axl.plot(h_ts, series[idx], linewidth=1.1, color=color, label=name)
        axl.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
        axl.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
        for side in ("top", "right"):
            axl.spines[side].set_visible(False)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--charts-only", action="store_true",
                    help="re-render from the saved zoo parquet instead of refitting")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "open", "high", "low", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    train_mask = (ts <= TRAIN_END).to_numpy()
    r288 = np.full(len(close), np.nan)
    r288[288:] = np.log(close[288:] / close[:-288])
    oracle_dir, _ = zigzag_oracle(close, threshold=THETA)

    zoo = pd.read_parquet(STATES_PATH)[["timestamp", "jm_lam32", "czz4", "hmm"]]
    zoo["timestamp"] = pd.to_datetime(zoo["timestamp"])
    merged = pd.merge_asof(pd.DataFrame({"timestamp": ts}), zoo.sort_values("timestamp"),
                           on="timestamp", direction="backward", tolerance=pd.Timedelta("10min")).fillna(1)
    states = {"jm": merged["jm_lam32"].to_numpy().astype(np.int8)}

    if args.charts_only:
        cached = pd.read_parquet(OUT_PARQUET)
        assert len(cached) == len(panel), "cached zoo parquet does not match the panel"
        for k in ("dc", "cnn", "qcml"):
            states[k] = cached[k].to_numpy().astype(np.int8)
        p_bull = cached["cnn_p_bull"].to_numpy()
        obs = cached[["qcml_berry", "qcml_spec_entropy", "qcml_purity", "qcml_h_sens"]].to_numpy()
        print("loaded cached classifier states", flush=True)
    else:
        print("stage=dc", flush=True)
        states["dc"] = build_dc(close, r288, train_mask)
        print("stage=cnn", flush=True)
        states["cnn"], p_bull = build_cnn(panel, close, oracle_dir, r288, train_mask, device)
        print("stage=qcml", flush=True)
        states["qcml"], obs = build_qcml(close, r288, train_mask)

    windows = {
        "week": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=args.days)).to_numpy()),
        "full": np.arange(len(close)),
        "val_2025Q4": np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()),
        "oos_2026Q1": np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()),
    }
    scorecard = {}
    for wtag, idx in windows.items():
        scorecard[wtag] = {k: score(v, oracle_dir, idx) for k, v in states.items()}
        scorecard[wtag]["czz4 (참고)"] = score(merged["czz4"].to_numpy().astype(np.int8), oracle_dir, idx)
        scorecard[wtag]["old HMM (참고)"] = score(merged["hmm"].to_numpy().astype(np.int8), oracle_dir, idx)
    (OUT_DIR / "scorecard.json").write_text(json.dumps(scorecard, indent=2, ensure_ascii=False))
    print(json.dumps(scorecard["week"], indent=2, ensure_ascii=False), flush=True)

    widx = windows["week"]
    extras = {
        "cnn": [("p(bull)", p_bull, "#2563EB")],
        "qcml": [("berry rate", obs[:, 0], "#7C3AED"), ("spec entropy", obs[:, 1] / 3, "#0E7C66"),
                 ("purity", obs[:, 2], "#D9542B"), ("H sensitivity", obs[:, 3], "#B45309")],
    }
    for tag, named in states.items():
        week_chart(tag, named, ts, close, widx, scorecard["week"][tag],
                   OUT_DIR / f"week_{tag}.png", extra_line=extras.get(tag))

    # combined comparison
    h_ts = ts.to_numpy()[widx]
    order = ["jm", "dc", "cnn", "qcml"]
    fig, axes = plt.subplots(1 + len(order) + 1, 1, figsize=(15, 8), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.8] * (len(order) + 1), "hspace": 0.08})
    axes[0].plot(h_ts, close[widx], color=INK, linewidth=1.1)
    axes[0].set_title("BTC 최근 7일 — HMM 대체 후보 4계열 레짐 분류 비교", loc="left", fontsize=13, color=INK)
    axes[0].grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(False)
    axes[0].legend(handles=[Patch(facecolor=c, label=l) for l, c in
                            (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
                   loc="upper left", frameon=False, fontsize=9, ncol=3)
    strips = [(states[k][widx], LABELS[k].split(". ")[1] + "  ") for k in order]
    strips.append((merged["hmm"].to_numpy().astype(np.int8)[widx], "구 HMM (퇴역)  "))
    for sax, (arr, label) in zip(axes[1:], strips):
        for s, e, stt in contiguous_runs(arr):
            sax.axvspan(h_ts[s], h_ts[min(e + 1, len(widx) - 1)], color=REGIME_COLORS[stt], linewidth=0)
        sax.set_yticks([])
        sax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            sax.spines[side].set_visible(False)
    fig.savefig(OUT_DIR / "week_all.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'week_all.png'}", flush=True)

    out = pd.DataFrame({"timestamp": ts, "close": close, "oracle_dir": oracle_dir,
                        "czz4": merged["czz4"].to_numpy(), "hmm_old": merged["hmm"].to_numpy(),
                        **states, "cnn_p_bull": p_bull})
    for i, nm in enumerate(["qcml_berry", "qcml_spec_entropy", "qcml_purity", "qcml_h_sens"]):
        out[nm] = obs[:, i]
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
