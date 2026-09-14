"""**피쳐 확장 선별** — 새 피쳐군을 gym 라벨 위에서 IC 로 먼저 거른다 (2026-09-14, 사용자 *"피쳐를 좀 더 추가해보자"*).

PPO 를 다시 돌리기 전에, 캐시된 gym 라벨(`direction_labels.npz`, 방향수익 y = ½(r_long−r_short)·1e4)
위에서 **모델 없이** 잰다: 피쳐별 창별 IC · 부호 유지 · anti-stable, 그리고 군 단위 쌍둥이(GBM) 표본외 IC
증분(43열 기준 대비). 손익분기 IC 는 0.098~0.119 이고 현재 최대는 0.030 이다 -- 그 거리를 좁히는 군만 넣는다.

## 후보군 (전부 klines 5분봉 index 에 **인과 정렬** — 봉 t 의 값은 봉 t 마감(t+5분)까지의 정보만)
  deriv   Binance metrics 5분: OI 로그변화 12/48/288 · OI z288 · 상위트레이더 롱숏비(건수/포지션) ·
          전체 롱숏비 · 테이커 매수/매도 비 (로그) + z48                                    ~10열
  tape    1분 체결 테이프 → 5분 합/평균: 불균형 · 대형체결 비중/불균형 · 초대형 불균형 · kyle λ ·
          단위충격 · 부호전환율 · 런 수 · 평균체결크기 · 최대체결 + z288                          ~12열
  book    30초 호가 깊이 → 봉 마감 직전 스냅샷(bd_ok=1): OBI ±0.2/1/2/5% · 총깊이 로그 · OBI1% z288   ~6열
  dvol    Deribit DVOL 시간봉(1h 지연 정렬): 수준 · 24h 변화 · VRP(DVOL−실현) · z288 (2026-08-04 까지)  4열
  horizon klines 긴 지평: 수익 288/576/1440 · 레인지 위치 288/1440 · 20일 고저 거리                  ~6열
  deployed 배포 모델 출력: 안전MAE 240 LONG · 1440/60 비 · 전방변동성 예측(사이징 모델)                3열
  (btc·time 은 env 에 이미 군으로 있다 — 같은 선별에 넣는다)

## 선별 규칙 (실행 전 고정)
군이 상태에 들어가려면 둘 중 하나: (a) 그 군 안에 |TRAIN IC| ≥ 0.05 이고 부호유지 3/3 인 열이 있다,
(b) 쌍둥이 표본외 IC 증분(43열 대비)이 OOS·TEST **둘 다** +0.01 이상. 둘 다 아니면 안 넣는다.
⚠️OOS 한 창만 좋은 군은 이 저장소가 세 번 본 «시기 한정」이라 (b) 가 두 창을 요구한다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402

OUT = P.OUT
EXT_PARQUET = OUT / "ext_features.parquet"
EXT_FAMILIES = OUT / "ext_families.json"
METRICS = ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv"
TAPE = ROOT / "data/research/eth_tape_1m_20260906.parquet"
BOOK = ROOT / "data/research/eth_bookdepth_30s_20260908.parquet"
DVOL = ROOT / "data/derivatives/deribit_dvol/ETH_dvol_hourly.csv"
FIVE = pd.Timedelta("5min")


def z(s: pd.Series, w: int) -> pd.Series:
    r = s.rolling(w, min_periods=max(20, w // 4))
    return (s - r.mean()) / r.std().replace(0, np.nan)


def asof_close(d: pd.DataFrame, other: pd.DataFrame, cols: list[str], lag: pd.Timedelta = pd.Timedelta(0)) -> pd.DataFrame:
    """봉 마감시각(t+5분) 이전에 알 수 있는 마지막 값을 붙인다. lag 는 «그 값이 확정되는 지연»."""
    o = other[["ts"] + cols].copy(); o["ts"] = o["ts"] + lag
    o = o.sort_values("ts")
    key = pd.DataFrame({"ts": d.timestamp + FIVE})
    m = pd.merge_asof(key, o, on="ts", direction="backward", allow_exact_matches=True)
    return m[cols].reset_index(drop=True)


def fam_deriv(d: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_csv(METRICS, parse_dates=["create_time"]).rename(columns={"create_time": "ts"})
    m["oi"] = np.log(m["sum_open_interest"].clip(lower=1))
    for c in ("count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
              "count_long_short_ratio", "sum_taker_long_short_vol_ratio"):
        m[c] = np.log(m[c].clip(lower=1e-6))
    a = asof_close(d, m, ["oi", "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                          "count_long_short_ratio", "sum_taker_long_short_vol_ratio"])
    f = pd.DataFrame(index=d.index)
    for w in (12, 48, 288):
        f[f"dv_oi_chg{w}"] = a["oi"].diff(w)
    f["dv_oi_z288"] = z(a["oi"], 288)
    f["dv_top_ls_n"] = a["count_toptrader_long_short_ratio"]
    f["dv_top_ls_pos"] = a["sum_toptrader_long_short_ratio"]
    f["dv_glob_ls"] = a["count_long_short_ratio"]
    f["dv_taker_ls"] = a["sum_taker_long_short_vol_ratio"]
    f["dv_top_ls_pos_z48"] = z(a["sum_toptrader_long_short_ratio"], 48)
    f["dv_taker_ls_z48"] = z(a["sum_taker_long_short_vol_ratio"], 48)
    return f


def fam_tape(d: pd.DataFrame) -> pd.DataFrame:
    t = pd.read_parquet(TAPE, columns=["ts", "volume", "signed_vol", "lg_share", "lg_imbalance", "xl_imbalance",
                                       "kyle_lambda", "impact_per_vol", "switch_rate", "n_runs", "avg_trade_size", "q_max"])
    t["ts"] = pd.to_datetime(t["ts"])
    t["bin"] = t["ts"].dt.floor("5min")
    g = t.groupby("bin")
    agg = pd.DataFrame({
        "tp_imb": g["signed_vol"].sum() / g["volume"].sum().replace(0, np.nan),
        "tp_lg_share": g["lg_share"].mean(), "tp_lg_imb": g["lg_imbalance"].mean(),
        "tp_xl_imb": g["xl_imbalance"].mean(), "tp_kyle": g["kyle_lambda"].mean(),
        "tp_impact": g["impact_per_vol"].mean(), "tp_switch": g["switch_rate"].mean(),
        "tp_runs": g["n_runs"].sum(), "tp_avg_size": np.log(g["avg_trade_size"].mean().clip(lower=1e-9)),
        "tp_qmax": np.log(g["q_max"].max().clip(lower=1e-9))}).reset_index().rename(columns={"bin": "ts"})
    # 5분 bin [t, t+5) 의 1분봉은 t+5 마감까지 확정 → 봉 t 에 그대로 붙는다(키 = 봉 시작)
    m = d[["timestamp"]].merge(agg, left_on="timestamp", right_on="ts", how="left").drop(columns=["ts", "timestamp"])
    f = m.copy()
    for c in ("tp_imb", "tp_lg_imb", "tp_kyle", "tp_impact", "tp_avg_size", "tp_qmax"):
        f[f"{c}_z288"] = z(m[c], 288)
    return f.reset_index(drop=True)


def fam_book(d: pd.DataFrame) -> pd.DataFrame:
    b = pd.read_parquet(BOOK)
    b["ts"] = pd.to_datetime(b["ts"])
    b = b[b["bd_ok"] == 1]
    for lv in ("0p2", "1p0", "2p0", "5p0"):
        b[f"bk_obi_{lv}"] = (b[f"dm{lv}"] - b[f"d{lv}"]) / (b[f"dm{lv}"] + b[f"d{lv}"]).replace(0, np.nan)
    b["bk_depth1_log"] = np.log((b["dm1p0"] + b["d1p0"]).clip(lower=1))
    cols = ["bk_obi_0p2", "bk_obi_1p0", "bk_obi_2p0", "bk_obi_5p0", "bk_depth1_log"]
    a = asof_close(d, b, cols)
    a["bk_obi_1p0_z288"] = z(a["bk_obi_1p0"], 288)
    a["bk_depth1_z288"] = z(a["bk_depth1_log"], 288)
    return a


def fam_dvol(d: pd.DataFrame) -> pd.DataFrame:
    v = pd.read_csv(DVOL, parse_dates=["timestamp"]).rename(columns={"timestamp": "ts"})
    v["dvol"] = v["close"]
    a = asof_close(d, v, ["dvol"], lag=pd.Timedelta("1h"))     # 시간봉은 마감 뒤에야 안다
    f = pd.DataFrame(index=d.index)
    f["dvol_level"] = np.log(a["dvol"].clip(lower=1e-6))
    f["dvol_chg24"] = f["dvol_level"].diff(288)
    rv_ann = d["rv288"] * np.sqrt(288 * 365) * 100.0          # 실현변동성 연율화(%) — DVOL 단위
    f["dvol_vrp"] = a["dvol"] - rv_ann
    f["dvol_z288"] = z(f["dvol_level"], 288)
    return f


def fam_horizon(d: pd.DataFrame) -> pd.DataFrame:
    c = d["close"]; f = pd.DataFrame(index=d.index)
    for w in (288, 576, 1440):
        f[f"hz_ret{w}"] = np.log(c / c.shift(w))
    for w in (288, 1440):
        hi = d["high"].rolling(w).max(); lo = d["low"].rolling(w).min()
        f[f"hz_pos{w}"] = (c - lo) / (hi - lo).replace(0, np.nan)
    hi20 = d["high"].rolling(5760).max(); lo20 = d["low"].rolling(5760).min()
    f["hz_dist_hi20d"] = np.log(hi20 / c); f["hz_dist_lo20d"] = np.log(c / lo20)
    return f


def fam_deployed(d: pd.DataFrame, sm: dict) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    f["dp_safe_mae240"] = np.log(np.clip(sm[(240, "LONG")], 1e-6, None))
    f["dp_mae_ratio"] = np.log(np.clip(sm[(1440, "LONG")], 1e-6, None)) - np.log(np.clip(sm[(60, "LONG")], 1e-6, None))
    art = svm.load_model()
    if art is not None:
        X = svm.build_features(d.timestamp, d.close.to_numpy(float), d.quote_volume.to_numpy(float),
                               d.trades.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float))
        f["dp_vol_forecast"] = np.log(np.clip(svm.predict_vol(art["models"], X), 1e-9, None))
    return f


def build_ext(d: pd.DataFrame, sm: dict) -> tuple[pd.DataFrame, dict]:
    fams = {"deriv": fam_deriv(d), "tape": fam_tape(d), "book": fam_book(d), "dvol": fam_dvol(d),
            "horizon": fam_horizon(d), "deployed": fam_deployed(d, sm)}
    ext = pd.concat([d[["timestamp"]]] + [v.reset_index(drop=True) for v in fams.values()], axis=1)
    return ext, {k: list(v.columns) for k, v in fams.items()}


def causality_check(d: pd.DataFrame, sm: dict, ext: pd.DataFrame, cut: int) -> None:
    """앞 cut 행만으로 다시 만들어 겹치는 구간이 같아야 한다(미래 행 의존 없음)."""
    sub = d.iloc[:cut].reset_index(drop=True)
    sm_sub = {k: (v[:cut] if isinstance(v, np.ndarray) else v) for k, v in sm.items()}
    e2, _ = build_ext(sub, sm_sub)
    a = ext.iloc[:cut].drop(columns=["timestamp"]).to_numpy(float); b = e2.drop(columns=["timestamp"]).to_numpy(float)
    m = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[m], b[m], rtol=1e-9, atol=1e-9) and (np.isfinite(a) == np.isfinite(b)).all(), "확장 피쳐가 미래 행에 의존한다"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=pathlib.Path, default=OUT / "direction_labels.npz",
                    help="라벨 npz. 지평/손절폭을 바꾼 라벨로 군을 다시 거르려면 여기에 준다")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    d, sm, win, S43, cols43, _ = P.prepare(G.DEFAULT_FAMILIES)
    if EXT_PARQUET.exists() and not a.rebuild:
        ext = pd.read_parquet(EXT_PARQUET); fams = json.load(open(EXT_FAMILIES))
    else:
        print("확장 피쳐 생성 …", flush=True)
        ext, fams = build_ext(d, sm)
        causality_check(d, sm, ext, cut=int(win["VAL"][1]))
        ext.to_parquet(EXT_PARQUET); json.dump(fams, open(EXT_FAMILIES, "w"), ensure_ascii=False, indent=1)
        print(f"  저장 {EXT_PARQUET} · 인과성 절단검사 통과", flush=True)
    # btc·time 은 env 군을 그대로 후보로 올린다
    fams = {**fams, "btc": G.BTC_COLS, "time": G.TIME_COLS}
    for k in ("btc", "time"):
        for c in fams[k]:
            ext[c] = d[c].to_numpy()
    assert (ext.timestamp.to_numpy() == d.timestamp.to_numpy()).all()

    z_ = np.load(a.labels, allow_pickle=True)
    lab = {k: (z_[f"{k}_idx"], z_[f"{k}_y"]) for k in z_["names"]}
    # 표류항 m 이 있으면 베팅 순손익까지 잰다(r_long = m + y). IC 증분만 보면 안 된다 --
    # 09-14 스윕에서 IC +0.016 인데 순손익 −4.67bp 인 셀이 나왔다.
    drift = {k: z_[f"{k}_m"] for k in z_["names"]} if f"OOS_m" in z_ else None
    print(f"라벨: {a.labels.name} · TRAIN {len(lab['TRAIN'][0]):,} · 순손익 읽기값 {'있음' if drift else '없음'}")
    tr_idx, tr_y = lab["TRAIN"]
    norm_cols = [c for k in fams for c in fams[k]]
    normx = G.fit_normalizer(ext, norm_cols, *win["TRAIN"])
    SX = G.build_state(ext, normx)
    col_pos = {c: i for i, c in enumerate(norm_cols)}
    cover = {c: float(np.isfinite(ext[c].to_numpy(float)[tr_idx]).mean()) for c in norm_cols}

    # ── 피쳐별 IC ─────────────────────────────────────────────────────────
    rows = []
    for c in norm_cols:
        j = col_pos[c]; r = {"feature": c, "cover": cover[c]}
        for w in ("TRAIN", "VAL", "OOS", "TEST"):
            idx, y = lab[w]; x = SX[idx, j]
            r[w] = float(spearmanr(x, y).statistic) if np.std(x) > 0 else np.nan
        r["sign_keep"] = int(sum(np.sign(r[w]) == np.sign(r["TRAIN"]) for w in ("VAL", "OOS", "TEST") if np.isfinite(r[w]) and np.isfinite(r["TRAIN"])))
        rows.append(r)
    df = pd.DataFrame(rows)
    fam_of = {c: k for k in fams for c in fams[k]}
    df["family"] = df.feature.map(fam_of)

    # ── 군 단위 쌍둥이 IC 증분 (43열 기준 vs 43열+군) ─────────────────────────
    from sklearn.ensemble import HistGradientBoostingRegressor
    SEEDS3 = P.SEEDS[:3]              # 🔴짝지은 씨드 -- 단일 씨드 비교는 이 저장소에서 두 번 뒤집혔다
    def twin_ic(Xtr, Xev: dict) -> dict:
        ic = {w: [] for w in Xev}; net = {w: [] for w in Xev}
        for sd in SEEDS3:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=200, l2_regularization=1.0, random_state=sd)
            m.fit(Xtr, tr_y)
            for w, X in Xev.items():
                pr = m.predict(X); y = lab[w][1]
                ic[w].append(float(spearmanr(pr, y).statistic))
                if drift is not None:
                    sg = np.sign(pr); sg[sg == 0] = 1.0
                    net[w].append(float((drift[w] + sg * y).mean()))
        out = {w: float(np.mean(v)) for w, v in ic.items()}
        if drift is not None:
            out |= {f"net_{w}": float(np.mean(v)) for w, v in net.items()}
        return out
    ev_w = ("VAL", "OOS", "TEST")
    base = twin_ic(S43[tr_idx], {w: S43[lab[w][0]] for w in ev_w})
    fam_res = {"base43": base}
    for k, cs in fams.items():
        js = [col_pos[c] for c in cs]
        Xtr = np.hstack([S43[tr_idx], SX[tr_idx][:, js]])
        fam_res[k] = twin_ic(Xtr, {w: np.hstack([S43[lab[w][0]], SX[lab[w][0]][:, js]]) for w in ev_w})
    allj = list(range(SX.shape[1]))
    fam_res["all"] = twin_ic(np.hstack([S43[tr_idx], SX[tr_idx]]), {w: np.hstack([S43[lab[w][0]], SX[lab[w][0]]]) for w in ev_w})

    # ── 보고 ─────────────────────────────────────────────────────────────
    print("\n군별 요약 (열 수 · 학습창 커버리지 · 최대|TRAIN IC| · 3/3 부호유지 열 수 · 쌍둥이 IC 증분 VAL/OOS/TEST)")
    verdict = {}
    for k in fams:
        sub = df[df.family == k]
        best = sub.loc[sub["TRAIN"].abs().idxmax()] if sub["TRAIN"].notna().any() else None
        inc = {w: fam_res[k][w] - base[w] for w in ev_w}
        netinc = ({w: fam_res[k][f"net_{w}"] - base[f"net_{w}"] for w in ev_w} if drift is not None else None)
        a_rule = bool(((sub["TRAIN"].abs() >= 0.05) & (sub["sign_keep"] == 3)).any())
        b_rule = bool(inc["OOS"] >= 0.01 and inc["TEST"] >= 0.01)
        verdict[k] = {"n": len(sub), "coverage": float(sub.cover.mean()),
                      "max_abs_train_ic": float(sub["TRAIN"].abs().max()), "best": None if best is None else best.feature,
                      "sign_keep3": int((sub.sign_keep == 3).sum()), "twin_inc": inc, "rule_a": a_rule, "rule_b": b_rule,
                      "net_inc": netinc, "rule_c": (netinc is not None and netinc["OOS"] > 0 and netinc["TEST"] > 0),
                      "select": a_rule or b_rule}
        v = verdict[k]
        print(f"  {k:<9} {v['n']:>3}열 · 커버 {100*v['coverage']:.0f}% · max|IC| {v['max_abs_train_ic']:.4f}({v['best']})"
              f" · 3/3 {v['sign_keep3']} · 증분 {inc['VAL']:+.4f}/{inc['OOS']:+.4f}/{inc['TEST']:+.4f}"
              f" · {'⭐선택' if v['select'] else '탈락'}"
              + ("" if netinc is None else f" · 순손익증분 {netinc['VAL']:+.2f}/{netinc['OOS']:+.2f}/{netinc['TEST']:+.2f}bp"
                                           f" {'✔' if v['rule_c'] else '✘'}"))
    inc_all = {w: fam_res["all"][w] - base[w] for w in ev_w}
    print(f"  {'all':<9} {SX.shape[1]:>3}열 · 증분 {inc_all['VAL']:+.4f}/{inc_all['OOS']:+.4f}/{inc_all['TEST']:+.4f}"
          f"  (기준 43열 쌍둥이 IC {base['VAL']:+.4f}/{base['OOS']:+.4f}/{base['TEST']:+.4f})")
    print("\n새 피쳐 |TRAIN IC| 상위 12")
    top = df.sort_values("TRAIN", key=lambda s: s.abs(), ascending=False).head(12)
    for _, r in top.iterrows():
        print(f"  {r.feature:<22}{r.family:<9}{r.TRAIN:>+8.4f}{r.VAL:>+8.4f}{r.OOS:>+8.4f}{r.TEST:>+8.4f}  {r.sign_keep}/3  커버{100*r.cover:.0f}%")
    sel = [k for k, v in verdict.items() if v["select"]]
    print(f"\n⇒ 선택된 군: {sel if sel else '없음'}")
    rep = {"families": verdict, "twin_ic": fam_res, "all_inc": inc_all, "selected": sel,
           "per_feature": df.to_dict("records")}
    json.dump(rep, open(OUT / "report_feature_expansion.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(f"저장 {OUT / 'report_feature_expansion.json'}")
    # 자체점검
    assert abs(base["OOS"]) < 0.2 and np.isfinite(base["OOS"]), "기준 쌍둥이 IC 가 비정상"
    assert df.cover.between(0, 1).all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
