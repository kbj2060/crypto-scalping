#!/usr/bin/env python3
"""경보 신호등 3개(단변량 임계) → **모델**로 올릴 수 있는가 (2026-09-11, 사용자 요청).

배포된 경보는 압축 봉에서 z 임계 q99 를 넘는지 보는 **단변량 규칙 3개**다. 상호작용을 못 본다
— "체결은 급등했는데 거래대금은 안 늘었다"와 "둘 다 늘었다"를 구분하지 못한다.
데이터가 전부 kline 이므로 TabPFN 으로 같은 타깃을 학습시킬 수 있다.

타깃  압축 봉(volexp<0.70) 중 **앞 2시간(24봉) 실현변동성 상위 5%**.
      fwd_rv(H) = std(lr[t+1 : t+H]) — 앞만 본다. 기저가 지평 무관 5% 로 고정된다.
      ⚠️분위는 **창별**로 낸다: 규칙과 모델이 같은 타깃을 받아야 비교가 성립한다.
      학습 라벨만 TRAIN 분위로 고정(그 임계를 VAL/OOS/FWD 에 그대로 적용한 판도 함께 보고).

세 팔 — 모델 효과와 표본 효과를 가른다(2026-09-09 기록: GBM 프록시는 표본이 상한을 넘으면
        개선폭이 소멸했다. 이 분리를 안 하면 "TabPFN 이 졌다"가 "표본이 작아 졌다"와 안 갈린다):
    (1) HGB(전체)      학습셋 전부
    (2) HGB(부분표집)   TabPFN 과 **같은 컨텍스트**만        ⭐(1)-(2) = 표본 효과
    (3) TabPFN         in-context, 컨텍스트 상한 강제        ⭐(2)-(3) = 모델 효과

판정  **매칭 커버리지**에서 lift 를 잰다 — 배포 규칙이 q99 이므로 모델도 상위 1% 만 발동시킨다.
      커버리지를 안 맞추면 더 자주 켜는 쪽이 그냥 달라 보인다(2026-09-08 기록).
      기준선: 배포 최고 신호등(체결속도 3봉지속 z2016) · 3등 OR · 무작위 귀무.
      차이가 **시드 폭 안이면 무승부**. 두 창(OOS·FWD) 다 이겨야 승격 후보다.

⚠️서버 GPU 는 대시보드 TabPFN 재적합과 공유된다 — 도는 동안 대시보드가 느려질 수 있다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
KL5 = ROOT / "binance_data" / "klines" / "ETHUSDT" / "ETHUSDT-5m-api.csv"
OUT = ROOT / "tmp" / "eth_breakout_alert_tabpfn_20260911"

H = 24                       # 앞 2시간 — 배포 경보의 최고 지평
COMPRESS = 0.70
TOPQ = 0.95                  # 타깃 = 상위 5%
FIRE = 0.01                  # 매칭 커버리지 = 배포 q99 와 같은 발동률
EXPAND = 1.80                # 전환 확정선 — 카드가 실제로 주장하는 사건(431건을 센 정의)
EMBARGO = 48                 # 창 경계 금지대(봉). 라벨 창 24봉의 2배
# volexp 계열 — 전환 라벨을 쓸 때 순환 점검용으로 통째로 빼본다.
# 라벨 사건이 volexp 로 정의되고 rv12[t+k](k<12)가 rv12[t]와 봉을 공유하기 때문이다.
VOLEXP_FEATS = ("volexp", "volexp_d1", "volexp_d12", "comp_depth")
SEEDS = (11, 22, 33, 44, 55)
WINDOWS = [("TRAIN", "2021-12-01", "2025-08-31"), ("VAL", "2025-09-01", "2025-12-31"),
           ("OOS", "2026-01-01", "2026-03-31"), ("FWD", "2026-04-01", "2026-12-31")]
EVAL_WINS = ("VAL", "OOS", "FWD")
# 배포 신호등 3개 — (표시명, 원시열, z창, 평활봉)
DEPLOYED = [("체결속도 3봉지속", "n", 2016, 3), ("체결속도", "n", 864, 1), ("거래대금", "qv", 2016, 1)]


# ────────────────────────────────────────────────────────────── 패널
def build(kl: Path = KL5, h: int = H) -> pd.DataFrame:
    d = pd.read_csv(kl, usecols=["timestamp", "open", "high", "low", "close", "quote_volume",
                                  "trades", "taker_buy_quote"], parse_dates=["timestamp"])
    d = d.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    op = d.open.to_numpy(float)
    qv = d.quote_volume.to_numpy(float); n = d.trades.to_numpy(float)
    tbq = d.taker_buy_quote.to_numpy(float)

    lr = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(c[0], 1e-12)))
    S = pd.Series(lr)
    rv12, rv288 = S.rolling(12).std(), S.rolling(288).std()
    volexp = (rv12 / rv288).to_numpy()
    comp = (volexp < COMPRESS) & np.isfinite(volexp)

    tr = np.maximum.reduce([hi - lo, np.abs(hi - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))])
    atr = pd.Series(tr).rolling(96).mean().to_numpy() / np.maximum(c, 1e-9)
    ats = qv / np.maximum(n, 1)                                  # 평균 체결 크기
    tbr = np.abs(tbq / np.maximum(qv, 1e-9) - 0.5)               # 테이커 |쏠림|

    F: dict[str, np.ndarray] = {}
    for nm, s in (("n", n), ("qv", qv), ("ats", ats), ("tbr", tbr)):
        ss = pd.Series(s)
        for W in (96, 288, 864, 2016):
            z = ((ss - ss.rolling(W).mean()) / ss.rolling(W).std()).to_numpy()
            F[f"z_{nm}_{W}"] = z
            if W in (864, 2016) and nm in ("n", "qv"):
                # 배포 경보의 "3봉 지속" 아이디어 — 한 봉 튀는 것과 유지되는 것을 가른다
                F[f"z_{nm}_{W}_min3"] = pd.Series(z).rolling(3).min().to_numpy()
    F["volexp"] = volexp
    F["volexp_d1"] = np.r_[0.0, np.diff(volexp)]
    F["volexp_d12"] = volexp - np.r_[np.full(12, np.nan), volexp[:-12]]
    F["comp_depth"] = COMPRESS - volexp                           # 얼마나 깊이 눌렸나
    F["atr_pct"] = atr
    F["atr_rank"] = pd.Series(atr).rolling(288).rank(pct=True).to_numpy()
    F["bbw_rank"] = S.rolling(48).std().rolling(288).rank(pct=True).to_numpy()
    F["range_atr"] = (hi - lo) / np.maximum(c * atr, 1e-12)
    F["body_range"] = np.abs(c - op) / np.maximum(hi - lo, 1e-9)
    F["absret_atr"] = np.abs(lr) / np.maximum(atr, 1e-9)
    # 압축이 몇 봉째인가 — 오래 눌릴수록 터질 때 크다는 통념을 모델이 쓸 수 있게 준다
    grp = (~comp).cumsum()
    F["comp_age"] = pd.Series(comp.astype(int)).groupby(grp).cumsum().to_numpy()
    hh = d.timestamp.dt.hour.to_numpy() + d.timestamp.dt.minute.to_numpy() / 60.0
    F["hod_sin"] = np.sin(2 * np.pi * hh / 24); F["hod_cos"] = np.cos(2 * np.pi * hh / 24)

    out = pd.DataFrame(F)
    out.attrs["features"] = list(F.keys())      # ⭐피쳐는 **명시**한다
    out.attrs["h"] = h
    out.insert(0, "timestamp", d.timestamp)
    out["compressed"] = comp
    # 타깃 원재료: 앞만 본다. rolling(H).std().shift(-H) = std(lr[t+1 : t+H])
    out["fwd_rv"] = S.rolling(h).std().shift(-h).to_numpy()
    # ⭐사용자 지적(2026-09-11): 카드가 주장하는 사건은 RV 분위가 아니라 **전환**이다.
    #   앞 H봉 안에 volexp 가 EXPAND 를 상향 교차하는가. 대리 타깃이 아닌 그 사건 자체.
    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    fwd_cross = pd.Series(cross[::-1]).rolling(h, min_periods=1).max()[::-1].to_numpy()
    out["fwd_trans"] = np.r_[fwd_cross[1:], False].astype(bool)   # (t, t+H] — 자기 봉 제외
    # ⭐라벨 3: **얼마나 멀리 가는가**(2026-09-11 차트 점검에서 추가).
    #   volexp 는 rv12/rv288 **비율**이라 배경 변동성이 이미 높으면 큰 움직임이 와도 안 오른다.
    #   부드러운 추세는 단기 변동성을 안 튀긴다 — 2025-05-19 는 창 안에서 +2% 가는데 라벨 0 이었다.
    #   손실은 «변동성이 튀어서»가 아니라 «반대로 멀리 가서» 난다. 그래서 최대 이탈폭을 잰다.
    logc = np.log(np.maximum(c, 1e-12))
    fmax = pd.Series(logc[::-1]).rolling(h, min_periods=1).max()[::-1].to_numpy()
    fmin = pd.Series(logc[::-1]).rolling(h, min_periods=1).min()[::-1].to_numpy()
    up = np.r_[fmax[1:], np.nan] - logc                     # (t, t+H] 최대 상승
    dn = logc - np.r_[fmin[1:], np.nan]                     # (t, t+H] 최대 하락
    out["fwd_move"] = np.maximum(up, dn) / np.maximum(atr, 1e-9)   # ATR 대비 최대 이탈
    # 🔴ATR 정규화는 **게이트 비교를 오염시킨다** — 저ATR 봉을 고르는 게이트는 분모가 작아
    #   기계적으로 기저가 오른다. 절대 이탈폭(로그수익 기준)도 같이 둔다.
    out["fwd_move_abs"] = np.maximum(up, dn)
    # 배포 규칙 재현용 원시 z (피쳐와 같은 값이지만 이름을 분리해 의도를 남긴다)
    for label, col, W, sm in DEPLOYED:
        key = f"z_{col}_{W}" + ("_min3" if sm > 1 else "")
        out[f"rule::{label}"] = out[key]
    return out


def feature_cols(p: pd.DataFrame) -> list[str]:
    """build() 가 선언한 목록만 쓴다.

    🔴제외 목록으로 유도하지 않는다. 첫 판이 그렇게 했다가 label() 이 나중에 붙인
      `y_fixed`(= fwd_rv >= 임계 = 라벨 그 자체)가 피쳐로 새어 lift 가 이론 최대 20.0 에
      붙었다. 두 HGB 팔이 컨텍스트 148k/10k 인데 소수점까지 같았던 게 신호였다.
    """
    f = p.attrs.get("features")
    assert f, "build() 가 features 를 선언하지 않았다"
    assert not ({"y", "y_fixed", "fwd_rv", "win", "compressed"} & set(f)), f
    return list(f)


def _trail_q(x: np.ndarray, q: float, win: int = 2016) -> np.ndarray:
    """후행 분위. shift(1) 로 자기 봉을 안 본다(배포 규칙과 같은 규약)."""
    return pd.Series(x).rolling(win, min_periods=200).quantile(q).shift(1).to_numpy()


def apply_gate(p: pd.DataFrame, gate: str) -> np.ndarray:
    """«횡보»를 무엇으로 볼 것인가. 현행 volexp<0.70 은 비율 위의 절대 임계라
    배경 변동성이 높으면 큰 움직임이 와도 안 걸린다(2026-09-11 차트에서 확인)."""
    if gate == "rule3":
        # ⭐AFML 메타라벨링: 배포 경보 3종 중 하나라도 발동한 봉만 모집단으로 쓴다.
        #   라벨은 여전히 시장 결과 — «이 발동이 맞았나»를 배우게 된다(순환 없음).
        fire = np.zeros(len(p), bool)
        for label, col, W, sm in DEPLOYED:
            key = f"z_{col}_{W}" + ("_min3" if sm > 1 else "")
            x = p[key].to_numpy()
            fire |= np.isfinite(x) & (x >= _trail_q(x, 0.99))
        return fire
    if gate == "none":
        return np.isfinite(p["atr_pct"].to_numpy())
    if gate == "volexp":
        return p["compressed"].to_numpy()
    if gate == "volexpq":            # 같은 양이지만 **후행 분위** — 레짐 드리프트에 적응
        v = p["volexp"]
        return (v <= v.rolling(2016, min_periods=500).quantile(0.40)).to_numpy()
    col = {"atr": "atr_rank", "bbw": "bbw_rank"}[gate]
    return (p[col] <= 0.40).to_numpy()


def label(p: pd.DataFrame, kind: str = "rv", cand: np.ndarray | None = None) -> pd.DataFrame:
    """kind='rv'  앞 H봉 실현변동성 창별 상위 5% (기저 5% 고정 — 지평 비교용 대리 타깃)
       kind='trans' 앞 H봉 안에 volexp 가 1.80 상향 교차 (카드가 주장하는 **사건**)
       kind='move'  앞 H봉 ATR 대비 **최대 이탈폭** 창별 상위 5% (**손실과 직결되는 것**)
    ⚠️trans 는 기저가 창마다 다르다 — lift 는 그 창 자기 기저로 나눈다."""
    feats = p.attrs.get("features")
    p = p.copy()
    p.attrs["features"] = feats                  # copy 가 attrs 를 잃어도 잃지 않게
    p["win"] = ""
    for nm, a, b in WINDOWS:
        m = (p.timestamp >= a) & (p.timestamp <= b)
        p.loc[m, "win"] = nm
    # 🔴후보는 **게이트가 정한다**. 여기서 p.compressed 를 고정으로 쓰면, 게이트를 바꿨을 때
    #   양성이 압축 봉에만 남아 «압축 봉인가»가 타깃에 새어든다. 2026-09-11 실제 사고:
    #   --gate none 인데 규칙 lift 가 세 창 모두 정확히 0.00 이었고(규칙은 비압축에서 발동),
    #   모델은 압축 판별로 lift 11~18 을 벌었다(순열 1·2위가 atr_pct·comp_age 였다).
    base_cand = p.compressed.to_numpy() if cand is None else np.asarray(cand)
    ok = pd.Series(base_cand, index=p.index) & np.isfinite(p.fwd_rv) & (p.win != "")
    p["y"] = False
    thr_train = np.nan
    if kind == "trans":
        p["y"] = ok & p.fwd_trans
    else:
        col = {"rv": "fwd_rv", "move": "fwd_move", "moveabs": "fwd_move_abs"}[kind]
        ok = ok & np.isfinite(p[col])
        for nm, _, _ in WINDOWS:
            m = ok & (p.win == nm)
            if m.sum() < 500:
                continue
            thr = float(np.nanquantile(p.loc[m, col], TOPQ))
            p.loc[m, "y"] = p.loc[m, col] >= thr
            if nm == "TRAIN":
                thr_train = thr
    p.attrs["thr_train"] = thr_train
    p.attrs["label_kind"] = kind
    return p


# ────────────────────────────────────────────────────────────── 평가
def lift_at(score: np.ndarray, y: np.ndarray, cov: float | None = None) -> tuple[float, int, float]:
    """상위 cov 비율만 발동시켰을 때의 lift. 커버리지를 고정해야 비교가 성립한다.

    ⚠️cov 기본값을 `FIRE` 로 두면 **정의 시점에 묶여** --fire 인자가 조용히 무시된다.
    """
    cov = FIRE if cov is None else cov
    ok = np.isfinite(score)
    if ok.sum() < 200 or y.mean() <= 0:
        return (np.nan, 0, np.nan)
    k = max(int(round(ok.sum() * cov)), 1)
    cut = np.partition(score[ok], -k)[-k]
    fire = ok & (score >= cut)
    prec = float(y[fire].mean())
    return (prec / float(y.mean()), int(fire.sum()), prec)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(score)
    s, yy = score[ok], y[ok]
    if yy.sum() == 0 or yy.sum() == len(yy):
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1 = float(yy.sum()); n0 = len(yy) - n1
    return float((r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def null_lift(y: np.ndarray, rng, cov: float | None = None, B: int = 400) -> tuple[float, float]:
    """무작위로 같은 개수를 발동시킨 귀무 — lift 1 근처지만 표본이 작으면 폭이 넓다."""
    cov = FIRE if cov is None else cov
    k = max(int(round(len(y) * cov)), 1)
    base = y.mean()
    v = [y[rng.choice(len(y), k, replace=False)].mean() / base for _ in range(B)]
    return (float(np.quantile(v, 0.95)), float(np.max(v)))


# ────────────────────────────────────────────────────────────── 모델
def hgb(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, random_state=seed)


def tabpfn(seed, dev, cap, n_est=1):
    """n_est = 피쳐/라벨 순열 앙상블. 1이면 시드 분산이 크다 — 무작위 시드에서 VAL 폭이
    [3.04, 9.24] 로 벌어져 게이트를 못 넘었다(균등 증분 시드로는 [5.54,7.47] 로 좁아 보였다)."""
    from tabpfn import TabPFNClassifier
    # 사전학습 한도는 10k행x100피쳐지만 ignore_pretraining_limits 로 넘길 수 있다.
    # 3070 Ti 는 8GB 라 컨텍스트를 키우면 VRAM 이 먼저 걸린다 — memory_saving_mode 로 완화한다.
    return TabPFNClassifier(device=dev, random_state=seed, n_estimators=n_est,
                            ignore_pretraining_limits=cap > 10000,
                            memory_saving_mode="auto" if cap > 10000 else False)


def sub_idx(nrows: int, cap: int, rng) -> np.ndarray:
    """컨텍스트 상한 부분표집. 시간 순서를 유지한다(최근 편향을 인위로 넣지 않는다)."""
    if nrows <= cap:
        return np.arange(nrows)
    return np.sort(rng.choice(nrows, cap, replace=False))


def main() -> int:
    global FIRE                          # lift_at/null_lift 기본값이 이 상수를 본다
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=10000, help="TabPFN 컨텍스트 상한")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--skip-tabpfn", action="store_true")
    ap.add_argument("--klines", default=str(KL5), help="5분봉 CSV (서버 아카이브가 더 짧다)")
    ap.add_argument("--label", choices=("rv", "trans", "move", "moveabs"), default="rv")
    ap.add_argument("--no-volexp", action="store_true", help="volexp 계열 제외(전환 라벨 순환 점검)")
    ap.add_argument("--random-seeds", type=int, default=0,
                    help=">0 이면 그만큼 **무작위 추출**한다. Seed-Diversity 게이트는 고정 증분을 금지한다")
    ap.add_argument("--walkforward", action="store_true",
                    help="월 단위 재적합. TabPFN 은 in-context 라 재적합이 공짜다 — 배포 규칙의 후행 분위와 같은 적응성")
    ap.add_argument("--wf-context", type=int, default=10000, help="walk-forward 재적합이 볼 직전 압축봉 수")
    ap.add_argument("--perm", action="store_true", help="순열 중요도(최고 팔)")
    ap.add_argument("--eval-stride", type=int, default=1,
                    help="평가행을 k 봉마다 하나씩만. 게이트 없음(전 봉)이면 창당 25~47k 라 "
                         "TabPFN 예측이 감당이 안 된다 — 커버리지 1%%면 stride 3 에서도 발동 80~160건")
    ap.add_argument("--n-est", type=int, default=1, help="TabPFN 앙상블 크기")
    ap.add_argument("--gate", choices=("volexp", "volexpq", "atr", "bbw", "none", "rule3"),
                    default="volexp",
                    help="후보(«횡보») 정의. volexp<0.70 은 **유도된 적 없는 상수**다 — 대안과 대조한다. "
                         "gate 를 바꾸면 라벨 임계도 후보에 따라 달라지므로 --global-thr 로 고정한다")
    ap.add_argument("--fire", type=float, default=FIRE,
                    help="매칭 커버리지. 메타라벨(gate=rule3)은 모집단이 이미 걸러져 있어 "
                         "1%%로 두면 창당 발동이 5~11건뿐이라 판정 불가 — 0.3 정도를 쓴다")
    ap.add_argument("--global-thr", action="store_true",
                    help="라벨 임계를 후보가 아니라 **창 전체 봉**에서 낸다 — 게이트끼리 비교하려면 필수")
    ap.add_argument("--ctx-recent", action="store_true",
                    help="컨텍스트를 TRAIN **최근** cap 행으로. 기본은 전 기간 무작위 추출. "
                         "어느 쪽을 얼릴지가 배포 결정이라 직접 잰다")
    a = ap.parse_args()
    FIRE = a.fire
    OUT.mkdir(parents=True, exist_ok=True)

    raw = build(Path(a.klines))
    gate = apply_gate(raw, a.gate)                 # ⭐라벨보다 **먼저** 후보를 정한다
    p = label(raw, a.label, cand=gate)
    if a.global_thr:
        # 게이트 비교용: 라벨 임계를 후보가 아니라 창 전체에서 낸다(타깃을 게이트와 독립시킨다)
        col = {"rv": "fwd_rv", "move": "fwd_move", "moveabs": "fwd_move_abs"}.get(a.label)
        if col:
            y = np.zeros(len(p), bool)
            for nm, _, _ in WINDOWS:
                m = (p.win == nm).to_numpy() & np.isfinite(p[col]).to_numpy()
                if m.sum() > 500:
                    y |= m & (p[col].to_numpy() >= np.nanquantile(p.loc[m, col], TOPQ))
            p["y"] = y
    ok = pd.Series(gate, index=p.index) & np.isfinite(p.fwd_rv) & (p.win != "")
    feats = [f for f in feature_cols(p) if not (a.no_volexp and f in VOLEXP_FEATS)]
    # 창 경계 금지대: 라벨이 앞 24봉을 보므로 경계 근처 행은 두 창에 걸친다
    bnd = np.zeros(len(p), bool)
    for nm, aa, bb in WINDOWS:
        idx = np.flatnonzero((p.timestamp >= aa) & (p.timestamp <= bb))
        if len(idx):
            bnd[idx[-EMBARGO:]] = True
    use = ok & ~bnd & np.isfinite(p[feats]).all(axis=1)
    if a.eval_stride > 1:      # TRAIN 은 그대로 두고 **평가 창만** 솎는다(학습 표본을 줄이지 않는다)
        keep = (np.arange(len(p)) % a.eval_stride == 0) | (p.win == "TRAIN").to_numpy()
        use = use & keep

    print(f"[패널] {len(p):,}봉 · 게이트«{a.gate}» {int(gate.sum()):,} · 사용 {int(use.sum()):,} "
          f"· 피쳐 {len(feats)}개 · 금지대 {EMBARGO}봉")
    tname = {"trans": "앞 24봉 안에 volexp 1.80 상향교차 (사건)",
             "move": f"앞 {H}봉 ATR 대비 최대 이탈폭 창별 상위 {1-TOPQ:.0%}",
             "moveabs": f"앞 {H}봉 **절대** 최대 이탈폭 창별 상위 {1-TOPQ:.0%} (ATR 혼입 없음)",
             "rv": f"앞 {H}봉 실현변동성 창별 상위 {1-TOPQ:.0%} (대리)"}[a.label]
    print(f"[타깃] {tname}" + ("  · volexp 계열 제외" if a.no_volexp else ""))
    for nm, _, _ in WINDOWS:
        m = use & (p.win == nm)
        if m.sum():
            print(f"  {nm:6s} {int(m.sum()):7,}행 · 기저 {p.loc[m,'y'].mean()*100:5.2f}% "
                  f"· {p.loc[m,'timestamp'].min():%Y-%m-%d}~{p.loc[m,'timestamp'].max():%Y-%m-%d}")

    tr = np.flatnonzero(use & (p.win == "TRAIN"))
    X = p[feats].to_numpy(np.float32)
    rows = []

    # ── 기준선: 배포 규칙 ──────────────────────────────────────
    print(f"\n[기준선] 매칭 커버리지 상위 {FIRE:.0%} · lift = 정밀도 / 기저")
    print(f"{'창':6s} {'기준선':22s} {'lift':>6s} {'발동':>6s} {'정밀도':>7s} {'AUC':>6s}")
    for w in EVAL_WINS:
        m = np.flatnonzero(use & (p.win == w))
        if not len(m):
            continue
        y = p.y.to_numpy()[m]
        rng = np.random.default_rng(7 + hash(w) % 1000)
        q95, mx = null_lift(y, rng)
        arms = {f"규칙 {lb}": p[f"rule::{lb}"].to_numpy()[m] for lb, *_ in DEPLOYED}
        # 3등 OR = 세 z 를 각자 창 내 백분위로 바꿔 최대값 — 커버리지를 맞추려면 순위여야 한다
        pr = np.column_stack([pd.Series(v).rank(pct=True).to_numpy() for v in arms.values()])
        arms["규칙 3등 OR(최대)"] = pr.max(axis=1)
        for nm, s in arms.items():
            lf, k, pc = lift_at(s, y)
            print(f"{w:6s} {nm:22s} {lf:6.2f} {k:6d} {pc*100:6.2f}% {auc(s, y):6.3f}")
            rows.append({"win": w, "arm": nm, "seed": -1, "lift": lf, "fires": k,
                         "prec": pc, "auc": auc(s, y)})
        print(f"{w:6s} {'무작위 귀무 q95 / 최대':22s} {q95:6.2f} {'':6s} {'':7s} (lift 상한 {mx:.2f})")
        rows.append({"win": w, "arm": "귀무 q95", "seed": -1, "lift": q95, "fires": 0,
                     "prec": np.nan, "auc": np.nan})

    # ── 세 팔 ────────────────────────────────────────────────
    if a.random_seeds:
        seeds = tuple(int(x) for x in np.random.default_rng(20260911).choice(10**6, a.random_seeds,
                                                                            replace=False))
    else:
        seeds = SEEDS[:a.seeds]
    arms = [("HGB 전체", "hgb_full"), ("HGB 부분표집", "hgb_sub")]
    if not a.skip_tabpfn:
        arms.append((f"TabPFN(ctx {a.cap//1000}k, n{a.n_est})", "tabpfn"))
    print(f"\n[학습] TRAIN {len(tr):,}행 · 컨텍스트 상한 {a.cap:,} "
          f"({'최근 창' if a.ctx_recent else '전 기간 무작위'}) · 시드 {list(seeds)}")
    ytr = p.y.to_numpy()[tr]
    ens: dict = {}                                    # (팔, 창) -> 시드별 점수 순위
    allrows = np.flatnonzero(use)                     # walk-forward 컨텍스트 후보(창 무관, 시간순)
    yall = p.y.to_numpy()
    for disp, kind in arms:
        for sd in seeds:
            rng = np.random.default_rng(sd)
            mdl = None
            if not a.walkforward:
                if kind == "hgb_full":
                    idx = tr
                elif a.ctx_recent:
                    idx = tr[-a.cap:]                       # 최근 창만
                else:
                    idx = tr[sub_idx(len(tr), a.cap, rng)]  # 전 기간 무작위(시간순 유지)
                mdl = hgb(sd) if kind.startswith("hgb") else tabpfn(sd, a.device, a.cap, a.n_est)
                mdl.fit(X[idx], yall[idx])
            for w in EVAL_WINS:
                m = np.flatnonzero(use & (p.win == w))
                if not len(m):
                    continue
                if a.walkforward:
                    # 월 단위로 끊어, 그 달 시작보다 EMBARGO 봉 이상 앞선 압축봉만 컨텍스트로 쓴다
                    s = np.full(len(m), np.nan)
                    mon = p.timestamp.to_numpy()[m].astype("datetime64[M]")
                    for mm in np.unique(mon):
                        sel = np.flatnonzero(mon == mm)
                        cut = m[sel[0]] - EMBARGO
                        ctx = allrows[allrows <= cut]
                        if len(ctx) < 1000:
                            continue
                        ctx = ctx[-a.wf_context:] if kind != "hgb_full" else ctx
                        md = hgb(sd) if kind.startswith("hgb") else tabpfn(sd, a.device, a.cap, a.n_est)
                        md.fit(X[ctx], yall[ctx])
                        s[sel] = md.predict_proba(X[m[sel]])[:, 1]
                else:
                    s = mdl.predict_proba(X[m])[:, 1]
                y = p.y.to_numpy()[m]
                lf, k, pc = lift_at(s, y)
                rows.append({"win": w, "arm": disp, "seed": int(sd), "lift": lf, "fires": k,
                             "prec": pc, "auc": auc(s, y)})
                ens.setdefault((disp, w), []).append(pd.Series(s).rank(pct=True).to_numpy())
            print(f"  {disp:18s} seed {sd:3d} · ctx {len(idx):,} · "
                  + " · ".join(f"{w} lift {[r for r in rows if r['win']==w and r['arm']==disp and r['seed']==sd][0]['lift']:.2f}"
                               for w in EVAL_WINS if (use & (p.win == w)).sum()), flush=True)

    # 시드 평균 팔 — 확률 대신 **창 내 순위**를 평균한다(시드마다 보정이 달라 확률 평균은 위험)
    for (disp, w), lst in ens.items():
        if len(lst) < 2:
            continue
        m = np.flatnonzero(use & (p.win == w))
        sc = np.mean(np.vstack(lst), axis=0)
        lf, k, pc = lift_at(sc, yall[m])
        rows.append({"win": w, "arm": f"{disp} ⊕시드평균", "seed": -2, "lift": lf, "fires": k,
                     "prec": pc, "auc": auc(sc, yall[m])})

    if a.perm:
        # 순열 중요도 — 배포 경보 3종을 모델이 **실제로** 쓰는지 본다.
        # 피쳐를 창 안에서 섞고 매칭 커버리지 lift 가 얼마나 내려가는지로 잰다.
        sd = seeds[0]
        rng = np.random.default_rng(sd)
        idx = tr[sub_idx(len(tr), a.cap, rng)]
        mdl = tabpfn(sd, a.device, a.cap, a.n_est) if not a.skip_tabpfn else hgb(sd)
        mdl.fit(X[idx], yall[idx])
        RULE_FEATS = {"z_n_2016_min3": "경보1 체결속도3봉지속", "z_n_864": "경보2 체결속도",
                      "z_qv_2016": "경보3 거래대금"}
        print(f"\n[순열 중요도] 시드 {sd} · 피쳐를 섞었을 때 lift 하락폭 (클수록 많이 쓴다)")
        imp = {}
        for w in ("OOS", "FWD"):
            m = np.flatnonzero(use & (p.win == w))
            if not len(m):
                continue
            y = yall[m]
            base = lift_at(mdl.predict_proba(X[m])[:, 1], y)[0]
            for fi, fn in enumerate(feats):
                Xp = X[m].copy()
                Xp[:, fi] = Xp[rng.permutation(len(m)), fi]
                imp.setdefault(fn, {})[w] = base - lift_at(mdl.predict_proba(Xp)[:, 1], y)[0]
            print(f"  {w} 기준 lift {base:.2f}", flush=True)
        iv = pd.DataFrame(imp).T
        iv["평균"] = iv.mean(axis=1)
        iv = iv.sort_values("평균", ascending=False)
        iv.to_csv(OUT / f"perm_{a.label}_{a.gate}.csv")
        print(f"  {'피쳐':22s} {'OOS':>7s} {'FWD':>7s} {'평균':>7s}")
        for fn, rr in iv.head(12).iterrows():
            star = f"  ← {RULE_FEATS[fn]}" if fn in RULE_FEATS else ""
            print(f"  {fn:22s} {rr.get('OOS', np.nan):7.2f} {rr.get('FWD', np.nan):7.2f} "
                  f"{rr['평균']:7.2f}{star}")
        print("  --- 배포 경보 3종 순위 ---")
        for fn, disp in RULE_FEATS.items():
            if fn in iv.index:
                print(f"  {disp:22s} 전체 {len(iv)}개 중 {list(iv.index).index(fn)+1}위 "
                      f"· 평균 하락 {iv.loc[fn,'평균']:.2f}")

    r = pd.DataFrame(rows)
    ftag = a.label + ("_novolexp" if a.no_volexp else "") + ("_wf" if a.walkforward else "") + f"_n{a.n_est}" + ("_recent" if a.ctx_recent else "") + f"_g{a.gate}"
    r.to_csv(OUT / f"arms_{ftag}.csv", index=False)

    # ── 판정 ─────────────────────────────────────────────────
    print(f"\n=== 판정 (매칭 커버리지 {FIRE:.0%} lift · 시드 중앙 [최소, 최대]) ===")
    print(f"{'팔':22s} " + " ".join(f"{w:>22s}" for w in EVAL_WINS))
    best_rule = {}
    for w in EVAL_WINS:
        sub = r[(r.win == w) & r.arm.str.startswith("규칙")]
        if len(sub):
            best_rule[w] = float(sub.lift.max())
    ens_arms = [(f"{d} ⊕시드평균", None) for d, _ in arms]
    for disp, _ in ([("규칙 " + DEPLOYED[0][0], None), ("규칙 3등 OR(최대)", None)]
                    + arms + ens_arms):
        cells = []
        for w in EVAL_WINS:
            sub = r[(r.win == w) & (r.arm == disp)]
            if not len(sub):
                cells.append(f"{'-':>22s}"); continue
            v = sub.lift.to_numpy()
            cells.append(f"{np.median(v):8.2f} [{v.min():5.2f},{v.max():5.2f}]")
        print(f"{disp:22s} " + " ".join(cells))
    print(f"\n최고 규칙 lift: " + " · ".join(f"{w} {best_rule.get(w, float('nan')):.2f}" for w in EVAL_WINS))
    verdict = {}
    for disp, _ in arms + ens_arms:
        wins = []
        for w in EVAL_WINS:
            sub = r[(r.win == w) & (r.arm == disp)]
            if not len(sub) or w not in best_rule:
                continue
            wins.append(bool(sub.lift.min() > best_rule[w]))     # 시드 최악도 이겨야 승
        verdict[disp] = wins
        mark = "✅" if wins and all(wins) else "❌"
        print(f"  {mark} {disp:20s} 최악시드도 최고규칙 초과: " +
              " ".join(f"{w}={'O' if x else 'X'}" for w, x in zip(EVAL_WINS, wins)))
    (OUT / f"verdict_{ftag}.json").write_text(json.dumps(
        {"best_rule": best_rule, "verdict": {k: v for k, v in verdict.items()},
         "cap": a.cap, "seeds": list(seeds), "H": H, "fire": FIRE,
         "label": a.label, "no_volexp": bool(a.no_volexp)}, ensure_ascii=False, indent=2))
    print(f"\n산출물: {OUT}")
    print("⚠️두 창(OOS·FWD) 다 이기고 시드 폭이 안 겹쳐야 승격 후보다. 하나만 이기면 무승부로 읽는다.")
    return 0


def _self_check() -> None:
    """lift/AUC/커버리지 산식이 아는 답을 내는지."""
    rng = np.random.default_rng(0)
    y = np.zeros(10000, bool); y[:500] = True; rng.shuffle(y)          # 기저 5%
    perfect = y.astype(float) + rng.normal(0, 1e-6, len(y))
    lf, k, pc = lift_at(perfect, y, 0.01)
    assert k == 100 and pc == 1.0 and abs(lf - 20.0) < 1e-6, (lf, k, pc)  # 상위1%가 전부 양성 → 20x
    assert auc(perfect, y) > 0.999, auc(perfect, y)
    noise = rng.normal(0, 1, len(y))
    assert 0.45 < auc(noise, y) < 0.55, auc(noise, y)
    q95, mx = null_lift(y, np.random.default_rng(1), 0.01, B=200)
    assert 1.2 < q95 < 4.0, q95                                        # 무작위는 1 근처, 폭만 있다
    lf2, _, _ = lift_at(noise, y, 0.01)
    assert lf2 <= mx + 1e-9, (lf2, mx)                                 # 잡음은 귀무 최대 안
    # 전환 라벨의 창 산술 — (t, t+H] 이고 자기 봉을 포함하면 안 된다
    import pandas as _pd
    for hit in (1, 5, 24):
        v = np.full(60, 0.3); v[30 + hit] = 2.0                        # t=30 기준 hit 봉 뒤에 교차
        cross = (v >= EXPAND) & np.r_[False, v[:-1] < EXPAND]
        fc = _pd.Series(cross[::-1]).rolling(H, min_periods=1).max()[::-1].to_numpy()
        ft = np.r_[fc[1:], False].astype(bool)
        assert ft[30], f"{hit}봉 뒤 교차를 못 잡는다"
    v = np.full(60, 0.3); v[30 + H + 1] = 2.0                          # 창 밖(H+1봉 뒤)
    cross = (v >= EXPAND) & np.r_[False, v[:-1] < EXPAND]
    fc = _pd.Series(cross[::-1]).rolling(H, min_periods=1).max()[::-1].to_numpy()
    assert not np.r_[fc[1:], False].astype(bool)[30], "창 밖 교차를 잡으면 안 된다"
    v = np.full(60, 0.3); v[30] = 2.0                                  # 자기 봉에서 교차
    cross = (v >= EXPAND) & np.r_[False, v[:-1] < EXPAND]
    fc = _pd.Series(cross[::-1]).rolling(H, min_periods=1).max()[::-1].to_numpy()
    assert not np.r_[fc[1:], False].astype(bool)[30], "자기 봉 교차를 포함하면 미래참조가 아니라 동시참조다"
    print("self-check OK  (완전예측 20x · 잡음 AUC~0.5 · 귀무 폭 · 커버리지 정확 · 전환창 (t,t+24])")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
