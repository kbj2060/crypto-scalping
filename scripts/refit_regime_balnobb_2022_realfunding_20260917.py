#!/usr/bin/env python3
"""B단계 — balnobb 을 2022-01~2023-12 에 재적합. (2026-09-17)

왜: Omega 부모를 2022~ 최장창으로 재학습하려면 그 구간의 레짐 6열이 필요한데, 배포 balnobb 은
2024-01~2026-06 에 적합돼 있어 2022~23 에 쓰면 **미래를 본 모델이 과거를 판정**하는 꼴이 된다.
같은 라벨·같은 config 로 구간별 적합본을 만들어 walk-forward 로 잇는다.

🔴결정적 검증: 재적합본(2022~23만 봄)과 배포본(2024~26 봄)을 **둘 다 2024 에 돌려** 예측
일치율을 낸다. 기준선 = 2026-09-10 기록의 s12k3↔balnobb **80.3%**(서로 **다른 라벨**인데도).
같은 라벨인 우리 두 판이 그보다 낮으면 구간 차이가 라벨 차이보다 크다는 뜻이고, 그건 장기창
재학습 자체에 대한 경고다.

B2 판 (2026-09-17 재실행): 입력이 **펀딩 복구본** 프레임으로 바뀌었다.
옛 프레임은 2022~2024 펀딩이 전량 중앙값이라 파생 9열이 같이 상수였다(D단계 §3).
모델 config·라벨식·검증 절차는 한 글자도 바꾸지 않았다 -- 바뀐 건 입력뿐이다.
"""
import os, sys, json, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT = Path(os.environ.get("ZEUS_ROOT") or ("/home/llewyn/crypto-scalping" if Path("/home/llewyn/crypto-scalping").exists() else Path.home()/"crypto-scalping")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
# 🔴원본 라벨 함수의 ADX 폴백을 그대로 쓴다. 첫 실행에서 이걸 빼고 fillna(0) 로 때웠더니
#   adx 가 전부 0 -> `adx < weak_adx_max(12)` 가 항상 참 -> **라벨이 100% chop** 이 됐다.
# `_adx` 만 쓰는데 그 모듈이 최상단에서 mamba_ssm 을 import 한다(GPU 전용, dev 에 없음).
# _adx 는 순수 pandas/numpy 라 Mamba 와 무관하므로 import 만 통과시킨다 -- 코드는 복사하지 않는다.
import importlib.util as _ilu, types as _types  # noqa: E402
if _ilu.find_spec("mamba_ssm") is None:
    _stub = _types.ModuleType("mamba_ssm"); _stub.Mamba = None
    sys.modules.setdefault("mamba_ssm", _stub)
from train_regime3_hmm_mamba_20260529 import _adx  # noqa: E402
OUT = ROOT/"tmp/omega461_longwindow_20260917"
SRC = OUT/"features_136_2022_2026_realfunding.parquet"

art = joblib.load(ROOT/"tmp/eth_regime_balnobb_20260910/model.joblib")
FC, MED, CLS, CFG = art["feature_cols"], art["feature_medians"], art["classes"], art["config"]
LS = art["label_spec"]["config"]

# ── A단계 산출물 검증 (통과 못 하면 여기서 멈춘다) ──
assert SRC.exists(), f"A단계 산출물 없음: {SRC}"
F = pd.read_parquet(SRC)
F["timestamp"] = pd.to_datetime(F["timestamp"])
miss = [c for c in FC if c not in F.columns]
assert not miss, f"136 열 결손: {miss}"
nan = int(F[FC].isna().sum().sum())
assert nan == 0, f"136 열에 NaN {nan}개 잔존"
print(f"✅A단계 검증: {len(F):,}행 · 136/136 · NaN 0 · {F.timestamp.min()} ~ {F.timestamp.max()}", flush=True)

# ── balnobb 라벨 (결정식, 후행지표만) ──
def balnobb_labels(d):
    close = pd.to_numeric(d["close"], errors="coerce")
    ema21 = close.ewm(span=21, adjust=False).mean()
    slope = ((ema21 - ema21.shift(5)) / (close * 5.0 + 1e-12)).replace([np.inf,-np.inf], np.nan).fillna(0.0).to_numpy()
    adx = pd.to_numeric(d.get("adx_14", pd.Series(np.nan, index=d.index)), errors="coerce")
    if adx.isna().all():                      # 프레임에 adx_14 가 없다 -- 원본과 동일하게 직접 계산
        adx = _adx(pd.to_numeric(d["high"], errors="coerce"),
                   pd.to_numeric(d["low"], errors="coerce"), close)
    adx = adx.fillna(0.0).to_numpy()
    lab = np.full(len(d), 2, dtype=np.int64)          # 2 = chop
    trending = adx >= float(LS["trend_adx_min"])
    lab[trending & (slope >  float(LS["slope_min"]))] = 0   # bull
    lab[trending & (slope < -float(LS["slope_min"]))] = 1   # bear
    lab[adx < float(LS["weak_adx_max"])] = 2                # chop override (BB 항 없음)
    return lab

F["label"] = balnobb_labels(F)
tr = F[(F.timestamp >= "2022-01-01") & (F.timestamp <= "2023-12-31 23:55:00")]
shares = np.bincount(tr.label, minlength=3)/len(tr)
assert shares.min() > 0.02, f"라벨 퇴화: {dict(zip(CLS, shares.round(4)))} -- 여기서 멈춘다"
print(f"학습 {len(tr):,}봉 · 라벨 비중 {dict(zip(CLS, np.bincount(tr.label, minlength=3)/len(tr)))}", flush=True)

clf = HistGradientBoostingClassifier(
    max_depth=CFG["max_depth"], learning_rate=CFG["learning_rate"],
    max_iter=CFG["max_iter"], l2_regularization=CFG["l2_regularization"], random_state=20260917)
clf.fit(pd.DataFrame(tr[FC].to_numpy(np.float64), columns=FC), tr.label.to_numpy())
print("적합 완료", flush=True)

# ── 결정적 검증: 2024 구간에서 두 모델 나란히 ──
ev = F[(F.timestamp >= "2024-01-01") & (F.timestamp <= "2024-12-31 23:55:00")]
X = pd.DataFrame(ev[FC].to_numpy(np.float64), columns=FC)
p_new, p_old = clf.predict(X), art["model"].predict(X)
agree = float((p_new == p_old).mean())
print(f"\n=== 2024 구간 {len(ev):,}봉 — 재적합본 vs 배포본 ===")
print(f"  ⭐예측 일치율 {agree*100:.2f}%   (기준선: 09-10 s12k3↔balnobb **다른 라벨** 80.3%)")
for i,c in enumerate(CLS):
    print(f"    {c:5s} 재적합 {float((p_new==i).mean())*100:5.1f}%  배포본 {float((p_old==i).mean())*100:5.1f}%")
acc_new = float((p_new == ev.label.to_numpy()).mean()); acc_old = float((p_old == ev.label.to_numpy()).mean())
print(f"  결정적 라벨 대비 정확도: 재적합 {acc_new*100:.2f}% (표본외) · 배포본 {acc_old*100:.2f}% (표본내)")

joblib.dump({"model_id":"eth_regime_balnobb_refit2022_realfunding_20260917", "classes":CLS,
             "feature_cols":FC, "feature_medians":MED, "model":clf, "config":CFG,
             "train_range":"2022-01-01T00:00:00 ~ 2023-12-31T23:55:00",
             "serving_debounce_k":0, "label_spec":art["label_spec"],
             "notes":{"purpose":"walk-forward regime for Omega4.6.1 long-window retrain",
                      "agreement_with_deployed_on_2024":agree,
                      "acc_vs_deterministic_2024":{"refit_oos":acc_new,"deployed_insample":acc_old}}},
            OUT/"regime_balnobb_refit2022_realfunding.joblib")
print(f"\n저장 {OUT/'regime_balnobb_refit2022_realfunding.joblib'}")
