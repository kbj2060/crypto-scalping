"""**Step 0 — 분위 기반 «지금 넣을까» 규칙** (2026-09-14, 사용자 요청 A안).

신경망을 짜기 전에, **이미 검증된 분위 모델**만으로 오라클 천장의 몇 %를 잡는지 본다.
못 잡으면 신경망도 못 잡는다(정보가 없는 것). 잡으면 신경망이 정밀도를 올릴 자리가 있다.

## 규칙 — 최적 정지의 두 조건
포지션이 e0 에 열려 있고 남은 칸 j, 남은 봉 r 이다. 매 봉:

    ① 물타기 조건:  현재가 < e0        (롱 기준. 불타기는 산술적으로 지므로 제외)
    ② 바닥 조건:    d_q(r) <= eps * atr_pct

`d_q(r)` 는 **지금부터 남은 r 봉 동안 더 떨어질 폭**의 q 분위 -- MAE 분위 모델이 바로 그 값을
예측한다(타깃이 같다). 더 떨어질 여지가 작으면 «바닥 근처」라 본다.
⭐두 조건이 다른 일을 한다: ①은 «싸게», ②는 «지금」. ①만 쓰면 아무 하락에서나 사고,
②만 쓰면 이익 중에도 산다(불타기).

## 체결 -- 낙관을 걷어낸다
오라클은 원하는 가격에 항상 체결된다고 봤다. 여기서는 모델이 **봉 종가에 결정**하고 그 종가로
체결한다(지정가를 걸어두는 게 아니다). 비용은 지정가 메이커(2.0)가 아니라 **peg(2.95)** 로
잡는다 -- 즉시 결정이라 메이커로 남는다는 보장이 없다.
미체결 칸은 **소멸**한다(만기에 몰아 사지 않는다 -- 그게 더 보수적이다).

⚠️Step 0 이지 전략이 아니다. 통과 기준은 «단일을 의도명목당으로 3/3 이기고 acc 0.50 에서도
이긴다」이고, 노출당은 노출만 줄여도 이기므로 기준에서 뺐다.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_mae_quantile_model_20260913 as maq  # noqa: E402
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402

_KL = pathlib.Path("/home/kbj20/crypto-scalping/binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
assert _KL.exists(), _KL
maq.KLINES = _KL

PEG_BP, EXIT_BP = 2.95, 2.93          # 추가도 peg 로 문다(메이커 가정을 안 쓴다)
SEED = 20260914
WINDOWS = {"VAL(2025-09~12)⚠": ("2025-09-01", "2025-12-31"),
           "OOS(2026-01~03)": ("2026-01-01", "2026-03-31"),
           "TEST(2026-04~09)": ("2026-04-01", "2026-09-10")}


def arm_single(c, idx, sides, w):
    net = np.array([1e4 * (s * (c[i + w] / c[i] - 1.0) - (PEG_BP + EXIT_BP) / 1e4)
                    for i, s in zip(idx, sides)])
    return {"net_bp": float(net.mean()), "expo": 1.0, "worst": float(net.min()),
            "adds": 0.0, "n": len(net)}


def arm_oracle(c, idx, sides, w, k):
    """천장. 창 안에서 진입가보다 유리한 **최고의 k−1 지점**을 미리 안다."""
    net, expo, adds = [], [], []
    for i, s in zip(idx, sides):
        e0, ex = c[i], c[i + w]
        seg = c[i + 1:i + 1 + w]
        px, tt = [e0], [0]
        better = np.where(seg < e0 if s > 0 else seg > e0)[0]
        if len(better):
            rank = better[np.argsort(seg[better] if s > 0 else -seg[better])][:k - 1]
            for t in np.sort(rank):
                px.append(float(seg[t])); tt.append(int(t))
        size = 1.0 / k
        net.append(1e4 * sum(size * (s * (ex / p - 1.0) - (PEG_BP + EXIT_BP) / 1e4) for p in px))
        expo.append(float(sum(size * (w - t) / w for t in tt)))
        adds.append(len(px) - 1)
    return {"net_bp": float(np.mean(net)), "expo": float(np.mean(expo)),
            "worst": float(np.min(net)), "adds": float(np.mean(adds)), "n": len(net)}


def heikin_bollinger(df, *, n=20, m=2.0, recent=12):
    """봉별 «지금 바닥 신호인가» 벡터 4 개. 전부 봉 t 종가까지만 본다(자기 봉 포함 = 저장소 규약).

    HA 전환 = 직전 봉이 음(ha_close<=ha_open)이었다가 이번 봉이 양 -> 하락이 꺾인 자리.
    BB 는 (20, 2) -- `features/engineering.py:272` 와 같은 파라미터."""
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_c)):                      # 재귀라 벡터화 불가
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    bull = ha_c > ha_o
    flip_up = np.r_[False, bull[1:] & ~bull[:-1]]      # 음 -> 양
    flip_dn = np.r_[False, ~bull[1:] & bull[:-1]]
    mid = pd.Series(c).rolling(n, min_periods=n).mean().to_numpy()
    sd = pd.Series(c).rolling(n, min_periods=n).std(ddof=0).to_numpy()
    up, dn = mid + m * sd, mid - m * sd
    touch_lo, touch_hi = c <= dn, c >= up
    back_lo = np.r_[False, touch_lo[:-1] & ~touch_lo[1:]]   # 이탈했다가 복귀
    back_hi = np.r_[False, touch_hi[:-1] & ~touch_hi[1:]]
    # 🔴같은 봉에서 «HA 양전 & 종가가 하단 밖」은 49 만 봉 중 12 회뿐이다 -- 정의상 거의 배타라
    # 교집합 팔은 결과가 아니라 «0 칸」이 나온다. 실제 규칙은 **하단을 찍고 난 뒤** 양전이므로
    # 최근 recent 봉 안에 터치가 있었는지로 본다(현재 봉 포함 = 인과적).
    near_lo = pd.Series(touch_lo).rolling(recent, min_periods=1).max().to_numpy().astype(bool)
    near_hi = pd.Series(touch_hi).rolling(recent, min_periods=1).max().to_numpy().astype(bool)
    return {"HA전환": (flip_up, flip_dn), "BB터치": (touch_lo, touch_hi),
            "BB복귀": (back_lo, back_hi),
            "HA전환&최근BB": (flip_up & near_lo, flip_dn & near_hi)}


def arm_rule(c, idx, sides, w, k, z, eps, use_model=True, gate=None):
    """분위 규칙. `z[e, t]` = 예측 잔여 하락폭을 **그 시점의 전형값으로 나눈 값**.

    🔴첫 판은 문턱을 `eps * atr_pct` 로 잡았는데 d_q(240분)=0.806% 대 eps=2 에서 0.22% 라
    **7배 어긋나** 추가가 0.00 칸이었다(= 1/k 크기 단일 진입). 단위가 다른 두 양을 비교했다.
    ⭐게다가 d_q(r) 은 r 이 줄면 **기계적으로** 작아진다 -- 절대값으로 문턱을 잡으면 그건
    «창 끝에 사라」는 시계 규칙이지 모델이 아니다. 그 시점의 전형값으로 나눠 **«이 시점 치고
    유난히 작은가»** 만 남긴다. `use_model=False` 는 그 정규화를 빼고 시간+물타기 조건만 보는
    **대조군**이다 -- 모델이 시계 이상을 하는지 이 둘의 차이로 판정한다.
    `gate[r, t]` 은 추가 시점을 더 좁히는 **외부 신호**(HA/BB)다. 물타기 조건 위에 얹는다 --
    그래야 «신호가 보태는 것」만 대조군과의 차이로 읽힌다."""
    net, expo, adds = [], [], []
    size = 1.0 / k
    for r, (i, s) in enumerate(zip(idx, sides)):
        e0, ex = c[i], c[i + w]
        px, tt = [e0], [0]
        j = k - 1
        for t in range(w - 1):
            if j <= 0:
                break
            p = c[i + 1 + t]
            cond = (z[r, t] <= eps) if use_model else True
            if gate is not None and not gate[r, t]:
                continue
            if (p < e0 if s > 0 else p > e0) and cond:
                px.append(float(p)); tt.append(t); j -= 1
        net.append(1e4 * sum(size * (s * (ex / p - 1.0) - (PEG_BP + EXIT_BP) / 1e4) for p in px))
        expo.append(float(sum(size * (w - t) / w for t in tt)))
        adds.append(len(px) - 1)
    return {"net_bp": float(np.mean(net)), "expo": float(np.mean(expo)),
            "worst": float(np.min(net)), "adds": float(np.mean(adds)), "n": len(net)}


def _self_check() -> None:
    w, k = 12, 3
    c = np.full(60, 100.0)
    idx, sd = np.array([5]), np.array([1.0])
    a = arm_single(c, idx, sd, w)
    assert abs(a["net_bp"] + PEG_BP + EXIT_BP) < 1e-6, a
    # 평평하면 «현재가 < e0» 가 거짓이라 한 칸도 안 넣는다 -> 노출 1/k
    z0 = np.zeros((1, w))
    r0 = arm_rule(c, idx, sd, w, k, z0, 1.0)
    assert r0["adds"] == 0 and abs(r0["expo"] - 1.0 / k) < 1e-9, r0
    # 하락 경로 + z 가 작으면 두 칸 다 들어간다
    c2 = np.concatenate([np.full(6, 100.0), np.linspace(99.9, 95.0, 54)])
    r1 = arm_rule(c2, idx, sd, w, k, z0, 1.0)
    assert r1["adds"] == k - 1, r1
    # z 가 크면(= 이 시점 치고 아직 많이 남았다) 한 칸도 안 넣는다
    r3 = arm_rule(c2, idx, sd, w, k, np.full((1, w), 9.0), 1.0)
    assert r3["adds"] == 0, r3
    # 🔴대조군은 z 를 무시한다 -- z 가 커도 물타기 조건만 맞으면 넣는다
    r4 = arm_rule(c2, idx, sd, w, k, np.full((1, w), 9.0), 1.0, use_model=False)
    assert r4["adds"] == k - 1, r4
    # 게이트가 전부 거짓이면 한 칸도 안 들어간다 / 전부 참이면 대조군과 같다
    g0 = np.zeros((1, w), bool)
    assert arm_rule(c2, idx, sd, w, k, z0, 1.0, use_model=False, gate=g0)["adds"] == 0
    assert arm_rule(c2, idx, sd, w, k, z0, 1.0, use_model=False,
                    gate=~g0)["adds"] == r4["adds"]
    # HA/BB: 단조 상승이면 HA 는 계속 양이라 **전환이 없다**(첫 봉 제외)
    n = 80
    up = pd.DataFrame({"open": np.arange(n) + 100.0, "high": np.arange(n) + 100.5,
                       "low": np.arange(n) + 99.5, "close": np.arange(n) + 100.2})
    sig = heikin_bollinger(up)
    # 🔴봉 1 의 전환은 **초기화 과도기**다(ha_open[0] 을 (o+c)/2 로 잡아 첫 봉만 음이 된다).
    # 실데이터에서는 워밍업에 묻히지만 자체점검에서는 보이므로 봉 2 부터 본다.
    assert sig["HA전환"][0][2:].sum() == 0, np.where(sig["HA전환"][0])
    # V 자면 바닥 뒤에 롱 전환이 정확히 한 번 생긴다
    v = np.r_[np.linspace(120, 100, 40), np.linspace(100, 120, 40)]
    vd = pd.DataFrame({"open": v, "high": v + 0.5, "low": v - 0.5, "close": v})
    s2 = heikin_bollinger(vd)
    assert s2["HA전환"][0].sum() == 1 and s2["HA전환"][0][38:44].any(), np.where(s2["HA전환"][0])
    # 교집합은 «터치가 최근에 있었나」라 단독 HA 전환보다 많을 수 없고, 터치가 없으면 0 이다
    assert s2["HA전환&최근BB"][0].sum() <= s2["HA전환"][0].sum()
    print("통과 — 비용 항등 · 물타기 조건 · 바닥 조건 · 예산 상한 · 게이트 · HA 전환")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--hold-bars", type=int, default=48)
    ap.add_argument("--acc", default="0.60,0.50")
    ap.add_argument("--every", type=int, default=48)
    ap.add_argument("--q", type=float, default=0.6, help="바닥 조건에 쓸 분위")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check(); return 0

    import joblib
    art = ROOT / "data" / "live" / f"eth_ladder_depth_q{a.k}.joblib"
    assert art.exists(), f"분위 모델이 없다 -- research_entry_ladder_ml_depth 를 먼저 돌려라: {art}"
    model = joblib.load(art)["models"][a.q]

    df = maq._load_klines()
    c = df.close.to_numpy(float)
    sig = heikin_bollinger(df)
    X = svm.build_features(df.ts, c, df.quote_volume.to_numpy(float), df.trades.to_numpy(float),
                           df.high.to_numpy(float), df.low.to_numpy(float))
    atrp = (pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(288, min_periods=200).mean()
            / c).to_numpy()
    ts = df.ts.to_numpy()
    ok = np.isfinite(X.to_numpy(float)).all(1) & np.isfinite(atrp)
    w = a.hold_bars
    print(f"5분봉 {len(df):,} · 보유 {w*5}분 · k={a.k} · 분위 q={a.q} · {a.every*5}분마다 표집")
    print("⚠️추가도 peg(2.95bp) 로 문다 · 미체결 칸은 소멸 · 만기 몰아사기 없음\n")
    rng = np.random.default_rng(SEED)
    for acc in [float(x) for x in a.acc.split(",")]:
        print(f"=== 정확도 {acc} ===")
        print(f"{'창':>18} {'팔':>16} {'의도명목당bp':>12} {'노출':>6} {'추가칸':>7} "
              f"{'최악':>9} {'천장 포착률':>11}")
        for wname, (w0, w1) in WINDOWS.items():
            lo_i = max(int(np.searchsorted(ts, np.datetime64(w0))), svm.WARMUP)
            hi_i = int(np.searchsorted(ts, np.datetime64(w1 + "T23:59:59"))) - w - 1
            if hi_i - lo_i < 1000:
                continue
            idx = np.array([i for i in range(lo_i, hi_i, a.every) if ok[i]])
            truth = np.where(c[idx + w] >= c[idx], 1.0, -1.0)
            sides = np.where(rng.random(len(idx)) < acc, truth, -truth)
            # depth[e, t]: 봉 i+1+t 에서 **남은 r=w-1-t 봉** 동안의 추가 하락폭 q 분위
            rows = []
            for t in range(w - 1):
                f = X.iloc[idx + 1 + t].copy()
                f["log_h"] = np.log(max(1.0, (w - 1 - t) * 5.0))
                f["side"] = sides
                rows.append(f[maq.FEATURES])
            big = pd.concat(rows, ignore_index=True)   # DataFrame 유지 = 컬럼 이름 검증
            pred = model.predict(big).reshape(w - 1, len(idx)).T     # (진입, 봉)
            # 🔴그 시점의 전형값으로 나눈다 -- d_q(r) 의 기계적 축소를 빼고 «유난히 작은가」만.
            med = np.median(pred, axis=0, keepdims=True)
            z = pred / np.maximum(med, 1e-12)
            base = arm_single(c, idx, sides, w)
            orc = arm_oracle(c, idx, sides, w, a.k)
            ctl = arm_rule(c, idx, sides, w, a.k, z, 0.0, use_model=False)
            bars = idx[:, None] + 1 + np.arange(w - 1)[None, :]
            long_side = sides[:, None] > 0
            gates = {nm: np.where(long_side, lg[bars], sh[bars])
                     for nm, (lg, sh) in sig.items()}
            cc = ((ctl["net_bp"] - base["net_bp"]) / (orc["net_bp"] - base["net_bp"])
                  if orc["net_bp"] > base["net_bp"] else float("nan"))
            print(f"{wname:>18} {'대조군(모델없음)':>16} {ctl['net_bp']:>12.2f} "
                  f"{ctl['expo']:>6.2f} {ctl['adds']:>7.2f} {ctl['worst']:>9.1f} "
                  f"{100*cc:>10.1f}%")
            for nm, g in gates.items():
                r = arm_rule(c, idx, sides, w, a.k, z, 0.0, use_model=False, gate=g)
                cap = ((r["net_bp"] - base["net_bp"]) / (orc["net_bp"] - base["net_bp"])
                       if orc["net_bp"] > base["net_bp"] else float("nan"))
                print(f"{wname:>18} {'+' + nm:>16} {r['net_bp']:>12.2f} {r['expo']:>6.2f} "
                      f"{r['adds']:>7.2f} {r['worst']:>9.1f} {100*cap:>10.1f}%")
            for eps in (0.7, 0.85, 1.0, 1.2):
                r = arm_rule(c, idx, sides, w, a.k, z, eps)
                cap = ((r["net_bp"] - base["net_bp"]) / (orc["net_bp"] - base["net_bp"])
                       if orc["net_bp"] > base["net_bp"] else float("nan"))
                print(f"{wname:>18} {'규칙 eps=' + str(eps):>16} {r['net_bp']:>12.2f} "
                      f"{r['expo']:>6.2f} {r['adds']:>7.2f} {r['worst']:>9.1f} {100*cap:>10.1f}%")
            print(f"{wname:>18} {'단일':>16} {base['net_bp']:>12.2f} {base['expo']:>6.2f} "
                  f"{0.0:>7.2f} {base['worst']:>9.1f} {0.0:>10.1f}%")
            print(f"{wname:>18} {'오라클(천장)':>16} {orc['net_bp']:>12.2f} {orc['expo']:>6.2f} "
                  f"{'-':>7} {orc['worst']:>9.1f} {100.0:>10.1f}%")
        print()
    print("⚠️포착률이 0 근처면 **신경망도 못 잡는다**(정보 부재). 단일을 3/3 으로 이겨야 통과다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
