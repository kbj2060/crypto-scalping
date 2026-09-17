#!/usr/bin/env python3
"""Zeus — **라우팅이 있는 것 vs 없는 것**, 그리고 **롱/숏 전문가 분리** (2026-09-17)

## 왜 학습이 필요한가
추론 절제(`--routing`)는 **이미 라우팅으로 특화되어 학습된** 전문가를 다르게 쓰는 것뿐이라
「애초에 라우팅 없이 하나로 배웠다면?」에 답하지 못한다. 그건 균등 가중으로 새로 학습해야 한다.
(그 절제 결과: 하드 라우팅이 **무작위 배정 5시드 전부보다 낮다** — 건수맞춤 후 −4.68bp.)

## 팔 (사전 지정)
  N0  현행 — 3 레짐 전문가 · 가중 `balanced × route_prob` · 하드 라우팅        모델 3
  N1  **라우팅 없음** — 1 모델 · 가중 `balanced` 만 · 전 봉 동일 모델          모델 1
  N1b 라우팅 없음 · N1 의 3시드 **앙상블** ← ⭐**용량 대조**(추가 학습 0)       모델 3
  N2  **측면 분리** — 롱 전문가(LONG+CASH 행만) + 숏 전문가(SHORT+CASH 행만)   모델 2
      ⚠️측면은 «예측 대상»이라 라우팅 키가 못 된다. 둘 다 전 봉을 채점하고 확률을 비교한다.

🔴**용량 대조가 필수다.** 전문가를 나누면 파라미터가 늘어 「분리가 좋다」와 「모델이 커서
좋다」가 섞인다. N1b(같은 용량·라우팅만 없음)가 그 둘을 가른다.
🔴**건수를 맞춘다.** 확신도가 높을수록 p 가 높으므로(stageN: D1 40.51%→D10 47.53%)
게이트 통과 수가 다르면 「서열」과 「선별성」이 뒤섞인다.

라벨은 배포와 동일한 zigzag(품질=방향, `same_as_direction`). 청산은 Zeus 더블 배리어.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "trading_bot_modules"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_side_skill_decomposition_20260917 as K  # noqa: E402
import train_eval_zeus_baseline_quality_head_20260917 as Z  # noqa: E402  (두-타깃 fit)

# --qlabel=<파일명 조각> : 품질 머리에만 «다른» 타깃을 준다(N4 용). 방향은 zigzag 유지.
# 배포 h48qual 의 구조(quality_mode=quality_label_action)를 balnobb 일관 라인에 옮기는 것이다.
QLABEL = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--qlabel=")),
              "h48cons_deployed_42bp")

SEEDS = (next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--seeds=")), None)
         or ["613042", "27851", "904377"])
SEEDS = [int(x) for x in SEEDS]
FOLDS = [f for f in K.FOLDS if f[0] in ("F1", "F2", "F3", "CAND")]
TARGET_N = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--match=")), 3700))
ARMS = (next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--arms=")), None)
        or ["N0", "N1", "N2"])
# 🔴모르는 팔 이름을 «조용히 무시»하면 빈 결과가 표에서 그냥 빠진다(2026-09-17 실제 발생:
# 서버에 N5 블록이 없는 옛 판이 있었는데 에러 없이 N5 행만 사라졌다).
_KNOWN = {"N0", "N1", "N1b", "N2", "N3", "N4", "N5", "N6", "N0x2", "N7", "N8"}
assert set(ARMS) <= _KNOWN, f"모르는 팔: {sorted(set(ARMS) - _KNOWN)} (스크립트 판이 오래됐을 수 있다)"
# --frame=HMM : 레짐 6열을 HMM 판으로 바꾼다(열 이름은 같으므로 학습 코드는 그대로).
# ⚠️두 프레임을 섞지 않도록 캐시/산출물 이름을 분리한다.
FRAME = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--frame=")), "balnobb")
if FRAME == "HMM":
    E.PARQUET = E.PARQUET.parent / "features_with_regime_2022_2026_HMM.parquet"
    assert E.PARQUET.exists(), f"HMM 프레임 없음: {E.PARQUET}"
# --noregime : balnobb 레짐 6열을 **입력에서 뺀다**(누수 절제, 2026-09-17).
# 🔴그 6열은 구간별 레짐 분류기가 만든 것인데 F1(2023H2)·F2·F3 의 TEST 창이 각 분류기의
#   **학습창 안에** 있다(refit2022 는 2022-01~2023-12, 배포본은 2024-01~2025-09).
#   표본 외인 폴드는 CAND 하나뿐이다. 이 절제가 「F1 우위 = 누수」인지 가른다.
NOREG = "--noregime" in sys.argv
# --purge=<일> : TRAIN 끝을 그만큼 잘라낸다(라벨 경계 누수 절제, 2026-09-17).
# 🔴zigzag 피벗은 **사후 확정**이라 TRAIN 끝 근처 라벨이 TEST 창의 가격으로 정해진다.
#   기록상 「지그재그 라벨은 시장을 한 달 뒤따른다」이므로 침범 폭이 한 달 규모일 수 있다.
PURGE = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--purge=")), 0))
# --dirlabel=<파일명 조각> : **방향·품질 두 머리의 타깃**을 zigzag 대신 그 라벨로 바꾼다.
# ⭐--qlabel(품질만) 과 다르다 -- 이건 두 머리를 «같이» 바꾼다(same_as_direction 유지).
# 계기: 라벨 서열을 누수된 레짐 6열이 있는 구성에서 세웠으므로, 깨끗한 구성에서 다시 잰다.
DIRLABEL = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dirlabel=")), None)
FSUF = (("" if FRAME == "balnobb" else f"_{FRAME}") + ("_noreg" if NOREG else "")
        + (f"_purge{PURGE}" if PURGE else "") + (f"_dl{DIRLABEL}" if DIRLABEL else ""))
OUTJ = E.OUT / f"stageP_routing_side_experts{FSUF}.json"
CACHE = E.OUT / f"stageP_probs{FSUF}.npz"
TPB, SLB, COST = K.BASE_TP * 1e4, K.BASE_SL * 1e4, 1.02


def _qpath(name):
    """라벨을 h48 · 더블배리어 두 디렉터리에서 찾는다."""
    for d in ("zeus_h48_quality_labels_20260917", "zeus_double_barrier_labels_20260917"):
        c = E.OUT.parent / d / f"{name}.parquet"
        if c.exists():
            return c
    raise FileNotFoundError(f"라벨 없음: {name}")


DPATH = _qpath(DIRLABEL) if DIRLABEL else None
QPATH = _qpath(QLABEL) if any(a.startswith("--arms=") and
                              any(x in a for x in ("N4", "N5", "N7")) for a in sys.argv) else None
# 🔴캐시 키에 라벨이 없으면 «다른 라벨로 재학습»해도 옛 캐시가 조용히 재사용된다
# (2026-09-17: N3 가 N2 와 같은 숫자를 낸 사고와 같은 부류). 키에 라벨 이름을 붙인다.
QTAG = f"@{QPATH.stem}" if QPATH is not None else ""
# 🔴N4/N5/N7 은 QPATH «하나»를 공유한다 -- N5 와 N7 을 한 프로세스에 같이 넣으면 둘이
# 같은 라벨로 학습돼 「h48 양두」와 「더블배리어 양두」가 동일 모델이 된다. 갈라서 돌린다.
assert not ({"N5", "N7"} <= set(ARMS)), "N5 와 N7 은 라벨 인자를 공유한다 -- 따로 실행할 것"


def log(*a): print(*a, flush=True)


def gate_score(D, Q):
    da = D.argmax(1)
    qf = np.where(da > 0, Q[np.arange(len(Q)), da], Q[:, 0])
    return da, qf


def realize(te, side, idx):
    hi = pd.to_numeric(te["high"]).to_numpy(np.float64)
    lo = pd.to_numeric(te["low"]).to_numpy(np.float64)
    cl = pd.to_numeric(te["close"]).to_numpy(np.float64)
    r, h, _rs, _rn, _m = K._first_touch_open(idx, side, hi, lo, cl, K.BASE_TP, K.BASE_SL, K.MAXBARS)
    return r * 1e4 - COST, h.astype(float), te.timestamp.dt.floor("D").to_numpy()[idx]


def main() -> int:
    E.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} · seeds={SEEDS} · arms={ARMS} · 건수맞춤 {TARGET_N:,} · "
        f"더블배리어 TP{K.BASE_TP*100:g}%/SL{K.BASE_SL*100:g}%")
    log(f"라벨(N4/N5/N7) = {QPATH}")
    log(f"방향·품질 타깃 = {DPATH or 'zigzag_action (기본)'}")
    df, base_cols = E.load()
    if NOREG:
        from omega4_6_2_source_parent_live import CURRENT_PREFIX
        drop = [c for c in base_cols if c.startswith(CURRENT_PREFIX)]
        assert len(drop) == 6, f"레짐 6열이 아니라 {len(drop)}열: {drop}"
        base_cols = [c for c in base_cols if c not in drop]
        log(f"⭐--noregime: 레짐 6열 제거 → 입력 {len(base_cols)}열 ({drop})")
    cache = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}
    store = {a: [] for a in ("N0", "N1", "N1b", "N2", "N3", "N4", "N5", "N6", "N0x2", "N7", "N8")}

    for name, t0, t1, v0, v1 in FOLDS:
        t1e = (pd.Timestamp(t1) - pd.Timedelta(days=PURGE)).strftime("%Y-%m-%d") if PURGE else t1
        tm = (df.timestamp >= t0) & (df.timestamp <= t1e + " 23:59:59")
        vm = (df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")
        tr, te = df[tm].reset_index(drop=True), df[vm].reset_index(drop=True)
        assert tr.timestamp.max() < te.timestamp.min(), f"{name} TRAIN 이 TEST 를 침범"
        if PURGE:
            gap = (te.timestamp.min() - tr.timestamp.max()).days
            assert gap >= PURGE, f"{name} purge 간격 {gap}일 < {PURGE}일"
        yt = pd.to_numeric(tr["zigzag_action"]).to_numpy(np.int64)
        if DPATH is not None:                      # 두 머리의 타깃을 통째로 교체
            _dl = pd.read_parquet(DPATH, columns=["timestamp", "tb_action"])
            _dl["timestamp"] = pd.to_datetime(_dl["timestamp"])
            yt = (tr[["timestamp"]].merge(_dl, on="timestamp", how="left")
                  .tb_action.fillna(0).to_numpy(np.int64))
            assert len(yt) == len(tr) and np.bincount(yt, minlength=3).min() > 100, "방향 라벨 퇴화"
        rt = tabm._route_probs(tr)
        ev = tabm._route_probs(te).argmax(1)
        xs, scaler = tabm._standardize_fit(tabm._base_input(tr, base_cols))
        xv = tabm._standardize_apply(tabm._base_input(te, base_cols), scaler)
        n = len(tr); split = max(int(n * 0.85), min(n - 1, 512))
        bal = compute_sample_weight("balanced", y=yt).astype(np.float32)
        log(f"\n{'='*92}\n=== {name} TRAIN {t0}~{t1e}{f' (purge {PURGE}일)' if PURGE else ''} {n:,}"
            f" · TEST {v0}~{v1} {len(te):,}\n{'='*92}")

        def fit_get(key, ytr, wtr, rows=None):
            """캐시된 (D,Q) 를 주거나 학습한다. rows 가 있으면 그 부분집합만 학습한다."""
            ck = f"{name}|{key}"
            if ck in cache:
                z = dict(cache[ck].item()); return z["D"], z["Q"]
            sel = np.ones(n, bool) if rows is None else rows
            si = np.where(sel[:split])[0]; sv = np.where(sel[split:])[0]
            # ⭐부분집합 학습이면 클래스 가중치를 «그 부분집합에서» 다시 계산한다.
            # 전체 3클래스 기준 가중치를 쓰면 롱 전문가(SHORT 를 안 봄)의 균형이 어긋난다.
            # 🔴그리고 **전달받은 가중(라우팅)을 버리면 안 된다** -- 버리면 N3 의 레짐 가중이
            # 사라져 세 레짐 모델이 동일해지고 N3 가 N2 로 붕괴한다(2026-09-17 실제 발생).
            if rows is None:
                wi, wv = wtr[:split][si], wtr[split:][sv]
            else:
                cw = compute_sample_weight("balanced", y=ytr[sel]).astype(np.float32)
                w_ = cw * wtr[sel]                      # 부분집합 클래스균형 × 전달 가중(라우팅)
                wi, wv = w_[:len(si)], w_[len(si):]
                assert w_.sum() > 0, "부분집합 가중치 합이 0"
            assert len(wi) == len(si) and len(wv) == len(sv), "부분집합 가중치 길이 불일치"
            m, _ = E.fit_expert(xs[:split][si], ytr[:split][si], wi,
                                xs[split:][sv], ytr[split:][sv], wv,
                                seed=int(key.split("s")[-1]), ei=0, device=device)
            D, Q = E.heads(m, xv, device)
            cache[ck] = np.array({"D": D, "Q": Q}, dtype=object); np.savez(CACHE, **cache)
            return D, Q

        # ── N0: 3 레짐 전문가 · 라우팅 가중 · 하드 라우팅 ──
        if "N0" in ARMS:
            for sd in SEEDS:
                D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
                for ei, en in enumerate(("bull", "bear", "chop")):
                    ck = f"{name}|N0{en}s{sd}"
                    if ck in cache:
                        z = dict(cache[ck].item()); Dd, Qq = z["D"], z["Q"]
                    else:
                        w = bal * rt[:, ei].astype(np.float32)
                        m, _ = E.fit_expert(xs[:split], yt[:split], w[:split],
                                            xs[split:], yt[split:], w[split:],
                                            seed=sd, ei=ei, device=device)
                        Dd, Qq = E.heads(m, xv, device)
                        cache[ck] = np.array({"D": Dd, "Q": Qq}, dtype=object); np.savez(CACHE, **cache)
                    s_ = ev == ei
                    D[s_], Q[s_] = Dd[s_], Qq[s_]
                store["N0"].append((name, te, D, Q)); log(f"  N0/seed {sd} 완료")

        # ── N1: 라우팅 없음, 단일 모델 ──
        if "N1" in ARMS:
            Ds, Qs = [], []
            for sd in SEEDS:
                D, Q = fit_get(f"N1s{sd}", yt, bal)
                Ds.append(D); Qs.append(Q)
                store["N1"].append((name, te, D, Q)); log(f"  N1/seed {sd} 완료")
            store["N1b"].append((name, te, np.mean(Ds, 0), np.mean(Qs, 0)))   # 용량 대조(공짜)

        # ── N4: N0 구조 + 품질 타깃만 h48 (배포 h48qual 을 balnobb 라인에 옮긴 것) ──
        if "N4" in ARMS:
            ql = pd.read_parquet(QPATH, columns=["timestamp", "tb_action"])
            ql["timestamp"] = pd.to_datetime(ql["timestamp"])
            yq_t = (tr[["timestamp"]].merge(ql, on="timestamp", how="left")
                    .tb_action.fillna(0).to_numpy(np.int64))
            assert len(yq_t) == n, "품질 라벨 조인 길이 불일치"
            for sd in SEEDS:
                D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
                for ei in range(3):
                    ck = f"{name}|N4{ei}s{sd}{QTAG}"
                    if ck in cache:
                        z = dict(cache[ck].item()); Dd, Qq = z["D"], z["Q"]
                    else:
                        w = bal * rt[:, ei].astype(np.float32)
                        mm, _ = Z.fit(xs[:split], yt[:split], yq_t[:split], w[:split],
                                      xs[split:], yt[split:], yq_t[split:], w[split:],
                                      seed=sd, ei=ei, device=device)
                        Dd, Qq = E.heads(mm, xv, device)
                        cache[ck] = np.array({"D": Dd, "Q": Qq}, dtype=object); np.savez(CACHE, **cache)
                    m_ = ev == ei
                    D[m_], Q[m_] = Dd[m_], Qq[m_]
                store["N4"].append((name, te, D, Q)); log(f"  N4/seed {sd} 완료")

        # ── N5: 방향·품질 **둘 다 h48** (N0 구조 · same_as_direction 을 h48 라벨로) ──
        # N4(품질만 h48)가 −1.63 으로 실패했다. 이 세션의 반복 패턴은 «두 머리가 같아야
        # 확신도가 서열이 된다»이므로, h48 라벨을 **방향 타깃으로도** 줘서 가른다.
        # 이게 「h48 라벨 자체가 쓸 만한가」와 「타깃을 가른 게 문제였나」를 분리한다.
        if "N5" in ARMS:
            ql5 = pd.read_parquet(QPATH, columns=["timestamp", "tb_action"])
            ql5["timestamp"] = pd.to_datetime(ql5["timestamp"])
            y5 = (tr[["timestamp"]].merge(ql5, on="timestamp", how="left")
                  .tb_action.fillna(0).to_numpy(np.int64))
            assert len(y5) == n and np.bincount(y5, minlength=3).min() > 100, "h48 방향 라벨 퇴화"
            bal5 = compute_sample_weight("balanced", y=y5).astype(np.float32)
            for sd in SEEDS:
                D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
                for ei in range(3):
                    ck = f"{name}|N5{ei}s{sd}{QTAG}"
                    if ck in cache:
                        z = dict(cache[ck].item()); Dd, Qq = z["D"], z["Q"]
                    else:
                        w = bal5 * rt[:, ei].astype(np.float32)
                        mm, _ = E.fit_expert(xs[:split], y5[:split], w[:split],
                                             xs[split:], y5[split:], w[split:],
                                             seed=sd, ei=ei, device=device)
                        Dd, Qq = E.heads(mm, xv, device)
                        cache[ck] = np.array({"D": Dd, "Q": Qq}, dtype=object); np.savez(CACHE, **cache)
                    m_ = ev == ei
                    D[m_], Q[m_] = Dd[m_], Qq[m_]
                store["N5"].append((name, te, D, Q)); log(f"  N5/seed {sd} 완료")

        # ── N7: 방향·품질 **둘 다 더블배리어 라벨** (N5 의 형제 -- 라벨만 다르다) ──
        if "N7" in ARMS:
            ql7 = pd.read_parquet(QPATH, columns=["timestamp", "tb_action"])
            ql7["timestamp"] = pd.to_datetime(ql7["timestamp"])
            y7 = (tr[["timestamp"]].merge(ql7, on="timestamp", how="left")
                  .tb_action.fillna(0).to_numpy(np.int64))
            assert len(y7) == n and np.bincount(y7, minlength=3).min() > 100, "더블배리어 방향 라벨 퇴화"
            bal7 = compute_sample_weight("balanced", y=y7).astype(np.float32)
            for sd in SEEDS:
                D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
                for ei in range(3):
                    ck = f"{name}|N7{ei}s{sd}{QTAG}"
                    if ck in cache:
                        z = dict(cache[ck].item()); Dd, Qq = z["D"], z["Q"]
                    else:
                        w = bal7 * rt[:, ei].astype(np.float32)
                        mm, _ = E.fit_expert(xs[:split], y7[:split], w[:split],
                                             xs[split:], y7[split:], w[split:],
                                             seed=sd, ei=ei, device=device)
                        Dd, Qq = E.heads(mm, xv, device)
                        cache[ck] = np.array({"D": Dd, "Q": Qq}, dtype=object); np.savez(CACHE, **cache)
                    m_ = ev == ei
                    D[m_], Q[m_] = Dd[m_], Qq[m_]
                store["N7"].append((name, te, D, Q)); log(f"  N7/seed {sd} 완료")

        # ── N3: **레짐 × 측면** (3 레짐 × 2 측면 = 6 모델) ──
        # 레짐 가중 학습(라우팅 유지) + 측면별 부분집합. 추론은 레짐 하드 라우팅으로 전문가
        # 쌍을 고르고, 그 안에서 롱/숏 확률을 비교한다(측면은 예측 대상이라 라우팅 키가 못 된다).
        if "N3" in ARMS:
            for sd in SEEDS:
                D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
                for ei in range(3):
                    w = bal * rt[:, ei].astype(np.float32)
                    DL, QL = fit_get(f"N3L{ei}s{sd}", yt, w, rows=(yt != 2))
                    DS, QS = fit_get(f"N3S{ei}s{sd}", yt, w, rows=(yt != 1))
                    d = np.stack([np.minimum(DL[:, 0], DS[:, 0]), DL[:, 1], DS[:, 2]], 1)
                    q = np.stack([np.minimum(QL[:, 0], QS[:, 0]), QL[:, 1], QS[:, 2]], 1)
                    d /= np.maximum(d.sum(1, keepdims=True), 1e-12)
                    q /= np.maximum(q.sum(1, keepdims=True), 1e-12)
                    m_ = ev == ei
                    D[m_], Q[m_] = d[m_], q[m_]
                store["N3"].append((name, te, D, Q)); log(f"  N3/seed {sd} 완료")

        # ── N2: 측면 분리 (LONG+CASH 행 / SHORT+CASH 행) ──
        if "N2" in ARMS:
            for sd in SEEDS:
                DL, QL = fit_get(f"N2Ls{sd}", yt, bal, rows=(yt != 2))
                DS, QS = fit_get(f"N2Ss{sd}", yt, bal, rows=(yt != 1))
                # 둘 다 전 봉 채점 -> 확률 비교. 측면은 예측 대상이라 라우팅 키가 못 된다.
                D = np.stack([np.minimum(DL[:, 0], DS[:, 0]), DL[:, 1], DS[:, 2]], 1)
                D = D / np.maximum(D.sum(1, keepdims=True), 1e-12)
                Q = np.stack([np.minimum(QL[:, 0], QS[:, 0]), QL[:, 1], QS[:, 2]], 1)
                Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)
                store["N2"].append((name, te, D, Q)); log(f"  N2/seed {sd} 완료")

    # ── 파생 팔 (추가 학습 0, 캐시된 확률을 재조합만) ──
    # N6: 사용자 구상 -- balnobb 레짐 라우팅 후 **두 부모(zigzag · h48) 앙상블**.
    #     store 는 폴드-major, 시드-minor 로 쌓이므로 같은 인덱스끼리 짝지으면 된다.
    # N0x2: ⭐용량 대조 -- 같은 zigzag 라벨 부모 «2개»(시드 i, i+1) 앙상블. 모델 수가 N6 와
    #     같다(6개). 이게 「두 라벨을 섞어서」와 「모델을 2배 써서」를 가른다.
    if store["N0"] and store["N5"]:
        assert len(store["N0"]) == len(store["N5"]), "N0/N5 폴드×시드 길이 불일치"
        for (n0, te0, D0, Q0), (n5, _te5, D5, Q5) in zip(store["N0"], store["N5"]):
            assert n0 == n5, f"폴드 정렬 어긋남 {n0} vs {n5}"
            store["N6"].append((n0, te0, (D0 + D5) / 2.0, (Q0 + Q5) / 2.0))
        log(f"  N6 재조합 완료 ({len(store['N6'])}개 = 폴드×시드)")
    if store["N0"] and store["N7"]:
        assert len(store["N0"]) == len(store["N7"]), "N0/N7 길이 불일치"
        for (n0, te0, D0, Q0), (n7, _t7, D7, Q7) in zip(store["N0"], store["N7"]):
            assert n0 == n7, f"폴드 정렬 어긋남 {n0} vs {n7}"
            store["N8"].append((n0, te0, (D0 + D7) / 2.0, (Q0 + Q7) / 2.0))
        log(f"  N8(zigzag + 더블배리어 앙상블) 재조합 완료 ({len(store['N8'])}개)")
    if store["N0"]:
        S = len(SEEDS)
        for i, (n0, te0, D0, Q0) in enumerate(store["N0"]):
            j = (i // S) * S + ((i % S) + 1) % S          # 같은 폴드의 다음 시드
            _n, _t, D1, Q1 = store["N0"][j]
            store["N0x2"].append((n0, te0, (D0 + D1) / 2.0, (Q0 + Q1) / 2.0))
        log(f"  N0x2(용량 대조) 재조합 완료 ({len(store['N0x2'])}개)")

    # ── 건수 맞춘 평가 ──
    rows = []
    for arm, segs in store.items():
        if not segs:
            continue
        by_seed = {}
        for i, (name, te, D, Q) in enumerate(segs):
            by_seed.setdefault(i % max(len(segs) // len(FOLDS), 1), []).append((name, te, D, Q))
        for si, group in by_seed.items():
            allq = np.concatenate([gate_score(D, Q)[1][gate_score(D, Q)[0] != 0] for _n, _t, D, Q in group])
            thr = float(np.sort(allq)[::-1][min(TARGET_N, len(allq)) - 1])
            pnl, hold, days = [], [], []
            for _n, te, D, Q in group:
                da, qf = gate_score(D, Q)
                side = np.where((da == 1) & (qf >= thr), 1.0, np.where((da == 2) & (qf >= thr), -1.0, 0.0))
                idx = np.where(side != 0)[0]
                if len(idx) < 20:
                    continue
                a, b, c = realize(te, side, idx); pnl.append(a); hold.append(b); days.append(c)
            pnl = np.concatenate(pnl); hold = np.concatenate(hold); days = np.concatenate(days)
            lo_, hi_, nd = E.block_ci(pnl, days)
            g = float(pnl.mean()); pdy = 288.0 / max(hold.mean(), 1e-9)
            rows.append({"arm": arm, "seed_slot": si, "q": thr, "n": int(len(pnl)), "indep_days": nd,
                         "gross_bp": g, "p": (g + COST + SLB) / (TPB + SLB), "ci95": [lo_, hi_],
                         "median_hold": float(np.median(hold)), "per_day": pdy, "net_day": g * pdy})
    D_ = pd.DataFrame(rows)
    log(f"\n{'='*104}\n■ 라우팅 유무 · 측면 분리 (건수맞춤 {TARGET_N:,} · 더블배리어 · 4폴드)")
    log(f"{'팔':<6}{'모델수':>7}{'건수':>8}{'건당bp':>9}{'함축p':>8}{'CI':>20}{'건/일':>7}{'순/일':>8}")
    NM = {"N0": 3, "N1": 1, "N1b": 3, "N2": 2, "N3": 6, "N4": 3, "N5": 3, "N6": 6, "N0x2": 6, "N7": 3, "N8": 6}
    for arm, g in D_.groupby("arm", sort=False):
        log(f"{arm:<6}{NM[arm]:>7}{int(g.n.mean()):>8,}{g.gross_bp.mean():>+9.2f}{g.p.mean()*100:>7.2f}%"
            f"  [{g.ci95.apply(lambda x: x[0]).mean():+7.2f},{g.ci95.apply(lambda x: x[1]).mean():+7.2f}]"
            f"{g.per_day.mean():>7.2f}{g.net_day.mean():>8.1f}"
            f"   시드폭 {g.gross_bp.max()-g.gross_bp.min():.2f}")
    OUTJ.write_text(json.dumps(rows, indent=2, default=float))
    log(f"저장: {OUTJ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
