#!/usr/bin/env python3
"""**사건 라벨 경계 감사** -- 피쳐 창이 라벨 시작 봉을 만지는지 (2026-09-08).

## 왜 이 감사가 생겼나
2026-09-08 돌파/되돌림 분류에서 라벨은 트리거 분 `s1` **부터** 배리어를 탐색하는데
경로 피쳐 창이 `s1` 을 **포함**했다. 같은 1분봉을 피쳐와 라벨이 공유해 정확도가
**54.7~57.4% -> 50.0~54.3%** 로 4pp 부풀어 있었다. 정적 규칙 하나로 막을 수 있는 사고였다.

## 두 가지 검사
### A. 정적 스캔 -- 빌더 소스에서 경계 위반 패턴
라벨 `first_touch(..., START, ...)` 의 START 와 피쳐 창의 끝 인덱스가 **같은 심볼**이면 위반.
같은 파일 안에서 `mask = span <= X` / `[:X]` / `cl1[... + X]` 형태로 X 가 재사용되는지 본다.
휴리스틱이므로 **needs_review** 를 낼 뿐, 사람이 확인한다.

### B. ⭐수치 트립와이어 (본체) -- 피쳐를 한 칸 더 밀어 정확도가 유지되는가
`--data` 로 데이터셋을 주면, 피쳐를 **결정 시점 기준 한 단위 더 과거**로 밀 수 있는
컬럼(`--shiftable`)을 NaN 으로 만들어 재학습한다.
경계 누수가 있으면 정확도가 크게 떨어진다. **떨어지는 폭이 곧 누수 크기**다.
정적 스캔이 못 잡는 형태(파생 피쳐, 조인 시점)도 여기서 드러난다.

사용:
    python scripts/audit_event_label_feature_boundary_20260908.py --scan scripts/build_*.py
    python scripts/audit_event_label_feature_boundary_20260908.py --data <parquet> \
        --label y --feat-prefix f_,g_ --boundary-group g_mv_,g_obs_
"""
from __future__ import annotations
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL_START = re.compile(r"first_touch\s*\([^,]+,\s*[^,]+,\s*([A-Za-z_][A-Za-z0-9_]*)")
WIN_END = [re.compile(r"span\s*<=\s*([A-Za-z_][A-Za-z0-9_]*)"),
           re.compile(r"mask\s*=\s*[^=]*<=\s*([A-Za-z_][A-Za-z0-9_]*)"),
           re.compile(r"\[\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]")]


def _deps(src):
    """SYM = <rhs> 의존 그래프 (2단계 폐포용). 라벨 시작 심볼이 무엇에서 왔는지 추적한다."""
    dep = {}
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$", src, re.M):
        lhs, rhs = m.group(1), m.group(2)
        dep.setdefault(lhs, set()).update(re.findall(r"[A-Za-z_]\w*", rhs))
    return dep


def _closure(sym, dep, depth=3):
    seen = {sym}; frontier = {sym}
    for _ in range(depth):
        nxt = set()
        for s in frontier: nxt |= dep.get(s, set())
        nxt -= seen
        if not nxt: break
        seen |= nxt; frontier = nxt
    return seen


def scan(paths):
    """라벨 탐색 시작 심볼(또는 그 조상)이 피쳐 창의 끝으로도 쓰이면 경계 위반 후보.

    ⚠️`X - 1` 처럼 한 단위 당겨 쓴 형태는 안전하므로 제외한다."""
    bad = []
    for p in paths:
        try: src = Path(p).read_text()
        except Exception: continue
        starts = set(LABEL_START.findall(src))
        if not starts: continue
        dep = _deps(src)
        ends = []          # (심볼, 안전여부)
        for rx in WIN_END:
            for m in rx.finditer(src):
                line = src[max(0, m.start() - 120):m.end() + 40]
                safe = bool(re.search(re.escape(m.group(1)) + r"\s*-\s*1", src)) or "- 1" in line
                ends.append((m.group(1), safe))
        hit = []
        for st in starts:
            cl = _closure(st, dep)
            for sym, safe in ends:
                if sym in cl and not safe:
                    hit.append((st, sym))
        if hit: bad.append((str(p), sorted(set(hit))))
    print("=" * 96)
    print("A) 정적 스캔 -- 라벨 시작 인덱스(또는 그 조상)가 피쳐 창 끝으로도 쓰였는가")
    print("=" * 96)
    if not bad: print("   위반 후보 없음")
    for p, h in bad:
        print(f"   🔴 {p}")
        for st, sym in h:
            print(f"      라벨 시작 `{st}` <- 피쳐 창 끝 `{sym}` (같은 봉 공유)")
        print("      -> 피쳐 창을 `- 1` 로 줄이거나 라벨을 한 단위 뒤에서 시작할 것")
    return len(bad)


def tripwire(data, label, prefixes, boundary):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    A = pd.read_parquet(data)
    A["timestamp"] = pd.to_datetime(A["timestamp"])
    A = A.sort_values("timestamp").reset_index(drop=True)
    feats = [c for c in A.columns if c.startswith(tuple(prefixes))]
    bnd = [c for c in feats if c.startswith(tuple(boundary))]
    y = A[label].to_numpy(int); ts = A["timestamp"]
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    sp = A["split"].to_numpy() if "split" in A.columns else np.array(["ALL"] * len(A))
    print("\n" + "=" * 96)
    print(f"B) 수치 트립와이어 -- 경계 피쳐 {len(bnd)}개를 제거하면 정확도가 얼마나 떨어지는가")
    print("=" * 96)
    out = {}
    for tag, cols in (("전체", feats), ("경계피쳐 제거", [c for c in feats if c not in bnd])):
        X = A[cols].to_numpy(np.float32)
        pred = np.full(len(A), np.nan)
        for i, mo in enumerate(uniq):
            if i < 6: continue
            te = (months == mo).to_numpy()
            tr = (ts < ts[te].min() - pd.Timedelta(hours=4)).to_numpy()
            if tr.sum() < 2000 or te.sum() < 30: continue
            c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                               early_stopping=True, validation_fraction=0.15,
                                               random_state=20260908)
            c.fit(X[tr], y[tr]); pred[te] = c.predict_proba(X[te])[:, 1]
        m = np.isfinite(pred) & ~np.isin(sp, ["TRAIN", "PRE"])
        acc = ((pred[m] > 0.5).astype(int) == y[m]).mean()
        auc = roc_auc_score(y[m], pred[m])
        out[tag] = (acc, auc)
        print(f"   {tag:>14} ({len(cols):>3}피쳐): 정확도 {acc:.4f} AUC {auc:.4f} n={m.sum():,}")
    d = out["전체"][0] - out["경계피쳐 제거"][0]
    print(f"\n   ⭐경계 피쳐 기여 {d:+.4f}")
    if d > 0.02:
        print("   🔴 2pp 초과 -- 경계 누수를 강하게 의심한다. 피쳐 창 끝과 라벨 시작을 직접 확인할 것.")
    elif d > 0.01:
        print("   ⚠️ 1~2pp -- needs_review. 창을 한 단위 줄여 재측정할 것.")
    else:
        print("   ✅ 1pp 미만 -- 경계 누수 징후 없음(다른 형태의 누수는 별도 검사).")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", nargs="*", default=None)
    ap.add_argument("--data"); ap.add_argument("--label", default="y")
    ap.add_argument("--feat-prefix", default="f_,g_,sig_")
    ap.add_argument("--boundary-group", default="g_mv_,g_obs_")
    a = ap.parse_args()
    rc = 0
    if a.scan is not None:
        paths = [p for pat in (a.scan or ["scripts/build_*.py"]) for p in glob.glob(str(ROOT / pat))]
        rc += scan(sorted(set(paths)))
    if a.data:
        d = tripwire(a.data, a.label, a.feat_prefix.split(","), a.boundary_group.split(","))
        rc += int(d > 0.02)
    print(json.dumps({"violations": rc}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
