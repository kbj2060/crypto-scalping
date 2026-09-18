#!/usr/bin/env python3
"""Zeus 동결 부모의 **봉별 예측 + 데이터 계보**를 내보낸다. (2026-09-18)

## 왜
`Omega Artifact Integrity Promotion Gate` 의 실질은 두 가지다:
  ①**정확 임계값의 per-bar 부모 예측**이 파일로 있어야 한다(아래층이 재생성에 의존하지 않도록)
  ②`report.json` 이 **어떤 정확한 바이트로** 만들어졌는지 선언하고 그게 매니페스트·디스크와
    일치해야 한다(2026-07-30 P0-2 -- 업스트림 zip 이 소급 수정돼 재현 불가였던 사고).
지금까지 Zeus 산출물에는 ①이 아예 없었다. 이 스크립트가 그것만 만든다.

🔴**게이트 자체는 «리스크 사이드카» 승격을 다룬다**(risk_sidecar.pkl·baseline_bundle 요구).
Zeus v4 는 크기가 고정이라 사이드카가 없으므로 그 체크는 구조적으로 통과할 수 없다 --
그 부분은 조문 개정 사항이고, 이 스크립트는 **개정과 무관하게 필요한 실질**을 만든다.

## 창 이름은 «실제로 무엇인지»로 붙인다
동결 아티팩트의 TRAIN 은 2022-01-01~2026-05-31(purge 30일)이다. 따라서
  train      = TRAIN 구간 (구성상 표본내 -- 그렇게 표시한다)
  validation = 2026-07-01~2026-08-19  (첫 전방 구간)
  oos        = 2026-08-20~            (라벨이 없어 어떤 분석도 닿은 적 없는 구간)
🔴계약의 기본 split(2025-09~12 / 2026-01~03)을 쓰지 않는다 -- 이 아티팩트의 TRAIN 이 그 두
창을 «덮기» 때문에 그 이름을 붙이면 표본내를 OOS 라 부르게 된다.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(os.environ.get("ZEUS_ROOT") or Path.home() / "crypto-scalping")
CODE = Path(__file__).resolve().parents[1]
for _p in (CODE, CODE / "scripts", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(_p))
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402

ART_NAME = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--art=")),
                "zeus_v4_shadow_20260918")
ART = ROOT / "data/live" / ART_NAME
MANIFEST = ROOT / "data/splits/DATASET_MANIFEST.json"
SPLITS = {"train": ("2022-01-01", "2026-05-31"),
          "validation": ("2026-07-01", "2026-08-19"),
          "oos": ("2026-08-20", "2099-01-01")}


def log(*a): print(*a, flush=True)


def _runner():
    sp = importlib.util.spec_from_file_location(
        "ZR", CODE / "scripts/live_zeus_shadow_runner_20260918.py")
    m = importlib.util.module_from_spec(sp)
    argv, sys.argv = sys.argv, ["x", f"--art={ART_NAME}"]
    try:
        sp.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    R = _runner()
    dev = torch.device("cpu")
    models, scaler, base_cols, spec = R.load_art(dev)
    tag = f"q{int(round(float(spec['rollq_q']) * 100)):03d}"
    log(f"{ART_NAME} · 입력 {len(base_cols)}열 · 점수 {spec.get('gate_score','q')} · 태그 {tag}")

    df, _ = E.load()
    feats = E.PARQUET
    digest = sha256(feats)
    rel = feats.resolve().relative_to(ROOT).as_posix()
    log(f"프레임 {rel} · sha256 {digest[:16]}… · {feats.stat().st_size/1e6:.1f}MB")

    extra_lineage: dict[str, dict] = {}
    # ── ①정확 임계값의 per-bar 예측 ─────────────────────────────────────────────
    for split, (t0, t1) in SPLITS.items():
        m = (df.timestamp >= t0) & (df.timestamp <= t1 + " 23:59:59")
        sub = df[m].reset_index(drop=True)
        if not len(sub) and split == "oos":
            # 🔴연구 parquet 은 라벨 끝(2026-08-19)에서 끝난다. 그 «뒤» 구간은 BV 패널에만
            #   있는데 패널은 매일 갱신되므로 해시를 등재하면 즉시 어긋난다(게이트가 잡으려는
            #   실패 그 자체). ⇒ **불변 스냅샷을 떠서** 아티팩트 안에 두고 그걸 등재한다.
            snap = ART / "oos_frame_20260918.parquet"
            if not snap.exists():
                F = R.build_frame(20000, live=True)
                F = F[F.timestamp >= t0].reset_index(drop=True)
                F.to_parquet(snap, index=False)
                log(f"  oos 스냅샷 생성 {len(F):,}행 → {snap.name}")
            sub = pd.read_parquet(snap)
            sub["timestamp"] = pd.to_datetime(sub["timestamp"])
            extra_lineage["oos"] = {"features_path": snap.resolve().relative_to(ROOT).as_posix(),
                                    "features_sha256": sha256(snap)}
        if not len(sub):
            log(f"  🔴{split}: 봉 0 -- 건너뜀"); continue
        da, sc = R.scores(sub, models, scaler, base_cols, dev, 0,
                          spec.get("gate_score", "q"))
        thr = R.threshold_series(sc, da, float(spec["rollq_q"]), int(spec["rollq_window"]))
        out = pd.DataFrame({
            "timestamp": sub.timestamp, "direction": da, "gate_score": sc,
            "gate_threshold": thr,
            "side": np.where((da == 1) & (sc >= thr), 1,
                             np.where((da == 2) & (sc >= thr), -1, 0)),
        })
        path = ART / f"{split}_predictions_{tag}.csv"
        out.to_csv(path, index=False)
        log(f"  {split:<10} {len(out):>7,}행 [{sub.timestamp.min()} ~ {sub.timestamp.max()}] "
            f"→ {path.name} ({path.stat().st_size/1e6:.1f}MB)")

    # ── ②데이터 계보 등재 ───────────────────────────────────────────────────────
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": {}, "schema_version": 1}
    prev = man["files"].get(rel)
    if prev and prev.get("sha256") != digest:
        log(f"  🔴매니페스트의 기존 해시와 다르다: {prev.get('sha256')[:16]}… -- 프레임이 바뀌었다")
        return 2
    man["files"][rel] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "generator_git_sha": os.popen(f"git -C {CODE} rev-parse HEAD").read().strip() or None,
        "rows": int(len(df)), "sha256": digest, "size_bytes": int(feats.stat().st_size),
        "ts_min": str(df.timestamp.min()), "ts_max": str(df.timestamp.max()),
    }
    for _sp, lin in extra_lineage.items():
        man["files"][lin["features_path"]] = {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "generator_git_sha": os.popen(f"git -C {CODE} rev-parse HEAD").read().strip() or None,
            "rows": -1, "sha256": lin["features_sha256"],
            "size_bytes": int((ROOT / lin["features_path"]).stat().st_size),
            "ts_min": None, "ts_max": None,
        }
        log(f"매니페스트 등재({_sp} 스냅샷): {lin['features_path']}")
    MANIFEST.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    log(f"매니페스트 등재: {rel}")

    # ── ③report.json ───────────────────────────────────────────────────────────
    meta = json.loads((ART / "meta.json").read_text())
    (ART / "report.json").write_text(json.dumps({
        "model_id": meta["model_id"],
        "contract": {"quality_threshold": float(spec["rollq_q"]),
                     "threshold_kind": "rolling_quantile",
                     "rollq_window": int(spec["rollq_window"]),
                     "gate_score": spec.get("gate_score", "q")},
        "label_contract": {"direction": "zigzag_action_labels_20260531",
                           "quality_mode": "same_as_direction"},
        "dataset_lineage": {"features_path": rel, "features_sha256": digest},
        "dataset_lineage_by_split": extra_lineage or None,
        "prediction_tag": tag,
        "precomputed_prediction_dir": str(ART.relative_to(ROOT)),
        "splits": {k: list(v) for k, v in SPLITS.items()},
        "risk_model": None,
        "risk_sizing_source": "fixed_live__half_kelly_logged_only",
        "orders": "NONE",
        "notes": "리스크 사이드카가 없으므로 Omega Artifact Integrity 의 사이드카 체크는 "
                 "구조적으로 적용되지 않는다. 이 파일은 그 게이트의 «실질»(정확 임계 예측 + "
                 "데이터 계보)을 충족시킨다.",
    }, indent=2, ensure_ascii=False))
    log(f"report.json 기록: {ART/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
