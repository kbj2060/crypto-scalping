"""사이드카 pkl 에 실제로 어떤 회귀기가 들어갔는지 확인한다.

2026-09-09: TabPFN 래퍼의 몽키패치 조건이 never-true 라 HGB 가 조용히 학습된 사고가 있었다
(스모크 결과가 HGB 판과 비트 단위로 동일해서 발견). 그 뒤로는 숫자만 보지 말고 **저장된 모델의
실제 타입**을 확인한다.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
BASE = ROOT / "tmp/causal_regen_20260516"

# TabPFN 사이드카 pkl 은 래퍼 클래스를 담고 있어 그 정의가 import 가능해야 로드된다.
# ⚠️ 이 자체가 발견 사항이다 -- 래퍼가 `__main__` 에 정의돼 있으면 pickle 이 `__main__.
# TabPFNRiskRegressor` 를 찾으므로 **다른 프로세스에서 로드가 안 된다**. 라이브 서빙 경로가
# 이 pkl 을 읽어야 하므로, 채택하려면 래퍼를 임포트 가능한 모듈로 옮겨야 한다.
try:
    import train_eval_omega461_risk_sidecar_tabpfn_20260909 as _tp  # noqa: E402
    import __main__
    __main__.TabPFNRiskRegressor = _tp.TabPFNRiskRegressor
except Exception as _e:  # 없으면 HGB 만 검사하고 계속한다
    print(f"[warn] TabPFN 래퍼 import 실패({type(_e).__name__}) -- TabPFN pkl 은 건너뛴다", flush=True)


def show(tag: str, d: str) -> None:
    p = pickle.load(open(BASE / d / "risk_sidecar.pkl", "rb"))
    m = p["model"]
    inner = list(m.values())[-1] if isinstance(m, dict) else m
    print(f"{tag}  pkl.model_kind={p.get('model_kind')!r}  실제 타입={type(inner).__name__}", flush=True)
    for k in ("n_fit_rows", "n_rows_after_weight", "n_estimators", "output"):
        if hasattr(inner, k):
            print(f"         {k}={getattr(inner, k)}", flush=True)
    r = json.loads((BASE / d / "report.json").read_text())
    s = r["selected"]
    v, o = s["validation"], s["oos"]
    print(f"         VAL pnl {v['pnl']:+8.2f} mdd {v['mdd']:+7.2f} tr {v['trades']:3d} "
          f"notl {v['avg_notional']:.3f}", flush=True)
    print(f"         OOS pnl {o['pnl']:+8.2f} mdd {o['mdd']:+7.2f} tr {o['trades']:3d}", flush=True)


def main() -> int:
    for tag, d in (("HGB   ", "omega4_2_trade_risk_sidecar_20260622_regimespine_balnobb_zig075_s615372041_20260909"),
                   ("TabPFN", "omega4_2_trade_risk_sidecar_20260622_tabpfnA_mean_balnobb_zig075_s615372041_20260909")):
        try:
            show(tag, d)
        except FileNotFoundError:
            print(f"{tag}  (없음: {d})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
