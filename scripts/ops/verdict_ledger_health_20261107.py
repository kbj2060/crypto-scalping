#!/usr/bin/env python3
"""판정 장부 건강 점검 (2026-11-07). 전진 판정이 조용히 무효가 되는 것을 판정일 전에 잡는다. 읽기 전용, 결과·적중은 보지 않는다.

11-01 재학습이 전부 실패한 채 6일이 지났고, 판정 코드가 브랜치에만 있었던 사고에서 나왔다.
  ① 모델 학습월이 이번 달인가(매달 1·2일 재학습 → 4일 뒤부터 경고) ② 전진 기록이 새로운가(파일 수정 시각·줄 수만)
  ③ 달력의 열린 항목이 가리키는 스크립트가 저장소에 있는가
    python scripts/ops/verdict_ledger_health_20261107.py            # 문제가 있으면 exit 1
    python scripts/ops/verdict_ledger_health_20261107.py --selftest
ponytail: 경보 전달은 하지 않는다 -- exit 코드만. cron·텔레그램은 등록 승인을 받은 뒤 붙인다.
"""
import json, re, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["eth_dir4h", "sol_dir4h", "xrp_dir4h", "eth_dirshort", "eth_dirshort65", "btc_dirshort65", "xrp_dirshort65"]
# (파일, 허용 지연 시간) -- 기록기 주기보다 넉넉히
FRESH = {"data/live/pred_record/ETH.jsonl": 1, "data/live/pred_record/xsec.jsonl": 6, "data/live/regime_band_forward.jsonl": 3,
         "data/live/eth_direction_forward.jsonl": 12, "data/live/liqmap_score.jsonl": 3, "data/live/maxpain_oos_ledger.jsonl": 30, "data/live/macro_bias_forward.jsonl": 24 * 8}


def check_models(root, now, load):
    if now.day < 4:
        return []
    want, out = now.strftime("%Y-%m"), []
    for m in MODELS:
        f = root / "data" / "live" / f"{m}_model.joblib"
        if not f.exists():
            out.append(f"모델 없음 {f.name}")
            continue
        got = str(load(f).get("month"))[:7]
        if got != want:
            out.append(f"재학습 안 됨 {f.name}: {got} (이번 달 {want})")
    return out


def check_fresh(root, now_ts, fresh=FRESH):
    out = []
    for p, hours in fresh.items():
        f = root / p
        if not f.exists():
            out.append(f"기록 없음 {p}")
        elif now_ts - f.stat().st_mtime > hours * 3600:
            out.append(f"기록 멈춤 {p}: {(now_ts - f.stat().st_mtime) / 3600:.1f}시간 전")
    return out


def check_scripts(root, cal):
    out = []
    for i in cal["items"]:
        if i.get("done"):
            continue
        for s in re.findall(r"scripts/[\w./-]+\.py", i.get("how", "")):
            if not (root / s).exists():
                out.append(f"판정 코드 없음 {i['id']}: {s}")
    return out


def main():
    import joblib
    now = datetime.now(timezone.utc)
    cal = json.loads((ROOT / "dashboard" / "verdict_calendar.json").read_text(encoding="utf-8"))
    problems = check_models(ROOT, now, joblib.load) + check_fresh(ROOT, time.time()) + check_scripts(ROOT, cal)
    n_open = sum(1 for i in cal["items"] if not i.get("done"))
    print(f"열린 판정 {n_open}건 · 문제 {len(problems)}건")
    for p in problems:
        print(" -", p)
    sys.exit(1 if problems else 0)


def selftest():
    d = Path(tempfile.mkdtemp())
    (d / "data" / "live").mkdir(parents=True)
    (d / "scripts").mkdir()
    for m in MODELS:
        (d / "data" / "live" / f"{m}_model.joblib").write_text("x")
    now = datetime(2026, 11, 7, tzinfo=timezone.utc)
    ok = lambda f: {"month": "2026-11-01"}
    old = lambda f: {"month": "2026-10-01"}
    assert check_models(d, now, ok) == [] and len(check_models(d, now, old)) == len(MODELS)
    assert check_models(d, datetime(2026, 11, 2, tzinfo=timezone.utc), old) == []     # 재학습 기간엔 조용히
    (d / "data" / "live" / "eth_dir4h_model.joblib").unlink()
    assert any("모델 없음" in x for x in check_models(d, now, ok))
    f = d / "data" / "live" / "a.jsonl"
    f.write_text("x")
    assert check_fresh(d, f.stat().st_mtime + 3600, {"data/live/a.jsonl": 2}) == []
    assert len(check_fresh(d, f.stat().st_mtime + 3 * 3600, {"data/live/a.jsonl": 2})) == 1
    assert len(check_fresh(d, 0, {"data/live/없음.jsonl": 2})) == 1
    (d / "scripts" / "있음.py").write_text("")
    cal = {"items": [{"id": "a", "how": "python scripts/있음.py"}, {"id": "b", "how": "python3 scripts/없음.py vol"}, {"id": "c", "how": "python scripts/없음2.py", "done": "닫음"}]}
    r = check_scripts(d, cal)
    assert len(r) == 1 and "b" in r[0], r
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
