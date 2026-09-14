#!/usr/bin/env python3
"""칩 등급의 기준 전환(2026-09-15) 자체점검. 프레임워크 없이 assert 만.

지키려는 계약 두 개:
① 등급은 **최근 30일 기준**(`ref_pred_recent`)으로 자른다 — 없으면 학습창 고정으로 떨어지고
   그 사실을 `ref_source` 로 말한다(조용히 떨어지지 않는다).
② **수량 배수는 어느 경우에도 고정 기준**(`ref_pred/pred`)이다 — 사이징 눈금은 안 움직인다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import server as s  # noqa: E402

FIX, REC = 0.0016, 0.0013          # 고정 기준 · 최근 30일 기준(조용한 레짐)


def chip(pred, recent=REC):
    sm = {"used": True, "pred_vol": pred, "ref_pred": FIX, "ref_recent_days": 30}
    if recent:
        sm["ref_pred_recent"] = recent
    return s.vol_level_item({"sizing_model": sm})


def main() -> int:
    # ① 같은 예측이 기준에 따라 등급이 갈린다 — 이게 이번 변경의 전부다.
    hot = REC * 1.5                                    # 최근 기준 1.50배 · 고정 기준 1.22배
    assert chip(hot)["grade"] == "주의", chip(hot)
    assert chip(hot, recent=None)["grade"] == "안정", chip(hot, recent=None)

    # 세 등급 경계(최근 기준). 컷은 1.39 / 1.88.
    assert chip(REC * 1.38)["grade"] == "안정"
    assert chip(REC * 1.40)["grade"] == "주의"
    assert chip(REC * 1.90)["grade"] == "위험"

    # ② 수량 배수는 기준이 바뀌어도 그대로다(= 고정 ref_pred / pred).
    for pred in (REC * 0.5, REC, REC * 2):
        assert abs(chip(pred)["qty_mult"] - FIX / pred) < 1e-12
        assert abs(chip(pred, recent=None)["qty_mult"] - FIX / pred) < 1e-12

    # 기준 출처를 숨기지 않는다.
    assert chip(REC)["ref_source"] == "recent" and chip(REC)["ref_days"] == 30
    assert chip(REC, recent=None)["ref_source"] == "train"
    for bad in (0, -1, "0.001", None):                 # 이상값이면 고정으로 떨어진다
        assert chip(REC, recent=bad)["ref_source"] == "train", bad

    # 예측이 없으면 등급도 없다.
    assert s.vol_level_item({"sizing_model": {"used": False}})["available"] is False
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
