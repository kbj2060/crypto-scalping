"""가격 꼬리표 세로 배치(app.js declutterTagY) -- 본문을 떼어 node 로 **실제로 돌린다** (2026-09-26 비평 P2).

비평: «↑ 2765» 가 2708.5 **아래**에 놓였다(순서 역전)·꼬리표끼리 겹쳤다.
    python3 -m pytest -q test/test_declutter_tag_y_20260926.py
"""
import json
import pathlib
import re
import subprocess

APP = pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "live" / "app.js"


def run(labels, lo=10, hi=300, gap=22):
    src = APP.read_text("utf-8")
    fn = re.search(r"^function declutterTagY\(.*?^}\n", src, re.S | re.M).group(0)
    js = fn + f"console.log(JSON.stringify(declutterTagY({json.dumps(labels)}, {lo}, {hi}, {gap})));"
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_offscreen_levels_keep_price_order():
    # 화면 위 밖 둘(2765 가 더 높다 = rawY 더 작다)이 같은 가장자리로 clamp · 화면 안 2708.5 가 그 근처.
    # 삽입순은 «가까운 것 먼저»라 옛 코드는 2765 를 더 아래에 쌓았다.
    got = run([{"name": "2708.5", "rawY": 18, "realY": 18},
               {"name": "2720", "rawY": -40, "realY": 12},
               {"name": "2765", "rawY": -300, "realY": 12}])
    names = [g["name"] for g in got]
    assert names == ["2765", "2720", "2708.5"], names           # 위→아래 = 가격 높은→낮은
    ys = [g["adjustedY"] for g in got]
    assert all(b - a >= 22 - 1e-9 for a, b in zip(ys, ys[1:])), ys   # 안 겹친다


def test_bottom_stack_pushes_back_inside_without_overlap():
    got = run([{"name": "in", "rawY": 290, "realY": 290},
               {"name": "off1", "rawY": 400, "realY": 298},
               {"name": "off2", "rawY": 500, "realY": 298}], lo=10, hi=300)
    ys = [g["adjustedY"] for g in got]
    assert ys[-1] <= 300 and all(b - a >= 22 - 1e-9 for a, b in zip(ys, ys[1:])), ys
    assert [g["name"] for g in got] == ["in", "off1", "off2"]


def test_far_apart_labels_stay_at_their_price():
    got = run([{"name": "a", "rawY": 50, "realY": 50}, {"name": "b", "rawY": 200, "realY": 200}])
    assert [g["adjustedY"] for g in got] == [50, 200]
