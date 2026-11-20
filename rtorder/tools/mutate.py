"""rtorder 양성 대조(2026-11-20): 규칙을 하나씩 일부러 틀리게 바꿔 시험이 실패하는지 본다.
원본 소스는 건드리지 않는다 -- rtorder/target/mutwork 에 사본을 떠서 거기만 바꾸고, 끝에 원본 해시가 그대로인지 확인한다.
실행(저장소 뿌리): ( ulimit -v 4000000; timeout 1800 python3 -I rtorder/tools/mutate.py )
"""
import hashlib, os, shutil, subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
W = SRC / "target" / "mutwork"
P, X, F = "src/plan.rs", "src/exec.rs", "src/pyfmt.rs"
PLAN = [  # (이름, 파일, 원래, 바꾼 것) -- plan_fixture 로 잡혀야 한다
    ("내림 경계 +1e-9 → -1e-9", P, "Ok(int_trunc(value / grid + 1e-9)? * grid)", "Ok(int_trunc(value / grid - 1e-9)? * grid)"),
    ("올림 경계 -1e-9 → +1e-9", P, "Ok(py_ceil(value / grid - 1e-9)? * grid)", "Ok(py_ceil(value / grid + 1e-9)? * grid)"),
    ("롱 진입가 내림 → 올림", P, 'if a.side == "LONG" { floor_to(a.best_bid, f.tick)? }', 'if a.side == "LONG" { ceil_to(a.best_bid, f.tick)? }'),
    ("상한 여유 room <= 0 → < 0", P, "if room <= 0.0 {", "if room < 0.0 {"),
    ("가용 잔고 버퍼 (1-2%) → (1+2%)", P, "(1.0 - MARGIN_BUFFER)", "(1.0 + MARGIN_BUFFER)"),
    ("비율을 상한 전 권고에 곱함", P, "qty = available_qty * a.fraction;", "qty = pyfmt::max(0.0, a.recommended_qty) * a.fraction;"),
    ("규칙 첫 진입 1/3 → 0.75", P, "cap * REC_FIRST_SHARE", "cap * 0.75"),
    ("실효 L 0.9×레버리지 → 1.0×", P, "RULE_MARGIN_MAX * order_leverage", "1.0 * order_leverage"),
    ("청산 수량 min → max", P, "pyfmt::min(floor_to(want, f.step)?, floor_to(held, f.step)?)", "pyfmt::max(floor_to(want, f.step)?, floor_to(held, f.step)?)"),
    ("청산 시장가 문턱 >= → >", P, "v >= EXIT_TAKER_VOL_BPM", "v > EXIT_TAKER_VOL_BPM"),
    ("clear 여유 무장+5초 → +0초", P, "account_ts > armed.armed_at + 5.0", "account_ts > armed.armed_at + 0.0"),
    ("선 손절 닿음 >= → >", P, "if if up { hi >= sl } else { lo <= sl } {", "if if up { hi > sl } else { lo < sl } {"),
    ("선 익절가 롱 올림 → 내림", P, "if long { py_ceil(tp / tick - 1e-9)? } else", "if long { py_floor(tp / tick + 1e-9)? } else"),
    ("분할 익절 carry 버림", P, "carry = raw - q;", "carry = 0.0;"),
    ("merge: 쏜 틱의 tp_resting 규칙 뒤집음", P, 'if !(py_truthy(m.get("fired")) && !py_truthy(cur.get("fired"))) {', 'if py_truthy(m.get("fired")) && !py_truthy(cur.get("fired")) {'),
    ("commit_rule_l 동점이면 뒤의 것", P, "if c.1 < who.1 {", "if c.1 <= who.1 {"),
    ("파이썬 max 동점 → 뒤의 것(-0.0)", F, "if b > a { b } else { a }", "if b >= a { b } else { a }"),
    ("repr 지수 경계 16 → 17", F, "if (-4..16).contains(&exp) {\n        let point", "if (-4..17).contains(&exp) {\n        let point"),
    ("round(x,8) → 7자리", F, 'format!("{:.*}", n, x).parse().unwrap()', 'format!("{:.*}", n.min(7), x).parse().unwrap()'),
    ("int(round()) 동점 짝수 → 0.5 올림", F, "x.round_ties_even()", "x.round()"),
    ("0 나누기를 inf 로 흘림", P, "if b == 0.0 { Err(PlanError::ZeroDivision) } else { Ok(a / b) }", "Ok(a / b)"),
]
EXEC = [  # exec_invariants 로 잡혀야 한다
    ("취소 확인 실패해도 계속 재호가(2026-09-13 41개)", X, "None => return (r8(done + this_filled), price, repegs, Some((Phase::Error, cancel_msg(oid)))),", "None => {}"),
    ("PARTIALLY_FILLED 를 종료로", X, 'pub const TERMINAL: [&str; 4] = ["FILLED", "CANCELED", "EXPIRED", "REJECTED"];', 'pub const TERMINAL: [&str; 5] = ["FILLED", "CANCELED", "EXPIRED", "REJECTED", "PARTIALLY_FILLED"];'),
    ("-5022 판정이 코드 앞머리만 봄(09-20)", X, 'err.contains("5022") || err.contains("could not be executed as maker")', 'err.starts_with("-5022")'),
    ("앞지름 부등호 뒤집음", X, 'if order_side == "BUY" { bid > price } else { ask < price }', 'if order_side == "BUY" { bid < price } else { ask > price }'),
    ("알림에 깨도 매번 리페그 판단(E16)", X, "if ex.now() - last_chk < POLL_SEC {", "if ex.now() - last_chk < 0.0 {"),
    ("불명 접수의 표식 조회 빼기(E6)", X, "if !e.starts_with(|ch: char| ch.is_ascii_digit()) {\n                // 🔴", "if false {\n                // 🔴"),
    ("잔량 대신 전량 재주문", X, "let remaining = r8(total - done);\n        let cid = format!(\"{PEG_TAG}", "let remaining = total;\n        let cid = format!(\"{PEG_TAG}"),
    ("부분 체결 뒤에도 수정 적용(D4)", X, "if g.action.is_none() && done == 0.0 {", "if g.action.is_none() {"),
    ("감시 진입 maker_only 무시", X, "if plan.maker_only {", "if false {"),
    ("on_fill 을 매 조회마다", X, "ex.on_fill().await; // 첫 체결을 본 순간 한 번(서버가 손절을 바로 건다 -- 60분 무방비 금지)\n            on_fill = false;", "ex.on_fill().await;"),
    ("오류 때 마지막 체결량 버림(E9)", X, "None => return (r8(done + this_filled), price, repegs, Some((Phase::Error, cancel_msg(oid)))),", "None => return (r8(done), price, repegs, Some((Phase::Error, cancel_msg(oid)))),"),
    ("지정가 대기 5xx 를 확정 거부로", X, "|| e0.starts_with('5') || e0.contains(\"-1007\")", "|| e0.contains(\"-1007\")"),
    ("거래소 취소를 취소로 안 봄(기다리기)", X, "let outside_cancel = terminal(&status) && status != \"FILLED\";", "let outside_cancel = false;"),
    ("대기 취소를 무시하고 따라가기 계속", X, "let cancel = |s: &Shared| st(s).action == Some(Action::Cancel);", "let cancel = |_s: &Shared| false;"),
]


def digest():
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SRC / "src").glob("*.rs"))}


def main() -> int:
    before = digest()
    env = dict(os.environ, CARGO_TARGET_DIR=str(SRC / "target" / "mut"), PATH=f"{Path.home()}/.cargo/bin:" + os.environ["PATH"])
    caught = total = 0
    only = sys.argv[1] if len(sys.argv) > 1 else ""   # "plan" | "exec" | "" = 둘 다
    for test, muts in (("plan_fixture", PLAN), ("exec_invariants", EXEC)):
        if only and not test.startswith(only):
            continue
        for name, f, a, b in muts:
            shutil.rmtree(W, ignore_errors=True)
            shutil.copytree(SRC, W, ignore=shutil.ignore_patterns("target"))
            p = W / f
            s = p.read_text()
            assert s.count(a) >= 1, (name, a)
            p.write_text(s.replace(a, b, 1))
            r = subprocess.run(["cargo", "test", "-j", "2", "--offline", "--test", test], cwd=W, env=env, capture_output=True, text=True, timeout=600)
            out = r.stdout + r.stderr
            failed = sorted({l.split()[1] for l in out.splitlines() if l.startswith("test ") and l.endswith("FAILED")})
            ok = r.returncode != 0
            caught, total = caught + ok, total + 1
            print(f"{'잡음' if ok else '🔴못 잡음'} | {test} | {name} | {', '.join(failed)[:110]}", flush=True)
    shutil.rmtree(W, ignore_errors=True)
    assert digest() == before, "원본 소스가 바뀌었다"
    print(f"{caught}/{total} 잡음 · 원본 소스 해시 그대로")
    return 0 if caught == total else 1


if __name__ == "__main__":
    sys.exit(main())
