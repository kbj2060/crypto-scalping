#!/usr/bin/env python3
"""양성 대조: 규칙을 하나씩 틀리게 바꾼 rtserve 사본으로 시험(tests/http_parity.rs · 단위 시험)을 돌려 실패하는지 본다 (2026-11-21).

사본 = <작업 폴더>/mut (원본 rtserve 는 안 건드린다 -- 끝에 src·tests 바이트를 처음과 대조한다). 빌드 산출물은 <작업 폴더>/mut_target
하나를 같이 쓰되 변이마다 `cargo clean -p rtserve`(rtcollect mutate_check 의 교훈: 안 지우면 직전 변이 산출물을 그대로 쓴다).
시험 실행만 `ulimit -v 12000000` + timeout.

    python3 rtserve/tools/mutate_check.py <작업 폴더> [이름 일부 ...]
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
# (이름, 파일, 찾을 글자, 바꿀 글자) -- 찾을 글자는 그 파일에 정확히 한 번
M = [
    ("무변경(대조: 통과해야 함)", "src/lib.rs", "pub mod api;", "pub mod api;"),
    # ── 공통 동작 ──
    ("압축 협상 gzip 먼저", "src/lib.rs", 'if ae.contains("deflate") { Some("deflate") } else if ae.contains("gzip") { Some("gzip") }',
     'if ae.contains("gzip") { Some("gzip") } else if ae.contains("deflate") { Some("deflate") }'),
    ("압축 수준 6→5", "src/lib.rs", "Compress::new(Compression::new(6), true)", "Compress::new(Compression::new(5), true)"),
    ("gzip 압축 수준 6→9", "src/lib.rs", "Compress::new_gzip(Compression::new(6), 15)", "Compress::new_gzip(Compression::new(9), 15)"),
    ("ETag 16→15자", "src/lib.rs", 'hex(&Sha256::digest(r.body.as_ref().unwrap()))[..16]', 'hex(&Sha256::digest(r.body.as_ref().unwrap()))[..15]'),
    ("If-None-Match 공백 안 벗김", "src/lib.rs", "let c = c.trim_matches(|ch: char| ch.is_whitespace());", "let c = c;"),
    ("json.dumps 구분자 압축형", "src/lib.rs", "rtcollect::py_dumps_default(v).into_bytes()", "serde_json::to_string(v).unwrap().into_bytes()"),
    ("JSON 304 에 Cache-Control 빼기", "src/lib.rs", 'headers: vec![("ETag".into(), etag), ("Cache-Control".into(), cc)]', 'headers: vec![("ETag".into(), etag)]'),
    ("예외 응답도 미들웨어 지남", "src/lib.rs", "if !r.exc {", "if true {"),
    ("ttf 캐시 1년→1일", "src/lib.rs", '"public, max-age=31536000, immutable"', '"public, max-age=86400, immutable"'),
    ("html 이름 no-cache 빼기", "src/lib.rs", '} else if name.ends_with(".html") {\n        r.set("Cache-Control", "no-cache");', '} else if name.ends_with(".html") {'),
    ("Last-Modified ceil→floor", "src/lib.rs", "st_mtime.ceil() as u64", "st_mtime.floor() as u64"),
    ("정적 ETag 크기 빼기", "src/lib.rs", 'format!("{:x}-{:x}", mtime_ns, meta.size())', 'format!("{:x}", mtime_ns)'),
    ("정적 Accept-Ranges 빼기", "src/lib.rs", '            ("Accept-Ranges".into(), "bytes".into()),\n', ""),
    ("본문 없는 404 기본 형식 빼기", "src/lib.rs", 'r.set("Content-Type", "application/octet-stream");', "{}"),
    ("예외 본문 «404 Not Found»", "src/lib.rs", 'Some(format!("{status}: {reason}").into_bytes())', 'Some(format!("{status} {reason}").into_bytes())'),
    ("405 Allow 순서", "src/lib.rs", '("Allow", "GET,HEAD")', '("Allow", "HEAD,GET")'),
    ("js 형식 text/javascript", "src/lib.rs", '"js" | "mjs" => "application/javascript",', '"js" | "mjs" => "text/javascript",'),
    ("버스터 12→11자", "src/lib.rs", "hex(&h.finalize())[..12]", "hex(&h.finalize())[..11]"),
    ("탭 숨김: 이미 hidden 검사 빼기", "src/lib.rs", "|| tag.contains(\" hidden\")", ""),
    ("🔴정적 폴더 밖 검사 빼기", "src/lib.rs", "if !resolved.starts_with(&dir) {", "if false {"),
    # 절대 경로 검사(`rel.starts_with('/')`)를 빼는 변이는 동등 변이다 -- 붙인 뒤 폴더 밖 검사가 같은 404 를 낸다(두 겹 방어라 남겨 둔다).
    ("폴더 = 403 → 파일 응답", "src/lib.rs", "    if resolved.is_dir() {\n        return http_exc(403, None, &[]);\n    }\n", ""),
    # ── 엔드포인트 ──
    ("메모: 사전 아니어도 냄", "src/api.rs", 'py::loads(&s).ok()).filter(Value::is_object);', 'py::loads(&s).ok());'),
    ("레이더 age 소수 1→2", "src/api.rs", '"age_s": f(py::round_nd(age, 1))', '"age_s": f(py::round_nd(age, 2))'),
    ("레이더 지연 180→100000초", "src/api.rs", '"stale": age > 180.0', '"stale": age > 100000.0'),
    ("레이더 계수 파일 바꿔 끼우기 빼기", "src/api.rs", "if is_params_name(&nm) && root.join(\"dashboard\").join(&nm).exists() {", "if false {"),
    ("레이더 코인 열쇠 소문자 빼기", "src/api.rs", "coins.insert(k.to_lowercase(), Value::Object(e));", "coins.insert(k.clone(), Value::Object(e));"),
    ("레벨 5분 꼬리 잘린 첫 줄 안 버림", "src/api.rs", "skip(usize::from(size > n))", "skip(0)"),
    ("레벨 5분 늦은 판정도 그림", "src/api.rs", 'if k == "ev" && lag_pos()? {', 'if k == "ev" && false {'),
    ("레벨 5분 중복 열쇠에서 grp 빼기", "src/api.rs", 'py::hkey(&json!([e["a"], e["side"], e["grp"]]))', 'py::hkey(&json!([e["a"], e["side"]]))'),
    ("레벨 5분 stale 45→1e9초", "src/api.rs", "&json!(45))?", "&json!(1_000_000_000))?"),
    ("레벨 5분 최근 30→29", "src/api.rs", "let skip = done.len().saturating_sub(30);", "let skip = done.len().saturating_sub(29);"),
    ("띠 p_trend 문자열 못 읽음", "src/api.rs", 'o.insert("p_trend".into(), f(py::to_float(item(&r, "p_trend")?)?));',
     'o.insert("p_trend".into(), item(&r, "p_trend")?.clone());'),
    ("청산 탭 asset 소문자 빼기", "src/api.rs", 'let a = req.q("asset").unwrap_or("eth").to_lowercase();', 'let a = req.q("asset").unwrap_or("eth").to_string();'),
    ("아이스버그 깨진 줄 건너뜀(원본은 500)", "src/api.rs", "let e = py::loads(ln)?;", "let Ok(e) = py::loads(ln) else { continue };"),
    ("아이스버그 24h→25h", "src/api.rs", "let cut = f(now - 86400.0);", "let cut = f(now - 90000.0);"),
    ("ob5m 진행 봉 30→3초", "src/api.rs", "(now * 1000.0).trunc() as i64 - 30_000", "(now * 1000.0).trunc() as i64 - 3_000"),
    ("ob5m 벽 최대 → 최소", "src/api.rs", "if py::gt(&usd0, &cur)? {", "if py::lt(&usd0, &cur)? {"),
    ("ob5m tilt 소수 2→1", "src/api.rs", "py::round_v(&py::mul(imb, &json!(100))?, 2)?", "py::round_v(&py::mul(imb, &json!(100))?, 1)?"),
    ("ob5m 확정 봉 임시 표식 안 거름", "src/api.rs", "if !known.contains(&py::hkey(item(m, \"bar_t\")?)) {", "if true {"),
    ("국채 표시 창 >= → >", "src/api.rs", ".filter(|x| x.0 as f64 >= lo)", ".filter(|x| x.0 as f64 > lo)"),
    ("xyz full 늘 참", "src/api.rs", '"full": since == 0', '"full": true'),
    ("xyz 음수 since 그대로", "src/api.rs", ".clamp(0, 1 << 40)", ".clamp(-1 << 40, 1 << 40)"),
    ("오늘 칸 진행 중 1분봉 포함", "src/api.rs", "AND ts_sec + 60 <= ?", "AND ts_sec + 0 <= ?"),
    ("달력 신선도 90→150분", "src/api.rs", "crate::api2::worker_payload(path, 90.0,", "crate::api2::worker_payload(path, 150.0,"),
    # ── 둘째 조각(api2 · hub · py) ──
    ("워커 낡음 글자", "src/api2.rs", '("error", json!("worker_stale"))', '("error", json!("worker_old"))'),
    ("bare_missing 에도 tone", "src/api2.rs", "    if !o.bare_missing {\n        base.push", "    if true {\n        base.push"),
    ("gex 블록 8→7건", "src/api2.rs", "a.iter().take(8)", "a.iter().take(7)"),
    ("gex 최대 같은 값이면 뒤의 것", "src/api2.rs", "|(_, bk)| py::gt(&k, bk)", "|(_, bk)| py::ge(&k, bk)"),
    ("state ETag repr 공백", "src/api2.rs", 'format!("(({a}, {b}),)")', 'format!("(({a},{b}),)")'),
    ("state ensure_ascii 켬", "src/api2.rs", 'py::dumps(&json!({"state": st}), (",", ":"), false, None)', 'py::dumps(&json!({"state": st}), (",", ":"), true, None)'),
    ("칩 burst 거르기 빼기", "src/api2.rs", ' && py::get(&r, "k")? == Some(&json!("burst"))', ""),
    ("알림 최근 순서 그대로", "src/api2.rs", "    recent.reverse();\n", ""),
    ("알림 최근 30→29", "src/api2.rs", "ls.len().saturating_sub(30)", "ls.len().saturating_sub(29)"),
    ("뉴스 제목 엔티티 안 풂", "src/api2.rs", ".map(crate::text::unescape).unwrap_or_default()", ".map(str::to_string).unwrap_or_default()"),
    ("뉴스 키 링크 안 뗌", "src/text.rs", "if skip > 0 && n > 0 {", "if false {"),
    ("뉴스 키 12→20자", "src/text.rs", "if k.len() >= 12 {", "if k.len() >= 20 {"),
    ("급변 z 4→3", "src/api2.rs", "(r / sd).abs() >= 4.0", "(r / sd).abs() >= 3.0"),
    ("급변 NQ 개장 시간 안 거름", "src/api2.rs", 'if nm != "NQ" || cme_open(ts) {', "if true {"),
    ("stdev 모분산(n−1 → n)", "src/py.rs", "let mut den = &n * BigInt::from(xs.len() - 1);", "let mut den = &n * BigInt::from(xs.len());"),
    ("버블 반올림 6→5", "src/api2.rs", "f(py::round_nd(v[0], 6))", "f(py::round_nd(v[0], 5))"),
    ("버블 창 −5초 → −4초", "src/api2.rs", "let mut lo = end - sec - 5;", "let mut lo = end - sec - 4;"),
    ("적중 구멍 30→20분", "src/api2.rs", "const GAP_MS: i64 = 30 * 60_000;", "const GAP_MS: i64 = 20 * 60_000;"),
    ("적중 신호 80→81", "src/api2.rs", "py::ge(p, &json!(80))?", "py::ge(p, &json!(81))?"),
    ("순위 같은 값 평균 빼기", "src/api2.rs", "let r = (k + e) as f64 / 2.0 + 1.0;", "let r = k as f64 + 1.0;"),
    ("메모 쓰기 ensure_ascii 켬", "src/api2.rs", "write_atomic(&path, \"tmp\", &py::dumps(&Value::Object(notes), (\", \", \": \"), false, None))", "write_atomic(&path, \"tmp\", &py::dumps(&Value::Object(notes), (\", \", \": \"), true, None))"),
    ("오답 공백 안 접음", "src/api2.rs", '("text", json!(py_split_join(&text)))', '("text", json!(text))'),
    ("구독 들여쓰기 2→4", "src/api2.rs", "(\", \", \": \"), false, Some(2))", "(\", \", \": \"), false, Some(4))"),
    ("구독 옛 시각 안 지킴", "src/api2.rs", '("subscribed_utc", old_at.unwrap_or(json!(stamp)))', '("subscribed_utc", json!(stamp))'),
    ("client-timing 0.2초 상한 빼기", "src/api2.rs", "t.elapsed().as_secs_f64() < 0.2", "t.elapsed().as_secs_f64() < 0.0"),
    ("client-timing bool 받음", "src/api2.rs", "        _ => true,\n    });", "        Value::Bool(_) => false,\n        _ => true,\n    });"),
    ("주문 기록 쿼리 마지막 값", "src/api2.rs", "q.entry(k.clone()).or_insert_with(|| json!(v));", "q.insert(k.clone(), json!(v));"),
    ("주문 기록 UA 200→199", "src/api2.rs", '.chars().take(200).collect();\n    let ip', '.chars().take(199).collect();\n    let ip'),
    ("주문 기록 IP X-Forwarded-For 먼저", "src/api2.rs", 'hdr("cf-connecting-ip").or(hdr("x-forwarded-for"))', 'hdr("x-forwarded-for").or(hdr("cf-connecting-ip"))'),
    ("413 상태 줄 글자 빼기", "src/lib.rs", '        413 => "Request Entity Too Large", // 파이썬 HTTPStatus 글자(hyper 는 «Payload Too Large»)\n', ""),
    ("수급 창 660→600초", "src/hub.rs", 'let floor = q_int(q, "since").max(newest - SUPPLY_S);', 'let floor = q_int(q, "since").max(newest - 600);'),
    ("풋프린트 since 이상 → 초과", "src/hub.rs", "if !full && *t < since {", "if !full && *t <= since {"),
    ("풋프린트 기본 창 12→24", "src/hub.rs", "const FP_BARS: i64 = 12;", "const FP_BARS: i64 = 24;"),
    ("풋프린트 dhl 없음을 안 지움", "src/hub.rs", "        e.1 = b.dhl;", "        if b.dhl.is_some() { e.1 = b.dhl; }"),
    ("스트림 커서 안 옮김", "src/hub.rs", "q.insert(name.into(), cur.max(last[0].as_i64().unwrap_or(0)).to_string());", "let _ = (name, cur, last);"),
    ("client-timing 머지 대기 2칸 빼기", "src/api2.rs", '"rt_reconn_n",\n    "rt_draw_s", "rt_back_n"];', '"rt_reconn_n",\n    "x_a", "x_b"];'),
    ("events state 세 블록 → 둘", "src/hub.rs", 'for k in ["session", "microstructure", "tail_risk"] {', 'for k in ["session", "microstructure"] {'),
    ("events ts 마이크로초 → 밀리초", "src/hub.rs", 'format!("{base}.{frac:06}+00:00")', 'format!("{base}.{:03}+00:00", frac / 1000)'),
    # ── 셋째 조각(api3 · swr) ──
    ("세션 경보 창 30→31분", "src/api3.rs", "(-30.0..=30.0).contains(&minutes)", "(-31.0..=31.0).contains(&minutes)"),
    ("세션 경보 시간대 없는 시각 건너뜀(원본 500)", "src/api3.rs", '            return err("TypeError", "can\'t subtract offset-naive and offset-aware datetimes");', "            continue;"),
    ("세션 경보 high 거르기 빼기", "src/api3.rs", 'if py::get(e, "importance")? != Some(&json!("high")) {', "if false {"),
    ("청산 게이지 봉 30→15분", "src/api3.rs", "const BAR_MIN: i64 = 30;", "const BAR_MIN: i64 = 15;"),
    ("청산 게이지 봉 시작 >= → >", "src/api3.rs", ".filter(|r| r.0 >= bar_start)", ".filter(|r| r.0 > bar_start)"),
    ("청산 게이지 다시 읽기 1.6→0.6초", "src/api3.rs", "[0.0, 0.4, 0.8, 1.6]", "[0.0, 0.4, 0.8, 0.6]"),
    ("청산 게이지 마이크로초 안 씀", "src/api3.rs", 'if frac == 0 { format!("{base}+00:00") } else { format!("{base}.{frac:06}+00:00") }', 'format!("{base}+00:00")'),
    ("swr 신선 < → <=", "src/swr.rs", "                if age < ttl {", "                if age <= ttl {"),
    ("swr 갱신 실패 때 옛 값 버림", "src/swr.rs", "Err(PyErr(m)) => (self.log)(&format!(\"cache refresh failed for {key} (still serving stale): {m}\")),", "Err(_) => self.store(key, &V::Null),"),
    ("swr 묵음 창 빼기", "src/swr.rs", "if age < ttl + max_stale {", "if false {"),
    ("파이썬 round 짝수 → 0에서 먼 쪽", "src/py.rs", "Ok(x.round_ties_even() as i64)", "Ok(x.round() as i64)"),
    ("splitlines \\r 안 끊음", "src/py.rs", "matches!(c, '\\n' | '\\r' |", "matches!(c, '\\n' |"),
]


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted([*root.glob("src/**/*.rs"), *root.glob("tests/**/*")]):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + p.read_bytes())
    return h.hexdigest()


def main() -> int:
    work = Path(sys.argv[1]).resolve()
    picks = sys.argv[2:]
    mut, target = work / "mut", work / "mut_target"
    before = tree_hash(SRC)
    env = {**os.environ, "CARGO_TARGET_DIR": str(target)}
    results = []
    for name, rel, old, new in M:
        if picks and not any(p in name for p in picks):
            continue
        shutil.rmtree(mut, ignore_errors=True)
        shutil.copytree(SRC, mut, ignore=shutil.ignore_patterns("target"))
        man = mut / "Cargo.toml"
        # 사본은 저장소 밖 -- 형제 크레이트(rtcollect·rtcore: path 의존성 · 시험이 rtcore 정답지를 읽는다)를 작업 폴더에 심볼릭 링크로 ·
        #   위 폴더의 Cargo.toml 을 작업 공간으로 잘못 잡지 않게 [workspace]
        #   (path 의존성은 절대 경로로 -- 링크로 두면 cargo 가 링크 위 폴더에서 작업 공간을 찾다 깨진다 · 링크는 시험의 정답지 읽기용)
        if not (work / "rtcore").exists():
            (work / "rtcore").symlink_to(SRC.parent / "rtcore")
        t = man.read_text()
        for sib in ("rtcollect", "rtcore"):
            t = t.replace(f'path = "../{sib}"', f'path = "{SRC.parent / sib}"')
        # 디버그 정보 없이(duckdb 묶음을 넣은 뒤 시험 실행 파일 링크가 메모리 상한에 걸려 «빌드 실패»가 났다 -- 2026-11-21)
        man.write_text(t + "\n[workspace]\n\n[profile.dev]\ndebug = 0\n")
        f = mut / rel
        s = f.read_text()
        if s.count(old) != 1:
            results.append((name, f"찾을 글자 {s.count(old)}번 -- 변이 못 함"))
            continue
        f.write_text(s.replace(old, new))
        subprocess.run(["cargo", "clean", "-p", "rtserve", "-q"], cwd=mut, env=env, check=False)
        b = subprocess.run(["cargo", "test", "--no-run", "-q"], cwd=mut, env=env, capture_output=True, text=True)
        if b.returncode != 0:
            if name.startswith("무변경"):
                print(b.stderr[-3000:])
                print("🔴무변경 사본이 빌드되지 않는다 -- 대조를 멈춘다")
                return 1
            results.append((name, "빌드 실패(= 잡힘 · 컴파일러)"))
            continue
        t = subprocess.run(["bash", "-c", "ulimit -v 12000000; timeout 600 cargo test -q 2>&1"], cwd=mut, env=env, capture_output=True, text=True)
        failed = t.returncode != 0
        if failed and name.startswith("무변경"):   # 무변경이 실패하면 어느 시험인지 남긴다(흔들리는 시험 찾기)
            print("\n".join(t.stdout.splitlines()[-40:]))
        want_fail = not name.startswith("무변경")
        results.append((name, ("잡힘" if failed else "🔴못 잡음") if want_fail else ("통과(정상)" if not failed else "🔴무변경인데 실패")))
    shutil.rmtree(mut, ignore_errors=True)
    for name, r in results:
        print(f"{r:24} {name}")
    caught = sum(r.startswith(("잡힘", "빌드")) for n, r in results if not n.startswith("무변경"))
    total = sum(not n.startswith("무변경") for n, _ in results)
    print(f"양성 대조 {caught}/{total} · 원본 rtserve 바이트 {'그대로' if tree_hash(SRC) == before else '🔴바뀜'}")
    noop_ok = all(r == "통과(정상)" for n, r in results if n.startswith("무변경"))
    return 0 if caught == total and noop_ok and tree_hash(SRC) == before else 1


if __name__ == "__main__":
    sys.exit(main())
