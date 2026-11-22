#!/usr/bin/env python3
"""양성 대조: 규칙을 하나씩 틀리게 바꾼 rtserve 사본으로 시험(tests/http_parity.rs · 단위 시험)을 돌려 실패하는지 본다 (2026-11-21).

사본 = <작업 폴더>/mut (원본 rtserve 는 안 건드린다 -- 끝에 src·tests 바이트를 처음과 대조한다 · 끝나면 사본·링크를 지운다).
2026-11-22: 빌드는 --release 만 · 산출물은 rtserve/target 하나(새 빌드 폴더를 안 만든다 -- 개발기 디스크가 한 번 꽉 찼다).
  사본은 새 수정 시각으로 복사한다(직전 변이 산출물을 «새것»으로 잘못 보지 않게 -- 예전 `cargo clean -p rtserve` 대신 · 같은 target 의 원본 산출물을 안 지운다).
  변이마다 시험 하나만 고를 수 있다(다섯째 칸 = --test 이름). 시험 실행만 `ulimit -v 12000000` + timeout · cargo -j 2.

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
    # ── 넷째 조각(order · oi-5m · 청산 원 · 모델 핸들러) -- 이름 «④» ──
    ("④무변경(대조: 통과해야 함)", "src/order.rs", "pub fn is_proxied", "pub fn is_proxied", "order_proxy"),
    ("④X-Forwarded-For 접속 주소 안 넣음", "src/order.rs", ".map_or_else(|| req.peer.clone(), str::to_string)", ".map_or_else(String::new, str::to_string)", "order_proxy"),
    ("④User-Agent 안 넘김", "src/order.rs", '["user-agent", "cf-connecting-ip", "content-type"]', '["cf-connecting-ip", "content-type"]', "order_proxy"),
    ("④Cache-Control 안 돌려줌", "src/order.rs", '["Content-Type", "Cache-Control", "Allow"]', '["Content-Type", "Allow"]', "order_proxy"),
    ("④쿼리 원문 안 넘김", "src/order.rs", 'format!("{}?{}", req.path, req.raw_query)', "req.path.clone()", "order_proxy"),
    ("④문 닫힘 502→503", "src/order.rs", "r.status = 502;", "r.status = 503;", "order_proxy"),
    ("④acct_seq +1", "src/order.rs", '["seq"].as_i64()', '["seq"].as_i64().map(|s| s + 1)', "order_proxy"),
    ("④oi OKX first_bar 초과 → 이상", "src/hub.rs", "okx_d.get(&t).filter(|_| t > okx_from)", "okx_d.get(&t).filter(|_| t >= okx_from)", "http4_parity"),
    ("④oi 공백 300→600초", "src/hub.rs", "let gap = prev_bar.is_none_or(|p| bar - p > 300);", "let gap = prev_bar.is_none_or(|p| bar - p > 600);", "http4_parity"),
    # «현재 OI = 마지막 봉 끝 값» 변이는 동등 변이다(hot 마지막 행 = 마지막 봉의 끝 행) -- 뺐다.
    ("④oi 기본 48→24봉", "src/hub.rs", "map_or(Some(48),", "map_or(Some(24),", "http4_parity"),
    ("④청산 기록 시작 봉도 실음", "src/hub.rs", ".filter(|b| *b > from_bar)", ".filter(|b| *b >= from_bar)", "http4_parity"),
    ("④청산 OKX hot 시작 봉 안 거름", "src/hub.rs", "hot_first.is_some_and(|h| t > h)", "hot_first.is_some()", "http4_parity"),
    ("④청산 Bybit 시작 봉도 더함", "src/hub.rs", "            if t > first {", "            if t >= first {", "http4_parity"),
    ("④청산 HL 주소 5→6", "src/hub.rs", "users[..users.len().min(5)]", "users[..users.len().min(6)]", "http4_parity"),
    ("④청산 거래소 몫 round → 버림", "src/hub.rs", "json!(py::round0(x).unwrap_or(0))", "json!(x.trunc() as i64)", "http4_parity"),
    ("④Liq5m 밀기가 앞 봉을 버림", "src/hub.rs", "old.bars.drain(..).map(|r| (r.0, r)).collect();", "{ old.bars.clear(); BTreeMap::new() };", "http4_parity"),
    ("④Liq5m okx 갈아 끼우기 안 함", "src/hub.rs", "    if l.okx.is_some() {", "    if l.okx.is_none() {", "http4_parity"),
    ("④모델 밴 가드 빼기", "src/model.rs", "    if left > 0.0 {", "    if false {", "model_parity"),
    ("④모델 낡음 검사 빼기", "src/model.rs", "if let Some(why) = rtmodel::model_stale(&files[0], srcs(&hit), src) {", "if let Some(why) = None::<String> {", "model_parity"),
    ("④dir-short 코인 표에서 BTC 빼기", "src/model.rs", 'const DS: [&str; 4] = ["ETH", "BTC", "SOL", "XRP"];', 'const DS: [&str; 4] = ["ETH", "DOGE", "SOL", "XRP"];', "model_parity"),
    ("④vol HGB 바이낸스 밴 문구 None", "src/model.rs", 'reason.map_or(Ok(None), |_| Err("HTTPServiceUnavailable: binance_ip_ban".into()))', "Ok(None)", "model_parity"),
    ("④vol 쪽 대체를 QX4 실패에도 안 함", "src/model.rs", "        Ok(None) => {}", "        Ok(None) => return Ok(json!({\"ok\": false, \"reason\": \"artifact_missing\"})),", "model_parity"),
    # ── 여섯째 조각(래스터 읽기 · 히트맵 · supply oi) -- 이름 «⑥» ──
    ('⑥무변경(대조: 통과해야 함)', 'src/heatmap.rs', 'pub fn np_sum_f32', 'pub fn np_sum_f32', 'heatmap_parity'),
    ('⑥백분위 q 를 f64 로', 'src/heatmap.rs', 'let vi = (n as f32 - 1.0) * (pct / 100f32);', 'let vi = ((n as f64 - 1.0) * (pct as f64 / 100.0)) as f32;', 'heatmap_parity'),
    ('⑥재깔림 합 행 우선', 'src/heatmap.rs', '        for &j in &sel {\n            for r in i - lb..i {', '        for r in i - lb..i {\n            for &j in &sel {', 'heatmap_parity'),
    ('⑥블록 문턱 절반 → 0.6', 'src/heatmap.rs', 'let thr = csum[csum.len() - 1] * 0.5;', 'let thr = csum[csum.len() - 1] * 0.6;', 'heatmap_parity'),
    ('⑥접근 near 0.35 → 0.36', 'src/heatmap.rs', 'let near = x < 0.35f32;', 'let near = x < 0.36f32;', 'heatmap_parity'),
    ('⑥read_window 구멍(ts 0) 통과', 'src/raster_read.rs', 'mid[i].is_finite() && ts[i] > 0', 'mid[i].is_finite()', 'heatmap_parity'),
    ('⑥agg 묶음 첫 최대 → 끝 최대', 'src/raster_read.rs', 'if v > best {', 'if v >= best {', 'heatmap_parity'),
    ('⑥supply oi 660초 자르기 빼기', 'src/hub.rs', '    ring.retain(|k, _| *k >= cut);\n', '', 'http4_parity'),
    # ── 다섯째 조각(병행 고정 · 풋프린트 라우팅) -- 이름 «⑤» ──
    ('⑤무변경(대조: 통과해야 함)', 'src/api3.rs', 'fn fpd_of', 'fn fpd_of', 'fpd_http'),
    ('⑤읽기 전용 스위치 무시', 'src/lib.rs', 'if cfg.readonly && req.method == "POST"', 'if false && req.method == "POST"', 'model_parity'),
    ('⑤읽기 전용인데 접수 기록 적음', 'src/lib.rs', '&& !r.upstream && !cfg.readonly {', '&& !r.upstream {', 'model_parity'),
    ('⑤REST 꺼짐 무시', 'src/model.rs', 'if cfg.rest.is_empty() { Err(', 'if false { Err(', 'model_parity'),
    ('⑤풋프린트 asset 소문자로', 'src/api3.rs', 'let a = req.q("asset").unwrap_or("eth").to_string();', 'let a = req.q("asset").unwrap_or("eth").to_lowercase();', 'fpd_http'),
    ('⑤풋프린트 빈 since 통과', 'src/api3.rs', 'req.q(k).map(|v| crate::fpd::iso_day(v)', 'req.q(k).filter(|v| !v.is_empty()).map(|v| crate::fpd::iso_day(v)', 'fpd_http'),
    ('⑤일봉 cells range → 500', 'src/api3.rs', 'Err(PyErr(m)) if m == "range" => crate::http_exc(400, Some("range"), &[]),', '', 'fpd_http'),
    ('⑤4h cells from_to 를 bad_key 보다 먼저', 'src/api3.rs', 'let (frm, to) = (tri!(iso_h4_q(req, "from")), tri!(iso_h4_q(req, "to")));', 'if req.q("from").is_none() || req.q("to").is_none() { return crate::http_exc(400, Some("from_to"), &[]); }\n    let (frm, to) = (tri!(iso_h4_q(req, "from")), tri!(iso_h4_q(req, "to")));', 'fpd_http'),
    # ── 아홉째 조각(바이낸스 한 문 · trend · cycle-path) -- 이름 «⑨» ──
    ('⑨무변경(대조: 통과해야 함)', 'src/rest.rs', 'fn market_symbol', 'fn market_symbol', 'rest_parity'),
    ('⑨가중치 문턱 무시', 'src/model.rs', 'if let Some(w) = r.1.filter(|w| *w > limit) {', 'if let Some(w) = r.1.filter(|w| *w > limit * 2.0) {', 'rest_parity'),
    ('⑨문턱 거절 없이 보냄', 'src/model.rs', '    if st.1 > t {\n        return Err(Refused::Hold(st.1 - t));\n    }\n', '', 'rest_parity'),
    ('⑨trend 형성 봉 포함', 'src/rest.rs', '(r[6].as_i64().unwrap_or(i64::MAX) as f64) < now_ms).collect();\n        Ok(trend_payload', '(r[6].as_i64().unwrap_or(i64::MAX) as f64) < now_ms + 86_400_000.0).collect();\n        Ok(trend_payload', 'rest_parity'),
    ('⑨trend 표본 분산 n−1 → n', 'src/rest.rs', '/ 19.0).sqrt()', '/ 20.0).sqrt()', 'rest_parity'),
    ('⑨trend 크기 상한 빼기', 'src/rest.rs', 'let m = if v < 2.0 { v } else { 2.0 }; if m > -2.0 { m } else { -2.0 }', 'v', 'rest_parity'),
    ('⑨trend 나이 경계 90 → 89', 'src/rest.rs', 'for i in (90..n).rev() {', 'for i in (89..n).rev() {', 'rest_parity'),
    ('⑨trend sma50 14 → 13', 'src/rest.rs', '(49.max(n - 14)..n)', '(49.max(n - 13)..n)', 'rest_parity'),
    ('⑨cycle 1m 경계 <= → <', 'src/rest.rs', 'if span_min <= 1490.0 { "1m" }', 'if span_min < 1490.0 { "1m" }', 'rest_parity'),
    ('⑨cycle 15m 경계 7450 → 7449', 'src/rest.rs', 'else if span_min <= 7450.0 { "5m" }', 'else if span_min <= 7449.0 { "5m" }', 'rest_parity'),
    ('⑨cycle 밑줄 숫자 거절', 'src/rest.rs', "let v: i64 = d.replace('_', \"\").parse().ok()?;", 'let v: i64 = d.parse().ok()?;', 'rest_parity'),
    ('⑨cycle 30일 하한 빼기', 'src/rest.rs', 'Some(s) if now_ms - 30 * 86_400_000 <= s && s <= now_ms => s,', 'Some(s) if s <= now_ms => s,', 'rest_parity'),
    ('⑨cycle 주문 심볼 빼기', 'src/rest.rs', '&& !exec_symbols().contains(&symbol)', '', 'rest_parity'),
    ('⑨swing 창 50 → 49', 'src/rest.rs', 'const SWING_N: usize = 50;', 'const SWING_N: usize = 49;', 'rest_parity'),
    ('⑨swing 깨짐 > → >=', 'src/rest.rs', '.filter(|(px, _)| c[i] > *px)', '.filter(|(px, _)| c[i] >= *px)', 'rest_parity'),
    ('⑨swing CHoCH 판정 뒤집음', 'src/rest.rs', '"kind": if trend == -1 { "CHoCH" } else { "BOS" }', '"kind": if trend == 1 { "CHoCH" } else { "BOS" }', 'rest_parity'),
    ('⑨swing 한쪽 실패 partial 빼기', 'src/rest.rs', '            out.insert("partial".into(), json!(errs));\n', '', 'rest_parity'),
    ('⑨cycle 실패 reason 통째', 'src/rest.rs', 'm.split_once(": ").map_or(m.as_str(), |x| x.1)', 'm.as_str()', 'rest_parity'),
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
    mut, target = work / "mut", SRC / "target"
    before = tree_hash(SRC)
    env = {**os.environ, "CARGO_TARGET_DIR": str(target)}
    results = []
    for name, rel, old, new, *only in M:
        if picks and not any(p in name for p in picks):
            continue
        shutil.rmtree(mut, ignore_errors=True)
        shutil.copytree(SRC, mut, ignore=shutil.ignore_patterns("target"), copy_function=shutil.copy)   # 새 수정 시각
        man = mut / "Cargo.toml"
        # 사본은 저장소 밖 -- 형제 크레이트(rtcollect·rtcore: path 의존성 · 시험이 rtcore 정답지를 읽는다)를 작업 폴더에 심볼릭 링크로 ·
        #   위 폴더의 Cargo.toml 을 작업 공간으로 잘못 잡지 않게 [workspace]
        #   (path 의존성은 절대 경로로 -- 링크로 두면 cargo 가 링크 위 폴더에서 작업 공간을 찾다 깨진다 · 링크는 시험의 정답지 읽기용)
        for sib in ("rtcore", "rtmodel", "rtorder"):   # 시험이 읽는 형제 정답지·실행 파일(../rtmodel/tests/fixtures · ../rtorder/target/release/rtorder)
            if not (work / sib).exists():
                (work / sib).symlink_to(SRC.parent / sib)
        t = man.read_text()
        for sib in ("rtcollect", "rtcore", "rtmodel", "rtorder"):
            t = t.replace(f'path = "../{sib}"', f'path = "{SRC.parent / sib}"')
        man.write_text(t + "\n[workspace]\n")
        f = mut / rel
        s = f.read_text()
        if s.count(old) != 1:
            results.append((name, f"찾을 글자 {s.count(old)}번 -- 변이 못 함"))
            continue
        f.write_text(s.replace(old, new))
        sel = ["--test", only[0]] if only else []
        b = subprocess.run(["cargo", "test", "--release", "-j", "2", "--offline", "--no-run", "-q", *sel], cwd=mut, env=env, capture_output=True, text=True)
        if b.returncode != 0:
            if "무변경" in name:
                print(b.stderr[-3000:])
                print("🔴무변경 사본이 빌드되지 않는다 -- 대조를 멈춘다")
                return 1
            results.append((name, "빌드 실패(= 잡힘 · 컴파일러)"))
            continue
        t = subprocess.run(["bash", "-c", f"ulimit -v 12000000; timeout 600 cargo test --release -j 2 --offline -q {' '.join(sel)} 2>&1"], cwd=mut, env=env, capture_output=True, text=True)
        failed = t.returncode != 0
        if failed and "무변경" in name:   # 무변경이 실패하면 어느 시험인지 남긴다(흔들리는 시험 찾기)
            print("\n".join(t.stdout.splitlines()[-40:]))
        want_fail = "무변경" not in name
        results.append((name, ("잡힘" if failed else "🔴못 잡음") if want_fail else ("통과(정상)" if not failed else "🔴무변경인데 실패")))
    shutil.rmtree(mut, ignore_errors=True)
    # 같은 target 이라 마지막 사본의 시험 실행 파일이 원본 자리 이름으로 남는다(CARGO_MANIFEST_DIR = 지운 사본) -- 원본을 다시 빌드하게 수정 시각만 올린다
    for p in [SRC / "src" / "lib.rs", *SRC.glob("tests/*.rs")]:
        os.utime(p)
    for sib in ("rtcore", "rtmodel", "rtorder"):
        if (work / sib).is_symlink():
            (work / sib).unlink()
    for name, r in results:
        print(f"{r:24} {name}")
    caught = sum(r.startswith(("잡힘", "빌드")) for n, r in results if "무변경" not in n)
    total = sum("무변경" not in n for n, _ in results)
    print(f"양성 대조 {caught}/{total} · 원본 rtserve 바이트 {'그대로' if tree_hash(SRC) == before else '🔴바뀜'}")
    noop_ok = all(r == "통과(정상)" for n, r in results if "무변경" in n)
    return 0 if caught == total and noop_ok and tree_hash(SRC) == before else 1


if __name__ == "__main__":
    sys.exit(main())
