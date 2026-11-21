//! 응답 서버(rtserve)가 부를 «요청 때 계산하는» 핸들러 몫 -- 한 핸들러에 공개 함수 하나 (2026-11-21).
//! 입력 = 받아 둔 자료(캐시)·fetch 콜백 · 출력 = 본문(JSON 값 또는 글자). 캐시 시간(swr)·모델 다시 읽기(mtime)·밴 가드·예외 → {"ok": false, "reason"} 감싸기는 부르는 쪽.
//! 🔴Value 를 내는 것(dir4h·dir_short·QX4)은 json! 에 원본 dict 순서대로 키를 적었다 -- rtserve 처럼 serde_json `preserve_order` 를 켠 크레이트에서
//!   직렬화해야 원본과 같은 글자 순서가 된다(같은 빌드에서 기능이 합쳐진다). 실수 글자(1e-05 등)는 rtserve 의 파이썬 서식으로.
//! 계산 없이 파일만 읽는 핸들러(/api/regime-band · /api/risk-radar · /api/sr-lean · /api/pred-record)는 rtserve/src/api.rs 에 이미 있다 -- 여기 없음.
use crate::dirrest::{daily_closes, Get, Rest};
use crate::signals::{open_rev, vol4h, vol4h_rows, Fetch, Svm};
use crate::{dir4h, dirshort, qx4};
use serde_json::Value;
use std::collections::HashMap;

/// GET /api/dir4h (server.py dir4h_cached → d4.live_payload): REST 조립(rest 캐시) → 피쳐 65 → payload.
/// 적중 기록(pred_record_safe)은 부르는 쪽이 `Record::h4` 로(실패가 응답을 깨지 않게).
pub fn dir4h_body(rest: &mut Rest, get: &mut Get, now_ms: i64, wall_ms: f64, coin: &str, art: &dir4h::Art) -> Result<Value, String> {
    let fr = rest.frames(get, now_ms, wall_ms, coin)?;
    let dc = daily_closes(get, now_ms, coin)?;
    Ok(dir4h::payload(art, &dir4h::build_features(&dir4h::frames_for(&fr, coin), &dc)?))
}

/// GET /api/dir-short (server.py dir_short_cached → ds.live_payload, 최근 13봉). book = ETH 75열판의 호가 깊이 한 봉(`dirshort::book_status` 결과) ·
/// 65열판(BTC·SOL·XRP)은 None. 🔴원본은 4h 와 rest 캐시가 따로다(모듈이 다름) -- rest 도 따로 넘긴다.
pub fn dir_short_body(rest: &mut Rest, get: &mut Get, now_ms: i64, wall_ms: f64, coin: &str, art: &dirshort::Art, oos: &[(&str, Vec<f64>)],
                      book: Option<&dyn Fn(i64) -> (Option<Vec<f64>>, &'static str)>) -> Result<Value, String> {
    let fr = rest.frames(get, now_ms, wall_ms, coin)?;
    let dc = daily_closes(get, now_ms, coin)?;
    let f = dir4h::build_features(&dir4h::frames_for(&fr, coin), &dc)?;
    Ok(match book {
        Some(b) => dirshort::payload_book(art, &f, 13, oos, b),
        None => dirshort::payload(art, &f, 13, oos),
    })
}

/// GET /api/vol-forecast 의 QX4 쪽(server.py qx4_payload): 4코인 행 이어 받기(rows = 부르는 쪽이 들고 있는 상태) → qx4_from_klines.
/// None = HGB 대체로(이어 받기 실패·ok 아님). 반환 Value 에 "asset" 을 붙이는 것도 여기서({**q4, "asset": asset}).
/// 🔴QX4I(ETH·BTC 옵션 ATM IV 칸)는 Deribit 옵션 체결 받기(iv_now)를 안 옮겨 빠진다 -- `qx4::atm_iv`·`add_iv` 는 있다(체결 목록을 받으면 붙일 수 있음).
pub fn vol_forecast_qx4(fetch: &mut dyn FnMut(&[(&'static str, String)]) -> Vec<Value>, rows: &mut HashMap<String, Vec<Value>>, art: &qx4::Art, now_ms: f64, asset: &str) -> Option<Value> {
    if !qx4::refresh(fetch, rows) {
        return None;
    }
    let mut res = qx4::payload(rows, art, now_ms, &asset.to_uppercase());
    if res.get("ok") != Some(&Value::Bool(true)) {
        return None;
    }
    res.as_object_mut()?.insert("asset".into(), Value::String(asset.into()));
    Some(res)
}

/// GET /api/vol-forecast 의 HGB 대체 쪽(server.py api_vol_forecast produce): 5분봉 1000 × 10쪽 → vol4h_from_klines + "asset". 본문 글자.
pub fn vol_forecast_hgb(fetch: &mut Fetch, base: &str, symbol: &str, asset: &str, art: &Svm, now_ms: f64) -> Result<String, String> {
    let rows = vol4h_rows(fetch, base, symbol)?;
    let body = vol4h(&rows, art, now_ms);
    Ok(format!("{}, \"asset\": {}}}", &body[..body.len() - 1], crate::signals::jstr_ascii(asset)))
}

/// GET /api/open-rev (server.py api_open_rev): ETH 1일봉 93 · 5분봉 600 → open_rev_from_klines. 서버 시계는 실수 ms(time.time()*1000).
pub fn open_rev_body(fetch: &mut Fetch, base: &str, now_ms: f64) -> Result<String, String> {
    let q = |iv: &str, n: usize| format!("{base}/fapi/v1/klines?symbol=ETHUSDT&interval={iv}&limit={n}");
    let daily = fetch(&q("1d", 93))?;
    let m5 = fetch(&q("5m", 600))?;
    let (Some(d), Some(m)) = (daily.as_array(), m5.as_array()) else { return Err("klines".into()) };
    open_rev(d, m, now_ms, true)
}

/// GET /api/dir-signals (server.py api_dir_signals → dir_signals_payload) = `signals::dir_signals` 그대로(FRED 키는 부르는 쪽 환경에서).
pub use crate::signals::dir_signals as dir_signals_body;
