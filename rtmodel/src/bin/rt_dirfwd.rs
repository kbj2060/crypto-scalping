//! 방향 전진 기록기 cron 한 회차 -- scripts/live_eth_direction_forward_recorder_20261010.py `--record` 를 옮긴 것(00·08·16 UTC 31분 뒤 · 원본처럼 부르고 끝난다).
//! 3자산(ETH·BTC·SOL) 1일봉 93(형성 봉 뺌) + 5분봉 320(닫힌 봉) → rtmodel::signals::decide → 줄 하나씩 덧붙임.
//!
//! 환경변수(기본값):
//!   RT_DIRFWD_OUT     data/live/rt_dirfwd    eth_direction_forward.jsonl 을 쓰는 디렉터리. 🔴data/live(= 원본 자리)면 종료코드 2(전환 때만 RT_DIRFWD_TAKEOVER=1)
//!   RT_DIRFWD_REST    https://fapi.binance.com
//!   RT_SR_BAN_FILE    data/live/binance_ban_until   (밴 중이면 보내지 않고 실패)
//!   RT_DIRFWD_NOW_MS  (시험만) 고정 시각
//! 종료코드: 0 정상 · 1 실패(시각 밖 · 봉 빠짐 · REST -- 앞 자산 줄은 남는다, 원본과 같음) · 2 설정 거부.
use rtmodel::gate::{Fail, Gate};
use rtmodel::signals::record;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.into())
}
fn die(code: i32, msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(code)
}
fn wall() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// 원본 urlopen(timeout 15) · json. 밴 중이면 보내지 않는다.
fn get(gate: &mut Gate, url: &str) -> Result<Value, String> {
    let (status, body) = gate.send(url, true).map_err(|e| match e {
        Fail::Banned(left) => format!("binance IP ban: {left:.0}s left -- request not sent"),
        Fail::Conn(e) | Fail::Body(e) => e,
    })?;
    if status >= 400 {
        return Err(format!("HTTPError {status}"));
    }
    serde_json::from_str(&body).map_err(|e| format!("JSONDecodeError {e}"))
}

fn main() {
    let out = PathBuf::from(env("RT_DIRFWD_OUT", "data/live/rt_dirfwd"));
    let python = out.file_name() == Some("live".as_ref()) && out.parent().and_then(|d| d.file_name()) == Some("data".as_ref());
    if python && env("RT_DIRFWD_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 원본 기록기의 자리다(eth_direction_forward.jsonl) -- 그림자는 다른 경로에(기본 data/live/rt_dirfwd). 전환이면 RT_DIRFWD_TAKEOVER=1", out.display()));
    }
    let now: i64 = std::env::var("RT_DIRFWD_NOW_MS").ok().map_or_else(|| (wall() * 1000.0) as i64, |v| v.parse().unwrap_or_else(|_| die(2, "RT_DIRFWD_NOW_MS")));
    let (base, ban) = (env("RT_DIRFWD_REST", "https://fapi.binance.com"), PathBuf::from(env("RT_SR_BAN_FILE", "data/live/binance_ban_until")));
    let mut gate = Gate::new(Duration::from_secs(15), ban); // 모든 바깥 요청의 한 문(rtmodel::gate)
    let mut g = |url: &str| get(&mut gate, url);
    match record(&mut g, &base, now, &out.join("eth_direction_forward.jsonl")) {
        Ok(n) => println!("기록 {n}줄 {}", rtmodel::risk_radar::iso(now)),
        Err(e) => die(1, &format!("FAIL {e}")),
    }
}
