//! «띠 안·이탈» 전진 기록 cron 한 회차 -- scripts/live_regime_band_cron_20261102.py `main()` 을 옮긴 것(원본처럼 매시 02분 cron 이 부르고 끝난다).
//! 4코인 5분봉(1500 + 그 앞 1500 · 닫힌 봉만 · 꼬리 2700) → 마지막 정시 마감 봉까지 → 예측 기록 + 4시간 지난 행 채점 + 요약(rtmodel::regime_band).
//!
//! 환경변수(기본값):
//!   RT_REGIME_OUT     data/live/rt_regime_band   regime_band_forward.jsonl · _scored.jsonl · _summary.json. 🔴data/live(= 파이썬 크론 자리)면 종료코드 2(전환 때만 RT_REGIME_TAKEOVER=1)
//!   RT_REGIME_MODEL   <RT_REGIME_OUT>/regime_band.json   (python rtmodel/tools/export_regime.py)
//!   RT_REGIME_SRC_DIR (있으면) 원본 joblib 디렉터리 -- 그 joblib 이 모델 JSON 보다 새것이면(재학습 뒤 내보내기 안 함) 아무것도 안 쓰고 종료코드 1
//!   RT_REGIME_REST    https://fapi.binance.com
//!   RT_SR_BAN_FILE    data/live/binance_ban_until   (밴 중이면 보내지 않고 실패)
//!   RT_REGIME_NOW_MS  (시험만) 고정 시각
//! 종료코드: 0 정상(출력 JSON 한 줄) · 1 실패(원본처럼 아무것도 안 쓰고 "FAIL …") · 2 설정 거부.
use rtmodel::ban::{ban_left, ban_note};
use rtmodel::read_json;
use rtmodel::regime_band::{cron, Bundle};
use serde_json::Value;
use std::path::{Path, PathBuf};
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

/// 원본 rest_klines: urlopen(timeout 20) · json. 밴 중이면 보내지 않는다.
fn get(agent: &ureq::Agent, base: &str, ban: &Path, path: &str, params: &[(&'static str, String)]) -> Result<Value, String> {
    let left = ban_left(ban, wall());
    if left > 0.0 {
        return Err(format!("binance IP ban: {left:.0}s left -- request not sent"));
    }
    let q: Vec<String> = params.iter().map(|(k, v)| format!("{k}={v}")).collect();
    let mut r = agent.get(&format!("{base}{path}?{}", q.join("&"))).call().map_err(|e| format!("{e}"))?;
    let h = |n: &str| r.headers().get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
    let (status, retry) = (r.status().as_u16(), h("retry-after"));
    let body = r.body_mut().read_to_string().map_err(|e| format!("{e}"))?;
    ban_note(ban, status, &body, retry.as_deref(), wall());
    if status >= 400 {
        return Err(format!("HTTPError {status}"));
    }
    serde_json::from_str(&body).map_err(|e| format!("JSONDecodeError {e}"))
}

fn main() {
    let out = PathBuf::from(env("RT_REGIME_OUT", "data/live/rt_regime_band"));
    let python = out.file_name() == Some("live".as_ref()) && out.parent().and_then(|d| d.file_name()) == Some("data".as_ref());
    if python && env("RT_REGIME_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 크론의 기록 자리다(regime_band_forward*) -- 그림자는 다른 경로에(기본 data/live/rt_regime_band). 전환이면 RT_REGIME_TAKEOVER=1", out.display()));
    }
    let model = PathBuf::from(env("RT_REGIME_MODEL", &out.join("regime_band.json").to_string_lossy()));
    let mj = read_json::<Value>(&model).unwrap_or_else(|e| die(2, &format!("모델: {e} -- python rtmodel/tools/export_regime.py")));
    if let Ok(d) = std::env::var("RT_REGIME_SRC_DIR") {
        if let Some(why) = rtmodel::model_stale(&model, &mj, Path::new(&d)) {
            die(1, &format!("FAIL {why}")); // 원본과 다른 값을 기록하지 않는다
        }
    }
    let b = Bundle::from_json(&mj).unwrap_or_else(|e| die(2, &format!("모델: {e}")));
    let now: i64 = std::env::var("RT_REGIME_NOW_MS").ok().map_or_else(|| (wall() * 1000.0) as i64, |v| v.parse().unwrap_or_else(|_| die(2, "RT_REGIME_NOW_MS")));
    let (base, ban) = (env("RT_REGIME_REST", "https://fapi.binance.com"), PathBuf::from(env("RT_SR_BAN_FILE", "data/live/binance_ban_until")));
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(20))).http_status_as_error(false).proxy(None).build().into();
    let mut g = |path: &str, params: &[(&'static str, String)]| get(&agent, &base, &ban, path, params);
    let f = |n: &str| out.join(n);
    match cron(&mut g, now, &b, &f("regime_band_forward.jsonl"), &f("regime_band_forward_scored.jsonl"), &f("regime_band_forward_summary.json")) {
        Ok(s) => println!("{s}"),
        Err(e) => die(1, &format!("FAIL {e}")),
    }
}
