//! 4h·15분/1시간 방향 예측 루프 -- 서버 dashboard/server.py `pred_record_loop`(:2841)의 몫을 따로 돈다(그림자 · rtmodel/SHADOW.md).
//! 5분 봉이 닫힌 뒤 +20초마다 코인(ETH·BTC·SOL·XRP · 사이 3초)마다 15분·1시간(dir_short) → 4h(dir4h) 예측을 내고
//! 적중 기록(pred_record/{코인}.jsonl -- 전진 검정 fwd_dirshort 가 읽는 꼴)과 마지막 payload(dir4h_<코인>.json · dir_short_<코인>.json)를 쓴다.
//! 🔴ETH 15분·1시간(75열판 · 호가 깊이 sqlite)은 아직 못 옮겼다 -- 건너뛴다. 상대 강도(xsec.jsonl)도 아직.
//!
//! 환경변수(기본값):
//!   RT_DIR_OUT      data/live/rt_dirpred     쓰는 곳. 🔴data/live(= 파이썬 서버의 data/live/pred_record)면 종료코드 2(전환 때만 RT_DIR_TAKEOVER=1)
//!   RT_DIR_MODELS   <RT_DIR_OUT>             dir4h_<코인>.json(tools/export_dir4h.py) · ds_<코인>.json·oos_<코인>.json(tools/export_dirshort.py)
//!   RT_DIR_REST     https://fapi.binance.com
//!   RT_SR_BAN_FILE  data/live/binance_ban_until   (공용 밴 파일 -- 밴 중이면 보내지 않고 그 코인은 이번 회차 실패)
//!   RT_DIR_NOW_MS   (시험만) 고정 시각으로 한 회차만 돌고 끝낸다
use rtmodel::ban::{ban_left, ban_note};
use rtmodel::dirrest::{closes, daily_closes, Record, Rest};
use rtmodel::{dir4h, dirshort, read_json};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Duration;

const COINS: [&str; 4] = ["ETH", "BTC", "SOL", "XRP"];

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

/// 원본 _get: requests.get(FAPI + path, params, timeout=20) · raise_for_status · json. 밴 중이면 보내지 않는다.
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

fn write_atomic(p: &Path, s: &str) {
    let tmp = p.with_extension("tmp");
    if let Err(e) = std::fs::write(&tmp, s).and_then(|_| std::fs::rename(&tmp, p)) {
        eprintln!("{}: {e}", p.display());
    }
}

fn main() {
    let out = PathBuf::from(env("RT_DIR_OUT", "data/live/rt_dirpred"));
    let python = out.file_name() == Some("live".as_ref()) && out.parent().and_then(|d| d.file_name()) == Some("data".as_ref());
    if python && env("RT_DIR_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 서버의 기록 자리다(pred_record) -- 그림자는 다른 경로에(기본 data/live/rt_dirpred). 전환이면 RT_DIR_TAKEOVER=1", out.display()));
    }
    let models = PathBuf::from(env("RT_DIR_MODELS", &out.to_string_lossy()));
    let (base, ban) = (env("RT_DIR_REST", "https://fapi.binance.com"), PathBuf::from(env("RT_SR_BAN_FILE", "data/live/binance_ban_until")));
    let fixed: Option<i64> = std::env::var("RT_DIR_NOW_MS").ok().map(|v| v.parse().unwrap_or_else(|_| die(2, "RT_DIR_NOW_MS")));
    std::fs::create_dir_all(out.join("pred_record")).unwrap_or_else(|e| die(2, &format!("{}: {e}", out.display())));
    let a4: Vec<(String, dir4h::Art)> = COINS.iter().filter_map(|c| dir4h::Art::load(&models.join(format!("dir4h_{}.json", c.to_lowercase()))).ok().map(|a| (c.to_string(), a))).collect();
    let mut ass: Vec<(String, dirshort::Art, Vec<(&str, Vec<f64>)>)> = vec![];
    for c in COINS {
        let (Ok(m), Ok(o)) = (read_json::<Value>(&models.join(format!("ds_{}.json", c.to_lowercase()))), read_json::<Value>(&models.join(format!("oos_{}.json", c.to_lowercase())))) else { continue };
        let art = dirshort::Art::from_json(&m).unwrap_or_else(|e| die(2, &format!("ds_{c}: {e}")));
        let oos = dirshort::NAMES.iter().map(|(_, nm)| (*nm, o[*nm].as_array().map_or(vec![], |a| a.iter().map(|b| f64::from_bits(b.as_u64().unwrap_or(0))).collect()))).collect();
        ass.push((c.to_string(), art, oos));
    }
    println!("시작 · 4h {:?} · 15분·1시간 {:?} · 기록 {}", a4.iter().map(|x| &x.0).collect::<Vec<_>>(), ass.iter().map(|x| &x.0).collect::<Vec<_>>(), out.display());
    if a4.is_empty() && ass.is_empty() {
        die(2, &format!("모델이 하나도 없다: {}", models.display()));
    }
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(20))).http_status_as_error(false).proxy(None).build().into();
    let mut g = |path: &str, params: &[(&'static str, String)]| get(&agent, &base, &ban, path, params);
    let (mut r4, mut rs, mut rec) = (Rest::default(), Rest::default(), Record::default()); // 서버의 두 모듈 캐시
    loop {
        if fixed.is_none() {
            std::thread::sleep(Duration::from_secs_f64(300.0 - wall() % 300.0 + 20.0));
        }
        for c in COINS {
            if fixed.is_none() {
                std::thread::sleep(Duration::from_secs(3)); // 재기동 직후 첫 호출은 코인마다 31일치 REST -- 한꺼번에 몰리지 않게
            }
            for kind in ["dir_short", "dir4h"] {
                let now = fixed.unwrap_or_else(|| (wall() * 1000.0) as i64);
                let res: Result<(), String> = (|| {
                    let (r, pay) = if kind == "dir_short" {
                        let Some((_, art, oos)) = ass.iter().find(|x| x.0 == c) else { return Ok(()) };
                        let fr = rs.frames(&mut g, now, now as f64, c)?;
                        let dc = daily_closes(&mut g, now, c)?;
                        (&rs, dirshort::payload(art, &dir4h::build_features(&dir4h::frames_for(&fr, c), &dc)?, 13, oos))
                    } else {
                        let Some((_, art)) = a4.iter().find(|x| x.0 == c) else { return Ok(()) };
                        let fr = r4.frames(&mut g, now, now as f64, c)?;
                        let dc = daily_closes(&mut g, now, c)?;
                        (&r4, dir4h::payload(art, &dir4h::build_features(&dir4h::frames_for(&fr, c), &dc)?))
                    };
                    write_atomic(&out.join(format!("{kind}_{}.json", c.to_lowercase())), &pay.to_string());
                    // pred_record_safe: 기록 실패가 예측을 깨지 않게(삼키고 로그)
                    if let Some(cl) = closes(r, c) {
                        let w = if kind == "dir4h" { rec.h4(&out.join("pred_record"), c, &pay, &cl) } else { rec.short(&out.join("pred_record"), c, &pay, &cl) };
                        if let Err(e) = w {
                            eprintln!("[pred_record] {kind} {c}: {e}");
                        }
                    }
                    Ok(())
                })();
                if let Err(e) = res {
                    eprintln!("[pred_record_loop] {kind} {c}: {e}");
                }
            }
        }
        if fixed.is_some() {
            return;
        }
    }
}
