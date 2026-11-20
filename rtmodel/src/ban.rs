//! 바이낸스 IP 밴 공용 파일(scripts/binance_ban_guard.py 와 같은 파일 · 같은 규칙) -- 실행 파일들이 REST 를 보내기 전에 본다 (2026-11-21).
//! rtcollect/src/rest.rs 의 ban_left·ban_note 와 같은 식(크레이트가 달라 옮겨 적었다).
use std::path::Path;

/// binance_ban_guard.ban_remaining -- rtcollect/src/rest.rs ban_left 와 같은 식.
pub fn ban_left(file: &Path, now: f64) -> f64 {
    std::fs::read_to_string(file).ok().and_then(|s| s.trim().parse::<f64>().ok()).map_or(0.0, |u| (u - now).max(0.0))
}

/// binance_ban_guard.note -- rtcollect/src/rest.rs ban_note 와 같은 식: 418 = 본문 «banned until <13자리 ms>», 429 = Retry-After(없으면 30초 · 418 은 120초).
pub fn ban_note(file: &Path, status: u16, body: &str, retry_after: Option<&str>, now: f64) {
    if status != 418 && status != 429 {
        return;
    }
    let banned = body.find("banned until ").and_then(|i| {
        let d = &body[i + 13..];
        (d.len() >= 13 && d[..13].bytes().all(|b| b.is_ascii_digit())).then(|| d[..13].parse::<f64>().unwrap() / 1000.0)
    });
    let until = banned.unwrap_or_else(|| match retry_after.filter(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit())) {
        Some(r) => now + r.parse::<f64>().unwrap(),
        None => now + if status == 418 { 120.0 } else { 30.0 },
    });
    if until <= now + ban_left(file, now) {
        return;
    }
    if let Some(d) = file.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let tmp = file.with_extension(format!("{}.tmp", std::process::id()));
    if std::fs::write(&tmp, format!("{until:.3}\n")).and_then(|_| std::fs::rename(&tmp, file)).is_err() {
        eprintln!("⚠️밴 기록 실패: {}", file.display());
    }
}

