//! 파이썬 워커(scripts/live_macro_smile_worker_20261128.py)가 원본 핸들러 본문을 그대로 쓴 파일을 읽어 답한다(2026-11-28).
//! 파일 = 원본 `json.dumps` 바이트 그대로 → 본문도 그대로. 없거나 원본 swr 수명(ttl + max_stale)보다 낡으면 원본 실패 갈래 모양.
use crate::{Config, Handler, Req, Resp};
use serde_json::json;

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    match p {
        "/api/macro-trend" => vec![("GET", api_macro_trend)],
        "/api/opt-smile" => vec![("GET", api_opt_smile)],
        _ => Vec::new(),
    }
}

/// 파일 바이트(수정 시각이 `max_age` 초 안일 때만) · Err = 실패 갈래에 실을 사유
fn read_fresh(cfg: &Config, name: &str, max_age: f64) -> Result<Vec<u8>, String> {
    let p = cfg.live_dir().join(name);
    let (m, body) = match (std::fs::metadata(&p), std::fs::read(&p)) {
        (Ok(m), Ok(b)) => (m, b),
        _ => return Err(format!("FileNotFoundError: data/live/{name}")),
    };
    let mt = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    let age = (cfg.now)() - mt;
    if age > max_age {
        return Err(format!("stale: data/live/{name} age {age:.0}s"));
    }
    Ok(body)
}

/// 원본 api_macro_trend: swr 3600 · max_stale 6h · 실패 = macro_trend_payload 의 except 모양
fn api_macro_trend(cfg: &Config, _: &Req) -> Resp {
    match read_fresh(cfg, "macro_trend.json", 3600.0 + 6.0 * 3600.0) {
        Ok(b) => Resp::bytes("application/json; charset=utf-8", b),
        Err(e) => Resp::json(&json!({"ok": false, "error": e, "rows": [], "summary": {}, "errors": {}})),
    }
}

/// 원본 api_opt_smile: asset 허용 eth·btc·xrp · swr 60 · max_stale STALE_GRACE_SECONDS(600) · 실패 = deribit_unavailable
fn api_opt_smile(cfg: &Config, req: &Req) -> Resp {
    let a = req.q("asset").unwrap_or("eth").to_lowercase();
    if !matches!(a.as_str(), "eth" | "btc" | "xrp") {
        return Resp::json(&json!({"ok": false, "reason": "eth_btc_xrp_only"}));
    }
    match read_fresh(cfg, &format!("opt_smile_{a}.json"), 60.0 + 600.0) {
        Ok(b) => Resp::bytes("application/json; charset=utf-8", b),
        Err(_) => Resp::json(&json!({"ok": false, "reason": "deribit_unavailable"})),
    }
}
