//! 텔레그램 알림(2026-11-21) -- 원본 `scripts/ops_watchdog.send_telegram` + 부르는 자리 `bracket_tick_alert`(html.escape 해서 보냄).
//! 원본에서 주문 경로가 텔레그램을 보내는 자리는 이것 하나다(감시가 건너뛴 칸 · 칸·예외 종류마다 한 번 -- `Watcher::tick_alert`).
//! 본문 바이트는 원본 `json.dumps`(ensure_ascii · ", " ": ") 그대로. 시험 = tests/telegram.rs(127.0.0.1 가짜 서버 · 원본을 돌린 정답지).

use crate::pyfmt::json_str;
use std::time::Duration;

pub const API: &str = "https://api.telegram.org";
pub const TIMEOUT_S: f64 = 8.0;

#[derive(Clone)]
pub struct Telegram {
    /// 운영 = API · 시험 = http://127.0.0.1:…
    pub base: String,
    token: String,
    chat_id: String,
}

/// 토큰이 로그·Debug 로 새지 않게.
impl std::fmt::Debug for Telegram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Telegram({}, 토큰 {}자)", self.base, self.token.len())
    }
}

impl Telegram {
    /// 원본: TELEGRAM_BOT_TOKEN·TELEGRAM_CHAT_ID 둘 중 하나라도 비면 None(«설정 안 됨» -- 보내지 않는다).
    pub fn from_env(get: &dyn Fn(&str) -> Option<String>) -> Option<Telegram> {
        let (token, chat_id) = (get("TELEGRAM_BOT_TOKEN").unwrap_or_default(), get("TELEGRAM_CHAT_ID").unwrap_or_default());
        if token.is_empty() || chat_id.is_empty() {
            return None;
        }
        Some(Telegram { base: get("RTORDER_TELEGRAM_BASE").filter(|s| !s.is_empty()).unwrap_or_else(|| API.into()), token, chat_id })
    }

    pub fn body(&self, text: &str) -> String {
        format!("{{\"chat_id\": {}, \"text\": {}, \"parse_mode\": \"HTML\"}}", json_str(&self.chat_id), json_str(text))
    }

    /// 원본 send_telegram(이미 html.escape 된 문구) -- true = 보냄 · false = 실패(네트워크·HTTP 오류 -- 원본 OSError). 막는 호출(뒤에서 부를 것).
    pub fn send(&self, text: &str) -> bool {
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs_f64(TIMEOUT_S))).build().into();
        agent.post(format!("{}/bot{}/sendMessage", self.base, self.token)).header("Content-Type", "application/json").send(self.body(text).as_bytes())
            .and_then(|mut r| r.body_mut().read_to_vec()).is_ok()
    }
}

/// 파이썬 html.escape(quote=True).
pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#x27;")
}
