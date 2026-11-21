//! 바깥 REST 의 한 문 (2026-11-22) -- rtmodel 실행 파일(rt_dirpred·rt_regime·rt_dirfwd·rt_radar·rt_sr_lean)의 모든 HTTP GET 이 여기를 지난다.
//! 이 서버 IP 는 주문 경로와 바이낸스 한도(가중치 2,400/분)를 같이 쓴다(평소 최고 1,431 이상 실측) -- 그래서:
//!   ① 공용 밴 파일(ban.rs · scripts/binance_ban_guard.py)이 밴 중이면 보내지 않는다 · 418/429 를 받으면 그 파일에 적어 다음 요청부터 즉시 멈춘다.
//!   ② 응답 머리 `x-mbx-used-weight-1m` 이 문턱(RT_REST_WEIGHT_LIMIT · 기본 1,200)을 넘으면 다음 분 경계(+0.5초)까지 아무것도 안 보낸다.
//!   ③ 요청 사이 최소 간격(RT_REST_GAP_MS · 기본 200ms).
//! 원본 파이썬(rest_frames 등)은 쉬지 않고 연달아 부른다 -- 여기 ②③은 원본보다 느리게만 만든다(값은 그대로).
use crate::ban::{ban_left, ban_note};
use std::path::PathBuf;
use std::time::Duration;

pub enum Fail {
    /// 밴 중이라 안 보냄(남은 초)
    Banned(f64),
    /// 연결·전송 실패
    Conn(String),
    /// 본문 읽기 실패
    Body(String),
}

pub struct Gate {
    agent: ureq::Agent,
    ban: PathBuf,
    limit: f64,
    gap: f64,
    last: f64,
    hold_until: f64,
    now: Box<dyn FnMut() -> f64>,
    nap: Box<dyn FnMut(f64)>,
    /// 보낸 요청 수 · 마지막으로 본 가중치 · 문턱 때문에 쉰 초(로그·시험용)
    pub sent: u64,
    pub weight: Option<f64>,
    pub held_s: f64,
}

fn wall() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

impl Gate {
    pub fn new(timeout: Duration, ban: PathBuf) -> Gate {
        let num = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
        Gate {
            agent: ureq::Agent::config_builder().timeout_global(Some(timeout)).http_status_as_error(false).proxy(None).build().into(),
            ban,
            limit: num("RT_REST_WEIGHT_LIMIT", 1200.0),
            gap: num("RT_REST_GAP_MS", 200.0) / 1000.0,
            last: f64::NEG_INFINITY,
            hold_until: 0.0,
            now: Box::new(wall),
            nap: Box::new(|s| std::thread::sleep(Duration::from_secs_f64(s.max(0.0)))),
            sent: 0,
            weight: None,
            held_s: 0.0,
        }
    }
    /// 시험용: 시계·잠을 바꾼다(가짜 시계를 잠이 앞으로 민다).
    pub fn with_clock(mut self, now: Box<dyn FnMut() -> f64>, nap: Box<dyn FnMut(f64)>) -> Gate {
        (self.now, self.nap) = (now, nap);
        self
    }
    pub fn ban_file(&self) -> &std::path::Path {
        &self.ban
    }

    /// GET 한 번 → (상태, 본문). binance = 바이낸스 주소(밴 파일·가중치 머리를 본다 · Deribit 등은 간격만).
    pub fn send(&mut self, url: &str, binance: bool) -> Result<(u16, String), Fail> {
        let t = (self.now)();
        if self.hold_until > t {
            let s = self.hold_until - t;
            eprintln!("[gate] 가중치 {:.0} > 문턱 {:.0} -- 다음 분 경계까지 {s:.1}초 쉼", self.weight.unwrap_or(0.0), self.limit);
            (self.nap)(s);
            self.held_s += s;
        }
        if binance {
            let left = ban_left(&self.ban, (self.now)());
            if left > 0.0 {
                return Err(Fail::Banned(left));
            }
        }
        let t = (self.now)();
        if t - self.last < self.gap {
            (self.nap)(self.gap - (t - self.last));
        }
        self.last = (self.now)();
        self.sent += 1;
        let mut r = self.agent.get(url).call().map_err(|e| Fail::Conn(e.to_string()))?;
        let h = |n: &str| r.headers().get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
        let (status, retry, w) = (r.status().as_u16(), h("retry-after"), h("x-mbx-used-weight-1m"));
        let body = r.body_mut().read_to_string().map_err(|e| Fail::Body(e.to_string()))?;
        if binance {
            ban_note(&self.ban, status, &body, retry.as_deref(), (self.now)());
            if let Some(w) = w.and_then(|x| x.trim().parse::<f64>().ok()) {
                self.weight = Some(w);
                if w > self.limit {
                    self.hold_until = ((self.now)() / 60.0).floor() * 60.0 + 60.5;
                }
            }
        }
        Ok((status, body))
    }
}
