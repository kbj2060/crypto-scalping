//! 원본 swr_cached(server.py:1463) 의 «보이는 동작»: 신선(나이 < ttl) = 그대로 · 묵음(ttl ≤ 나이 < ttl + max_stale) = 옛 값을 바로 내고
//! 뒤에서 한 번 갱신(키마다 하나만 · 갱신이 실패하면 로그만 남기고 옛 값 유지) · 그보다 오래됨/없음 = 키 잠금을 잡고 기다려 새로 만든다
//! (만드는 중 예외는 부른 쪽으로 · 캐시에 안 남는다). 나이는 단조 시계(원본 time.monotonic). 값 None(null)은 «없음»으로 본다(원본과 같다).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::py::{PyErr, R, V};

pub type Produce = Arc<dyn Fn() -> R<V> + Send + Sync>;

#[derive(Default)]
struct Slot {
    ts: f64,
    payload: Option<V>,
    refreshing: bool,
    lock: Arc<Mutex<()>>,
}

pub struct Swr {
    slots: Mutex<HashMap<String, Slot>>,
    /// 단조 시계(초) -- 시험이 바꿔 끼운다
    pub mono: Arc<dyn Fn() -> f64 + Send + Sync>,
    pub log: Arc<dyn Fn(&str) + Send + Sync>,
}

impl Default for Swr {
    fn default() -> Swr {
        let t0 = std::time::Instant::now();
        Swr { slots: Mutex::default(), mono: Arc::new(move || t0.elapsed().as_secs_f64() + 1e6), log: Arc::new(|l| println!("{l}")) }
    }
}

impl Swr {
    pub fn get(self: &Arc<Self>, key: &str, ttl: f64, max_stale: f64, produce: Produce) -> R<V> {
        let now = (self.mono)();
        let lock = {
            let mut g = self.slots.lock().unwrap();
            let s = g.entry(key.to_string()).or_default();
            if let Some(p) = &s.payload {
                let age = now - s.ts;
                if age < ttl {
                    return Ok(p.clone());
                }
                if age < ttl + max_stale {
                    let p = p.clone();
                    if !s.refreshing {
                        s.refreshing = true;
                        let (me, key, lock, produce) = (self.clone(), key.to_string(), s.lock.clone(), produce);
                        std::thread::spawn(move || me.refresh(&key, ttl, &lock, &produce));
                    }
                    return Ok(p);
                }
            }
            s.lock.clone()
        };
        let _held = lock.lock().unwrap();
        {
            let g = self.slots.lock().unwrap();
            if let Some(s) = g.get(key) {
                if let Some(p) = &s.payload {
                    if (self.mono)() - s.ts < ttl {
                        return Ok(p.clone());
                    }
                }
            }
        }
        let v = produce()?;
        self.store(key, &v);
        Ok(v)
    }

    /// 원본 api_footprint_daily_cells 의 칸 키 정리(server.py:3874~3877): 저장된 값이 max 개를 넘으면 접두사가 맞는 키 중 오래된(ts) 절반을 지운다
    pub fn prune_oldest_half(&self, max: usize, prefixes: &[&str]) {
        let mut g = self.slots.lock().unwrap();
        if g.values().filter(|s| s.payload.is_some()).count() <= max {
            return;
        }
        let mut old: Vec<(f64, String)> = g.iter().filter(|(k, s)| s.payload.is_some() && prefixes.iter().any(|p| k.starts_with(p))).map(|(k, s)| (s.ts, k.clone())).collect();
        old.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (_, k) in &old[..old.len() / 2] {
            g.remove(k);
        }
    }

    fn store(&self, key: &str, v: &V) {
        let mut g = self.slots.lock().unwrap();
        let s = g.entry(key.to_string()).or_default();
        s.payload = if v.is_null() { None } else { Some(v.clone()) };
        s.ts = (self.mono)();
    }

    /// 원본 _schedule_refresh._run: 잠금을 잡고 · 그사이 누가 갱신했으면 그만 · 실패는 로그만
    fn refresh(&self, key: &str, ttl: f64, lock: &Mutex<()>, produce: &Produce) {
        {
            let _held = lock.lock().unwrap();
            let fresh = self.slots.lock().unwrap().get(key).is_some_and(|s| (self.mono)() - s.ts < ttl);
            if !fresh {
                match produce() {
                    Ok(v) => self.store(key, &v),
                    Err(PyErr(m)) => (self.log)(&format!("cache refresh failed for {key} (still serving stale): {m}")),
                }
            }
        }
        if let Some(s) = self.slots.lock().unwrap().get_mut(key) {
            s.refreshing = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicI64, Ordering};

    fn rig() -> (Arc<Swr>, Arc<Mutex<f64>>, Arc<Mutex<Vec<String>>>) {
        let t = Arc::new(Mutex::new(0.0));
        let logs: Arc<Mutex<Vec<String>>> = Default::default();
        let (tc, lc) = (t.clone(), logs.clone());
        (Arc::new(Swr { slots: Mutex::default(), mono: Arc::new(move || *tc.lock().unwrap()), log: Arc::new(move |l| lc.lock().unwrap().push(l.into())) }), t, logs)
    }

    fn wait_idle(s: &Arc<Swr>, key: &str) {
        for _ in 0..500 {
            if !s.slots.lock().unwrap().get(key).is_some_and(|x| x.refreshing) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("갱신이 안 끝났다");
    }

    #[test]
    fn fresh_stale_cold_and_failure() {
        let (s, t, logs) = rig();
        let n = Arc::new(AtomicI64::new(0));
        let fail = Arc::new(Mutex::new(false));
        let (nc, fc) = (n.clone(), fail.clone());
        let p: Produce = Arc::new(move || {
            if *fc.lock().unwrap() {
                return Err(PyErr("ValueError: boom".into()));
            }
            Ok(json!(nc.fetch_add(1, Ordering::SeqCst)))
        });
        assert_eq!(s.get("k", 10.0, 20.0, p.clone()).unwrap(), json!(0)); // 없음 → 만든다
        *t.lock().unwrap() = 9.9;
        assert_eq!(s.get("k", 10.0, 20.0, p.clone()).unwrap(), json!(0)); // 신선
        *t.lock().unwrap() = 15.0;
        assert_eq!(s.get("k", 10.0, 20.0, p.clone()).unwrap(), json!(0)); // 묵음 → 옛 값 + 뒤에서 갱신
        wait_idle(&s, "k");
        assert_eq!(s.get("k", 10.0, 20.0, p.clone()).unwrap(), json!(1)); // 갱신된 값(시각 15 기준 신선)
        *fail.lock().unwrap() = true;
        *t.lock().unwrap() = 30.0;
        assert_eq!(s.get("k", 10.0, 20.0, p.clone()).unwrap(), json!(1)); // 묵음 · 갱신 실패 → 옛 값 유지
        wait_idle(&s, "k");
        assert_eq!(logs.lock().unwrap().len(), 1);
        assert_eq!(s.get("k", 10.0, 20.0, p.clone()).unwrap(), json!(1));
        *t.lock().unwrap() = 46.0; // 나이 31 ≥ ttl + max_stale → 기다려 새로 · 실패는 부른 쪽으로
        assert!(s.get("k", 10.0, 20.0, p.clone()).is_err());
        *fail.lock().unwrap() = false;
        assert_eq!(s.get("k", 10.0, 0.0, p.clone()).unwrap(), json!(2));
        *t.lock().unwrap() = 56.0; // max_stale 0: 나이 = ttl 이면 곧바로 새로(옛 값 안 냄)
        assert_eq!(s.get("k", 10.0, 0.0, p).unwrap(), json!(3));
    }
}
