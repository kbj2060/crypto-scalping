//! 자료 흐름 일치(compare/feed.py): 같은 «서버 응답 열 + 체결 열»(시나리오 JSON)을 feed::Feed 에 넣고 단계마다 봉·칸 상태를 JSON 으로 낸다.
//! cargo run --release --example feed_replay -- <시나리오.json>   → 표준출력 = [덤프…]
use rtui::feed::{Feed, Kind, Req, Source, Urls, history_retry_due};
use rtui::view::{tf_candles, tf_levels};
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn main() {
    let sc: Value = serde_json::from_str(&std::fs::read_to_string(std::env::args().nth(1).expect("시나리오.json")).unwrap()).unwrap();
    let mut f = Feed::new("eth", "ETHUSDT", Source::Poll, Urls::default(), false);
    let (mut out, mut since) = (vec![], vec![]);
    let load = sc["load"].as_f64().unwrap();
    // 페이지를 연 순간의 tick 한 번(풋프린트·이력 묻기는 404 · 소켓은 열림)
    f.tick(load, false, false);
    f.on_footprint(None);
    f.on_history(None, load);
    for e in sc["events"].as_array().unwrap() {
        let now = e["now"].as_f64().unwrap_or(0.0);
        let body = || e["body"].as_str().map(str::to_owned);
        match e["k"].as_str().unwrap() {
            "fp" => {
                f.out.clear();
                f.refresh_footprint(now);
                for r in f.out.drain(..) { if let Req::Get { kind: Kind::Footprint, url } = r { since.push(url.rsplit("since=").next().unwrap().to_owned()); } }
                f.on_footprint(body().as_deref());
            }
            "hist" => f.on_history(body().as_deref(), now),
            "trade" => f.on_ws(e["msg"].as_str().unwrap(), now),
            "reconnect" => { f.on_ws_close(); f.ensure_ws(now, false); }
            "retry" => out.push(json!({"retry": e["cases"].as_array().unwrap().iter().map(|c| history_retry_due(c[0].as_i64().unwrap(), c[1].as_f64(), c[2].as_f64().unwrap(), c[3].as_f64().unwrap())).collect::<Vec<_>>()})),
            "dump" => {
                let k = e["tf"].as_i64().unwrap_or(1);
                let want = k > 1;
                let merged = f.for_chart(want);
                let bars = f.bars_now(want);
                let by_time = merged.as_ref().map(|(bucket, m)| {
                    let mut by5: BTreeMap<i64, &Vec<Vec<f64>>> = bars.iter().filter(|b| !b.levels.is_empty()).map(|b| (b.time, &b.levels)).collect();
                    if let Some((t, lv)) = m { by5.insert(*t, lv); }
                    if k > 1 { json!(tf_levels(&by5, *bucket, k).into_iter().collect::<Vec<_>>()) } else { json!(by5.into_iter().collect::<Vec<_>>()) }
                });
                let bucket = merged.as_ref().map_or(0.5, |m| m.0);
                let lane = if k > 1 { json!(tf_levels(&bars.iter().map(|b| (b.time, &b.levels)).collect(), bucket, k).into_iter().map(|(t, l)| json!([t, l, null])).collect::<Vec<_>>()) }
                           else { json!(bars.iter().map(|b| json!([b.time, b.levels, b.dhl])).collect::<Vec<_>>()) };
                let cs: Vec<_> = tf_candles(&f.chart_full(), k).iter().map(|c| json!([c.time, c.open, c.high, c.low, c.close])).collect();
                let l = &f.live;
                out.push(json!({"tf": k, "byTime": by_time, "lane": if f.fp.is_some() { lane } else { json!(null) }, "candles": cs,
                    "live": {"barStart": l.bar_start, "since": if l.since.is_finite() { json!(l.since) } else { json!(null) }, "cells": l.cells.iter().map(|(k, c)| json!([k, c])).collect::<Vec<_>>()},
                    "price": f.price, "fast": f.last_fast, "full": f.last_full, "nbars": f.fp_bars.len()}));
            }
            k => panic!("모르는 사건 {k}"),
        }
    }
    println!("{}", json!({"dumps": out, "since": since}));
}
