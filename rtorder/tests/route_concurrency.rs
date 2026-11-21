// G7(조립 뒤 칸 다시 보기)·G8(주문 잠금)의 동시 요청 시험: 청산 접수가 계좌 조회(await)에 멈춘 사이
// ① 같은 잠금 안 주소(청산 접수 둘째)는 409 already_working(원본 busy_or) ② 잠금 밖 주소(손절 해제)는 칸을 바꾸지 못하고 기다렸다가 뒤에 돈다
// ③ 그래서 첫 요청의 «조립 사이 바뀜»이 생기지 않고(Rust 는 요청이 DeskCell 을 쥔다) 원본 정답지와 같은 200 · 두 번째 칸 차지 없음.
#[path = "common/desk_io.rs"]
mod desk_io;

use desk_io::*;
use rtorder::assemble::DeskIo;
use rtorder::route::Front;
use rtorder::runner::DeskCell;
use rtorder::userstream::Hub;
use serde_json::Value;
use std::sync::Arc;

/// 계좌 조회 한 번을 문이 열릴 때까지 붙잡는 바깥(나머지는 정답지 가짜 그대로).
struct Gate {
    inner: FakeDeskIo,
    entered: Arc<tokio::sync::Notify>,
    open: Arc<tokio::sync::Notify>,
    hold: bool,
}
impl DeskIo for Gate {
    fn now(&self) -> f64 {
        self.inner.now()
    }
    fn exec_enabled(&self) -> bool {
        self.inner.exec_enabled()
    }
    async fn account(&mut self, fresh: bool) -> Result<Value, String> {
        if std::mem::take(&mut self.hold) {
            self.entered.notify_one();
            self.open.notified().await;
        }
        self.inner.account(fresh).await
    }
    fn sec_prices(&mut self, a: &str) -> Option<Vec<(f64, f64)>> {
        self.inner.sec_prices(a)
    }
    async fn clear_bracket(&mut self, s: &str, p: &str, a: bool, b: bool) -> Result<Value, String> {
        self.inner.clear_bracket(s, p, a, b).await
    }
    async fn place_bracket(&mut self, spec: &Value, s: &str, p: &str, strict: bool) -> Result<Value, String> {
        self.inner.place_bracket(spec, s, p, strict).await
    }
    async fn place_tp_ladder(&mut self, legs: &Value, s: &str, p: &str) -> Result<Value, String> {
        self.inner.place_tp_ladder(legs, s, p).await
    }
}

fn case(fx: &Value, name: &str) -> Value {
    fx["cases"].as_array().unwrap().iter().find(|c| c["name"] == name).unwrap().clone()
}
fn pairs(inp: &Value) -> Vec<(String, String)> {
    inp["pairs"].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into())).collect()
}

#[tokio::test(flavor = "current_thread")]
async fn exit_held_in_assembly_blocks_second_exit_and_serializes_disarm() {
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/route_fixture.json")).unwrap()).unwrap();
    let c = case(&fx, "xs_exec_limit_ok");
    let inp = c["in"].clone();
    let (desk, _d) = desk_from(&inp);
    let dir = desk.w.dir.clone();
    // 손절 해제 정답지의 무장 파일을 같은 폴더에 둔다(해제가 실제로 파일을 바꾸게)
    let dis = case(&fx, "dis_already");
    std::fs::write(rtorder::watch::bracket_path(&dir), serde_json::to_string(&dis["in"]["file0"]).unwrap()).unwrap();
    let (cell, hub, front) = (Arc::new(DeskCell::new(desk)), Arc::new(Hub::default()), Arc::new(Front::default()));
    let (entered, open) = (Arc::new(tokio::sync::Notify::new()), Arc::new(tokio::sync::Notify::new()));
    let local = tokio::task::LocalSet::new();
    local.run_until(async {
        let (c1, h1, f1, i1) = (cell.clone(), hub.clone(), front.clone(), inp.clone());
        let mut g = Gate { inner: FakeDeskIo::new(&inp, dir.clone()), entered: entered.clone(), open: open.clone(), hold: true };
        let first = tokio::task::spawn_local(async move {
            let mut q = FakeQuotes::new(&i1);
            f1.call(&c1, &h1, &mut q, &mut g, "POST", "/api/manual-exit/submit", &pairs(&i1)).await
        });
        entered.notified().await; // 첫 요청이 조립(계좌 조회) 안에서 멈췄다
        // ① 둘째 청산(잠금 안) = 바로 409
        let mut io2 = FakeDeskIo::new(&inp, dir.clone());
        let mut q2 = FakeQuotes::new(&inp);
        let r2 = tokio::time::timeout(std::time::Duration::from_secs(5), front.call(&cell, &hub, &mut q2, &mut io2, "POST", "/api/manual-exit/submit", &pairs(&inp)))
            .await.expect("409 는 바로 와야 한다(잠금을 쥔 요청의 칸을 기다리면 교착)");
        assert_eq!(r2.status, 409);
        assert!(r2.body.contains("\"already_working\""), "{}", r2.body);
        assert!(io2.calls.is_empty(), "409 는 조립도 안 한다");
        // ② 손절 해제(잠금 밖) = 칸을 못 잡아 기다린다
        let (c3, h3, f3, d3) = (cell.clone(), hub.clone(), front.clone(), dis["in"].clone());
        let dir3 = dir.clone();
        let third = tokio::task::spawn_local(async move {
            let mut io = FakeDeskIo::new(&d3, dir3);
            f3.call(&c3, &h3, &mut FakeQuotes::new(&d3), &mut io, "POST", "/api/manual-bracket/disarm", &pairs(&d3)).await
        });
        for _ in 0..5 {
            tokio::task::yield_now().await;
        }
        assert!(!third.is_finished(), "잠금 밖 요청도 조립 중인 칸을 건드리지 못한다(DeskCell)");
        open.notify_one();
        let r1 = first.await.unwrap();
        assert_eq!(r1.status, 200, "{}", r1.body);
        assert_eq!(r1.task.len(), 1, "작업 하나만");
        let r3 = third.await.unwrap();
        assert_eq!(r3.status, 200, "{}", r3.body);
        assert!(r3.body.contains("\"sl_off\""), "{}", r3.body);
    }).await;
    let d = cell.lock().await;
    assert_eq!(d.running.iter().filter(|k| k.starts_with("manual_")).count(), 1, "칸 하나만 잡혔다 {:?}", d.running);
}
