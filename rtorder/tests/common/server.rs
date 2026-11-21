// 가짜 HTTP 서버는 라이브러리(`rtorder::fakesrv`)로 옮겼다(2026-11-21 · 그림자 재생도 쓴다). 시험 파일들이 `include!` 하던 이름 그대로.
#[allow(unused_imports)]
use rtorder::fakesrv::{local_only, serve, Act, Got, Handler, Srv};
#[allow(unused_imports)]
use std::sync::atomic::{AtomicUsize, Ordering};
