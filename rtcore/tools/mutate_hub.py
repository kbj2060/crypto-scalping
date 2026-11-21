#!/usr/bin/env python3
"""문 단속(hub.rs) 시험의 양성 대조 (2026-11-20): mutate_footprint.py 의 «바꿈 → 빌드 → 실행 → 사본으로 되돌림 → 바이트 비교»를 그대로 쓴다.
    python3 rtcore/tools/mutate_hub.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutate_footprint as mf  # noqa: E402

H = "src/hub.rs"
mf.FILES = [H]
mf.TESTS = ["--test", "hub_gate"]
mf.M = [
    ("경로 안 봄", H, "if req.uri().path() != cfg.path {", "if false {"),
    ("Origin 안 봄", H, "if !same && !cfg.origins.iter().any(|a| a.as_bytes() == o.as_bytes()) {", "if false {"),
    # 2026-11-21 같은 호스트
    ("같은 호스트 늘 참", H, "    matches!((hp(auth), hp(&h)), (Some(a), Some(b)) if a == b)", "    let _ = (hp(auth), hp(&h));\n    true"),
    ("같은 호스트 안 봄(목록만)", H, "if !same && !cfg.origins", "if !cfg.origins"),
    ("포트 안 비교", H, "(Some(a), Some(b)) if a == b)", "(Some(a), Some(b)) if a.0 == b.0)"),
    ("대소문자 그대로", H, "    let o = origin.trim().to_ascii_lowercase();", "    let o = origin.trim().to_string();"),
    ("기본 포트 채우지 않음", H, "                None => (a.to_string(), def),", "                None => (a.to_string(), \"\"),"),
    ("http·https 아닌 것도 받음", H, '        _ => return false,\n    };', '        _ => "443",\n    };'),
    ("사용자정보(@) 허용", H, "a.contains(['/', '@', '?', '#', ' '])", "a.contains(['/', '?', '#', ' '])"),
    ("Host 없으면 Origin 만으로 통과", H, ".is_some_and(|(o, h)| same_host(o, h))", ".is_none_or(|(o, h)| same_host(o, h))"),
    ("Origin 앞부분만 맞으면 통과", H, "a.as_bytes() == o.as_bytes()", "o.as_bytes().starts_with(a.as_bytes())"),
    ("연결 상한 안 봄", H, "    if !has_slot {", "    if false {"),
    ("칸을 핸드셰이크 뒤 바로 돌려줌", H, "    let mut rx = state.tx.subscribe();", "    drop(slot);\n    let mut rx = state.tx.subscribe();"),
    ("구독 상한 안 봄", H, "if req.subscribe.len() > cfg.max_subs {", "if false {"),
    ("받는 메시지 상한 없음", H, ".max_message_size(Some(MAX_IN)).max_frame_size(Some(MAX_IN))", ".max_message_size(None).max_frame_size(None)"),
    ("pong 없어도 안 끊음", H, "                if !heard {", "                if false {"),
    ("ping 안 보냄", H, "out.send(Message::Ping(Vec::new().into())).await?;", "let _ = &mut out;"),
]

if __name__ == "__main__":
    mf.main()
