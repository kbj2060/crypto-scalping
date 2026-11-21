//! 뉴스 글자 처리 -- 파이썬 `html.unescape`(rtcollect::html) 와 server.py news_key(570) 의 정규식 다섯 줄을 손으로.

/// 파이썬 `html.unescape` -- 표와 규칙은 rtcollect/src/html.rs 한 곳(뉴스 수집기와 같이 쓴다)
pub use rtcollect::html::unescape;

/// 파이썬 re 의 `\w`(유니코드): 글자·숫자·밑줄 -- str.isalnum() 근사(Rust 알파벳 성질은 일부 결합 기호를 더 넣는다 · 뉴스 제목에서는 차이 없음)
fn is_w(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// 원본 news_key(server.py:562): 출처 머리말·링크를 떼고 영숫자만 소문자로 80자 · 12자 미만이면 ""
pub fn news_key(title: &str) -> String {
    // re.sub(r"https?://\S+", " ", unescape(title))
    let t = unescape(title);
    let mut a = String::with_capacity(t.len());
    let mut rest = t.as_str();
    while let Some(k) = rest.find("http") {
        let after = &rest[k + 4..];
        let skip = if after.starts_with("s://") { 4 } else if after.starts_with("://") { 3 } else { 0 };
        let tail = &after[skip..];
        let n = tail.chars().take_while(|c| !c.is_whitespace()).map(char::len_utf8).sum::<usize>();
        if skip > 0 && n > 0 {
            a.push_str(&rest[..k]);
            a.push(' ');
            rest = &tail[n..];
        } else {
            a.push_str(&rest[..k + 1]);
            rest = &rest[k + 1..];
        }
    }
    a.push_str(rest);
    // re.sub(r"\[No Title\] - Post from [^\n]*", " ", t)
    let mut b = String::with_capacity(a.len());
    let mut rest = a.as_str();
    while let Some(k) = rest.find("[No Title] - Post from ") {
        b.push_str(&rest[..k]);
        b.push(' ');
        let tail = &rest[k + 23..];
        rest = &tail[tail.find('\n').unwrap_or(tail.len())..];
    }
    b.push_str(rest);
    // re.sub(r"^[^:\n]{1,60}\(@\w+\):\s*", "", t.strip()) -- 탐욕 {1,60}: 조건을 맞추는 가장 긴 머리
    let mut t: &str = b.trim_matches(char::is_whitespace);
    let cs: Vec<(usize, char)> = t.char_indices().collect();
    let run = cs.iter().take(60).take_while(|(_, c)| *c != ':' && *c != '\n').count();
    for p in (1..=run).rev() {
        let at = cs.get(p).map_or(t.len(), |x| x.0);
        let s = &t[at..];
        if let Some(r) = s.strip_prefix("(@") {
            let w: usize = r.chars().take_while(|c| is_w(*c)).map(char::len_utf8).sum();
            if w > 0 && r[w..].starts_with("):") {
                t = r[w + 2..].trim_start_matches(char::is_whitespace);
                break;
            }
        }
    }
    // re.sub(r"^(?:[A-Z0-9][A-Z0-9 .'&-]{0,30}|FinancialJuice|Tree News|AggrNews|BWEnews|Binance EN):\s*", "", t)
    let up = |c: char| c.is_ascii_uppercase() || c.is_ascii_digit();
    let mut cut = None;
    if t.chars().next().is_some_and(up) {
        let n = t.chars().skip(1).take(30).take_while(|c| up(*c) || " .'&-".contains(*c)).count();
        // 탐욕 뒤 되짚기: ':' 가 바로 뒤에 오는 가장 긴 길이(글자 모임에 ':' 가 없으니 그 길이 하나뿐)
        let pre: usize = t.chars().take(1 + n).map(char::len_utf8).sum();
        if t[pre..].starts_with(':') {
            cut = Some(pre + 1);
        }
    }
    if cut.is_none() {
        for nm in ["FinancialJuice", "Tree News", "AggrNews", "BWEnews", "Binance EN"] {
            if t.starts_with(nm) && t[nm.len()..].starts_with(':') {
                cut = Some(nm.len() + 1);
                break;
            }
        }
    }
    if let Some(c) = cut {
        t = t[c..].trim_start_matches(char::is_whitespace);
    }
    // re.sub(r"[\W_]+", "", t.lower()) → 앞 80자 · 12자 미만이면 ""
    let k: Vec<char> = t.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect();
    if k.len() >= 12 { k[..k.len().min(80)].iter().collect() } else { String::new() }
}
