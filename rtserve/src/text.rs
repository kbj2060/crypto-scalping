//! 뉴스 글자 처리 -- 파이썬 `html.unescape`(표 = html_entities.rs) 와 server.py news_key(570) 의 정규식 다섯 줄을 손으로.

use crate::html_entities::{HTML5, INVALID_CHARREFS, INVALID_CODEPOINTS};

fn html5(name: &str) -> Option<&'static str> {
    HTML5.binary_search_by(|(k, _)| (*k).cmp(name)).ok().map(|i| HTML5[i].1)
}

/// 파이썬 `html.unescape(s)` -- `&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)` 를 왼쪽부터 바꾼다
pub fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let ch: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < ch.len() {
        if ch[i] != '&' {
            out.push(ch[i]);
            i += 1;
            continue;
        }
        let j = i + 1;
        // 숫자 참조
        if ch.get(j) == Some(&'#') {
            // 갈래 순서: #[0-9]+ 먼저 · 안 되면 #[xX][0-9a-fA-F]+
            let dec = ch[j + 1..].iter().take_while(|c| c.is_ascii_digit()).count();
            let hexn = if dec == 0 && matches!(ch.get(j + 1), Some('x' | 'X')) { ch[j + 2..].iter().take_while(|c| c.is_ascii_hexdigit()).count() } else { 0 };
            let (start, n, hex) = if dec > 0 { (j + 1, dec, false) } else { (j + 2, hexn, true) };
            if n > 0 {
                let digits: String = ch[start..start + n].iter().collect();
                let mut end = start + n;
                if ch.get(end) == Some(&';') {
                    end += 1;
                }
                let num = u64::from_str_radix(digits.trim_start_matches('0'), if hex { 16 } else { 10 }).unwrap_or(if digits.trim_start_matches('0').is_empty() { 0 } else { u64::MAX });
                out.push_str(&charref(num));
                i = end;
                continue;
            }
            // '#' 는 셋째 갈래 글자에 없다 -- 이 '&' 는 그대로
            out.push('&');
            i += 1;
            continue;
        }
        let n = ch[j..].iter().take(32).take_while(|c| !matches!(c, '\t' | '\n' | '\u{0c}' | ' ' | '<' | '&' | '#' | ';')).count();
        if n == 0 {
            out.push('&');
            i += 1;
            continue;
        }
        let mut end = j + n;
        if ch.get(end) == Some(&';') {
            end += 1;
        }
        let name: String = ch[j..end].iter().collect();
        match html5(&name) {
            Some(v) => out.push_str(v),
            None => {
                let nc: Vec<char> = name.chars().collect();
                let mut done = false;
                for x in (2..nc.len()).rev() {
                    let pre: String = nc[..x].iter().collect();
                    if let Some(v) = html5(&pre) {
                        out.push_str(v);
                        out.extend(&nc[x..]);
                        done = true;
                        break;
                    }
                }
                if !done {
                    out.push('&');
                    out.push_str(&name);
                }
            }
        }
        i = end;
    }
    out
}

fn charref(num: u64) -> String {
    if let Ok(k) = INVALID_CHARREFS.binary_search_by(|(n, _)| (*n as u64).cmp(&num)) {
        return INVALID_CHARREFS[k].1.to_string();
    }
    if (0xD800..=0xDFFF).contains(&num) || num > 0x10FFFF {
        return "\u{FFFD}".into();
    }
    if INVALID_CODEPOINTS.binary_search(&(num as u32)).is_ok() {
        return String::new();
    }
    char::from_u32(num as u32).map(String::from).unwrap_or_default()
}

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
