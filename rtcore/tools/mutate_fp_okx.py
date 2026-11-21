#!/usr/bin/env python3
"""풋프린트 OKX 합산·OKX 봉 합·미시 호가 전 침묵의 양성 대조 (2026-11-21): mutate_footprint.py 의 틀 그대로.
    python3 rtcore/tools/mutate_fp_okx.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutate_footprint as mf  # noqa: E402

F, S, T = "src/footprint.rs", "src/supply.rs", "src/state.rs"
mf.FILES = [F, S, T]
mf.TESTS = ["--test", "fp_okx_parity", "--test", "micro_parity", "--test", "footprint_parity"]
mf.M = [
    ("OKX 칸 안 더함", F, "            if let Some(a) = a {\n                for j in 0..6 {\n                    c[j] += a[j];", "            if let Some(a) = a.filter(|_| false) {\n                for j in 0..6 {\n                    c[j] += a[j];"),
    ("OKX 만 있는 칸 빠짐", F, "bar.cells.keys().chain(add.into_iter().flat_map(|o| o.cells.keys()))", "bar.cells.keys().chain(add.into_iter().flat_map(|o| o.cells.keys()).filter(|_| false))"),
    ("first_bar 거르기 >= 로", F, "self.bars.range(self.from + 1..).rev()", "self.bars.range(self.from..).rev()"),
    ("first_bar 생겨도 전량 안 밂", S, "                (self.fp.from, self.fp.full_dirty) = (bar, true);", "                (self.fp.from, self.fp.full_dirty) = (bar, false);"),
    ("OKX 칸 바뀐 봉 안 실음", F, "            if self.bars.contains_key(&b) {\n                self.dirty.entry(b).or_default().extend(ks);", "            if false {\n                self.dirty.entry(b).or_default().extend(ks);"),
    ("빈 창이면 전부 실음(옛 동작)", F, "        let lo = self.window_lo().unwrap_or(i64::MAX);", "        let lo = self.window_lo().unwrap_or(i64::MIN);"),
    ("OKX 봉 합 바뀐 봉 표시 안 함", S, "                self.okx_sum_dirty.insert(bar);", ""),
    ("OKX 봉 합 first_bar 봉 넣음", S, "        let from = self.okx_first_bar + 1;", "        let from = self.okx_first_bar;"),
    ("미시 값을 호가 전에도 밂", T, "            if book.is_some() {", "            if true {"),
]

if __name__ == "__main__":
    mf.main()
