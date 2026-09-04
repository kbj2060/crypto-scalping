#!/usr/bin/env python3
"""Render 10 evenly distributed +/-1h chart examples for each 5m event label."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data/labels/eth_5m_valid_setup_tuned_20260829"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUTPUT = Path("/mnt/c/Users/kbj20/.codex/visualizations/2026/08/28/01a04864-f6f0-7243-9199-9ce732ef5ca1/eth-5m-valid-setup-tuned-examples.html")

GROUPS = {
    "0": "SWEEP",
    "1": "BREAKOUT",
}


def parse_time(value: str) -> datetime:
    value = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_bars() -> list[dict]:
    bars = []
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                timestamp = parse_time(row["timestamp"])
                bars.append({
                    "timestamp": timestamp.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    bars.sort(key=lambda row: row["timestamp"])
    return bars


def select_examples(labels: list[dict], bars: list[dict]) -> dict[str, list[dict]]:
    positions = {row["timestamp"]: index for index, row in enumerate(bars)}
    grouped: dict[str, list[dict]] = {label: [] for label in GROUPS}
    for row in labels:
        grouped[row["label"]].append(row)

    examples: dict[str, list[dict]] = {}
    for label, event_type in GROUPS.items():
        rows = sorted(grouped[label], key=lambda row: row["timestamp"])
        if len(rows) < 10:
            raise RuntimeError(f"label {label} has only {len(rows)} rows; need 10")
        selected_indices = [round(index * (len(rows) - 1) / 9) for index in range(10)]
        selected = [rows[index] for index in selected_indices]
        rendered = []
        for number, row in enumerate(selected, start=1):
            event_time = parse_time(row["timestamp"])
            position = positions.get(event_time.isoformat())
            if position is None:
                raise RuntimeError(f"missing source bar for {row['timestamp']}")
            start = max(0, position - 12)
            end = min(len(bars), position + 13)
            window = []
            for bar in bars[start:end]:
                bar = dict(bar)
                bar["minutes"] = round((parse_time(bar["timestamp"]) - event_time).total_seconds() / 60)
                window.append(bar)
            rendered.append({
                "number": number,
                "timestamp": row["timestamp"],
                "side": row["side"],
                "event_id": row["event_id"],
                "level_price": float(row["level_price"]),
                "bars": window,
            })
        examples[label] = rendered
    return examples


def build_html(examples: dict[str, list[dict]]) -> str:
    payload = json.dumps(examples, ensure_ascii=False, separators=(",", ":"))
    sections = []
    for label, event_type in GROUPS.items():
        panels = []
        for number in range(1, 11):
            panels.append(
                f'<figure class="plot-wrap"><figcaption>{label} · {event_type} · example {number}</figcaption>'
                f'<svg data-label="{label}" data-number="{number}" role="img" aria-label="{event_type} example {number}"></svg></figure>'
            )
        sections.append(f'<section><h2>label {label} · {event_type}</h2><div class="plot-grid">{"".join(panels)}</div></section>')

    return f'''<div id="eth-5m-event-examples">
  <style>
    #eth-5m-event-examples {{
      --foreground: light-dark(#17202a, #e8edf2);
      --border: light-dark(#c7d0d9, #46515d);
      --viz-series-1: #2fb7a8;
      --viz-series-2: #ef8c61;
      --viz-series-3: #7d9cff;
      --viz-series-4: #d76be8;
      --viz-series-5: #e2b84c;
      --viz-series-6: #82c95b;
      color: var(--foreground);
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      max-width: 1100px;
      margin: 0 auto;
    }}
    #eth-5m-event-examples h1 {{ font-size: 20px; margin: 0 0 4px; }}
    #eth-5m-event-examples h2 {{ font-size: 16px; margin: 24px 0 8px; }}
    #eth-5m-event-examples .subtitle {{ font-size: 13px; margin: 0 0 16px; opacity: .82; }}
    #eth-5m-event-examples .legend {{ font-size: 12px; margin: 0 0 10px; }}
    #eth-5m-event-examples .swatch {{ display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 12px; border-radius: 50%; }}
    #eth-5m-event-examples .swatch:first-child {{ margin-left: 0; }}
    #eth-5m-event-examples .swatch.s0 {{ background: var(--viz-series-4); }}
    #eth-5m-event-examples .swatch.s1 {{ background: var(--viz-series-5); }}
    #eth-5m-event-examples .swatch.s2 {{ background: var(--viz-series-6); }}
    #eth-5m-event-examples .plot-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 18px; }}
    #eth-5m-event-examples .plot-wrap {{ min-width: 0; margin: 0; }}
    #eth-5m-event-examples figcaption {{ font-size: 12px; margin-bottom: 3px; }}
    #eth-5m-event-examples svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
    #eth-5m-event-examples text {{ fill: var(--foreground); font-size: 12px; }}
    #eth-5m-event-examples .grid-line {{ stroke: var(--border); stroke-width: 1; opacity: .45; }}
    #eth-5m-event-examples .frame {{ fill: none; stroke: var(--border); stroke-width: 1; }}
    #eth-5m-event-examples .axis-title {{ font-size: 12px; }}
    #eth-5m-event-examples .event-line {{ stroke: var(--viz-series-4); stroke-width: 1.5; stroke-dasharray: 4 3; }}
    #eth-5m-event-examples .level-line {{ stroke: var(--viz-series-5); stroke-width: 1.25; stroke-dasharray: 5 3; }}
    #eth-5m-event-examples .close-line {{ fill: none; stroke: var(--viz-series-3); stroke-width: 1.5; }}
    @media (max-width: 640px) {{
      #eth-5m-event-examples .plot-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
  <h1>ETHUSDT 5분봉 이벤트 라벨 사례</h1>
  <p class="subtitle">각 패널은 이벤트 시점 기준 전후 1시간(5분봉 25개) · UTC · 수평선=과거 24시간 레벨 · 점선=이벤트 시점</p>
  <div class="legend"><span class="swatch s0"></span>label 0 liquidity_sweep <span class="swatch s1"></span>label 1 trend_breakout <span class="swatch s2"></span>label 2 fakeout_trap</div>
  {''.join(sections)}
  <script>
    (() => {{
      const root = document.getElementById('eth-5m-event-examples');
      const examples = {payload};
      const colors = {{'0': 'var(--viz-series-4)', '1': 'var(--viz-series-5)', '2': 'var(--viz-series-6)'}};
      const ns = 'http://www.w3.org/2000/svg';
      const make = (tag, attrs, parent) => {{
        const node = document.createElementNS(ns, tag);
        Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
        parent.appendChild(node);
        return node;
      }};
      const draw = (svg, example, label) => {{
        const width = Math.max(300, svg.parentElement.clientWidth || 360);
        const height = Math.round(width * 0.58);
        svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
        svg.replaceChildren();
        const margin = {{top: 14, right: 12, bottom: 34, left: 52}};
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;
        const bars = example.bars;
        const prices = bars.flatMap(bar => [bar.high, bar.low]);
        const minimum = Math.min(...prices);
        const maximum = Math.max(...prices);
        const pad = Math.max((maximum - minimum) * 0.08, 0.01);
        const lo = minimum - pad;
        const hi = maximum + pad;
        const x = index => margin.left + (index / Math.max(1, bars.length - 1)) * plotWidth;
        const y = price => margin.top + (hi - price) / (hi - lo) * plotHeight;
        make('rect', {{x: margin.left, y: margin.top, width: plotWidth, height: plotHeight, class: 'frame', 'data-chart-frame': 'true'}}, svg);
        [0, 0.5, 1].forEach(fraction => {{
          const yy = margin.top + fraction * plotHeight;
          make('line', {{x1: margin.left, x2: margin.left + plotWidth, y1: yy, y2: yy, class: 'grid-line'}}, svg);
          const tick = make('text', {{x: margin.left - 6, y: yy + 4, 'text-anchor': 'end'}}, svg);
          tick.textContent = (hi - fraction * (hi - lo)).toFixed(2);
        }});
        const eventIndex = bars.findIndex(bar => bar.minutes === 0);
        const eventX = x(eventIndex < 0 ? 12 : eventIndex);
        make('line', {{x1: eventX, x2: eventX, y1: margin.top, y2: margin.top + plotHeight, class: 'event-line'}}, svg);
        const levelY = y(example.level_price);
        make('line', {{x1: margin.left, x2: margin.left + plotWidth, y1: levelY, y2: levelY, class: 'level-line'}}, svg);
        const candleWidth = Math.max(2, Math.min(8, plotWidth / bars.length * .62));
        bars.forEach((bar, index) => {{
          const xx = x(index);
          const rising = bar.close >= bar.open;
          const color = rising ? 'var(--viz-series-1)' : 'var(--viz-series-2)';
          make('line', {{x1: xx, x2: xx, y1: y(bar.high), y2: y(bar.low), stroke: color, 'stroke-width': 1}}, svg);
          make('rect', {{x: xx - candleWidth / 2, y: Math.min(y(bar.open), y(bar.close)), width: candleWidth, height: Math.max(1, Math.abs(y(bar.close) - y(bar.open))), fill: color}}, svg);
        }});
        const closePath = bars.map((bar, index) => `${{index ? 'L' : 'M'}} ${{x(index).toFixed(2)}} ${{y(bar.close).toFixed(2)}}`).join(' ');
        make('path', {{d: closePath, class: 'close-line'}}, svg);
        make('circle', {{cx: eventX, cy: y(bars[eventIndex < 0 ? 12 : eventIndex].close), r: 4, fill: colors[label]}}, svg);
        [0, 6, 12, 18, 24].forEach(index => {{
          if (index >= bars.length) return;
          const xx = x(index);
          make('line', {{x1: xx, x2: xx, y1: margin.top + plotHeight, y2: margin.top + plotHeight + 4, class: 'grid-line'}}, svg);
          const tick = make('text', {{x: xx, y: height - 11, 'text-anchor': 'middle'}}, svg);
          tick.textContent = `${{bars[index].minutes > 0 ? '+' : ''}}${{bars[index].minutes}}m`;
        }});
        const xTitle = make('text', {{x: margin.left + plotWidth / 2, y: height - 1, 'text-anchor': 'middle', class: 'axis-title', 'data-axis': 'x'}}, svg);
        xTitle.textContent = 'event-relative time (minutes)';
        const yTitle = make('text', {{x: 12, y: margin.top + plotHeight / 2, 'text-anchor': 'middle', class: 'axis-title', 'data-axis': 'y', transform: `rotate(-90 12 ${{margin.top + plotHeight / 2}})`}}, svg);
        yTitle.textContent = 'price (USDT)';
      }};
      root.querySelectorAll('svg[data-label]').forEach(svg => {{
        const label = svg.dataset.label;
        const example = examples[label][Number(svg.dataset.number) - 1];
        draw(svg, example, label);
      }});
      const redraw = () => root.querySelectorAll('svg[data-label]').forEach(svg => draw(svg, examples[svg.dataset.label][Number(svg.dataset.number) - 1], svg.dataset.label));
      new ResizeObserver(redraw).observe(root);
    }})();
  </script>
</div>\n'''


def build_static_svg(example: dict, label: str) -> str:
    width, height = 360, 220
    left, right, top, bottom = 52, 12, 14, 34
    plot_width = width - left - right
    plot_height = height - top - bottom
    bars = example["bars"]
    prices = [value for bar in bars for value in (bar["high"], bar["low"])]
    minimum, maximum = min(prices), max(prices)
    padding = max((maximum - minimum) * 0.08, 0.01)
    lo, hi = minimum - padding, maximum + padding

    def x(index: int) -> float:
        return left + index / max(1, len(bars) - 1) * plot_width

    def y(price: float) -> float:
        return top + (hi - price) / (hi - lo) * plot_height

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="label {label} {example["number"]} chart">',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" class="frame" data-chart-frame="true"/>',
    ]
    for fraction in (0, 0.5, 1):
        yy = top + fraction * plot_height
        value = hi - fraction * (hi - lo)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{yy:.2f}" y2="{yy:.2f}" class="grid-line"/>')
        parts.append(f'<text x="{left - 6}" y="{yy + 4:.2f}" text-anchor="end">{value:.2f}</text>')

    event_index = next((index for index, bar in enumerate(bars) if bar["minutes"] == 0), min(12, len(bars) - 1))
    event_x = x(event_index)
    parts.append(f'<line x1="{event_x:.2f}" x2="{event_x:.2f}" y1="{top}" y2="{top + plot_height}" class="event-line"/>')
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(example["level_price"]):.2f}" y2="{y(example["level_price"]):.2f}" class="level-line"/>')

    candle_width = max(2, min(8, plot_width / len(bars) * 0.62))
    for index, bar in enumerate(bars):
        xx = x(index)
        color = "var(--viz-series-1)" if bar["close"] >= bar["open"] else "var(--viz-series-2)"
        body_y = min(y(bar["open"]), y(bar["close"]))
        body_height = max(1, abs(y(bar["close"]) - y(bar["open"])))
        parts.append(f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{y(bar["high"]):.2f}" y2="{y(bar["low"]):.2f}" stroke="{color}" stroke-width="1"/>')
        parts.append(f'<rect x="{xx - candle_width / 2:.2f}" y="{body_y:.2f}" width="{candle_width:.2f}" height="{body_height:.2f}" fill="{color}"/>')

    close_path = " ".join(
        f'{"M" if index == 0 else "L"} {x(index):.2f} {y(bar["close"]):.2f}'
        for index, bar in enumerate(bars)
    )
    parts.append(f'<path d="{close_path}" class="close-line"/>')
    parts.append(f'<circle cx="{event_x:.2f}" cy="{y(bars[event_index]["close"]):.2f}" r="4" fill="var(--label-{label})"/>')
    for index in (0, 6, 12, 18, 24):
        if index >= len(bars):
            continue
        xx = x(index)
        parts.append(f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{top + plot_height}" y2="{top + plot_height + 4}" class="grid-line"/>')
        minutes = bars[index]["minutes"]
        parts.append(f'<text x="{xx:.2f}" y="{height - 11}" text-anchor="middle">{"+" if minutes > 0 else ""}{minutes}m</text>')
    parts.append(f'<text x="{left + plot_width / 2}" y="{height - 1}" text-anchor="middle" class="axis-title" data-axis="x">event-relative time (minutes)</text>')
    center = top + plot_height / 2
    parts.append(f'<text x="12" y="{center}" text-anchor="middle" class="axis-title" data-axis="y" transform="rotate(-90 12 {center})">price (USDT)</text>')
    parts.append('</svg>')
    return "".join(parts)


def build_static_html(examples: dict[str, list[dict]]) -> str:
    sections = []
    for label, event_type in GROUPS.items():
        panels = []
        for example in examples[label]:
            panels.append(
                f'<figure class="plot-wrap"><figcaption>{label} · {event_type} · example {example["number"]}</figcaption>'
                f'{build_static_svg(example, label)}</figure>'
            )
        sections.append(f'<section><h2>label {label} · {event_type}</h2><div class="plot-grid">{"".join(panels)}</div></section>')
    return f'''<div id="eth-5m-event-examples">
  <style>
    #eth-5m-event-examples {{
      --foreground: #17202a; --border: #c7d0d9;
      --viz-series-1: #2fb7a8; --viz-series-2: #ef8c61; --viz-series-3: #7d9cff;
      --viz-series-4: #d76be8; --viz-series-5: #e2b84c; --viz-series-6: #82c95b;
      --label-0: var(--viz-series-4); --label-1: var(--viz-series-5); --label-2: var(--viz-series-6);
      color: var(--foreground); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; max-width: 1100px; margin: 0 auto;
    }}
    @media (prefers-color-scheme: dark) {{ #eth-5m-event-examples {{ --foreground: #e8edf2; --border: #46515d; }} }}
    #eth-5m-event-examples h1 {{ font-size: 20px; margin: 0 0 4px; }}
    #eth-5m-event-examples h2 {{ font-size: 16px; margin: 24px 0 8px; }}
    #eth-5m-event-examples .subtitle, #eth-5m-event-examples .legend {{ font-size: 12px; margin: 0 0 10px; opacity: .84; }}
    #eth-5m-event-examples .swatch {{ display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 12px; border-radius: 50%; }}
    #eth-5m-event-examples .swatch:first-child {{ margin-left: 0; }}
    #eth-5m-event-examples .s0 {{ background: var(--label-0); }} #eth-5m-event-examples .s1 {{ background: var(--label-1); }} #eth-5m-event-examples .s2 {{ background: var(--label-2); }}
    #eth-5m-event-examples .plot-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 18px; }}
    #eth-5m-event-examples .plot-wrap {{ min-width: 0; margin: 0; }}
    #eth-5m-event-examples figcaption {{ font-size: 12px; margin-bottom: 3px; }}
    #eth-5m-event-examples svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
    #eth-5m-event-examples text {{ fill: var(--foreground); font-size: 12px; }}
    #eth-5m-event-examples .grid-line {{ stroke: var(--border); stroke-width: 1; opacity: .45; }}
    #eth-5m-event-examples .frame {{ fill: none; stroke: var(--border); stroke-width: 1; }}
    #eth-5m-event-examples .event-line {{ stroke: var(--viz-series-4); stroke-width: 1.5; stroke-dasharray: 4 3; }}
    #eth-5m-event-examples .level-line {{ stroke: var(--viz-series-5); stroke-width: 1.25; stroke-dasharray: 5 3; }}
    #eth-5m-event-examples .close-line {{ fill: none; stroke: var(--viz-series-3); stroke-width: 1.5; }}
    @media (max-width: 640px) {{ #eth-5m-event-examples .plot-grid {{ grid-template-columns: 1fr; }} }}
  </style>
  <h1>ETHUSDT 5분봉 이벤트 라벨 사례</h1>
  <p class="subtitle">이벤트 시점 전후 1시간 · 5분봉 25개 · UTC · 수평선=과거 24시간 레벨 · 점선=이벤트 시점</p>
  <div class="legend"><span class="swatch s0"></span>0 liquidity_sweep <span class="swatch s1"></span>1 trend_breakout <span class="swatch s2"></span>2 fakeout_trap</div>
  {''.join(sections)}
</div>
'''


def main() -> None:
    with (LABEL_DIR / "eth_5m_valid_setup_labels.csv").open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    html = build_static_html(select_examples(labels, load_bars()))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "bytes": len(html.encode("utf-8"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
