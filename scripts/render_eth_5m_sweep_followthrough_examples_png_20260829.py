#!/usr/bin/env python3
"""Render ten +/-1h candle examples for each sweep follow-through label."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data/labels/eth_5m_support_reaction_20260829"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUTPUT_DIR = Path(
    r"C:\Users\kbj20\.codex\visualizations\2026\08\28\01a04864-f6f0-7243-9199-9ce732ef5ca1\eth-5m-support-reaction-examples"
)
GROUPS = {"0": "SWEEP_V_BOUNCE", "1": "SUPPORT_BREAKOUT"}


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


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


def load_examples(bars: list[dict]) -> dict[str, list[dict]]:
    with (LABEL_DIR / "eth_5m_support_reaction_labels.csv").open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    positions = {row["timestamp"]: index for index, row in enumerate(bars)}
    grouped = {label: sorted((row for row in labels if row["label"] == label), key=lambda row: row["timestamp"]) for label in GROUPS}
    examples = {}
    for label in GROUPS:
        rows = grouped[label]
        if len(rows) < 10:
            raise RuntimeError(f"label {label} has only {len(rows)} rows")
        chosen = [rows[round(index * (len(rows) - 1) / 9)] for index in range(10)]
        rendered = []
        for number, row in enumerate(chosen, 1):
            event_time = parse_time(row["timestamp"])
            position = positions.get(event_time.isoformat())
            if position is None:
                raise RuntimeError(f"source bar missing for {row['timestamp']}")
            window = []
            for bar in bars[max(0, position - 12):position + 13]:
                item = dict(bar)
                item["minutes"] = round((parse_time(bar["timestamp"]) - event_time).total_seconds() / 60)
                window.append(item)
            rendered.append({"number": number, **row, "bars": window})
        examples[label] = rendered
    return examples


def draw_panel(draw, example: dict, label: str, x0: int, y0: int, width: int, height: int) -> None:
    foreground = (35, 43, 52)
    border = (176, 188, 199)
    grid = (218, 225, 231)
    up_color = (35, 155, 139)
    down_color = (218, 105, 69)
    close_color = (71, 102, 193)
    label_color = (191, 83, 206) if label == "0" else (205, 153, 39)
    title_font, small_font = font(16, True), font(12)
    title = f"label {label}  {example['number']}  {example['timestamp']}  {example['side']}"
    draw.text((x0, y0), title, fill=foreground, font=title_font)
    left, right, top, bottom = x0 + 62, x0 + width - 12, y0 + 25, y0 + height - 27
    bars = example["bars"]
    values = [value for bar in bars for value in (bar["high"], bar["low"])]
    minimum, maximum = min(values), max(values)
    padding = max((maximum - minimum) * 0.08, 0.01)
    lo, hi = minimum - padding, maximum + padding

    def xx(index: int) -> float:
        return left + index / max(1, len(bars) - 1) * (right - left)

    def yy(price: float) -> float:
        return top + (hi - price) / (hi - lo) * (bottom - top)

    draw.rectangle((left, top, right, bottom), outline=border, width=1)
    for fraction in (0, 0.5, 1):
        y = top + fraction * (bottom - top)
        draw.line((left, y, right, y), fill=grid, width=1)
        draw.text((left - 8, y - 7), f"{hi - fraction * (hi - lo):.2f}", fill=foreground, font=small_font, anchor="ra")

    event_index = next(index for index, bar in enumerate(bars) if bar["minutes"] == 0)
    event_x = xx(event_index)
    for y in range(top, bottom, 8):
        draw.line((event_x, y, event_x, min(y + 4, bottom)), fill=label_color, width=2)
    level_y = yy(float(example["support_level"]))
    for x in range(left, right, 12):
        draw.line((x, level_y, min(x + 7, right), level_y), fill=(184, 135, 25), width=2)

    candle_width = max(2, min(9, int((right - left) / len(bars) * 0.62)))
    close_points = []
    for index, bar in enumerate(bars):
        x = xx(index)
        color = up_color if bar["close"] >= bar["open"] else down_color
        draw.line((x, yy(bar["high"]), x, yy(bar["low"])), fill=color, width=2)
        body_top, body_bottom = sorted((yy(bar["open"]), yy(bar["close"])))
        draw.rectangle((x - candle_width / 2, body_top, x + candle_width / 2, max(body_top + 1, body_bottom)), fill=color)
        close_points.append((x, yy(bar["close"])))
    draw.line(close_points, fill=close_color, width=2)
    draw.ellipse((event_x - 5, yy(bars[event_index]["close"]) - 5, event_x + 5, yy(bars[event_index]["close"]) + 5), fill=label_color)
    for index in (0, 6, 12, 18, 24):
        x = xx(index)
        draw.line((x, bottom, x, bottom + 4), fill=border, width=1)
        minutes = bars[index]["minutes"]
        draw.text((x, bottom + 7), f"{'+' if minutes > 0 else ''}{minutes}m", fill=foreground, font=small_font, anchor="ma")
    draw.text((left + (right - left) / 2, y0 + height - 2), "event-relative time (minutes)", fill=foreground, font=small_font, anchor="ms")


def render_sheet(examples: list[dict], label: str, path: Path, title: str) -> None:
    panel_width, panel_height = 780, 285
    image = Image.new("RGB", (panel_width, 165 + panel_height * len(examples)), (250, 252, 254))
    draw = ImageDraw.Draw(image)
    draw.text((30, 22), title, fill=(25, 32, 40), font=font(24, True))
    draw.text((30, 60), "±1 hour around sweep · 25 five-minute candles · UTC", fill=(75, 86, 98), font=font(16))
    draw.text((30, 88), "Dashed horizontal = swept level · dashed vertical = dashboard liquidity_sweep bar", fill=(75, 86, 98), font=font(16))
    for row, example in enumerate(examples):
        draw_panel(draw, example, label, 18, 125 + row * panel_height, panel_width - 35, panel_height - 10)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def main() -> None:
    examples = load_examples(load_bars())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, event_type in GROUPS.items():
        render_sheet(examples[label], label, OUTPUT_DIR / f"label-{label}-{event_type}.png", f"ETHUSDT 5m · label {label} · {event_type}")
        for example in examples[label]:
            render_sheet([example], label, OUTPUT_DIR / f"label-{label}-example-{example['number']:02d}.png", f"ETHUSDT 5m · label {label} · example {example['number']}")

    panel_width, panel_height = 780, 285
    image = Image.new("RGB", (1600, 300 + panel_height * 10), (250, 252, 254))
    draw = ImageDraw.Draw(image)
    draw.text((50, 28), "ETHUSDT 5m support-reaction examples", fill=(25, 32, 40), font=font(29, True))
    draw.text((50, 75), "10 examples per class · ±1 hour around sweep · 25 five-minute candles · UTC", fill=(75, 86, 98), font=font(17))
    draw.text((50, 108), "Dashed horizontal = swept level · dashed vertical = dashboard liquidity_sweep bar", fill=(75, 86, 98), font=font(17))
    for column, label in enumerate(("0", "1")):
        for row, example in enumerate(examples[label]):
            draw_panel(draw, example, label, 30 + column * panel_width, 145 + row * panel_height, panel_width - 35, panel_height - 15)
    combined = OUTPUT_DIR / "eth-5m-sweep-followthrough-examples.png"
    image.save(combined, format="PNG", optimize=True)
    print(f"{combined} ({combined.stat().st_size} bytes); individual_pngs=20; class_sheets=2")


if __name__ == "__main__":
    main()
