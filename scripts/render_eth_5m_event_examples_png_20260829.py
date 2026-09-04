#!/usr/bin/env python3
"""Render the 30 selected 5m event examples as a downloadable PNG."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(r"C:\Users\kbj20\.codex\visualizations\2026\08\28\01a04864-f6f0-7243-9199-9ce732ef5ca1\eth-5m-valid-setup-tuned-examples")
LABEL_DIR = ROOT / "data/labels/eth_5m_valid_setup_tuned_20260829"


def load_helpers():
    path = ROOT / "scripts/render_eth_5m_event_examples_20260828.py"
    spec = importlib.util.spec_from_file_location("event_examples", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_panel(draw: ImageDraw.ImageDraw, example: dict, label: str, x0: int, y0: int, width: int, height: int) -> None:
    foreground = (35, 43, 52)
    border = (176, 188, 199)
    grid = (218, 225, 231)
    up_color = (35, 155, 139)
    down_color = (218, 105, 69)
    close_color = (71, 102, 193)
    event_colors = {"0": (191, 83, 206), "1": (205, 153, 39), "2": (94, 160, 57)}
    title_font, small_font = font(18, True), font(14)
    draw.text((x0, y0), f"label {label}  {example['number']}  {example['timestamp']}  {example['side']}", fill=foreground, font=title_font)
    left, right, top, bottom = x0 + 62, x0 + width - 12, y0 + 28, y0 + height - 27
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
        draw.text((left - 8, y - 8), f"{hi - fraction * (hi - lo):.2f}", fill=foreground, font=small_font, anchor="ra")

    event_index = next((index for index, bar in enumerate(bars) if bar["minutes"] == 0), min(12, len(bars) - 1))
    event_x = xx(event_index)
    for y in range(top, bottom, 8):
        draw.line((event_x, y, event_x, min(y + 4, bottom)), fill=event_colors[label], width=2)
    level_y = yy(example["level_price"])
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
    draw.ellipse((event_x - 5, yy(bars[event_index]["close"]) - 5, event_x + 5, yy(bars[event_index]["close"]) + 5), fill=event_colors[label])
    for index in (0, 6, 12, 18, 24):
        if index >= len(bars):
            continue
        x = xx(index)
        draw.line((x, bottom, x, bottom + 4), fill=border, width=1)
        minutes = bars[index]["minutes"]
        draw.text((x, bottom + 7), f"{'+' if minutes > 0 else ''}{minutes}m", fill=foreground, font=small_font, anchor="ma")
    draw.text((left + (right - left) / 2, y0 + height - 2), "event-relative time (minutes)", fill=foreground, font=small_font, anchor="ms")


def render_image(examples: list[dict], label: str, output: Path, title: str) -> None:
    panel_width, panel_height = 780, 290
    image = Image.new("RGB", (panel_width, 170 + panel_height * len(examples)), (250, 252, 254))
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), title, fill=(25, 32, 40), font=font(26, True))
    draw.text((30, 67), "±1 hour around event · 25 five-minute candles · UTC", fill=(75, 86, 98), font=font(17))
    draw.text((30, 97), "Dashed horizontal line = prior 24h level · dashed vertical line = event bar", fill=(75, 86, 98), font=font(17))
    for row, example in enumerate(examples):
        draw_panel(draw, example, label, 18, 135 + row * panel_height, panel_width - 35, panel_height - 15)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def main() -> None:
    helpers = load_helpers()
    with (LABEL_DIR / "eth_5m_valid_setup_labels.csv").open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    examples = helpers.select_examples(labels, helpers.load_bars())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label in ("0", "1"):
        event_type = helpers.GROUPS[label]
        render_image(examples[label], label, OUTPUT_DIR / f"label-{label}-{event_type}.png", f"ETHUSDT 5m · label {label} · {event_type}")
        for example in examples[label]:
            render_image([example], label, OUTPUT_DIR / f"label-{label}-example-{example['number']:02d}.png", f"ETHUSDT 5m · label {label} · example {example['number']}")

    width, panel_width, panel_height = 1600, 780, 290
    image = Image.new("RGB", (width, 300 + panel_height * 10), (250, 252, 254))
    draw = ImageDraw.Draw(image)
    draw.text((50, 28), "ETHUSDT 5m structure-label examples", fill=(25, 32, 40), font=font(30, True))
    draw.text((50, 75), "10 examples per class · ±1 hour around event · 25 five-minute candles · UTC", fill=(75, 86, 98), font=font(18))
    draw.text((50, 108), "Dashed horizontal line = prior 24h level · dashed vertical line = event bar", fill=(75, 86, 98), font=font(18))
    for column, label in enumerate(("0", "1")):
        for row, example in enumerate(examples[label]):
            draw_panel(draw, example, label, 30 + column * panel_width, 145 + row * panel_height, panel_width - 35, panel_height - 15)
    combined = OUTPUT_DIR / "eth-5m-structure-event-examples.png"
    image.save(combined, format="PNG", optimize=True)
    print(f"{combined} ({combined.stat().st_size} bytes); individual_pngs=20; class_sheets=2")


if __name__ == "__main__":
    main()
