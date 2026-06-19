from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"


def is_border_like(pixel: tuple[int, int, int]) -> bool:
    r, g, b = pixel
    return b > 48 and g > 34 and (b - r) > 10


def is_bright(pixel: tuple[int, int, int]) -> bool:
    return max(pixel) > 92 or (pixel[1] > 70 and pixel[2] > 95)


def connected_boxes(mask: list[list[bool]], min_area: int = 420) -> list[dict[str, int]]:
    height = len(mask)
    width = len(mask[0])
    seen = [[False] * width for _ in range(height)]
    boxes: list[dict[str, int]] = []

    for y in range(height):
        for x in range(width):
            if seen[y][x] or not mask[y][x]:
                continue

            queue: deque[tuple[int, int]] = deque([(x, y)])
            seen[y][x] = True
            x1 = x2 = x
            y1 = y2 = y
            area = 0

            while queue:
                cx, cy = queue.popleft()
                area += 1
                x1, x2 = min(x1, cx), max(x2, cx)
                y1, y2 = min(y1, cy), max(y2, cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and not seen[ny][nx] and mask[ny][nx]:
                        seen[ny][nx] = True
                        queue.append((nx, ny))

            if area >= min_area:
                boxes.append({"x": x1, "y": y1, "w": x2 - x1 + 1, "h": y2 - y1 + 1, "area": area})

    return sorted(boxes, key=lambda item: (item["y"], item["x"]))


def measure(path: Path) -> dict:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    small = image.resize((width // 2, height // 2))
    pixels = small.load()

    border_mask = [[is_border_like(pixels[x, y]) and is_bright(pixels[x, y]) for x in range(small.width)] for y in range(small.height)]
    boxes = connected_boxes(border_mask)
    scaled_boxes = [
        {key: value * 2 if key in {"x", "y", "w", "h"} else value for key, value in box.items()}
        for box in boxes
        if box["w"] > 18 and box["h"] > 14
    ]

    quantized = image.resize((width // 4, height // 4)).quantize(colors=16, method=2)
    palette = quantized.getpalette()
    total = quantized.size[0] * quantized.size[1]
    dominant = []
    for count, index in sorted(quantized.getcolors(), reverse=True)[:12]:
        rgb = tuple(palette[index * 3 : index * 3 + 3])
        dominant.append({"hex": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}", "pct": round(count / total * 100, 2)})

    return {
        "file": path.name,
        "canvas": {"w": width, "h": height},
        "dominantColors": dominant,
        "detectedComponentBoxes": scaled_boxes[:80],
    }


if __name__ == "__main__":
    report = [measure(path) for path in sorted(PAPER.glob("*.png"))]
    print(json.dumps(report, ensure_ascii=False, indent=2))
