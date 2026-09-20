"""Построение графиков по результатам стенда производительности.

Запуск после tools/benchmark.py:

    python tools/charts.py

Читает docs/benchmark.json и сохраняет изображения в каталог docs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 130,
})

ACCENT = "#1f6f8b"
SECOND = "#b5533c"


def load() -> dict:
    path = DOCS / "benchmark.json"
    if not path.exists():
        print("Сначала выполните tools/benchmark.py")
        raise SystemExit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def chart_scaling(report: dict) -> Path:
    """Зависимость времени от размера изображения и размера ядра."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    rows = report["image_size"]
    megapixels = [row["pixels"] / 1e6 for row in rows]
    times = [row["median_ms"] for row in rows]

    left.plot(megapixels, times, marker="o", color=ACCENT)
    left.set_xlabel("Размер изображения, Мпикс")
    left.set_ylabel("Время свёртки, мс")
    left.set_title("Ядро 3 x 3, режим повтора края")

    rows = report["kernel_size"]
    coefficients = [row["coefficients"] for row in rows]
    times = [row["median_ms"] for row in rows]

    right.plot(coefficients, times, marker="s", color=SECOND)
    right.set_xlabel("Число коэффициентов ядра")
    right.set_ylabel("Время свёртки, мс")
    right.set_title("Изображение 1024 x 1024")

    figure.tight_layout()
    path = DOCS / "perf_scaling.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def chart_implementations(report: dict) -> Path:
    """Сравнение реализаций и устойчивость под нагрузкой."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    rows = report["implementations"]
    pixels = [row["pixels"] / 1000 for row in rows]

    left.plot(pixels, [row["naive_ms"] for row in rows],
              marker="o", color=SECOND, label="Наивная реализация")
    left.plot(pixels, [row["fast_ms"] for row in rows],
              marker="s", color=ACCENT, label="Реализация на NumPy")
    left.set_yscale("log")
    left.set_xlabel("Размер изображения, тысяч пикселей")
    left.set_ylabel("Время свёртки, мс (лог. шкала)")
    left.set_title("Сравнение реализаций")
    left.legend()

    load_data = report["load"]
    median = load_data["median_ms"]
    first = load_data["first_quarter_ms"]
    last = load_data["last_quarter_ms"]

    labels = ["Первая\nчетверть", "Медиана\nвсего прогона",
              "Последняя\nчетверть"]
    values = [first, median, last]

    bars = right.bar(labels, values,
                     color=[ACCENT, "#8fa3ad", ACCENT], width=0.55)
    right.set_ylabel("Время свёртки, мс")
    right.set_title(f"Нагрузка: {load_data['iterations']} обработок подряд")
    right.set_ylim(0, max(values) * 1.35)

    for bar, value in zip(bars, values):
        right.text(bar.get_x() + bar.get_width() / 2,
                   value + max(values) * 0.04,
                   f"{value:.0f} мс", ha="center", fontsize=9)

    figure.tight_layout()
    path = DOCS / "perf_load.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def main() -> int:
    report = load()
    for path in (chart_scaling(report), chart_implementations(report)):
        print("сохранено:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
