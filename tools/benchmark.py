"""Стенд измерения производительности приложения «Свёртка изображения».

Запуск:

    python tools/benchmark.py

Результаты сохраняются в docs/benchmark.json и выводятся в консоль.
Измеряется монотонными часами; для каждой точки выполняется прогрев
и несколько повторов, из которых берётся медиана — она устойчивее
среднего к случайным задержкам планировщика.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from imgconv import convolution, kernel  # noqa: E402
from imgconv.borders import BorderMode  # noqa: E402

#: Число повторов измерения для каждой точки
REPEATS = 5

#: Число прогревочных запусков, результат которых отбрасывается
WARMUP = 1


def make_image(height: int, width: int) -> np.ndarray:
    """Воспроизводимое случайное изображение."""
    rng = np.random.default_rng(2026)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def make_box_kernel(size: int) -> kernel.Kernel:
    """Усредняющее ядро заданного нечётного размера."""
    return kernel.Kernel([[1] * size for _ in range(size)],
                         name=f"Усреднение {size}x{size}")


def measure(function, *args, repeats: int = REPEATS) -> dict:
    """Медиана, минимум и максимум времени выполнения в миллисекундах."""
    for _ in range(WARMUP):
        function(*args)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        samples.append((time.perf_counter() - start) * 1000.0)

    return {
        "median_ms": round(statistics.median(samples), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
    }


# ---------------------------------------------------------------------
# Эксперимент 1. Зависимость времени от размера изображения
# ---------------------------------------------------------------------

def experiment_image_size() -> list[dict]:
    print("Эксперимент 1: зависимость от размера изображения")
    sizes = [(256, 256), (512, 512), (1024, 1024), (1536, 1536),
             (2048, 2048), (3000, 4000)]
    box = make_box_kernel(3)
    rows = []

    for height, width in sizes:
        image = make_image(height, width)
        stats = measure(convolution.convolve_fast, image, box,
                        BorderMode.CLAMP)
        pixels = height * width
        row = {
            "width": width, "height": height, "pixels": pixels,
            "ms_per_megapixel": round(
                stats["median_ms"] / (pixels / 1e6), 3),
            **stats,
        }
        rows.append(row)
        print(f"  {width:>5} x {height:<5} "
              f"{pixels / 1e6:6.2f} Мпикс  "
              f"{stats['median_ms']:9.2f} мс  "
              f"{row['ms_per_megapixel']:7.2f} мс/Мпикс")

    return rows


# ---------------------------------------------------------------------
# Эксперимент 2. Зависимость времени от размера ядра
# ---------------------------------------------------------------------

def experiment_kernel_size() -> list[dict]:
    print("Эксперимент 2: зависимость от размера ядра")
    image = make_image(1024, 1024)
    rows = []

    for size in (3, 5, 7, 9, 11, 15):
        box = make_box_kernel(size)
        stats = measure(convolution.convolve_fast, image, box,
                        BorderMode.CLAMP)
        rows.append({"kernel": size, "coefficients": size * size,
                     **stats})
        print(f"  ядро {size:>2} x {size:<2} "
              f"({size * size:3d} коэф.)  {stats['median_ms']:9.2f} мс")

    return rows


# ---------------------------------------------------------------------
# Эксперимент 3. Сравнение двух реализаций
# ---------------------------------------------------------------------

def experiment_implementations() -> list[dict]:
    print("Эксперимент 3: сравнение реализаций")
    box = make_box_kernel(3)
    rows = []

    for side in (64, 96, 128, 160, 200):
        image = make_image(side, side)
        fast = measure(convolution.convolve_fast, image, box,
                       BorderMode.CLAMP)
        naive = measure(convolution.convolve_naive, image, box,
                        BorderMode.CLAMP, repeats=2)
        speedup = naive["median_ms"] / fast["median_ms"]
        rows.append({
            "side": side, "pixels": side * side,
            "fast_ms": fast["median_ms"],
            "naive_ms": naive["median_ms"],
            "speedup": round(speedup, 1),
        })
        print(f"  {side:>3} x {side:<3}  быстрая {fast['median_ms']:8.2f} мс  "
              f"наивная {naive['median_ms']:9.2f} мс  "
              f"ускорение {speedup:6.0f}x")

    return rows


# ---------------------------------------------------------------------
# Эксперимент 4. Нагрузочное тестирование
# ---------------------------------------------------------------------

def experiment_load(iterations: int = 50) -> dict:
    print(f"Эксперимент 4: нагрузочное, {iterations} обработок подряд")
    image = make_image(1024, 1024)
    box = make_box_kernel(5)

    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        convolution.convolve_fast(image, box, BorderMode.CLAMP)
        samples.append((time.perf_counter() - start) * 1000.0)

    peak_growth = tracemalloc.get_traced_memory()[0] - baseline
    tracemalloc.stop()

    quarter = max(1, iterations // 4)
    first = statistics.median(samples[:quarter])
    last = statistics.median(samples[-quarter:])

    result = {
        "iterations": iterations,
        "median_ms": round(statistics.median(samples), 2),
        "min_ms": round(min(samples), 2),
        "max_ms": round(max(samples), 2),
        "first_quarter_ms": round(first, 2),
        "last_quarter_ms": round(last, 2),
        "drift_percent": round((last - first) / first * 100.0, 2),
        "memory_growth_kb": round(peak_growth / 1024.0, 1),
    }

    print(f"  медиана {result['median_ms']} мс, "
          f"разброс {result['min_ms']}–{result['max_ms']} мс")
    print(f"  дрейф времени {result['drift_percent']} %, "
          f"прирост памяти {result['memory_growth_kb']} КБ")
    return result


# ---------------------------------------------------------------------
# Эксперимент 5. Стрессовое тестирование
# ---------------------------------------------------------------------

def experiment_stress() -> list[dict]:
    print("Эксперимент 5: стрессовое, рост размера до отказа")
    box = make_box_kernel(5)
    rows = []

    for side in (2000, 3000, 4000, 5000, 6000, 8000):
        pixels = side * side
        estimate = convolution.estimate_memory_bytes(side, side, 3, 2)

        # Изображение не создаётся, если свёртка заведомо
        # отклонит его: иначе память тратится впустую
        if estimate > convolution.MEMORY_LIMIT_BYTES:
            rows.append({
                "side": side, "megapixels": round(pixels / 1e6, 1),
                "ms": None,
                "estimate_gb": round(estimate / 1024 ** 3, 2),
                "status": "отказ: превышен предел памяти",
            })
            print(f"  {side} x {side} ({pixels / 1e6:5.1f} Мпикс): "
                  f"отказ, требуется {estimate / 1024 ** 3:.2f} ГБ")
            break

        try:
            image = make_image(side, side)
            start = time.perf_counter()
            convolution.convolve_fast(image, box, BorderMode.CLAMP)
            elapsed = (time.perf_counter() - start) * 1000.0
            status = "успешно"
            del image
        except MemoryError as error:
            elapsed = float("nan")
            status = f"отказ: {type(error).__name__}"
        except Exception as error:   # noqa: BLE001
            elapsed = float("nan")
            status = f"отказ: {type(error).__name__}: {error}"

        rows.append({"side": side, "megapixels": round(pixels / 1e6, 1),
                     "ms": None if elapsed != elapsed else round(elapsed, 1),
                     "estimate_gb": round(estimate / 1024 ** 3, 2),
                     "status": status})
        print(f"  {side} x {side} ({pixels / 1e6:5.1f} Мпикс): {status}"
              + (f", {elapsed:.0f} мс" if elapsed == elapsed else ""))

        if status != "успешно":
            break

    return rows


def main() -> int:
    print("Стенд измерения производительности")
    print(f"Платформа: {platform.platform()}")
    print(f"Python: {platform.python_version()}, "
          f"NumPy: {np.__version__}")
    print()

    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "image_size": experiment_image_size(),
        "kernel_size": experiment_kernel_size(),
        "implementations": experiment_implementations(),
        "load": experiment_load(),
        "stress": experiment_stress(),
    }

    target = Path(__file__).resolve().parent.parent / "docs"
    target.mkdir(exist_ok=True)
    path = target / "benchmark.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    print(f"\nРезультаты сохранены: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
