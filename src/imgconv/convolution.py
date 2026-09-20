"""Свёртка изображения ядром.

Реализовано два алгоритма с одинаковым результатом:

* ``convolve_naive`` — прямая запись определения свёртки вложенными
  циклами. Медленный, но очевидно правильный, поэтому используется
  как эталон при проверке корректности;
* ``convolve_fast`` — та же свёртка средствами NumPy, где сложение
  выполняется сразу для всего изображения.
"""

from __future__ import annotations

import numpy as np

from .borders import BorderMode, map_index
from .kernel import Kernel

#: Максимальное значение канала для 8-битного изображения
MAX_LEVEL = 255

#: Ограничение наивной реализации: выше неё расчёт занимает минуты
NAIVE_PIXEL_LIMIT = 250_000

#: Размер одного отсчёта промежуточного массива, байт (float64)
BYTES_PER_SAMPLE = 8

#: Предельный объём памяти под одну свёртку.
#: Без такой проверки операционная система завершает процесс
#: при нехватке памяти, не давая приложению сообщить об ошибке.
MEMORY_LIMIT_BYTES = 2 * 1024 ** 3


def estimate_memory_bytes(height: int, width: int, channels: int,
                          radius: int) -> int:
    """Оценка пикового расхода памяти на свёртку.

    Одновременно существуют три вещественных массива: исходное
    изображение, дополненное полями и накопитель результата.
    """
    source = height * width * channels * BYTES_PER_SAMPLE
    padded = ((height + 2 * radius) * (width + 2 * radius)
              * channels * BYTES_PER_SAMPLE)
    return source + padded + source


class ImageError(ValueError):
    """Ошибка входного изображения."""


def validate_image(image) -> np.ndarray:
    """Проверяет изображение и приводит его к виду (высота, ширина, канал)."""
    if image is None:
        raise ImageError("Изображение не задано")

    array = np.asarray(image)

    if array.size == 0:
        raise ImageError("Изображение не может быть пустым")

    if array.ndim == 2:
        # Полутоновое изображение считается одноканальным
        array = array[:, :, np.newaxis]
    elif array.ndim != 3:
        raise ImageError(
            f"Изображение должно быть двумерным или трёхмерным, "
            f"получено измерений: {array.ndim}"
        )

    if array.shape[2] not in (1, 3, 4):
        raise ImageError(
            f"Поддерживаются 1, 3 или 4 канала, получено: "
            f"{array.shape[2]}"
        )

    return array.astype(np.float64, copy=False)


def _finalize(accumulator: np.ndarray, kernel: Kernel,
              squeeze: bool) -> np.ndarray:
    """Нормализация, сдвиг, отсечение диапазона и возврат к uint8."""
    result = accumulator / kernel.divisor + kernel.offset

    # Свёртка может вывести значение за пределы диапазона яркости,
    # поэтому результат отсекается, а не переполняется
    np.clip(result, 0.0, MAX_LEVEL, out=result)
    result = np.rint(result).astype(np.uint8)

    return result[:, :, 0] if squeeze else result


def _restore_alpha(result: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Возвращает исходный канал прозрачности в результат.

    Свёртка применяется только к цветовым каналам. Если пропустить
    через фильтр и четвёртый канал, размытие превратит чёткую
    границу прозрачности в полупрозрачную кайму, а тиснение сделает
    прозрачные области непрозрачными.
    """
    if result.ndim == 3 and result.shape[2] == 4:
        result[:, :, 3] = source[:, :, 3].astype(np.uint8)
    return result


def _border_indices(length: int, radius: int,
                    border: BorderMode) -> np.ndarray:
    """Индексы пикселей с полями шириной radius для режима границы.

    Векторный аналог :func:`borders.map_index` для режимов, в которых
    каждому индексу за пределами изображения соответствует реальный
    пиксель.
    """
    positions = np.arange(-radius, length + radius)

    if border is BorderMode.CLAMP:
        return np.clip(positions, 0, length - 1)

    if border is BorderMode.REFLECT:
        if length == 1:
            return np.zeros_like(positions)
        # Отражение без повтора крайнего пикселя
        period = 2 * length - 2
        folded = np.abs(positions) % period
        return np.where(folded < length, folded, period - folded)

    raise ImageError(f"Режим обработки границ не поддержан: {border}")


def convolve_naive(image, kernel: Kernel,
                   border: BorderMode = BorderMode.CLAMP) -> np.ndarray:
    """Эталонная свёртка прямыми циклами.

    Предназначена для небольших изображений и проверки корректности.
    """
    if not isinstance(kernel, Kernel):
        raise ImageError("Ожидалось ядро свёртки")

    source = validate_image(image)
    squeeze = np.asarray(image).ndim == 2
    height, width, channels = source.shape

    if height * width > NAIVE_PIXEL_LIMIT:
        raise ImageError(
            f"Наивная реализация рассчитана на изображения до "
            f"{NAIVE_PIXEL_LIMIT} пикселей, получено {height * width}"
        )

    coefficients = kernel.values
    radius = kernel.radius
    accumulator = np.zeros((height, width, channels), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                total = 0.0

                for ky in range(-radius, radius + 1):
                    source_y = map_index(y + ky, height, border)
                    if source_y is None:
                        continue

                    for kx in range(-radius, radius + 1):
                        source_x = map_index(x + kx, width, border)
                        if source_x is None:
                            continue

                        weight = coefficients[ky + radius, kx + radius]
                        total += weight * source[source_y, source_x,
                                                 channel]

                accumulator[y, x, channel] = total

    return _restore_alpha(_finalize(accumulator, kernel, squeeze),
                          source)


def convolve_fast(image, kernel: Kernel,
                  border: BorderMode = BorderMode.CLAMP) -> np.ndarray:
    """Свёртка средствами NumPy.

    Изображение дополняется по краям согласно режиму границы, после
    чего для каждого коэффициента ядра складывается сдвинутая копия
    всего изображения. Число операций пропорционально размеру ядра,
    а не числу пикселей, поэтому реализация во много раз быстрее
    наивной при совпадающем результате.
    """
    if not isinstance(kernel, Kernel):
        raise ImageError("Ожидалось ядро свёртки")

    source = validate_image(image)
    squeeze = np.asarray(image).ndim == 2
    height, width, channels = source.shape
    radius = kernel.radius

    needed = estimate_memory_bytes(height, width, channels, radius)
    if needed > MEMORY_LIMIT_BYTES:
        raise ImageError(
            f"Изображение слишком велико: для обработки требуется "
            f"около {needed / 1024 ** 3:.1f} ГБ памяти при пределе "
            f"{MEMORY_LIMIT_BYTES / 1024 ** 3:.1f} ГБ"
        )

    if border is BorderMode.ZERO:
        # Отсутствующие пиксели считаются нулевыми и не влияют на сумму
        padding = ((radius, radius), (radius, radius), (0, 0))
        padded = np.pad(source, padding, mode="constant",
                        constant_values=0.0)
    elif border in (BorderMode.CLAMP, BorderMode.REFLECT):
        # Индексы строятся теми же формулами, что и в наивной
        # реализации, поэтому обе версии совпадают на любых
        # изображениях, включая те, что меньше ядра
        rows = _border_indices(height, radius, border)
        columns = _border_indices(width, radius, border)
        padded = source[rows][:, columns]
    else:
        raise ImageError(f"Режим обработки границ не поддержан: {border}")

    coefficients = kernel.values
    accumulator = np.zeros_like(source)

    for ky in range(kernel.size):
        for kx in range(kernel.size):
            weight = coefficients[ky, kx]
            if weight == 0.0:
                continue
            accumulator += weight * padded[ky:ky + height,
                                           kx:kx + width, :]

    return _restore_alpha(_finalize(accumulator, kernel, squeeze),
                          source)


def convolve(image, kernel: Kernel,
             border: BorderMode = BorderMode.CLAMP,
             use_fast: bool = True) -> np.ndarray:
    """Свёртка изображения выбранной реализацией."""
    if use_fast:
        return convolve_fast(image, kernel, border)
    return convolve_naive(image, kernel, border)
