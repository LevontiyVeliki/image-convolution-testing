"""Тестирование методом белого ящика.

Тесты построены по графу потока управления функции
``borders.map_index`` и по ветвям вспомогательных функций ядра.
Цель — покрытие всех ветвей (branch coverage), то есть проход
по каждой дуге графа хотя бы один раз.

Нумерация вершин в комментариях соответствует графу потока
управления, приведённому в отчёте.
"""

from __future__ import annotations

import numpy as np
import pytest

from imgconv import borders, convolution, kernel
from imgconv.borders import BorderMode, map_index

# =====================================================================
# Независимые пути графа потока управления map_index
# Цикломатическая сложность V(G) = 8, поэтому путей тоже восемь
# =====================================================================


def test_path_1_non_positive_length():
    """Путь 1: вершины 1-2-3. Неположительный размер."""
    with pytest.raises(borders.BorderError) as info:
        map_index(0, 0, BorderMode.CLAMP)

    assert "положительным" in str(info.value)


@pytest.mark.parametrize("index", [0, 3, 9])
def test_path_2_index_inside(index):
    """Путь 2: вершины 1-2-4-5. Индекс внутри изображения.

    Проверяются обе границы допустимого диапазона и середина.
    """
    assert map_index(index, 10, BorderMode.ZERO) == index


@pytest.mark.parametrize("index", [-1, -7, 10, 25])
def test_path_3_zero_mode(index):
    """Путь 3: вершины 1-2-4-6-7. Режим заполнения нулями."""
    assert map_index(index, 10, BorderMode.ZERO) is None


@pytest.mark.parametrize(
    "index, expected",
    [(-1, 0), (-100, 0), (10, 9), (99, 9)],
)
def test_path_4_clamp_mode(index, expected):
    """Путь 4: вершины 1-2-4-6-8-9. Повтор крайнего пикселя."""
    assert map_index(index, 10, BorderMode.CLAMP) == expected


@pytest.mark.parametrize("index", [-5, -1, 1, 4])
def test_path_5_reflect_single_pixel(index):
    """Путь 5: вершины 1-2-4-6-8-10-11-12.

    Изображение шириной в один пиксель: отражать нечего,
    любой индекс отображается в единственный пиксель.
    """
    assert map_index(index, 1, BorderMode.REFLECT) == 0


@pytest.mark.parametrize(
    "index, expected",
    [(-1, 1), (-2, 2), (-3, 3)],
)
def test_path_6_reflect_folded_inside(index, expected):
    """Путь 6: вершины 1-2-4-6-8-10-11-13-14-15.

    Свёрнутый индекс попал внутрь изображения.
    Для length = 5 период равен 8: abs(-1) % 8 = 1 < 5.
    """
    assert map_index(index, 5, BorderMode.REFLECT) == expected


@pytest.mark.parametrize(
    "index, expected",
    [(5, 3), (6, 2), (7, 1)],
)
def test_path_7_reflect_folded_outside(index, expected):
    """Путь 7: вершины 1-2-4-6-8-10-11-13-14-16.

    Свёрнутый индекс вышел за изображение и отражается обратно.
    Для length = 5 период равен 8: 5 % 8 = 5 >= 5, значит 8 - 5 = 3.
    """
    assert map_index(index, 5, BorderMode.REFLECT) == expected


def test_path_8_unsupported_mode():
    """Путь 8: вершины 1-2-4-6-8-10-17.

    Значение режима не принадлежит перечислению. Ветка недостижима
    при корректном использовании, но защищает от ошибки вызова.
    """
    with pytest.raises(borders.BorderError) as info:
        map_index(-1, 10, "произвольная строка")

    assert "не поддержан" in str(info.value)


def test_reflect_boundary_equals_period():
    """Граница пути 6 и 7: свёрнутый индекс равен length - 1."""
    assert map_index(4, 5, BorderMode.REFLECT) == 4
    assert map_index(-4, 5, BorderMode.REFLECT) == 4


# =====================================================================
# Ветви вспомогательных функций свёртки
# =====================================================================


@pytest.mark.parametrize("mode", [BorderMode.CLAMP, BorderMode.REFLECT])
def test_border_indices_match_reference(mode):
    """Векторное построение индексов совпадает с эталонным."""
    length, radius = 7, 2

    produced = convolution._border_indices(length, radius, mode)
    expected = [map_index(i, length, mode)
                for i in range(-radius, length + radius)]

    assert list(produced) == expected


def test_border_indices_rejects_zero_mode():
    """Ветка отказа: режим нулей не имеет индексного представления."""
    with pytest.raises(convolution.ImageError):
        convolution._border_indices(5, 1, BorderMode.ZERO)


def test_border_indices_single_pixel():
    """Ветка length == 1 в векторном построении индексов."""
    produced = convolution._border_indices(1, 2, BorderMode.REFLECT)

    assert list(produced) == [0, 0, 0, 0, 0]


def test_convolve_rejects_unsupported_border():
    """Ветка отказа в convolve_fast при некорректном режиме."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(convolution.ImageError) as info:
        convolution.convolve_fast(image, kernel.IDENTITY, "не режим")

    assert "не поддержан" in str(info.value)


def test_naive_rejects_large_image():
    """Ветка защиты наивной реализации от больших изображений."""
    image = np.zeros((600, 600, 3), dtype=np.uint8)

    with pytest.raises(convolution.ImageError) as info:
        convolution.convolve_naive(image, kernel.IDENTITY)

    assert "пикселей" in str(info.value)


def test_naive_rejects_non_kernel():
    """Ветка проверки типа аргумента в наивной реализации."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(convolution.ImageError):
        convolution.convolve_naive(image, [[1]])


def test_zero_coefficients_skipped(sample_image):
    """Ветка пропуска нулевых коэффициентов ядра.

    Тождественное ядро состоит из нулей, кроме центра, поэтому
    ветка continue выполняется восемь раз из девяти.
    """
    result = convolution.convolve_fast(sample_image, kernel.IDENTITY)

    assert np.array_equal(result, sample_image)


def test_convolve_dispatches_to_naive(sample_image):
    """Ветка выбора реализации в диспетчере convolve."""
    fast = convolution.convolve(sample_image, kernel.BOX_BLUR,
                                use_fast=True)
    naive = convolution.convolve(sample_image, kernel.BOX_BLUR,
                                 use_fast=False)

    assert np.array_equal(fast, naive)


# =====================================================================
# Ветви построения ядра
# =====================================================================


def test_divisor_from_sum():
    """Ветка: делитель не задан, сумма коэффициентов ненулевая."""
    built = kernel.Kernel([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    assert built.divisor == 9.0
    assert built.is_normalized


def test_divisor_falls_back_to_one():
    """Ветка: сумма коэффициентов равна нулю, деление отключается."""
    built = kernel.Kernel([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    assert built.divisor == 1.0
    assert not built.is_normalized


def test_explicit_divisor_used():
    """Ветка: делитель задан явно и не пересчитывается по сумме."""
    built = kernel.Kernel([[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                          divisor=3.0)

    assert built.divisor == 3.0


def test_default_name_assigned():
    """Ветка подстановки названия по умолчанию."""
    assert kernel.Kernel([[1]]).name == "Без названия"


def test_kernel_equality_branches():
    """Ветки сравнения ядер между собой и с посторонним объектом."""
    first = kernel.Kernel([[1]])
    second = kernel.Kernel([[1]])
    other = kernel.Kernel([[2]])

    assert first == second
    assert first != other
    assert first.__eq__("не ядро") is NotImplemented


def test_kernel_repr_contains_name():
    """Ветка текстового представления ядра."""
    assert "Тождественный" in repr(kernel.IDENTITY)


def test_non_numeric_values_rejected():
    """Ветка обработки нечисловых коэффициентов."""
    with pytest.raises(kernel.KernelError) as info:
        kernel.Kernel([["a", "b", "c"], ["d", "e", "f"],
                       ["g", "h", "i"]])

    assert "не приводятся" in str(info.value)


# =====================================================================
# Проверка предела памяти
# =====================================================================


def test_memory_estimate_grows_with_size():
    """Оценка памяти растёт пропорционально числу пикселей."""
    small = convolution.estimate_memory_bytes(1000, 1000, 3, 1)
    large = convolution.estimate_memory_bytes(2000, 2000, 3, 1)

    assert large > 3.5 * small


def test_memory_estimate_accounts_for_padding():
    """Поля ядра учитываются в оценке."""
    narrow = convolution.estimate_memory_bytes(100, 100, 3, 1)
    wide = convolution.estimate_memory_bytes(100, 100, 3, 7)

    assert wide > narrow


def test_oversized_image_rejected(monkeypatch):
    """Ветка отказа при превышении предела памяти.

    Регрессионная проверка дефекта ДЕФ-08: ранее операционная
    система завершала процесс, не давая приложению сообщить
    об ошибке.
    """
    monkeypatch.setattr(convolution, "MEMORY_LIMIT_BYTES", 1024)
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(convolution.ImageError) as info:
        convolution.convolve_fast(image, kernel.BOX_BLUR)

    message = str(info.value)
    assert "слишком велико" in message
    assert "ГБ" in message


def test_limit_allows_reasonable_images(sample_image):
    """Штатный предел не мешает обработке обычных изображений."""
    result = convolution.convolve_fast(sample_image, kernel.BOX_BLUR)

    assert result.shape == sample_image.shape
