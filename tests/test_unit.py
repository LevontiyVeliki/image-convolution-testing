"""Модульное тестирование.

Набор демонстрирует приёмы модульного тестирования: изоляцию
проверяемой единицы, структуру «подготовка — действие — проверка»,
параметризацию и применение тестовых двойников.

В отличие от интеграционного набора, здесь каждая проверка
касается одной единицы кода, а её окружение при необходимости
заменяется подставными объектами.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt

from imgconv import borders, convolution, imageio, kernel
from imgconv.borders import BorderMode

# =====================================================================
# Изолированная проверка класса Kernel
# =====================================================================


class TestKernel:
    """Класс Kernel не обращается ни к файлам, ни к интерфейсу,
    поэтому проверяется полностью изолированно."""

    def test_size_and_radius(self):
        # Подготовка
        values = [[1] * 5 for _ in range(5)]

        # Действие
        result = kernel.Kernel(values)

        # Проверка
        assert result.size == 5
        assert result.radius == 2

    def test_divisor_defaults_to_sum(self):
        result = kernel.Kernel([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        assert result.divisor == 9.0

    def test_values_are_defensive_copy(self):
        """Возвращаемая матрица не связана с внутренним состоянием."""
        source = kernel.Kernel([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        borrowed = source.values
        borrowed[1, 1] = 42.0

        assert source.values[1, 1] == 1.0

    @pytest.mark.parametrize(
        "values, expected_normalized",
        [
            ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], True),
            ([[0, 0, 0], [0, 1, 0], [0, 0, 0]], True),
            ([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], False),
        ],
    )
    def test_normalization_flag(self, values, expected_normalized):
        assert kernel.Kernel(values).is_normalized is expected_normalized

    @pytest.mark.parametrize(
        "values",
        [
            [[1, 2], [3, 4]],
            [[1, 2, 3], [4, 5, 6]],
            [],
            [[[1]]],
        ],
    )
    def test_invalid_shapes_rejected(self, values):
        with pytest.raises(kernel.KernelError):
            kernel.Kernel(values)


# =====================================================================
# Изолированная проверка отображения индексов
# =====================================================================


class TestMapIndex:
    """Функция чистая: результат зависит только от аргументов."""

    @pytest.mark.parametrize(
        "index, length, mode, expected",
        [
            (0, 10, BorderMode.ZERO, 0),
            (9, 10, BorderMode.ZERO, 9),
            (-1, 10, BorderMode.ZERO, None),
            (10, 10, BorderMode.ZERO, None),
            (-1, 10, BorderMode.CLAMP, 0),
            (10, 10, BorderMode.CLAMP, 9),
            (-1, 5, BorderMode.REFLECT, 1),
            (5, 5, BorderMode.REFLECT, 3),
            (0, 1, BorderMode.REFLECT, 0),
        ],
    )
    def test_mapping(self, index, length, mode, expected):
        assert borders.map_index(index, length, mode) == expected

    def test_non_positive_length_rejected(self):
        with pytest.raises(borders.BorderError):
            borders.map_index(0, 0, BorderMode.CLAMP)


# =====================================================================
# Тестовые двойники
# =====================================================================


def test_stub_replaces_modal_dialog(main_window, tmp_path):
    """Заглушка вместо модального окна.

    Настоящий диалог ожидал бы нажатия кнопки, которое некому
    выполнить, и тест завис бы. Приспособление main_window
    подменяет метод показа диалога пустой функцией.
    """
    assert not main_window.open_path(str(tmp_path / "нет.png"))

    assert main_window.last_error is not None


def test_spy_records_call_arguments(main_window, sample_image,
                                    monkeypatch, apply_directly):
    """Шпион фиксирует аргументы вызова ядра.

    Проверяется не результат свёртки, а то, что окно передаёт
    в ядро именно выбранные пользователем параметры.
    """
    calls = []

    def spy(image, selected, border):
        calls.append((image, selected, border))
        return image

    monkeypatch.setattr(convolution, "convolve", spy)
    main_window.load_array(sample_image)
    main_window.filter_box.setCurrentText("Повышение резкости")
    main_window.border_box.setCurrentText("Зеркальное отражение")

    assert apply_directly(main_window)

    assert len(calls) == 1
    _, used_kernel, used_border = calls[0]
    assert used_kernel.name == "Повышение резкости"
    assert used_border is BorderMode.REFLECT


def test_mock_simulates_core_failure(main_window, sample_image,
                                     monkeypatch, apply_directly):
    """Подставной объект имитирует отказ ядра.

    Воспроизвести настоящую ошибку вычисления трудно, поэтому
    ядро заменяется функцией, которая всегда её выбрасывает.
    """
    def always_fails(image, selected, border):
        raise convolution.ImageError("искусственный отказ ядра")

    monkeypatch.setattr(convolution, "convolve", always_fails)
    main_window.load_array(sample_image)

    assert not apply_directly(main_window)

    assert "искусственный отказ" in main_window.last_error
    assert main_window.result_image is None
    assert not main_window.is_busy


def test_fake_file_dialog(qtbot, main_window, tmp_path, sample_image,
                          monkeypatch):
    """Подставной диалог выбора файла.

    Системный диалог требует участия человека, поэтому статический
    метод подменяется функцией, возвращающей заранее известный путь.
    """
    from PySide6.QtWidgets import QFileDialog

    path = tmp_path / "подставной.png"
    imageio.save_image(path, sample_image)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(path), ""))

    qtbot.mouseClick(main_window.open_button, Qt.MouseButton.LeftButton)

    assert np.array_equal(main_window.source_image, sample_image)


def test_monkeypatch_narrows_memory_limit(monkeypatch):
    """Подмена константы вместо создания огромного изображения.

    Проверить отказ по памяти на настоящем изображении означало бы
    занять несколько гигабайт. Вместо этого уменьшается предел.
    """
    monkeypatch.setattr(convolution, "MEMORY_LIMIT_BYTES", 4096)
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(convolution.ImageError) as info:
        convolution.convolve_fast(image, kernel.IDENTITY)

    assert "слишком велико" in str(info.value)


# =====================================================================
# Свойства, проверяемые на множестве входов
# =====================================================================


@pytest.mark.parametrize("side", [1, 2, 3, 5, 8, 13])
@pytest.mark.parametrize("mode", list(BorderMode))
def test_identity_is_neutral(side, mode, rng):
    """Тождественное ядро нейтрально при любом размере и режиме."""
    image = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)

    result = convolution.convolve_fast(image, kernel.IDENTITY, mode)

    assert np.array_equal(result, image)


@pytest.mark.parametrize("level", [0, 1, 127, 254, 255])
def test_blur_preserves_uniform_fill(level):
    """Размытие однородной заливки не меняет её ни при каком уровне."""
    image = np.full((6, 6, 3), level, dtype=np.uint8)

    result = convolution.convolve_fast(image, kernel.BOX_BLUR)

    assert np.all(result == level)


@pytest.mark.parametrize("name", kernel.preset_names())
def test_output_stays_in_range(name, sample_image):
    """Результат любого фильтра остаётся в диапазоне яркости."""
    result = convolution.convolve_fast(sample_image,
                                       kernel.get_preset(name))

    assert result.min() >= 0
    assert result.max() <= 255
    assert result.dtype == np.uint8


# =====================================================================
# Проверки, добавленные по результатам мутационного тестирования
# =====================================================================


@pytest.mark.parametrize(
    "values, shape",
    [
        ([[1, 2, 3], [4, 5, 6]], "2 x 3, строк меньше столбцов"),
        ([[1, 2], [3, 4], [5, 6]], "3 x 2, строк больше столбцов"),
        ([[1], [2], [3]], "3 x 1, один столбец"),
        ([[1, 2, 3]], "1 x 3, одна строка"),
    ],
)
def test_non_square_kernel_rejected_both_ways(values, shape):
    """Неквадратное ядро отвергается при любом соотношении сторон.

    Добавлено после мутационного тестирования: замена условия
    «строк не равно столбцам» на «строк меньше столбцов»
    не была обнаружена, поскольку все проверки использовали
    матрицы, у которых строк меньше.
    """
    with pytest.raises(kernel.KernelError) as info:
        kernel.Kernel(values)

    assert "квадратным" in str(info.value)


def test_offset_is_added_to_result():
    """Сдвиг прибавляется к результату, а не вычитается.

    Добавлено после мутационного тестирования: замена сложения
    на вычитание не обнаруживалась, так как сдвиг проверялся
    только косвенно, через совпадение двух реализаций.
    """
    # Нулевое ядро обнуляет вклад изображения, остаётся только сдвиг
    shifted = kernel.Kernel([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            divisor=1.0, offset=128.0)
    image = np.full((4, 4, 3), 200, dtype=np.uint8)

    result = convolution.convolve_fast(image, shifted)

    assert np.all(result == 128)


@pytest.mark.parametrize("offset, expected", [(0.0, 0), (64.0, 64),
                                              (200.0, 200)])
def test_offset_values_applied_exactly(offset, expected):
    """Значение сдвига переносится в результат без искажения."""
    shifted = kernel.Kernel([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            divisor=1.0, offset=offset)
    image = np.zeros((3, 3, 3), dtype=np.uint8)

    result = convolution.convolve_fast(image, shifted)

    assert np.all(result == expected)
