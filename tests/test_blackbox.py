"""Тестирование методом чёрного ящика.

Тесты построены по двум техникам проектирования:

* разбиение на классы эквивалентности — по одному представителю
  от каждого класса допустимых и недопустимых входных данных;
* анализ граничных значений — значения на границах классов
  и непосредственно за ними.

Внутреннее устройство функций при составлении тестов не
учитывалось: проверяется только соответствие входа выходу.

Идентификаторы тестов соответствуют таблицам тест-кейсов
в отчёте по лабораторной работе.
"""

from __future__ import annotations

import numpy as np
import pytest

from imgconv import borders, convolution, kernel

# =====================================================================
# Классы эквивалентности входа «ядро свёртки»
# =====================================================================


@pytest.mark.parametrize(
    "case_id, values",
    [
        ("ЧЯ-01", [[1]]),
        ("ЧЯ-02", [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        ("ЧЯ-03", [[1] * 5 for _ in range(5)]),
        ("ЧЯ-04", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        ("ЧЯ-05", [[0.5, 0.25, 0.5], [0.25, 1.0, 0.25],
                   [0.5, 0.25, 0.5]]),
    ],
)
def test_valid_kernel_accepted(case_id, values):
    """Допустимый класс: квадратное ядро нечётного размера."""
    result = kernel.Kernel(values)

    assert result.size == len(values)
    assert result.radius == len(values) // 2


@pytest.mark.parametrize(
    "case_id, values, fragment",
    [
        ("ЧЯ-06", [[1, 2], [3, 4]], "нечётным"),
        ("ЧЯ-07", [[1] * 4 for _ in range(4)], "нечётным"),
        ("ЧЯ-08", [[1, 2, 3], [4, 5, 6]], "квадратным"),
        ("ЧЯ-09", [], "пустым"),
        ("ЧЯ-10", [[[1]]], "двумерным"),
        ("ЧЯ-11", [[1, 2, 3]], "квадратным"),
        ("ЧЯ-12", None, "не задано"),
    ],
)
def test_invalid_kernel_rejected(case_id, values, fragment):
    """Недопустимые классы: ядро отвергается с внятным сообщением."""
    with pytest.raises(kernel.KernelError) as info:
        kernel.Kernel(values)

    assert fragment in str(info.value)


@pytest.mark.parametrize(
    "case_id, divisor",
    [("ЧЯ-13", 0), ("ЧЯ-14", 0.0)],
)
def test_zero_divisor_rejected(case_id, divisor):
    """Граница: нулевой делитель недопустим."""
    with pytest.raises(kernel.KernelError) as info:
        kernel.Kernel([[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                      divisor=divisor)

    assert "нулём" in str(info.value)


@pytest.mark.parametrize(
    "case_id, values",
    [
        ("ЧЯ-15", [[float("inf")] * 3 for _ in range(3)]),
        ("ЧЯ-16", [[float("nan")] * 3 for _ in range(3)]),
    ],
)
def test_non_finite_kernel_rejected(case_id, values):
    """Недопустимый класс: бесконечность и неопределённость."""
    with pytest.raises(kernel.KernelError) as info:
        kernel.Kernel(values)

    assert "конечными" in str(info.value)


def test_kernel_values_are_isolated():
    """Изменение возвращённой матрицы не меняет само ядро."""
    source = kernel.Kernel([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    borrowed = source.values
    borrowed[0, 0] = 99.0

    assert source.values[0, 0] == 0.0


# =====================================================================
# Классы эквивалентности входа «изображение»
# =====================================================================


@pytest.mark.parametrize(
    "case_id, shape",
    [
        ("ЧЯ-17", (4, 4)),
        ("ЧЯ-18", (4, 4, 1)),
        ("ЧЯ-19", (4, 4, 3)),
        ("ЧЯ-20", (4, 4, 4)),
    ],
)
def test_supported_channel_counts(case_id, shape):
    """Допустимый класс: 1, 3 и 4 канала, а также полутон без оси."""
    image = np.zeros(shape, dtype=np.uint8)

    result = convolution.convolve_fast(image, kernel.IDENTITY)

    assert result.shape == image.shape


@pytest.mark.parametrize(
    "case_id, image, fragment",
    [
        ("ЧЯ-21", None, "не задано"),
        ("ЧЯ-22", np.zeros((0, 0), dtype=np.uint8), "пустым"),
        ("ЧЯ-23", np.zeros((4, 4, 2), dtype=np.uint8), "канала"),
        ("ЧЯ-24", np.zeros((4, 4, 5), dtype=np.uint8), "канала"),
        ("ЧЯ-25", np.zeros((2, 2, 2, 2), dtype=np.uint8), "измерений"),
        ("ЧЯ-26", np.zeros(4, dtype=np.uint8), "измерений"),
    ],
)
def test_invalid_image_rejected(case_id, image, fragment):
    """Недопустимые классы изображения."""
    with pytest.raises(convolution.ImageError) as info:
        convolution.convolve_fast(image, kernel.IDENTITY)

    assert fragment in str(info.value)


def test_kernel_argument_type_checked():
    """ЧЯ-27: вместо ядра передан посторонний объект."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(convolution.ImageError):
        convolution.convolve_fast(image, "не ядро")


# =====================================================================
# Анализ граничных значений
# =====================================================================


@pytest.mark.parametrize(
    "case_id, shape",
    [
        ("ЧЯ-28", (1, 1, 3)),
        ("ЧЯ-29", (1, 5, 3)),
        ("ЧЯ-30", (5, 1, 3)),
        ("ЧЯ-31", (2, 2, 3)),
        ("ЧЯ-32", (3, 3, 3)),
    ],
)
@pytest.mark.parametrize("mode", list(borders.BorderMode))
def test_image_smaller_than_kernel(case_id, shape, mode):
    """Граница: изображение меньше или равно размеру ядра.

    Размер ядра 5x5 превышает сторону изображения, поэтому окно
    почти целиком выходит за его пределы.
    """
    image = np.full(shape, 100, dtype=np.uint8)

    result = convolution.convolve_fast(image, kernel.GAUSSIAN_BLUR_5,
                                       mode)

    assert result.shape == shape


@pytest.mark.parametrize(
    "case_id, level",
    [("ЧЯ-33", 0), ("ЧЯ-34", 1), ("ЧЯ-35", 254), ("ЧЯ-36", 255)],
)
def test_brightness_boundaries_preserved(case_id, level):
    """Граница диапазона яркости: 0 и 255 сохраняются тождественным
    ядром, соседние значения тоже."""
    image = np.full((4, 4, 3), level, dtype=np.uint8)

    result = convolution.convolve_fast(image, kernel.IDENTITY)

    assert np.all(result == level)


def test_upper_clipping():
    """ЧЯ-37: результат выше 255 отсекается, а не переполняется."""
    image = np.full((5, 5, 3), 250, dtype=np.uint8)
    image[2, 2] = 255

    result = convolution.convolve_fast(image, kernel.SHARPEN)

    assert result.max() == 255


def test_lower_clipping():
    """ЧЯ-38: отрицательный результат отсекается нулём."""
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[2, 2] = 255

    result = convolution.convolve_fast(image, kernel.EDGE_DETECT)

    assert result.min() == 0
    assert result.dtype == np.uint8


def test_minimal_kernel_is_identity():
    """ЧЯ-39: ядро 1x1 с единицей не изменяет изображение."""
    single = kernel.Kernel([[1]])
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)

    assert np.array_equal(convolution.convolve_fast(image, single),
                          image)


# =====================================================================
# Функциональные инварианты
# =====================================================================


def test_identity_preserves_image(sample_image):
    """ЧЯ-40: тождественное ядро возвращает исходное изображение."""
    result = convolution.convolve_fast(sample_image, kernel.IDENTITY)

    assert np.array_equal(result, sample_image)


@pytest.mark.parametrize("name", ["Размытие средним",
                                  "Размытие по Гауссу",
                                  "Размытие по Гауссу 5x5"])
def test_blur_preserves_flat_image(name, flat_image):
    """ЧЯ-41: нормализованное размытие не меняет однородную заливку."""
    result = convolution.convolve_fast(flat_image,
                                       kernel.get_preset(name))

    assert np.array_equal(result, flat_image)


def test_unknown_preset_rejected():
    """ЧЯ-42: запрос несуществующего фильтра."""
    with pytest.raises(kernel.KernelError) as info:
        kernel.get_preset("Фильтр, которого нет")

    assert "Неизвестный фильтр" in str(info.value)


def test_unknown_border_title_rejected():
    """ЧЯ-43: запрос несуществующего режима границ."""
    with pytest.raises(borders.BorderError) as info:
        borders.mode_by_title("Режим, которого нет")

    assert "Неизвестный режим" in str(info.value)


def test_all_presets_produce_valid_output(sample_image):
    """ЧЯ-44: каждый фильтр даёт изображение той же формы."""
    for name in kernel.preset_names():
        result = convolution.convolve_fast(sample_image,
                                           kernel.get_preset(name))

        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8


# =====================================================================
# Классы эквивалентности входа «файл изображения»
# =====================================================================


@pytest.mark.parametrize(
    "case_id, suffix",
    [("ЧЯ-45", ".png"), ("ЧЯ-46", ".bmp"), ("ЧЯ-47", ".tiff")],
)
def test_supported_formats_round_trip(case_id, suffix, tmp_path,
                                      sample_image):
    """Допустимый класс: поддерживаемые форматы без потерь."""
    from imgconv import imageio

    path = tmp_path / f"image{suffix}"
    imageio.save_image(path, sample_image)
    restored = imageio.load_image(path)

    assert np.array_equal(restored, sample_image)


def test_jpeg_accepted(tmp_path, sample_image):
    """ЧЯ-48: JPEG принимается, хотя сжатие теряет точность."""
    from imgconv import imageio

    path = tmp_path / "image.jpg"
    imageio.save_image(path, sample_image)
    restored = imageio.load_image(path)

    assert restored.shape == sample_image.shape


def test_missing_file_rejected(tmp_path):
    """ЧЯ-49: файла не существует."""
    from imgconv import imageio

    with pytest.raises(imageio.ImageIOError) as info:
        imageio.load_image(tmp_path / "отсутствует.png")

    assert "не найден" in str(info.value)


@pytest.mark.parametrize(
    "case_id, name",
    [("ЧЯ-50", "файл.txt"), ("ЧЯ-51", "файл.docx")],
)
def test_unsupported_suffix_rejected_on_load(case_id, name, tmp_path):
    """Недопустимый класс: расширение вне списка поддерживаемых."""
    from imgconv import imageio

    path = tmp_path / name
    path.write_text("содержимое", encoding="utf-8")

    with pytest.raises(imageio.ImageIOError) as info:
        imageio.load_image(path)

    assert "не поддерживается" in str(info.value)


def test_missing_suffix_reported_separately(tmp_path):
    """ЧЯ-52: файл без расширения получает отдельное пояснение.

    Регрессионная проверка дефекта ДЕФ-03: прежнее сообщение
    обрывалось на двоеточии и не подсказывало пользователю,
    какие форматы допустимы.
    """
    from imgconv import imageio

    path = tmp_path / "файл_без_расширения"
    path.write_text("содержимое", encoding="utf-8")

    with pytest.raises(imageio.ImageIOError) as info:
        imageio.load_image(path)

    message = str(info.value)
    assert "Не указано расширение" in message
    assert ".png" in message


def test_corrupted_file_rejected(tmp_path):
    """ЧЯ-53: расширение верное, содержимое не является картинкой."""
    from imgconv import imageio

    path = tmp_path / "повреждённый.png"
    path.write_bytes(bytes([0, 1, 2]) + "это не изображение".encode("utf-8"))

    with pytest.raises(imageio.ImageIOError) as info:
        imageio.load_image(path)

    assert "прочитать" in str(info.value)


def test_unsupported_suffix_rejected_on_save(tmp_path, sample_image):
    """ЧЯ-54: сохранение в неподдерживаемый формат."""
    from imgconv import imageio

    with pytest.raises(imageio.ImageIOError) as info:
        imageio.save_image(tmp_path / "результат.gif", sample_image)

    assert "не поддерживается" in str(info.value)


def test_empty_array_rejected_on_save(tmp_path):
    """ЧЯ-55: граница — сохранение пустого массива."""
    from imgconv import imageio

    with pytest.raises(imageio.ImageIOError) as info:
        imageio.save_image(tmp_path / "пусто.png",
                           np.zeros((0, 0), dtype=np.uint8))

    assert "пустое" in str(info.value)


def test_save_to_missing_directory_rejected(tmp_path, sample_image):
    """ЧЯ-56: каталог назначения не существует."""
    from imgconv import imageio

    target = tmp_path / "нет_такого_каталога" / "результат.png"

    with pytest.raises(imageio.ImageIOError) as info:
        imageio.save_image(target, sample_image)

    assert "сохранить" in str(info.value)
