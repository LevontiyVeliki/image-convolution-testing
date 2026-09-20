"""Интеграционное тестирование.

Проверяется взаимодействие компонентов, а не их поведение по
отдельности. Предметом проверки служат стыки: передача данных
между модулем работы с файлами, ядром свёртки и графическим
интерфейсом, а также распространение ошибок между слоями.

Идентификаторы соответствуют таблице сценариев в отчёте.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt

from imgconv import borders, convolution, imageio, kernel


@pytest.fixture
def photo(tmp_path, rng):
    """Цветное изображение, записанное на диск."""
    array = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    path = tmp_path / "фото.png"
    imageio.save_image(path, array)
    return path, array


@pytest.fixture
def transparent(tmp_path):
    """Изображение с чёткой границей прозрачности."""
    array = np.zeros((16, 16, 4), dtype=np.uint8)
    array[..., :3] = 200
    array[:, :8, 3] = 0
    array[:, 8:, 3] = 255

    path = tmp_path / "прозрачное.png"
    # Режим определяется формой массива, указывать его
    # параметром mode не нужно: он объявлен устаревшим
    Image.fromarray(array).save(path)
    return path, array


def click(qtbot, button):
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


# =====================================================================
# Стык «файл — ядро — файл»
# =====================================================================


def test_full_pipeline_round_trip(tmp_path, photo):
    """ИТ-01: чтение, обработка и запись сохраняют форму и тип."""
    path, original = photo

    loaded = imageio.load_image(path)
    processed = convolution.convolve(loaded, kernel.GAUSSIAN_BLUR)
    target = tmp_path / "результат.png"
    imageio.save_image(target, processed)
    restored = imageio.load_image(target)

    assert np.array_equal(loaded, original)
    assert np.array_equal(restored, processed)


def test_repeated_processing_chain(tmp_path, photo):
    """ИТ-02: результат обработки пригоден для повторной обработки."""
    path, _ = photo

    first = convolution.convolve(imageio.load_image(path),
                                 kernel.BOX_BLUR)
    middle = tmp_path / "шаг1.png"
    imageio.save_image(middle, first)

    second = convolution.convolve(imageio.load_image(middle),
                                  kernel.SHARPEN)

    assert second.shape == first.shape
    assert second.dtype == np.uint8


@pytest.mark.parametrize("suffix", [".png", ".bmp", ".tif", ".tiff"])
def test_lossless_formats_preserve_result(tmp_path, photo, suffix):
    """ИТ-03: форматы без потерь возвращают результат без изменений."""
    path, _ = photo
    processed = convolution.convolve(imageio.load_image(path),
                                     kernel.EDGE_DETECT)

    target = tmp_path / f"итог{suffix}"
    imageio.save_image(target, processed)

    assert np.array_equal(imageio.load_image(target), processed)


# =====================================================================
# Стык «интерфейс — ядро»
# =====================================================================


@pytest.mark.parametrize("name", kernel.preset_names())
def test_window_applies_selected_kernel(qtbot, apply_filter, main_window, photo, name):
    """ИТ-04: окно применяет именно выбранный в списке фильтр."""
    path, _ = photo
    main_window.open_path(str(path))
    main_window.filter_box.setCurrentText(name)

    assert apply_filter(main_window)

    expected = convolution.convolve(main_window.source_image,
                                    kernel.get_preset(name),
                                    borders.BorderMode.ZERO)
    assert np.array_equal(main_window.result_image, expected)


@pytest.mark.parametrize("mode", list(borders.BorderMode))
def test_window_passes_border_mode(qtbot, apply_filter, main_window, photo, mode):
    """ИТ-05: выбранный режим границ доходит до ядра без подмены."""
    path, _ = photo
    main_window.open_path(str(path))
    main_window.filter_box.setCurrentText("Размытие по Гауссу 5x5")
    main_window.border_box.setCurrentText(mode.title)

    assert apply_filter(main_window)

    expected = convolution.convolve(main_window.source_image,
                                    kernel.GAUSSIAN_BLUR_5, mode)
    assert np.array_equal(main_window.result_image, expected)


def test_window_saves_what_it_shows(qtbot, apply_filter, main_window, photo, tmp_path):
    """ИТ-06: сохранённый файл совпадает с показанным результатом."""
    path, _ = photo
    main_window.open_path(str(path))
    assert apply_filter(main_window)

    target = tmp_path / "из_окна.png"
    assert main_window.save_path(str(target))

    assert np.array_equal(imageio.load_image(target),
                          main_window.result_image)


# =====================================================================
# Распространение ошибок между слоями
# =====================================================================


def test_file_error_reaches_window(main_window, tmp_path):
    """ИТ-07: ошибка модуля файлов доходит до строки состояния."""
    assert not main_window.open_path(str(tmp_path / "нет.png"))

    assert main_window.last_error in main_window.status.currentMessage()


def test_core_error_reaches_window(apply_directly, main_window):
    """ИТ-08: ошибка ядра доходит до окна, не роняя приложение."""
    main_window.load_array(np.zeros((4, 4, 3), dtype=np.uint8))
    # Подменяем источник заведомо недопустимым значением
    main_window._source = np.zeros((2, 2, 5), dtype=np.uint8)

    assert not apply_directly(main_window)
    assert "канала" in main_window.last_error


def test_window_survives_corrupted_file(main_window, tmp_path):
    """ИТ-09: повреждённый файл не нарушает состояние окна."""
    path = tmp_path / "битый.png"
    path.write_bytes(bytes(range(16)))

    assert not main_window.open_path(str(path))
    assert main_window.source_image is None
    assert not main_window.apply_button.isEnabled()


# =====================================================================
# Согласованность двух реализаций свёртки
# =====================================================================


@pytest.mark.parametrize("name", kernel.preset_names())
def test_implementations_agree_on_real_file(photo, name):
    """ИТ-10: обе реализации совпадают на изображении из файла."""
    path, _ = photo
    image = imageio.load_image(path)
    selected = kernel.get_preset(name)

    naive = convolution.convolve_naive(image, selected)
    fast = convolution.convolve_fast(image, selected)

    assert np.array_equal(naive, fast)


# =====================================================================
# Прозрачность: регрессионные проверки дефектов ДЕФ-05 и ДЕФ-07
# =====================================================================


def test_alpha_survives_loading(transparent):
    """ИТ-11: канал прозрачности не теряется при чтении файла.

    Регрессионная проверка дефекта ДЕФ-05.
    """
    path, original = transparent

    loaded = imageio.load_image(path)

    assert loaded.shape[2] == 4
    assert np.array_equal(loaded[:, :, 3], original[:, :, 3])


@pytest.mark.parametrize("name", kernel.preset_names())
def test_alpha_unchanged_by_convolution(transparent, name):
    """ИТ-12: фильтр не изменяет канал прозрачности.

    Регрессионная проверка дефекта ДЕФ-07.
    """
    path, _ = transparent
    image = imageio.load_image(path)

    result = convolution.convolve(image, kernel.get_preset(name))

    assert np.array_equal(result[:, :, 3], image[:, :, 3])


def test_colour_channels_are_processed(transparent, rng):
    """ИТ-13: цветовые каналы при этом действительно обрабатываются.

    Заполнение шумом выбрано намеренно: линейный градиент был бы
    неподвижной точкой размытия, и проверка прошла бы вхолостую
    при любой реализации.
    """
    path, _ = transparent
    image = imageio.load_image(path)
    image[:, :, :3] = rng.integers(0, 256, size=(16, 16, 3),
                                   dtype=np.uint8)

    result = convolution.convolve(image, kernel.GAUSSIAN_BLUR)

    assert not np.array_equal(result[:, :, :3], image[:, :, :3])
    assert np.array_equal(result[:, :, 3], image[:, :, 3])


def test_alpha_survives_round_trip(tmp_path, transparent):
    """ИТ-14: прозрачность сохраняется при записи и повторном чтении."""
    path, _ = transparent
    processed = convolution.convolve(imageio.load_image(path),
                                     kernel.BOX_BLUR)

    target = tmp_path / "с_прозрачностью.png"
    imageio.save_image(target, processed)

    assert np.array_equal(imageio.load_image(target), processed)


def test_alpha_dropped_for_jpeg(tmp_path, transparent):
    """ИТ-15: при сохранении в JPEG прозрачность отбрасывается.

    Регрессионная проверка дефекта ДЕФ-06: ранее библиотека
    прерывала сохранение сообщением на английском языке.
    """
    path, _ = transparent
    image = imageio.load_image(path)

    target = tmp_path / "без_прозрачности.jpg"
    imageio.save_image(target, image)

    assert imageio.load_image(target).shape[2] == 3


def test_transparent_image_passes_through_window(qtbot, apply_filter, main_window,
                                                 transparent):
    """ИТ-16: изображение с прозрачностью проходит весь путь в окне."""
    path, _ = transparent

    assert main_window.open_path(str(path))
    assert apply_filter(main_window)

    assert main_window.source_image.shape[2] == 4
    assert main_window.result_image.shape[2] == 4
    assert not main_window.result_view.pixmap().isNull()


def test_grayscale_file_converted_to_colour(tmp_path):
    """ИТ-17: полутоновый файл приводится к трёхканальному виду."""
    path = tmp_path / "серое.png"
    Image.fromarray(np.full((8, 8), 70, dtype=np.uint8)).save(path)

    loaded = imageio.load_image(path)

    assert loaded.shape == (8, 8, 3)
    assert np.all(loaded == 70)
