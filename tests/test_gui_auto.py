"""Автоматизированное тестирование графического интерфейса.

Тесты воспроизводят действия пользователя программно: нажимают
кнопки настоящими событиями мыши, выбирают значения в выпадающих
списках и проверяют состояние виджетов. Используется библиотека
pytest-qt, предоставляющая приспособление qtbot.

Идентификаторы соответствуют таблице автоматизированных тестов
в отчёте по лабораторной работе.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt

from imgconv import borders, imageio, kernel


@pytest.fixture
def picture(tmp_path, sample_image):
    """Файл изображения на диске для сценариев открытия."""
    path = tmp_path / "исходное.png"
    imageio.save_image(path, sample_image)
    return path


def click(qtbot, button):
    """Нажатие кнопки настоящим событием мыши."""
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


# =====================================================================
# Состояние элементов управления
# =====================================================================


def test_at_start_actions_disabled(main_window):
    """АТ-01: до открытия файла действия недоступны."""
    assert not main_window.apply_button.isEnabled()
    assert not main_window.reset_button.isEnabled()
    assert not main_window.save_button.isEnabled()
    assert "Откройте изображение" in main_window.status.currentMessage()


def test_open_enables_apply(qtbot, apply_filter, main_window, picture):
    """АТ-02: после открытия файла доступно применение фильтра."""
    assert main_window.open_path(str(picture))

    assert main_window.apply_button.isEnabled()
    assert not main_window.save_button.isEnabled()
    assert "Загружено" in main_window.status.currentMessage()


def test_apply_enables_save_and_reset(qtbot, apply_filter, main_window, picture):
    """АТ-03: после свёртки доступны сохранение и сброс."""
    main_window.open_path(str(picture))

    assert apply_filter(main_window)

    assert main_window.result_image is not None
    assert main_window.save_button.isEnabled()
    assert main_window.reset_button.isEnabled()


def test_reset_clears_result(qtbot, apply_filter, main_window, picture):
    """АТ-04: сброс убирает результат, оставляя исходное изображение."""
    main_window.open_path(str(picture))
    assert apply_filter(main_window)

    click(qtbot, main_window.reset_button)

    assert main_window.result_image is None
    assert main_window.source_image is not None
    assert not main_window.save_button.isEnabled()
    assert main_window.apply_button.isEnabled()


def test_new_image_clears_previous_result(qtbot, apply_filter, main_window, picture,
                                          sample_image):
    """АТ-05: загрузка нового изображения убирает прежний результат."""
    main_window.open_path(str(picture))
    assert apply_filter(main_window)

    main_window.load_array(sample_image)

    assert main_window.result_image is None
    assert not main_window.save_button.isEnabled()


# =====================================================================
# Наполнение списков и выбор параметров
# =====================================================================


def test_filter_list_matches_presets(main_window):
    """АТ-06: список фильтров совпадает с набором ядра."""
    shown = [main_window.filter_box.itemText(i)
             for i in range(main_window.filter_box.count())]

    assert shown == kernel.preset_names()


def test_border_list_matches_modes(main_window):
    """АТ-07: список режимов границ совпадает с перечислением."""
    shown = [main_window.border_box.itemText(i)
             for i in range(main_window.border_box.count())]

    assert shown == borders.mode_titles()


@pytest.mark.parametrize("name", kernel.preset_names())
def test_every_filter_applies(qtbot, apply_filter, main_window, picture, name):
    """АТ-08: каждый фильтр из списка применяется без ошибок."""
    main_window.open_path(str(picture))
    main_window.filter_box.setCurrentText(name)

    assert apply_filter(main_window)

    assert main_window.result_image is not None
    assert name in main_window.status.currentMessage()


@pytest.mark.parametrize("title", borders.mode_titles())
def test_every_border_mode_applies(qtbot, apply_filter, main_window, picture, title):
    """АТ-09: каждый режим обработки границ применяется."""
    main_window.open_path(str(picture))
    main_window.border_box.setCurrentText(title)

    assert apply_filter(main_window)

    assert title in main_window.status.currentMessage()


def test_filter_choice_changes_result(qtbot, apply_filter, main_window, picture):
    """АТ-10: смена фильтра меняет результат свёртки."""
    main_window.open_path(str(picture))

    main_window.filter_box.setCurrentText("Тождественный")
    assert apply_filter(main_window)
    identity = main_window.result_image.copy()

    main_window.filter_box.setCurrentText("Выделение границ")
    assert apply_filter(main_window)

    assert not np.array_equal(identity, main_window.result_image)


# =====================================================================
# Предпросмотр и строка состояния
# =====================================================================


def test_previews_filled(qtbot, apply_filter, main_window, picture):
    """АТ-11: обе области предпросмотра заполнены после свёртки."""
    main_window.open_path(str(picture))
    assert apply_filter(main_window)

    assert not main_window.source_view.pixmap().isNull()
    assert not main_window.result_view.pixmap().isNull()


def test_status_reports_duration(qtbot, apply_filter, main_window, picture):
    """АТ-12: строка состояния показывает время обработки."""
    main_window.open_path(str(picture))

    assert apply_filter(main_window)

    assert "мс" in main_window.status.currentMessage()
    assert main_window.last_duration_ms >= 0.0


# =====================================================================
# Обработка ошибок
# =====================================================================


def test_missing_file_reported(main_window, tmp_path):
    """АТ-13: открытие несуществующего файла отклоняется."""
    assert not main_window.open_path(str(tmp_path / "нет.png"))

    assert "не найден" in main_window.last_error
    assert main_window.source_image is None


def test_corrupted_file_reported(main_window, tmp_path):
    """АТ-14: повреждённый файл отклоняется без аварии."""
    path = tmp_path / "битый.png"
    path.write_bytes(bytes([0, 1, 2, 3]))

    assert not main_window.open_path(str(path))
    assert "прочитать" in main_window.last_error


def test_save_without_suffix_reported(qtbot, apply_filter, main_window, picture,
                                      tmp_path):
    """АТ-15: сохранение без расширения сопровождается подсказкой.

    Регрессионная проверка дефекта ДЕФ-03.
    """
    main_window.open_path(str(picture))
    assert apply_filter(main_window)

    assert not main_window.save_path(str(tmp_path / "результат"))
    assert "Не указано расширение" in main_window.last_error


def test_error_flag_cleared_after_success(qtbot, apply_filter, main_window, picture,
                                          tmp_path):
    """АТ-16: признак ошибки снимается после успешной операции.

    Регрессионная проверка дефекта ДЕФ-04.
    """
    main_window.open_path(str(picture))
    assert apply_filter(main_window)

    main_window.save_path(str(tmp_path / "без_расширения"))
    assert main_window.last_error is not None

    assert main_window.save_path(str(tmp_path / "итог.png"))
    assert main_window.last_error is None


def test_saved_file_can_be_reopened(qtbot, apply_filter, main_window, picture,
                                    tmp_path):
    """АТ-17: сохранённый результат открывается обратно."""
    main_window.open_path(str(picture))
    assert apply_filter(main_window)
    target = tmp_path / "результат.png"

    assert main_window.save_path(str(target))

    restored = imageio.load_image(target)
    assert np.array_equal(restored, main_window.result_image)


@pytest.mark.parametrize("suffix", [".png", ".bmp", ".tif"])
def test_save_supported_formats(qtbot, apply_filter, main_window, picture, tmp_path,
                                suffix):
    """АТ-18: результат сохраняется во все заявленные форматы.

    Регрессионная проверка дефекта ДЕФ-02: формат TIFF
    поддерживается модулем, но отсутствовал в фильтре диалога.
    """
    main_window.open_path(str(picture))
    assert apply_filter(main_window)

    assert main_window.save_path(str(tmp_path / f"итог{suffix}"))


def test_save_filter_lists_all_supported_formats(main_window):
    """АТ-19: фильтр диалога сохранения перечисляет все форматы.

    Регрессионная проверка дефекта ДЕФ-02 на уровне интерфейса.
    """
    import inspect

    source = inspect.getsource(type(main_window).on_save_clicked)

    for suffix in imageio.SUPPORTED_SUFFIXES:
        assert f"*{suffix}" in source


# =====================================================================
# Отрисовка предпросмотра для разных видов изображений
# =====================================================================


@pytest.mark.parametrize(
    "case_id, shape",
    [("АТ-20", (12, 16)), ("АТ-21", (12, 16, 3)), ("АТ-22", (12, 16, 4))],
)
def test_preview_handles_channel_layouts(main_window, case_id, shape):
    """Полутоновое, цветное и цветное с прозрачностью изображения."""
    image = np.full(shape, 80, dtype=np.uint8)

    main_window.load_array(image)

    assert not main_window.source_view.pixmap().isNull()


# =====================================================================
# Обработчики системных диалогов
# =====================================================================


def test_open_dialog_loads_chosen_file(qtbot, apply_filter, main_window, picture,
                                       monkeypatch):
    """АТ-23: выбор файла в диалоге приводит к загрузке."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(picture), ""))

    click(qtbot, main_window.open_button)

    assert main_window.source_image is not None


def test_open_dialog_cancel_changes_nothing(qtbot, apply_filter, main_window,
                                            monkeypatch):
    """АТ-24: отмена диалога открытия не меняет состояния."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: ("", ""))

    click(qtbot, main_window.open_button)

    assert main_window.source_image is None
    assert not main_window.apply_button.isEnabled()


def test_save_dialog_writes_file(qtbot, apply_filter, main_window, picture, tmp_path,
                                 monkeypatch):
    """АТ-25: выбор имени в диалоге приводит к сохранению."""
    from PySide6.QtWidgets import QFileDialog

    target = tmp_path / "из_диалога.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(target), ""))

    main_window.open_path(str(picture))
    assert apply_filter(main_window)
    click(qtbot, main_window.save_button)

    assert target.exists()


def test_save_dialog_cancel_writes_nothing(qtbot, apply_filter, main_window, picture,
                                           tmp_path, monkeypatch):
    """АТ-26: отмена диалога сохранения не создаёт файлов."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: ("", ""))

    main_window.open_path(str(picture))
    assert apply_filter(main_window)
    click(qtbot, main_window.save_button)

    assert list(tmp_path.glob("*.png")) == [picture]


def test_save_without_result_refused(main_window):
    """АТ-27: сохранение без результата отклоняется."""
    assert not main_window.save_path("не_важно.png")
    assert "Нечего сохранять" in main_window.status.currentMessage()


def test_apply_without_image_refused(apply_directly, main_window):
    """АТ-28: применение фильтра без изображения отклоняется."""
    assert not apply_directly(main_window)
    assert "Сначала откройте" in main_window.status.currentMessage()


# =====================================================================
# Фоновый поток обработки
# =====================================================================


def test_worker_emits_result(qtbot, sample_image):
    """АТ-29: рабочий поток передаёт результат сигналом.

    Метод run вызывается напрямую: так проверяется его логика
    без запуска настоящего потока.
    """
    from imgconv_gui.worker import ConvolutionWorker

    worker = ConvolutionWorker(sample_image, kernel.IDENTITY,
                               borders.BorderMode.CLAMP)

    with qtbot.waitSignal(worker.succeeded, timeout=5000) as signal:
        worker.run()

    assert np.array_equal(signal.args[0], sample_image)


def test_worker_reports_failure(qtbot):
    """АТ-30: ошибка внутри потока передаётся сигналом, а не бросается.

    Исключение, покинувшее run, не может быть перехвачено
    в основном потоке и привело бы к аварийному завершению.
    """
    from imgconv_gui.worker import ConvolutionWorker

    broken = np.zeros((4, 4, 5), dtype=np.uint8)
    worker = ConvolutionWorker(broken, kernel.IDENTITY,
                               borders.BorderMode.CLAMP)

    with qtbot.waitSignal(worker.failed, timeout=5000) as signal:
        worker.run()

    assert "канала" in signal.args[0]


def test_apply_refused_while_busy(main_window, sample_image,
                                  monkeypatch):
    """АТ-31: повторный запуск во время обработки отклоняется.

    Иначе два потока писали бы в одно поле результата.
    """
    main_window.load_array(sample_image)
    monkeypatch.setattr(type(main_window), "is_busy",
                        property(lambda self: True))

    assert not main_window.on_apply_clicked()
    assert "уже выполняется" in main_window.status.currentMessage()


def test_progress_hidden_when_idle(main_window):
    """АТ-32: индикатор занятости скрыт, пока обработка не идёт."""
    assert main_window.progress.isHidden()
    assert not main_window.is_busy


def test_save_without_result_via_handler(main_window):
    """АТ-33: обработчик кнопки сохранения без результата."""
    main_window.on_save_clicked()

    assert "Нечего сохранять" in main_window.status.currentMessage()
