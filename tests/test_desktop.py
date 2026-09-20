"""Тестирование десктопного приложения.

Проверяются свойства, специфичные именно для настольной программы:
модель состояний окна, соответствие соглашениям рабочего стола,
работа с файловой системой и корректное завершение при наличии
фоновых потоков.

Идентификаторы соответствуют таблицам в отчёте.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence

from imgconv import imageio
from imgconv_gui.main_window import BASE_TITLE


@pytest.fixture
def picture(tmp_path, sample_image):
    path = tmp_path / "снимок.png"
    imageio.save_image(path, sample_image)
    return path


def buttons_state(window) -> tuple[bool, bool, bool, bool]:
    """Доступность четырёх кнопок в порядке их размещения."""
    return (window.open_button.isEnabled(),
            window.apply_button.isEnabled(),
            window.reset_button.isEnabled(),
            window.save_button.isEnabled())


# =====================================================================
# Модель состояний окна
# =====================================================================


def test_state_empty(main_window):
    """ДТ-01: состояние «Пусто» — доступно только открытие."""
    assert buttons_state(main_window) == (True, False, False, False)
    assert main_window.windowTitle() == BASE_TITLE
    assert main_window.progress.isHidden()


def test_state_loaded(main_window, picture):
    """ДТ-02: состояние «Загружено» — доступна обработка."""
    main_window.open_path(str(picture))

    assert buttons_state(main_window) == (True, True, False, False)


def test_state_ready(apply_filter, main_window, picture):
    """ДТ-03: состояние «Готово» — доступны сброс и сохранение."""
    main_window.open_path(str(picture))

    assert apply_filter(main_window)

    assert buttons_state(main_window) == (True, True, True, True)


def test_state_busy_blocks_everything(main_window, sample_image,
                                      monkeypatch):
    """ДТ-04: состояние «Обработка» — все кнопки заблокированы.

    Иначе пользователь мог бы запустить вторую свёртку или
    закрыть изображение во время вычисления.
    """
    main_window.load_array(sample_image)
    monkeypatch.setattr(type(main_window), "is_busy",
                        property(lambda self: True))

    main_window._update_actions()

    assert buttons_state(main_window) == (False, False, False, False)


def test_reset_returns_to_loaded(apply_filter, main_window, picture):
    """ДТ-05: сброс возвращает окно в состояние «Загружено»."""
    main_window.open_path(str(picture))
    apply_filter(main_window)

    main_window.on_reset_clicked()

    assert buttons_state(main_window) == (True, True, False, False)


# =====================================================================
# Соглашения настольных приложений
# =====================================================================


def test_title_shows_file_name(main_window, picture):
    """ДТ-06: заголовок окна содержит имя открытого файла.

    Регрессионная проверка дефекта ДЕФ-11.
    """
    main_window.open_path(str(picture))

    assert BASE_TITLE in main_window.windowTitle()
    assert "снимок.png" in main_window.windowTitle()


def test_title_reset_without_document(main_window):
    """ДТ-07: без открытого файла заголовок возвращается к исходному."""
    main_window.set_document_title("что-то.png")

    main_window.set_document_title(None)

    assert main_window.windowTitle() == BASE_TITLE


@pytest.mark.parametrize(
    "case_id, attribute, expected",
    [
        ("ДТ-08", "open_button", "Ctrl+O"),
        ("ДТ-09", "apply_button", "Ctrl+R"),
        ("ДТ-10", "save_button", "Ctrl+S"),
        ("ДТ-11", "reset_button", "Ctrl+Z"),
    ],
)
def test_keyboard_shortcuts_assigned(main_window, case_id, attribute,
                                     expected):
    """Кнопкам назначены привычные сокращения.

    Регрессионная проверка дефекта ДЕФ-12.
    """
    button = getattr(main_window, attribute)

    assert button.shortcut() == QKeySequence(expected)
    assert expected in button.toolTip()


def test_shortcuts_do_not_conflict(main_window):
    """ДТ-12: назначенные сокращения не совпадают между собой.

    Совпадение привело бы к тому, что одно из действий никогда
    не вызывалось бы с клавиатуры.

    Фактическое срабатывание сокращения автоматически не
    проверяется: в безэкранном режиме оконная система
    отсутствует и не доставляет события клавиатуры. Это свойство
    вынесено в ручной сценарий РД-03.
    """
    shortcuts = [main_window.open_button.shortcut().toString(),
                 main_window.apply_button.shortcut().toString(),
                 main_window.save_button.shortcut().toString(),
                 main_window.reset_button.shortcut().toString()]

    assert all(shortcuts)
    assert len(set(shortcuts)) == len(shortcuts)


def test_minimum_window_size(main_window):
    """ДТ-13: задан минимальный размер окна.

    Регрессионная проверка дефекта ДЕФ-13: без него окно сжималось
    до размеров, при которых элементы управления не помещались.
    """
    main_window.resize(100, 80)

    assert main_window.minimumWidth() >= 600
    assert main_window.minimumHeight() >= 400
    assert main_window.width() >= main_window.minimumWidth()


def test_progress_indicator_present(main_window):
    """ДТ-14: индикатор занятости размещён в строке состояния."""
    assert main_window.progress.minimum() == 0
    assert main_window.progress.maximum() == 0
    assert main_window.progress.isHidden()


# =====================================================================
# Завершение работы
# =====================================================================


def test_close_waits_for_worker(qtbot, main_window, rng):
    """ДТ-15: закрытие во время обработки дожидается потока.

    Регрессионная проверка дефекта ДЕФ-10: уничтожение объекта
    QThread во время работы потока завершало процесс аварийно.
    """
    image = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
    main_window.load_array(image)
    main_window.filter_box.setCurrentText("Размытие по Гауссу 5x5")

    main_window.on_apply_clicked()
    assert main_window.is_busy

    assert main_window.close()

    assert not main_window._worker.isRunning()


def test_close_without_worker(main_window):
    """ДТ-16: закрытие без запущенной обработки проходит сразу."""
    assert main_window.close()


# =====================================================================
# Работа с файловой системой
# =====================================================================


@pytest.mark.parametrize(
    "case_id, name",
    [
        ("ДТ-17", "снимок.png"),
        ("ДТ-18", "имя с пробелами.png"),
        ("ДТ-19", "кириллица_в_имени.png"),
        ("ДТ-20", "файл.с.точками.png"),
        ("ДТ-21", "ВЕРХНИЙ_РЕГИСТР.PNG"),
    ],
)
def test_file_names_handled(main_window, tmp_path, sample_image,
                            case_id, name):
    """Имена файлов, типичные для настольной системы."""
    path = tmp_path / name
    imageio.save_image(path, sample_image)

    assert main_window.open_path(str(path))
    assert np.array_equal(main_window.source_image, sample_image)


def test_directory_with_spaces_and_cyrillic(main_window, tmp_path,
                                            sample_image):
    """ДТ-22: каталог с пробелами и кириллицей в пути."""
    directory = tmp_path / "Мои документы" / "снимки экрана"
    directory.mkdir(parents=True)
    path = directory / "образец.png"
    imageio.save_image(path, sample_image)

    assert main_window.open_path(str(path))


def test_save_to_nested_directory(apply_filter, main_window, picture,
                                  tmp_path):
    """ДТ-23: сохранение во вложенный существующий каталог."""
    main_window.open_path(str(picture))
    apply_filter(main_window)
    directory = tmp_path / "результаты"
    directory.mkdir()

    assert main_window.save_path(str(directory / "итог.png"))


def test_save_to_missing_directory_refused(apply_filter, main_window,
                                           picture, tmp_path):
    """ДТ-24: сохранение в несуществующий каталог отклоняется."""
    main_window.open_path(str(picture))
    apply_filter(main_window)

    assert not main_window.save_path(
        str(tmp_path / "нет_каталога" / "итог.png"))
    assert main_window.last_error is not None
