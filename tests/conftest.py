"""Общие приспособления для тестов."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Пакеты лежат в каталоге src, добавляем его в путь поиска модулей
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def rng():
    """Генератор с фиксированным зерном: тесты воспроизводимы."""
    return np.random.default_rng(2024)


@pytest.fixture
def sample_image(rng):
    """Небольшое цветное изображение."""
    return rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8)


@pytest.fixture
def flat_image():
    """Однородное изображение: удобно для проверки инвариантов."""
    return np.full((8, 8, 3), 120, dtype=np.uint8)


@pytest.fixture
def main_window(qtbot):
    """Главное окно с подавленными модальными диалогами.

    Без подмены show_warning диалог ждал бы нажатия кнопки
    и тест завис бы.
    """
    from imgconv_gui.main_window import MainWindow

    window = MainWindow()
    window.show_warning = lambda title, message: None
    qtbot.addWidget(window)
    return window


@pytest.fixture
def apply_filter(qtbot):
    """Нажимает «Применить» и дожидается окончания обработки.

    Свёртка выполняется в отдельном потоке, поэтому результат
    появляется не сразу. Помощник ждёт сигнала processing_finished
    и возвращает признак успеха.
    """
    from PySide6.QtCore import Qt

    def _apply(window, timeout=30000):
        with qtbot.waitSignal(window.processing_finished,
                              timeout=timeout) as signal:
            qtbot.mouseClick(window.apply_button,
                             Qt.MouseButton.LeftButton)
        return bool(signal.args[0])

    return _apply


@pytest.fixture
def apply_directly(qtbot):
    """То же, но вызовом обработчика вместо нажатия кнопки.

    Нужен там, где кнопка недоступна: например, при проверке
    отказа из-за отсутствия изображения.
    """

    def _apply(window, timeout=30000):
        with qtbot.waitSignal(window.processing_finished,
                              timeout=timeout) as signal:
            window.on_apply_clicked()
        return bool(signal.args[0])

    return _apply
