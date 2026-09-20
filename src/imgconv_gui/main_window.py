"""Главное окно приложения «Свёртка изображения».

Окно не содержит логики обработки: оно только собирает параметры,
вызывает ядро из пакета ``imgconv`` и показывает результат. Такое
разделение позволяет тестировать обработку без запуска интерфейса,
а интерфейс — без реальных файлов.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QProgressBar,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pathlib import Path

from imgconv import borders, convolution, imageio, kernel

from .worker import ConvolutionWorker

#: Сторона области предпросмотра в пикселях
PREVIEW_SIZE = 360

#: Заголовок окна без открытого файла
BASE_TITLE = "Свёртка изображения"


def array_to_pixmap(array: np.ndarray) -> QPixmap:
    """Преобразует массив изображения в объект для показа на экране."""
    data = np.ascontiguousarray(array, dtype=np.uint8)

    if data.ndim == 2:
        height, width = data.shape
        image = QImage(data.data, width, height, width,
                       QImage.Format.Format_Grayscale8)
    else:
        height, width, channels = data.shape
        if channels == 4:
            image = QImage(data.data, width, height, 4 * width,
                           QImage.Format.Format_RGBA8888)
        else:
            image = QImage(data.data, width, height, 3 * width,
                           QImage.Format.Format_RGB888)

    # copy() отвязывает изображение от временного массива
    return QPixmap.fromImage(image.copy())


class MainWindow(QMainWindow):
    """Главное окно: выбор файла, фильтра и режима границ."""

    #: Обработка завершена; параметр — признак успеха.
    #: Сигнал нужен и интерфейсу, и автоматическим тестам,
    #: которым требуется дождаться окончания фоновой работы.
    processing_finished = Signal(bool)

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(BASE_TITLE)
        self.setObjectName("mainWindow")

        # ДЕФ-13: без нижней границы окно сжималось до размеров,
        # при которых элементы управления переставали помещаться
        self.setMinimumSize(640, 420)

        self._source: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._last_duration_ms: float = 0.0
        self._last_error: str | None = None
        self._worker: ConvolutionWorker | None = None
        self._busy = False
        self._started_at: float = 0.0
        self._kernel_name: str = ""
        self._border_title: str = ""

        self._build_ui()
        self._update_actions()

    # -- построение интерфейса ------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("centralWidget")

        self.open_button = QPushButton("Открыть", central)
        self.open_button.setObjectName("openButton")
        self.open_button.setShortcut(QKeySequence.StandardKey.Open)
        self.open_button.setToolTip("Открыть изображение (Ctrl+O)")
        self.open_button.clicked.connect(self.on_open_clicked)

        self.filter_box = QComboBox(central)
        self.filter_box.setObjectName("filterBox")
        self.filter_box.addItems(kernel.preset_names())

        self.border_box = QComboBox(central)
        self.border_box.setObjectName("borderBox")
        self.border_box.addItems(borders.mode_titles())

        self.apply_button = QPushButton("Применить", central)
        self.apply_button.setObjectName("applyButton")
        self.apply_button.setShortcut(QKeySequence("Ctrl+R"))
        self.apply_button.setToolTip("Применить фильтр (Ctrl+R)")
        self.apply_button.clicked.connect(self.on_apply_clicked)

        self.reset_button = QPushButton("Сбросить", central)
        self.reset_button.setObjectName("resetButton")
        self.reset_button.setShortcut(QKeySequence("Ctrl+Z"))
        self.reset_button.setToolTip("Сбросить результат (Ctrl+Z)")
        self.reset_button.clicked.connect(self.on_reset_clicked)

        self.save_button = QPushButton("Сохранить", central)
        self.save_button.setObjectName("saveButton")
        self.save_button.setShortcut(QKeySequence.StandardKey.Save)
        self.save_button.setToolTip("Сохранить результат (Ctrl+S)")
        self.save_button.clicked.connect(self.on_save_clicked)

        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(QLabel("Фильтр:", central))
        controls.addWidget(self.filter_box, 1)
        controls.addWidget(QLabel("Границы:", central))
        controls.addWidget(self.border_box, 1)
        controls.addWidget(self.apply_button)
        controls.addWidget(self.reset_button)
        controls.addWidget(self.save_button)

        self.source_view = self._make_preview("sourceView", central)
        self.result_view = self._make_preview("resultView", central)

        previews = QHBoxLayout()
        previews.addWidget(self._wrap("Исходное изображение",
                                      self.source_view, central))
        previews.addWidget(self._wrap("Результат свёртки",
                                      self.result_view, central))

        layout = QVBoxLayout(central)
        layout.addLayout(controls)
        layout.addLayout(previews, 1)

        self.setCentralWidget(central)

        self.status = QStatusBar(self)
        self.status.setObjectName("statusBar")
        self.setStatusBar(self.status)

        # Индикатор без определённого значения: длительность свёртки
        # заранее неизвестна, важен сам факт продолжающейся работы
        self.progress = QProgressBar(self)
        self.progress.setObjectName("progressBar")
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(160)
        self.progress.hide()
        self.status.addPermanentWidget(self.progress)

        self.status.showMessage("Откройте изображение")

    @staticmethod
    def _make_preview(name: str, parent: QWidget) -> QLabel:
        view = QLabel(parent)
        view.setObjectName(name)
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view.setMinimumSize(PREVIEW_SIZE, PREVIEW_SIZE)
        view.setStyleSheet("border: 1px solid #808080;")
        return view

    @staticmethod
    def _wrap(title: str, view: QLabel, parent: QWidget) -> QGroupBox:
        box = QGroupBox(title, parent)
        inner = QVBoxLayout(box)
        inner.addWidget(view)
        return box

    # -- состояние -------------------------------------------------------

    def _update_actions(self) -> None:
        """Кнопки доступны только тогда, когда действие осмысленно."""
        if self.is_busy:
            # Во время обработки повторный запуск запрещён,
            # иначе два потока писали бы в одно поле результата
            for button in (self.open_button, self.apply_button,
                           self.reset_button, self.save_button):
                button.setEnabled(False)
            return

        has_source = self._source is not None
        has_result = self._result is not None

        self.open_button.setEnabled(True)
        self.apply_button.setEnabled(has_source)
        self.reset_button.setEnabled(has_result)
        self.save_button.setEnabled(has_result)

    @property
    def is_busy(self) -> bool:
        """Выполняется ли свёртка в фоновом потоке.

        Признак выставляется до запуска потока: между вызовом
        start и фактическим стартом проходит некоторое время,
        и опрос состояния потока в этот момент дал бы неверный
        ответ, оставив кнопки доступными.
        """
        return self._busy

    def _show(self, view: QLabel, array: np.ndarray | None) -> None:
        if array is None:
            view.clear()
            return

        pixmap = array_to_pixmap(array)
        view.setPixmap(pixmap.scaled(
            PREVIEW_SIZE, PREVIEW_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _report(self, message: str) -> None:
        self.status.showMessage(message)

    def _succeed(self, message: str) -> None:
        """Сообщает об успехе и снимает признак прошлой ошибки."""
        self._last_error = None
        self._report(message)

    def show_warning(self, title: str, message: str) -> None:
        """Показывает модальное предупреждение.

        Единственное место, где интерфейс открывает блокирующий
        диалог. Автоматические тесты подменяют этот метод, иначе
        окно ждало бы нажатия кнопки и тест зависал бы.
        """
        QMessageBox.warning(self, title, message)

    def _fail(self, title: str, message: str) -> bool:
        """Сообщает об ошибке и возвращает признак неуспеха."""
        self._last_error = message
        self._report(message)
        self.show_warning(title, message)
        return False

    # -- свойства для тестов ---------------------------------------------

    @property
    def source_image(self) -> np.ndarray | None:
        return self._source

    @property
    def result_image(self) -> np.ndarray | None:
        return self._result

    @property
    def last_duration_ms(self) -> float:
        return self._last_duration_ms

    @property
    def last_error(self) -> str | None:
        """Текст последней ошибки, показанной пользователю."""
        return self._last_error

    def set_document_title(self, name: str | None) -> None:
        """Показывает имя открытого файла в заголовке окна.

        Принято в настольных приложениях: заголовок сообщает,
        с каким документом идёт работа.
        """
        self.setWindowTitle(f"{BASE_TITLE} — {name}" if name
                            else BASE_TITLE)

    def load_array(self, array: np.ndarray) -> None:
        """Загружает изображение из массива, минуя файловый диалог."""
        self._source = np.asarray(array, dtype=np.uint8)
        self._result = None
        self._last_error = None
        self._show(self.source_view, self._source)
        self._show(self.result_view, None)
        self._update_actions()
        height, width = self._source.shape[:2]
        self._report(f"Загружено изображение {width} x {height}")

    # -- обработчики -----------------------------------------------------

    def on_open_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть изображение", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        self.open_path(path)

    def open_path(self, path: str) -> bool:
        """Загружает файл. Возвращает признак успеха."""
        try:
            array = imageio.load_image(path)
        except imageio.ImageIOError as error:
            return self._fail("Ошибка открытия", str(error))

        self.load_array(array)
        self.set_document_title(Path(path).name)
        return True

    def on_apply_clicked(self) -> bool:
        """Запускает свёртку в фоновом потоке.

        Возвращает признак того, что обработка начата. Результат
        появится позже; его готовность отмечается сигналом
        processing_finished.
        """
        if self.is_busy:
            self._report("Обработка уже выполняется")
            return False

        if self._source is None:
            self._report("Сначала откройте изображение")
            self.processing_finished.emit(False)
            return False

        try:
            selected = kernel.get_preset(self.filter_box.currentText())
            mode = borders.mode_by_title(self.border_box.currentText())
        except (kernel.KernelError, borders.BorderError) as error:
            self._fail("Ошибка параметров", str(error))
            self.processing_finished.emit(False)
            return False

        self._kernel_name = selected.name
        self._border_title = mode.title
        self._started_at = time.perf_counter()

        self._worker = ConvolutionWorker(self._source, selected, mode,
                                         self)
        self._worker.succeeded.connect(self._on_worker_succeeded)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._on_worker_finished)

        self._busy = True
        self._update_actions()
        self.progress.show()
        self._report(f"Обработка фильтром «{selected.name}»...")
        self._worker.start()
        return True

    def _on_worker_succeeded(self, result) -> None:
        """Результат готов: показать и разблокировать интерфейс."""
        self._last_duration_ms = (
            time.perf_counter() - self._started_at) * 1000.0
        self._result = result
        self._show(self.result_view, self._result)
        self._succeed(
            f"Фильтр «{self._kernel_name}», "
            f"границы: {self._border_title}, "
            f"время: {self._last_duration_ms:.1f} мс"
        )

    def _on_worker_failed(self, message: str) -> None:
        """Свёртка прервана ошибкой."""
        self._fail("Ошибка обработки", message)

    def _on_worker_finished(self) -> None:
        """Поток завершился независимо от исхода."""
        succeeded = self._last_error is None
        self._worker = None
        self._busy = False
        self.progress.hide()
        self._update_actions()
        self.processing_finished.emit(succeeded)

    def on_reset_clicked(self) -> None:
        """Убирает результат, оставляя исходное изображение."""
        self._result = None
        self._show(self.result_view, None)
        self._update_actions()
        self._report("Результат сброшен")

    def closeEvent(self, event) -> None:
        """Закрытие окна во время обработки.

        Объект QThread нельзя уничтожать, пока поток выполняется:
        Qt завершает процесс аварийно. Поэтому перед закрытием
        окно дожидается окончания работы.
        """
        if self._worker is not None and self._worker.isRunning():
            self._report("Завершение обработки перед закрытием...")
            self._worker.wait()

        event.accept()

    def on_save_clicked(self) -> None:
        if self._result is None:
            self._report("Нечего сохранять")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", "result.png",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        self.save_path(path)

    def save_path(self, path: str) -> bool:
        """Сохраняет результат в файл. Возвращает признак успеха."""
        if self._result is None:
            self._report("Нечего сохранять")
            return False

        try:
            imageio.save_image(path, self._result)
        except imageio.ImageIOError as error:
            return self._fail("Ошибка сохранения", str(error))

        self._succeed(f"Сохранено: {path}")
        return True
