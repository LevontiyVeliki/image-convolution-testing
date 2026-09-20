"""Фоновое выполнение свёртки.

Свёртка большого изображения занимает секунды. Если выполнять её в
том же потоке, что и цикл обработки событий, окно перестаёт
перерисовываться и реагировать на действия пользователя. Поэтому
вычисление вынесено в отдельный поток, а результат возвращается
сигналом.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal

from imgconv import convolution
from imgconv.borders import BorderMode
from imgconv.kernel import Kernel


class ConvolutionWorker(QThread):
    """Поток, выполняющий свёртку одного изображения."""

    #: Свёртка завершена, передаётся результат
    succeeded = Signal(object)

    #: Свёртка прервана ошибкой, передаётся текст сообщения
    failed = Signal(str)

    def __init__(self, image: np.ndarray, kernel: Kernel,
                 border: BorderMode, parent=None) -> None:
        super().__init__(parent)
        self._image = image
        self._kernel = kernel
        self._border = border

    def run(self) -> None:
        """Тело потока.

        Исключение не должно покидать метод run: оно не может быть
        перехвачено в основном потоке и привело бы к аварийному
        завершению. Поэтому ошибка передаётся сигналом.
        """
        try:
            result = convolution.convolve(self._image, self._kernel,
                                          self._border)
        except Exception as error:   # noqa: BLE001 - передаём наружу
            self.failed.emit(str(error))
            return

        self.succeeded.emit(result)
