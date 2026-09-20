"""Ядро свёртки: представление, проверка корректности, наборы фильтров."""

from __future__ import annotations

import numpy as np


class KernelError(ValueError):
    """Ошибка построения ядра свёртки."""


class Kernel:
    """Квадратное ядро свёртки нечётного размера.

    Ядро хранит коэффициенты, делитель нормализации и сдвиг.
    Результат свёртки вычисляется как сумма произведений
    коэффициентов на значения пикселей, делённая на делитель,
    с прибавлением сдвига.
    """

    def __init__(self, values, divisor=None, offset=0.0, name=""):
        matrix = self._to_matrix(values)
        self._validate_shape(matrix)

        self._values = matrix
        self._divisor = self._resolve_divisor(matrix, divisor)
        self._offset = float(offset)
        self._name = name or "Без названия"

    # -- построение -----------------------------------------------------

    @staticmethod
    def _to_matrix(values) -> np.ndarray:
        if values is None:
            raise KernelError("Ядро не задано")

        try:
            matrix = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise KernelError(
                f"Коэффициенты ядра не приводятся к числам: {error}"
            ) from error

        return matrix

    @staticmethod
    def _validate_shape(matrix: np.ndarray) -> None:
        if matrix.size == 0:
            raise KernelError("Ядро не может быть пустым")

        if matrix.ndim != 2:
            raise KernelError(
                f"Ядро должно быть двумерным, получено измерений: "
                f"{matrix.ndim}"
            )

        rows, columns = matrix.shape

        if rows != columns:
            raise KernelError(
                f"Ядро должно быть квадратным, получено {rows}x{columns}"
            )

        if rows % 2 == 0:
            raise KernelError(
                f"Размер ядра должен быть нечётным, получено {rows}"
            )

        if not np.all(np.isfinite(matrix)):
            raise KernelError("Коэффициенты ядра должны быть конечными")

    @staticmethod
    def _resolve_divisor(matrix: np.ndarray, divisor) -> float:
        if divisor is None:
            total = float(matrix.sum())
            # Ядра выделения границ дают нулевую сумму коэффициентов,
            # делить на неё нельзя, поэтому нормализация отключается
            return total if total != 0.0 else 1.0

        value = float(divisor)
        if value == 0.0:
            raise KernelError("Делитель ядра не может быть нулём")
        return value

    # -- свойства -------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """Копия коэффициентов: ядро неизменяемо снаружи."""
        return self._values.copy()

    @property
    def size(self) -> int:
        return int(self._values.shape[0])

    @property
    def radius(self) -> int:
        return self.size // 2

    @property
    def divisor(self) -> float:
        return self._divisor

    @property
    def offset(self) -> float:
        return self._offset

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_normalized(self) -> bool:
        """Сохраняет ли ядро общую яркость изображения.

        Приведение к bool обязательно: numpy возвращает
        собственный логический тип, который не тождествен
        встроенным True и False, хотя и равен им.
        """
        return bool(np.isclose(
            float(self._values.sum()) / self._divisor, 1.0))

    def __repr__(self) -> str:
        return (
            f"Kernel(name={self._name!r}, size={self.size}, "
            f"divisor={self._divisor}, offset={self._offset})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Kernel):
            return NotImplemented
        return (
            np.array_equal(self._values, other._values)
            and self._divisor == other._divisor
            and self._offset == other._offset
        )


# -- предустановленные фильтры -------------------------------------------

IDENTITY = Kernel(
    [[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]],
    name="Тождественный",
)

BOX_BLUR = Kernel(
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]],
    name="Размытие средним",
)

GAUSSIAN_BLUR = Kernel(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]],
    name="Размытие по Гауссу",
)

SHARPEN = Kernel(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]],
    name="Повышение резкости",
)

EDGE_DETECT = Kernel(
    [[-1, -1, -1],
     [-1, 8, -1],
     [-1, -1, -1]],
    name="Выделение границ",
)

SOBEL_X = Kernel(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    divisor=1.0,
    offset=128.0,
    name="Оператор Собеля по X",
)

SOBEL_Y = Kernel(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]],
    divisor=1.0,
    offset=128.0,
    name="Оператор Собеля по Y",
)

EMBOSS = Kernel(
    [[-2, -1, 0],
     [-1, 1, 1],
     [0, 1, 2]],
    divisor=1.0,
    offset=128.0,
    name="Тиснение",
)

GAUSSIAN_BLUR_5 = Kernel(
    [[1, 4, 6, 4, 1],
     [4, 16, 24, 16, 4],
     [6, 24, 36, 24, 6],
     [4, 16, 24, 16, 4],
     [1, 4, 6, 4, 1]],
    name="Размытие по Гауссу 5x5",
)

PRESETS = {
    kernel.name: kernel
    for kernel in (
        IDENTITY,
        BOX_BLUR,
        GAUSSIAN_BLUR,
        GAUSSIAN_BLUR_5,
        SHARPEN,
        EDGE_DETECT,
        SOBEL_X,
        SOBEL_Y,
        EMBOSS,
    )
}


def preset_names() -> list[str]:
    """Названия доступных фильтров в порядке объявления."""
    return list(PRESETS.keys())


def get_preset(name: str) -> Kernel:
    """Возвращает фильтр по названию."""
    try:
        return PRESETS[name]
    except KeyError:
        raise KernelError(f"Неизвестный фильтр: {name}") from None
