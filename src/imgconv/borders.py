"""Режимы обработки границ изображения при свёртке.

У пикселей по краям изображения часть окна ядра выходит за его
пределы. Режим границы определяет, какое значение подставить
вместо отсутствующего пикселя.
"""

from __future__ import annotations

from enum import Enum


class BorderError(ValueError):
    """Ошибка выбора режима обработки границ."""


class BorderMode(Enum):
    """Способ доопределения пикселей за пределами изображения."""

    ZERO = "Заполнение нулями"
    CLAMP = "Повтор крайнего пикселя"
    REFLECT = "Зеркальное отражение"

    @property
    def title(self) -> str:
        return self.value


#: Соответствие режимов параметру mode функции numpy.pad
NUMPY_PAD_MODES = {
    BorderMode.ZERO: "constant",
    BorderMode.CLAMP: "edge",
    BorderMode.REFLECT: "reflect",
}


def mode_titles() -> list[str]:
    """Названия режимов для выпадающего списка интерфейса."""
    return [mode.title for mode in BorderMode]


def mode_by_title(title: str) -> BorderMode:
    """Возвращает режим по его названию."""
    for mode in BorderMode:
        if mode.title == title:
            return mode
    raise BorderError(f"Неизвестный режим обработки границ: {title}")


def map_index(index: int, length: int, mode: BorderMode):
    """Отображает индекс за пределами [0, length) внутрь изображения.

    Возвращает None, если пиксель отсутствует и должен считаться
    нулевым. Используется наивной реализацией свёртки.
    """
    if length <= 0:
        raise BorderError("Размер изображения должен быть положительным")

    if 0 <= index < length:
        return index

    if mode is BorderMode.ZERO:
        return None

    if mode is BorderMode.CLAMP:
        return min(max(index, 0), length - 1)

    if mode is BorderMode.REFLECT:
        # Отражение без повтора крайнего пикселя: abcd -> dcb|abcd|cba
        if length == 1:
            return 0
        period = 2 * length - 2
        folded = abs(index) % period
        return folded if folded < length else period - folded

    raise BorderError(f"Режим обработки границ не поддержан: {mode}")
