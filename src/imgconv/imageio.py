"""Загрузка и сохранение изображений."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

#: Расширения, которые принимает приложение
SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: Форматы, не поддерживающие канал прозрачности
OPAQUE_SUFFIXES = (".jpg", ".jpeg")


class ImageIOError(IOError):
    """Ошибка чтения или записи файла изображения."""


def _describe_suffix(suffix: str) -> str:
    """Поясняющий текст для неподдерживаемого расширения.

    Отдельная формулировка для файла без расширения: сообщение
    вида «Формат не поддерживается: » обрывалось на двоеточии
    и не подсказывало пользователю, что делать.
    """
    listing = ", ".join(SUPPORTED_SUFFIXES)
    if not suffix:
        return (f"Не указано расширение файла. "
                f"Поддерживаются: {listing}")
    return (f"Формат не поддерживается: {suffix}. "
            f"Поддерживаются: {listing}")


def _has_alpha(handle) -> bool:
    """Содержит ли открытое изображение канал прозрачности."""
    if handle.mode in ("RGBA", "LA", "PA"):
        return True
    # Палитровые изображения хранят прозрачность отдельным полем
    return "transparency" in handle.info


def load_image(path) -> np.ndarray:
    """Читает изображение в массив (высота, ширина, канал)."""
    file_path = Path(path)

    if not file_path.exists():
        raise ImageIOError(f"Файл не найден: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ImageIOError(_describe_suffix(file_path.suffix))

    try:
        with Image.open(file_path) as handle:
            # Изображения с прозрачностью сохраняют четвёртый канал:
            # ядро обработки и предпросмотр его поддерживают.
            # Остальные приводятся к RGB ради единого представления.
            target = "RGBA" if _has_alpha(handle) else "RGB"
            converted = handle.convert(target)
            return np.asarray(converted, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError) as error:
        raise ImageIOError(
            f"Не удалось прочитать изображение: {error}"
        ) from error


def save_image(path, array) -> None:
    """Сохраняет массив как изображение."""
    file_path = Path(path)

    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ImageIOError(_describe_suffix(file_path.suffix))

    data = np.asarray(array)
    if data.size == 0:
        raise ImageIOError("Нечего сохранять: изображение пустое")

    # Формат JPEG не хранит прозрачность, поэтому четвёртый канал
    # отбрасывается. Без этого библиотека прерывала сохранение
    # сообщением на английском языке, непонятным пользователю.
    if (data.ndim == 3 and data.shape[2] == 4
            and file_path.suffix.lower() in OPAQUE_SUFFIXES):
        data = data[:, :, :3]

    try:
        Image.fromarray(data.astype(np.uint8)).save(file_path)
    except (OSError, ValueError) as error:
        raise ImageIOError(
            f"Не удалось сохранить изображение: {error}"
        ) from error
