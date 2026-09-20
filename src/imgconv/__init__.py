"""Приложение «Свёртка изображения».

Пакет содержит ядро обработки, независимое от интерфейса:
описание ядра свёртки, режимы обработки границ, две реализации
свёртки и работу с файлами изображений.
"""

from .borders import BorderMode, mode_by_title, mode_titles
from .convolution import convolve, convolve_fast, convolve_naive
from .imageio import load_image, save_image
from .kernel import Kernel, KernelError, get_preset, preset_names

__all__ = [
    "BorderMode",
    "Kernel",
    "KernelError",
    "convolve",
    "convolve_fast",
    "convolve_naive",
    "get_preset",
    "load_image",
    "mode_by_title",
    "mode_titles",
    "preset_names",
    "save_image",
]

__version__ = "1.0.0"
