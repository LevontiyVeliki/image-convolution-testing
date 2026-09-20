"""Дымовые тесты: проверка, что основные сценарии вообще работают.

Подробные наборы тестов разрабатываются в лабораторных работах,
здесь только быстрая проверка собранного приложения.
"""

from __future__ import annotations

import numpy as np
import pytest

from imgconv import borders, convolution, kernel


def test_package_imports():
    assert kernel.preset_names()
    assert borders.mode_titles()


def test_identity_kernel_keeps_image(sample_image):
    result = convolution.convolve_fast(sample_image, kernel.IDENTITY)
    assert np.array_equal(result, sample_image)


def test_blur_keeps_flat_image(flat_image):
    result = convolution.convolve_fast(flat_image, kernel.BOX_BLUR)
    assert np.array_equal(result, flat_image)


@pytest.mark.parametrize("name", kernel.preset_names())
@pytest.mark.parametrize("mode", list(borders.BorderMode))
def test_implementations_agree(sample_image, name, mode):
    """Быстрая реализация обязана совпадать с эталонной."""
    selected = kernel.get_preset(name)
    naive = convolution.convolve_naive(sample_image, selected, mode)
    fast = convolution.convolve_fast(sample_image, selected, mode)
    assert np.array_equal(naive, fast)


def test_result_shape_and_type(sample_image):
    result = convolution.convolve_fast(sample_image, kernel.SHARPEN)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8


def test_even_kernel_rejected():
    with pytest.raises(kernel.KernelError):
        kernel.Kernel([[1, 2], [3, 4]])


def test_empty_image_rejected():
    with pytest.raises(convolution.ImageError):
        convolution.convolve_fast(np.zeros((0, 0)), kernel.IDENTITY)


def test_window_applies_filter(apply_directly, main_window,
                               sample_image):
    main_window.load_array(sample_image)
    assert main_window.apply_button.isEnabled()
    assert apply_directly(main_window)
    assert main_window.result_image.shape == sample_image.shape
    assert main_window.save_button.isEnabled()


def test_window_reports_missing_file(main_window):
    assert not main_window.open_path("не_существует.png")
    assert "не найден" in main_window.last_error.lower()
