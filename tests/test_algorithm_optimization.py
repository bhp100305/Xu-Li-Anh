"""Compare optimized kernels with the original scalar implementation.

The scalar references deliberately retain the old traversal and sum order.
"""
import importlib
import unittest
from unittest.mock import patch

import numpy as np

from src.convolution import convolution2d
from src.canny import canny, hysteresis, non_maximum_suppression


def scalar_convolution(image, kernel, padding='reflect'):
    image = image.astype(np.float32)
    kernel = np.flip(kernel.astype(np.float32), axis=(0, 1))
    kh, kw = kernel.shape
    padded = np.pad(image, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode=padding)
    result = np.zeros(image.shape, dtype=np.float32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result[y, x] = np.sum(padded[y:y + kh, x:x + kw] * kernel)
    return result


def scalar_nms(magnitude, direction):
    result = np.zeros(magnitude.shape, dtype=np.float32)
    angle = direction.copy()
    angle[angle < 0] += 180
    for y in range(1, magnitude.shape[0] - 1):
        for x in range(1, magnitude.shape[1] - 1):
            a, b = 0, 0
            if 0 <= angle[y, x] < 22.5 or 157.5 <= angle[y, x] <= 180:
                a, b = magnitude[y, x - 1], magnitude[y, x + 1]
            elif 22.5 <= angle[y, x] < 67.5:
                a, b = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            elif 67.5 <= angle[y, x] < 112.5:
                a, b = magnitude[y - 1, x], magnitude[y + 1, x]
            elif 112.5 <= angle[y, x] < 157.5:
                a, b = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            if magnitude[y, x] >= a and magnitude[y, x] >= b:
                result[y, x] = magnitude[y, x]
    return result


def scalar_hysteresis(image, weak_value=75, strong_value=255):
    result = image.copy()
    changed = True
    while changed:
        changed = False
        for y in range(1, image.shape[0] - 1):
            for x in range(1, image.shape[1] - 1):
                if result[y, x] == weak_value and np.any(result[y - 1:y + 2, x - 1:x + 2] == strong_value):
                    result[y, x] = strong_value
                    changed = True
    result[result != strong_value] = 0
    return result


class OptimizedAlgorithmTests(unittest.TestCase):
    def test_convolution_products_and_sums_are_bit_identical(self):
        rng = np.random.default_rng(21)
        for shape in ((1, 1), (3, 17), (31, 29)):
            image = rng.uniform(-20, 300, shape).astype(np.float32)
            for kernel_shape in ((1, 1), (3, 3), (5, 5), (3, 5)):
                kernel = rng.normal(size=kernel_shape).astype(np.float32)
                for padding in ('reflect', 'constant', 'edge'):
                    with self.subTest(shape=shape, kernel=kernel_shape, padding=padding):
                        np.testing.assert_array_equal(convolution2d(image, kernel, padding),
                                                      scalar_convolution(image, kernel, padding))

    def test_convolution_multiple_blocks_and_strided_input(self):
        image = np.random.default_rng(22).normal(size=(90, 850)).astype(np.float32)[:, ::-1]
        kernel = np.arange(81, dtype=np.float32).reshape(9, 9) / 81
        np.testing.assert_array_equal(convolution2d(image, kernel), scalar_convolution(image, kernel))

    def test_nms_preserves_boundary_angles_ties_and_output_borders(self):
        rng = np.random.default_rng(23)
        boundaries = [-180, -157.5, -112.5, -67.5, -22.5, 0, 22.5, 67.5, 112.5, 157.5, 180]
        for shape in ((1, 1), (2, 9), (17, 21)):
            for constant in (False, True):
                magnitude = np.full(shape, 50, np.float32) if constant else rng.integers(0, 300, shape).astype(np.float32)
                angles = rng.choice(boundaries, size=shape).astype(np.float32)
                np.testing.assert_array_equal(non_maximum_suppression(magnitude, angles), scalar_nms(magnitude, angles))

    def test_hysteresis_preserves_weak_chains_diagonals_and_border_rules(self):
        rng = np.random.default_rng(24)
        for shape in ((0, 3), (1, 7), (2, 2), (30, 35)):
            for weak, strong in ((75, 255), (17, 200)):
                for _ in range(5):
                    image = rng.choice([0, weak, strong], size=shape, p=[0.2, 0.75, 0.05]).astype(np.uint8)
                    np.testing.assert_array_equal(hysteresis(image, weak, strong), scalar_hysteresis(image, weak, strong))
        image = np.zeros((9, 50), np.uint8)
        image[4, 1:49] = 75
        image[4, 49] = 255  # Far border seed must propagate all the way left.
        image[0, 1:20] = 75  # Border weak pixels must not be promoted.
        np.testing.assert_array_equal(hysteresis(image), scalar_hysteresis(image))
        self.assertEqual(np.count_nonzero(hysteresis(image)[4]), 49)

    def test_entire_canny_matches_original_intermediate_and_final_arrays(self):
        module = importlib.import_module('src.canny')
        rng = np.random.default_rng(25)
        for image in (rng.integers(0, 256, (24, 31, 3), dtype=np.uint8),
                      np.pad(np.full((15, 20), 255, np.uint8), 10),
                      np.zeros((20, 30), np.uint8)):
            with patch.multiple(module, convolution2d=scalar_convolution,
                                non_maximum_suppression=scalar_nms, hysteresis=scalar_hysteresis):
                reference = canny(image)
            result = canny(image)
            for key in reference:
                with self.subTest(key=key, shape=image.shape):
                    np.testing.assert_array_equal(result[key], reference[key])


if __name__ == '__main__':
    unittest.main()
