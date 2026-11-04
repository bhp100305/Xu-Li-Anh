"""Headless regression checks: python -m unittest discover -s tests -v."""

import unittest

import cv2
import numpy as np

from src.canny import canny, double_threshold
from src.convolution import convolution2d
from src.prewitt import prewitt
from src.sobel import sobel


class ConvolutionTests(unittest.TestCase):
    def test_asymmetric_kernel_matches_opencv(self):
        image = np.arange(35, dtype=np.float32).reshape(5, 7)
        kernel = np.array([[1, 2, 3], [0, -2, 4], [5, 0, -1]], np.float32)
        # filter2D computes correlation, so flip its kernel for convolution.
        expected = cv2.filter2D(
            image, -1, np.flip(kernel), borderType=cv2.BORDER_REFLECT_101
        )
        actual = convolution2d(image, kernel)
        np.testing.assert_allclose(actual, expected, atol=1e-5)
        self.assertEqual(actual.shape, image.shape)
        self.assertEqual(actual.dtype, np.float32)
        self.assertFalse(np.array_equal(actual, cv2.filter2D(image, -1, kernel)))

    def test_matches_scipy_with_explicit_padding(self):
        try:
            from scipy.signal import convolve2d
        except ImportError:
            self.skipTest('SciPy is an optional verification dependency')
        rng = np.random.default_rng(5)
        image = rng.normal(size=(9, 11)).astype(np.float32)
        for shape in ((3, 3), (3, 5), (1, 3)):
            kernel = rng.normal(size=shape).astype(np.float32)
            for padding in ('reflect', 'constant', 'edge'):
                with self.subTest(shape=shape, padding=padding):
                    padded = np.pad(image, tuple((s // 2, s // 2) for s in shape),
                                    mode=padding)
                    expected = convolve2d(padded, kernel, mode='valid')
                    np.testing.assert_allclose(
                        convolution2d(image, kernel, padding), expected, atol=2e-6
                    )

    def test_gradient_magnitudes_stay_compatible_with_correlation(self):
        image = np.random.default_rng(7).integers(0, 256, (12, 14)).astype(np.float32)
        for detector, weight in ((sobel, 2), (prewitt, 1)):
            with self.subTest(detector=detector.__name__):
                kernel = np.array([[-1, 0, 1], [-weight, 0, weight], [-1, 0, 1]],
                                  dtype=np.float32)
                gx = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
                gy = cv2.filter2D(image, -1, kernel.T, borderType=cv2.BORDER_REFLECT_101)
                result = detector(image)
                np.testing.assert_allclose(result['gx'], -gx, atol=1e-5)
                np.testing.assert_allclose(result['gy'], -gy, atol=1e-5)
                np.testing.assert_allclose(result['magnitude'], np.hypot(gx, gy), atol=1e-3)


class CannyTests(unittest.TestCase):
    def test_constant_images_have_no_edges(self):
        for value in (0, 127, 255):
            for shape in ((50, 50), (15, 15, 3)):
                with self.subTest(value=value, shape=shape):
                    image = np.full(shape, value, dtype=np.uint8)
                    result = canny(image)
                    self.assertEqual(np.count_nonzero(result['edge']), 0)
                    self.assertEqual(np.count_nonzero(result['threshold']), 0)
                    reference = cv2.Canny(image, result['low_threshold'], result['high_threshold'])
                    np.testing.assert_array_equal(result['edge'], reference)

    def test_zero_threshold_excludes_zero_gradient(self):
        image = np.array([[0, 1, 20]], dtype=np.float32)
        result, _, strong = double_threshold(image, 0, 0)
        np.testing.assert_array_equal(result, [[0, strong, strong]])

    def test_zero_gradient_is_not_weak(self):
        result, weak, strong = double_threshold(np.array([[0, 5, 10]]), 0, 10)
        np.testing.assert_array_equal(result, [[0, weak, strong]])

    def test_threshold_boundaries(self):
        result, weak, strong = double_threshold(np.array([[0, 4, 5, 9, 10, 11]]), 5, 10)
        np.testing.assert_array_equal(result, [[0, 0, weak, weak, strong, strong]])

    def test_zero_median_image_still_detects_edges(self):
        image = np.zeros((50, 50), dtype=np.uint8)
        image[15:35, 15:35] = 255
        result = canny(image)
        self.assertEqual(result['high_threshold'], 0)
        self.assertGreater(np.count_nonzero(result['edge']), 0)
        self.assertLess(np.count_nonzero(result['edge']), image.size // 2)
        self.assertTrue(np.all(result['edge'][result['nms'] == 0] == 0))


if __name__ == '__main__':
    unittest.main()
