"""Compare actual algorithm edges through the shared document scan stages."""
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import app as web
from src.comparison import compare_algorithms
from src.scanner import scan_document


def document_photo():
    image = np.full((200, 160, 3), 30, np.uint8)
    cv2.rectangle(image, (25, 20), (140, 185), (240, 245, 250), -1)
    for y in range(48, 170, 25):
        cv2.putText(image, 'FORM 0123', (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (15, 15, 15), 1)
    return image


class ComparisonScanTests(unittest.TestCase):
    def test_each_scan_uses_its_own_edges_and_original_pixels_without_rerunning_canny(self):
        image = document_photo()
        calls = []

        def record(**options):
            calls.append(options)
            return scan_document(**options)

        with patch('src.comparison.scan_document', side_effect=record), \
                patch('src.scanner.canny', side_effect=AssertionError('Must use the supplied algorithm edges')):
            result = compare_algorithms(image, max_width=800, include_scans=True)
        self.assertEqual(len(calls), 6)
        for options, data in zip(calls, result['results'].values()):
            with self.subTest(algorithm=data['name']):
                _, expected = cv2.threshold(data['edge'], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                np.testing.assert_array_equal(options['edge_image'], expected)
                np.testing.assert_array_equal(options['image'], image)
                self.assertIsNotNone(data['scan'])
                self.assertIsNone(data['scan_error'])
                self.assertGreaterEqual(data['scan_time'], 0)
                self.assertGreater(data['scan']['final_height'], data['scan']['final_width'])
                np.testing.assert_allclose(data['scan']['corners'],
                                           [[25, 20], [140, 20], [140, 185], [25, 185]], atol=8)

    def test_canny_comparison_and_document_scanner_produce_the_same_scan(self):
        image = document_photo()
        expected = scan_document(image, detection_width=800, output_width=1000, verbose=False)
        comparison = compare_algorithms(image, max_width=800, include_scans=True)
        actual = comparison['results']['canny']['scan']
        np.testing.assert_array_equal(actual['corners'], expected['corners'])
        np.testing.assert_array_equal(actual['sharpened'], expected['sharpened'])

    def test_one_failed_algorithm_does_not_replace_or_discard_other_results(self):
        calls = 0

        def scan(**options):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('Không tìm được khung giấy cho Prewitt')
            return scan_document(**options)

        with patch('src.comparison.scan_document', side_effect=scan):
            result = compare_algorithms(document_photo(), include_scans=True)
        self.assertEqual(len(result['results']), 6)
        self.assertIsNone(result['results']['prewitt']['scan'])
        self.assertIn('Prewitt', result['results']['prewitt']['scan_error'])
        self.assertEqual(sum(data['scan'] is not None for data in result['results'].values()), 5)

    def test_precomputed_edges_must_match_detection_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'khớp kích thước'):
            scan_document(document_photo(), detection_width=80,
                          edge_image=np.zeros((200, 160), np.uint8), verbose=False)

    def test_comparison_page_keeps_edge_cards_and_scanned_overview(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(web, 'UPLOAD_FOLDER', Path(folder)), \
                patch.object(web, 'RESULT_FOLDER', Path(folder)):
            _, encoded = cv2.imencode('.png', document_photo())
            with web.app.test_client() as client, client.post('/?tab=compare', data={
                'action': 'compare', 'image': (io.BytesIO(encoded.tobytes()), 'form.png'),
            }) as response:
                self.assertEqual(response.status_code, 200)
                html = response.get_data(as_text=True)
            self.assertEqual(html.count('data-scan-status="ready"'), 6)
            self.assertEqual(html.count('class="result-download"'), 6)
            self.assertNotIn('data-comparison-view=', html)
            self.assertNotIn('class="result-download scan-download"', html)
            self.assertIn('6 / 6 phương pháp tạo được bản quét', html)
            self.assertIn('canny_scan.png', html)
            cards = re.findall(r'<article[^>]*data-comparison-card.*?</article>', html, re.S)
            self.assertEqual(len(cards), 6)
            for card in cards:
                key = re.search(r'data-comparison-key="([^"]+)"', card).group(1)
                self.assertIn(f'_{key}.jpg', card)
                self.assertNotIn('_scan.png', card)
            scans = list(Path(folder).glob('*_scan.png'))
            self.assertEqual(len(scans), 6)
            self.assertTrue(all(cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None for path in scans))

    def test_no_document_keeps_six_edge_results_and_reports_no_scan(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(web, 'UPLOAD_FOLDER', Path(folder)), \
                patch.object(web, 'RESULT_FOLDER', Path(folder)):
            _, encoded = cv2.imencode('.png', np.zeros((100, 80, 3), np.uint8))
            with web.app.test_client() as client, client.post('/?tab=compare', data={
                'action': 'compare', 'image': (io.BytesIO(encoded.tobytes()), 'black.png'),
            }) as response:
                self.assertEqual(response.status_code, 200)
                html = response.get_data(as_text=True)
            self.assertEqual(html.count('data-scan-status="failed"'), 6)
            self.assertEqual(html.count('class="result-download"'), 6)
            self.assertNotIn('class="result-download scan-download"', html)
            self.assertIn('0 / 6 phương pháp tạo được bản quét', html)
            self.assertEqual(len(list(Path(folder).glob('*_scan.png'))), 0)


if __name__ == '__main__':
    unittest.main()
