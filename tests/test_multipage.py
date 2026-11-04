"""Ensure the camera UI preserves the original document scanner output."""
import io
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import app as web
from src.scanner import scan_document


class MultipageTests(unittest.TestCase):
    def setUp(self):
        self.client = web.app.test_client()
        self.image = np.full((160, 200, 3), 30, np.uint8)
        cv2.rectangle(self.image, (30, 20), (170, 140), (240, 250, 255), -1)
        cv2.putText(self.image, 'DOC', (55, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        self.corners = [[30 / 199, 20 / 159], [170 / 199, 20 / 159],
                        [170 / 199, 140 / 159], [30 / 199, 140 / 159]]

    def post_image(self, route, **fields):
        _, encoded = cv2.imencode('.png', self.image)
        return self.client.post(route, data={
            'image': (io.BytesIO(encoded.tobytes()), 'document.png'), **fields,
        })

    def test_scanner_page_and_assets(self):
        response = self.client.get('/multipage')
        self.assertEqual(response.status_code, 200)
        self.assertIn('captureButton', response.get_data(as_text=True))
        for path in ('/static/js/multipage.js', '/static/css/capture.css'):
            with self.client.get(path) as asset:
                self.assertEqual(asset.status_code, 200)

    def test_detect_real_document_without_scanning(self):
        response = self.post_image('/api/multipage/detect')
        self.assertEqual(response.status_code, 200)
        corners = np.asarray(response.json['corners'])
        self.assertEqual(corners.shape, (4, 2))
        self.assertTrue(np.all((corners >= 0) & (corners <= 1)))
        np.testing.assert_allclose(corners, self.corners, atol=0.06)

    def test_blank_image_has_no_detection(self):
        self.image[:] = 0
        response = self.post_image('/api/multipage/detect')
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json['corners'])

    def test_returned_image_matches_original_scanner_pixel_for_pixel(self):
        for tilted in (False, True):
            with self.subTest(tilted=tilted):
                if tilted:
                    matrix = cv2.getRotationMatrix2D((100, 80), 8, 0.9)
                    self.image = cv2.warpAffine(self.image, matrix, (200, 160), borderValue=(30, 30, 30))
                expected = scan_document(image=self.image, detection_width=800,
                                         output_width=2400, verbose=False)['sharpened']
                response = self.post_image('/api/multipage/scan')
                self.assertEqual(response.status_code, 200)
                actual = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(actual, expected)

    def test_preview_corners_and_filters_cannot_replace_original_pipeline(self):
        expected = np.arange(23 * 31, dtype=np.uint8).reshape(23, 31)
        with patch('multipage.scan_document', return_value={'sharpened': expected}) as scanner:
            response = self.post_image('/api/multipage/scan',
                                       corners=json.dumps(self.corners), filter='bw')
        self.assertEqual(response.status_code, 200)
        scanner.assert_called_once()
        options = scanner.call_args.kwargs
        np.testing.assert_array_equal(options['image'], self.image)
        self.assertEqual(options['detection_width'], 800)
        self.assertEqual(options['output_width'], 2400)
        self.assertFalse(options['verbose'])
        actual = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(actual, expected)

    def test_scan_failure_does_not_return_a_guessed_crop(self):
        with patch('multipage.scan_document', side_effect=RuntimeError('Không phát hiện được vùng tài liệu.')):
            response = self.post_image('/api/multipage/scan')
        self.assertEqual(response.status_code, 422)
        self.assertIn('Không phát hiện', response.json['error'])

    def test_high_resolution_crop_keeps_more_detail_than_the_old_1600_limit(self):
        self.image = np.full((1500, 2100, 3), 35, np.uint8)
        cv2.rectangle(self.image, (100, 100), (2000, 1400), (245, 245, 245), -1)
        for y in range(180, 1320, 35):
            cv2.putText(self.image, 'SMALL DOCUMENT TEXT 0123456789', (180, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
        old = scan_document(image=self.image, detection_width=800,
                            output_width=1600, verbose=False)
        scanned_results = []

        def record_scan(**options):
            result = scan_document(**options)
            scanned_results.append(result)
            return result

        with patch('multipage.scan_document', side_effect=record_scan):
            response = self.post_image('/api/multipage/scan')
        self.assertEqual(response.status_code, 200)
        image = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(old['sharpened'].shape[1], 1600)
        np.testing.assert_array_equal(scanned_results[0]['corners'], old['corners'])
        self.assertGreater(image.shape[1], 1800)
        self.assertLessEqual(image.shape[1], 2400)

    def test_corrupt_upload_is_a_json_error(self):
        for route in ('/api/multipage/detect', '/api/multipage/scan'):
            response = self.client.post(route, data={'image': (io.BytesIO(b'bad'), 'bad.png')})
            self.assertEqual(response.status_code, 400)
            self.assertIn('error', response.json)

    def test_pdf_export_still_accepts_scanned_pages(self):
        _, encoded = cv2.imencode('.png', self.image)
        response = self.client.post('/api/multipage/pdf', data={
            'pages': [(io.BytesIO(encoded.tobytes()), 'page1.png'),
                      (io.BytesIO(encoded.tobytes()), 'page2.png')],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'application/pdf')
        self.assertTrue(response.data.startswith(b'%PDF'))


if __name__ == '__main__':
    unittest.main()
