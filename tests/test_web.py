"""Validate Flask UI flows without opening desktop windows."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import app as web


class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = web.app.test_client()
        self.temp = tempfile.TemporaryDirectory()
        folder = Path(self.temp.name)
        for name in ('UPLOAD_FOLDER', 'RESULT_FOLDER'):
            patcher = patch.object(web, name, folder)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.temp.cleanup)

    def upload(self, action='compare'):
        image = np.zeros((24, 30, 3), dtype=np.uint8)
        image[6:18, 8:22] = 255
        _, encoded = cv2.imencode('.png', image)
        return self.client.post('/?tab=' + action, data={
            'action': action,
            'image': (io.BytesIO(encoded.tobytes()), 'document.png'),
        })

    def test_navigation_and_assets(self):
        self.assertIn('Quét tài liệu', self.client.get('/').get_data(as_text=True))
        compare = self.client.get('/?tab=compare').get_data(as_text=True)
        self.assertIn('value="compare"', compare)
        self.assertNotIn('id="strengthRange"', compare)
        for path in ('/static/css/style.css', '/static/js/app.js'):
            with self.client.get(path) as response:
                self.assertEqual(response.status_code, 200)

    def test_missing_and_unsupported_uploads(self):
        for data in ({'action': 'compare'}, {
            'action': 'compare', 'image': (io.BytesIO(b'invalid'), 'bad.txt'),
        }):
            response = self.client.post('/?tab=compare', data=data)
            self.assertEqual(response.status_code, 400)
            self.assertIn('role="alert"', response.get_data(as_text=True))
            self.assertIn('value="compare"', response.get_data(as_text=True))

    def test_corrupt_image(self):
        response = self.client.post('/', data={
            'action': 'scan', 'image': (io.BytesIO(b'not an image'), 'bad.png'),
        })
        self.assertEqual(response.status_code, 400)

    def test_oversize_keeps_comparison_mode(self):
        with patch.dict(web.app.config, MAX_CONTENT_LENGTH=100):
            response = self.upload()
        self.assertEqual(response.status_code, 413)
        self.assertIn('value="compare"', response.get_data(as_text=True))

    def test_real_comparison_renders_six_results(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertEqual(html.count('class="result-download"'), 6)
        self.assertIn('OpenCV Canny', html)

    def test_scan_result_renders_download_and_stages(self):
        # Only isolate the costly scanner pipeline; exercise uploads, saving and rendering.
        image = np.zeros((24, 30, 3), dtype=np.uint8)
        result = dict(edge=image[:, :, 0], detection_preview=image,
                      warped=image, sharpened=image, corners=np.zeros((4, 2)),
                      final_width=30, final_height=24, low_threshold=0, high_threshold=0)
        with patch.object(web, 'scan_document', return_value=result):
            response = self.upload('scan')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('Tải bản quét JPG', html)
        self.assertEqual(html.count('class="stage-number"'), 4)


if __name__ == '__main__':
    unittest.main()
