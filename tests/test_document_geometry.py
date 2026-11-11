"""Regression photos for tilted forms with strong printed table borders."""
import itertools
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.document_detector import find_document_from_brightness, order_points, refine_document_region
from src.scanner import scan_document


def form_photo(background=220, quad=None):
    page = np.full((600, 420, 3), 245, np.uint8)
    cv2.rectangle(page, (45, 75), (375, 250), (15, 15, 15), 3)
    for y in range(105, 250, 30):
        cv2.line(page, (45, y), (375, y), (15, 15, 15), 2)
    cv2.line(page, (165, 75), (165, 250), (15, 15, 15), 2)
    for y in range(300, 560, 25):
        cv2.putText(page, 'COMPLETE FORM BODY', (40, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (15, 15, 15), 1, cv2.LINE_AA)
    for x, y in ((12, 12), (385, 12), (385, 565), (12, 565)):
        cv2.rectangle(page, (x, y), (x + 20, y + 20), (35, 35, 35), -1)
    expected = np.float32(quad if quad is not None else [[75, 25], [585, 95], [625, 855], [20, 880]])
    transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [419, 0], [419, 599], [0, 599]]), expected)
    photo = np.full((900, 650, 3), background, np.uint8)
    warped = cv2.warpPerspective(page, transform, (650, 900))
    mask = cv2.warpPerspective(np.full((600, 420), 255, np.uint8), transform, (650, 900))
    photo[mask > 0] = warped[mask > 0]
    return photo, expected


class DocumentGeometryTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DOCUSCAN_RECTIFIED_IMAGE'), 'Rectified original supplied locally')
    def test_reported_rectified_page_keeps_full_image_and_editor_coordinates(self):
        import app as web
        import json
        raw = Path(os.environ['DOCUSCAN_RECTIFIED_IMAGE']).read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (1221, 1000))
        result = scan_document(image, output_width=2400, verbose=False)
        np.testing.assert_array_equal(result['corners'], [[0, 0], [999, 0], [999, 1220], [0, 1220]])
        self.assertGreaterEqual(result['final_height'], 1219)
        self.assertGreaterEqual(result['final_width'], 998)
        with web.app.test_client() as client, client.post('/api/multipage/scan', data={
            'image': (io.BytesIO(raw), 'already-scanned.jpg'),
        }) as response:
            self.assertEqual(response.status_code, 200)
            np.testing.assert_array_equal(json.loads(response.headers['X-Document-Corners']),
                                          [[0, 0], [1, 0], [1, 1], [0, 1]])
            response.request.environ['wsgi.input'].close()

    def test_shadowed_full_page_with_large_printed_panels_keeps_all_contents(self):
        image = np.full((600, 420, 3), 220, np.int16)
        image -= np.linspace(60, 0, 420).astype(np.int16)[None, :, None]
        image = image.astype(np.uint8)
        cv2.rectangle(image, (15, 185), (170, 390), (60, 60, 60), 2)
        for y in range(225, 385, 30):
            cv2.line(image, (15, y), (170, y), (35, 35, 35), 2)
        cv2.rectangle(image, (230, 185), (395, 390), (85, 85, 85), -1)
        for y in range(40, 570, 45):
            cv2.putText(image, 'FORM CONTENT 0123456789', (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)
        result = scan_document(image, verbose=False)
        np.testing.assert_array_equal(result['corners'], [[0, 0], [419, 0], [419, 599], [0, 599]])
        self.assertEqual(result['detection_method'], 'full-frame')

    @unittest.skipUnless(os.environ.get('DOCUSCAN_DIAGONAL_IMAGE'), 'Diagonal original supplied locally')
    def test_reported_diagonal_photo_is_cropped_in_both_scanner_routes(self):
        import app as web
        raw = Path(os.environ['DOCUSCAN_DIAGONAL_IMAGE']).read_bytes()
        expected = [[1004, 400], [1845, 858], [980, 2148], [30, 1365]]
        with tempfile.TemporaryDirectory() as folder:
            client = web.app.test_client()
            for route, module in (('/', 'app'), ('/api/multipage/scan', 'multipage')):
                recorded = []

                def record(**options):
                    result = scan_document(**options)
                    recorded.append(result)
                    return result

                with self.subTest(route=route), patch(f'{module}.scan_document', side_effect=record), \
                        patch.object(web, 'UPLOAD_FOLDER', Path(folder)), \
                        patch.object(web, 'RESULT_FOLDER', Path(folder)):
                    with client, client.post(route, data={
                        'action': 'scan', 'image': (io.BytesIO(raw), 'diagonal.jpg'),
                    }) as response:
                        status = response.status_code
                        response.request.environ['wsgi.input'].close()
                self.assertEqual(status, 200)
                self.assertEqual(len(recorded), 1)
                np.testing.assert_allclose(recorded[0]['corners'], expected, atol=50)
                self.assertGreater(recorded[0]['final_height'], recorded[0]['final_width'])

    @unittest.skipUnless(os.environ.get('DOCUSCAN_REGRESSION_IMAGE'), 'Original photo supplied locally')
    def test_reported_exam_photo_excludes_keyboard_and_rectifies_the_sheet(self):
        # Keep the user's original outside the repository. This opt-in check
        # uses the actual reported 960x1280 photo, not a screenshot of it.
        raw = Path(os.environ['DOCUSCAN_REGRESSION_IMAGE']).read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (1280, 960))
        expected = [[149, 214], [861, 174], [929, 1237], [138, 1264]]
        # The hand touching the paper must not be included by different
        # foreground-model initializations in successive requests.
        for seed in (0, 6, 11):
            with self.subTest(initialization=seed):
                cv2.setRNGSeed(seed)
                result = scan_document(image, output_width=2400, verbose=False)
                np.testing.assert_allclose(result['corners'], expected, atol=8)
                self.assertEqual(result['detection_method'], 'paper-region')
                self.assertGreater(result['sharpened'].shape[0], 1000)

    def test_colour_region_removes_background_from_a_loose_keyboard_crop(self):
        photo, expected = form_photo(120)
        # A long horizontal laptop hinge is outside the slanted paper.
        photo[:24] = 35
        loose = np.float32([[5, 0], [630, 20], [649, 899], [0, 899]])
        refined = refine_document_region(photo, loose)
        self.assertIsNotNone(refined)
        np.testing.assert_allclose(refined, expected, atol=5)

    def test_region_refinement_keeps_an_already_tight_crop(self):
        photo, corners = form_photo(120)
        self.assertIsNone(refine_document_region(photo, corners))

    def test_light_desk_detects_outer_page_instead_of_internal_table(self):
        photo, expected = form_photo(220)
        result = scan_document(photo, detection_width=800, output_width=2400, verbose=False)
        np.testing.assert_allclose(result['corners'], expected, atol=12)
        self.assertGreater(result['final_height'], result['final_width'])
        self.assert_all_corner_markers_present(result['sharpened'])

    def test_dark_desk_still_uses_all_four_outer_corners(self):
        photo, expected = form_photo(35)
        result = scan_document(photo, detection_width=800, output_width=2400, verbose=False)
        np.testing.assert_allclose(result['corners'], expected, atol=12)
        self.assert_all_corner_markers_present(result['sharpened'])

    def test_paper_region_keeps_trapezoid_instead_of_a_rotated_rectangle(self):
        photo, expected = form_photo(220)
        corners = find_document_from_brightness(photo)
        self.assertIsNotNone(corners)
        np.testing.assert_allclose(corners, expected, atol=10)

    def test_45_degree_ties_do_not_repeat_or_drop_corners(self):
        points = np.float32([[100, 0], [200, 100], [100, 200], [0, 100]])
        for permutation in itertools.permutations(points):
            ordered = order_points(np.asarray(permutation))
            np.testing.assert_array_equal(ordered, points)
            self.assertEqual(len(np.unique(ordered, axis=0)), 4)

    def test_oblique_portrait_does_not_start_at_the_lower_left_extremum(self):
        points = np.float32([[1004, 400], [1845, 858], [980, 2148], [30, 1365]])
        self.assertLess(points[3].sum(), points[0].sum())  # Old x+y rule fails here.
        for permutation in itertools.permutations(points):
            np.testing.assert_array_equal(order_points(np.asarray(permutation)), points)

    def test_an_already_scanned_form_is_not_recropped_to_its_table(self):
        image = np.full((600, 420, 3), 245, np.uint8)
        cv2.rectangle(image, (50, 60), (370, 230), (10, 10, 10), 3)
        cv2.putText(image, 'TOP', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
        cv2.putText(image, 'BOTTOM', (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
        result = scan_document(image, verbose=False)
        np.testing.assert_allclose(result['corners'], [[0, 0], [419, 0], [419, 599], [0, 599]], atol=1)
        self.assertGreater(result['sharpened'].shape[0], 580)

    def test_no_paper_on_dark_image_still_reports_failure(self):
        with self.assertRaises(RuntimeError):
            scan_document(np.zeros((100, 100, 3), np.uint8), verbose=False)

    def test_batch_uploads_of_different_tilted_forms_keep_all_contents(self):
        import app as web
        client = web.app.test_client()
        samples = [(35, None), (220, None),
                   (220, [[130, 70], [560, 35], [590, 840], [65, 885]]),
                   (220, [[300, 20], [630, 360], [330, 875], [15, 530]])]
        for index, (background, quad) in enumerate(samples):
            with self.subTest(file=index):
                photo, expected = form_photo(background, quad)
                recorded = []

                def record(**options):
                    result = scan_document(**options)
                    recorded.append(result)
                    return result

                _, encoded = cv2.imencode('.png', photo)
                with patch('multipage.scan_document', side_effect=record):
                    response = client.post('/api/multipage/scan', data={
                        'image': (io.BytesIO(encoded.tobytes()), f'form_{index}.png'),
                    })
                self.assertEqual(response.status_code, 200)
                np.testing.assert_allclose(recorded[0]['corners'], expected, atol=12)
                result = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_UNCHANGED)
                self.assert_all_corner_markers_present(result)

    def assert_all_corner_markers_present(self, image):
        height, width = image.shape
        for region in (image[:height // 8, :width // 8], image[:height // 8, -width // 8:],
                       image[-height // 8:, :width // 8], image[-height // 8:, -width // 8:]):
            self.assertGreater(np.count_nonzero(region < 150), 25)


if __name__ == '__main__':
    unittest.main()
