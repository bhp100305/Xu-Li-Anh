"""Known planar targets inside larger, non-planar laptop outlines."""
import io
import unittest

import cv2
import numpy as np

from src.canny import canny
from src.document_detector import detect_document, find_supported_surface
from src.scanner import scan_document


def laptop_photo(light=False, quad=None):
    photo = np.full((900, 650, 3), (170, 190, 205), np.uint8)
    cv2.rectangle(photo, (260, 25), (635, 160), (175, 175, 175), 8)
    cv2.rectangle(photo, (267, 32), (628, 153), (55, 95, 75), -1)
    for center, radius in (((180, 145), 70), ((430, 170), 40), ((570, 176), 35)):
        cv2.circle(photo, center, radius, (32, 35, 38), -1)
    bezel = np.int32([[50, 210], [615, 222], [605, 625], [65, 615]])
    cv2.fillConvexPoly(photo, bezel, (12, 14, 16))
    keyboard = np.int32([[65, 615], [605, 625], [640, 805], [40, 790]])
    cv2.fillConvexPoly(photo, keyboard, (145, 150, 155))
    for row in range(5):
        for column in range(16):
            x, y = 73 + column * 32 - row * 2, 640 + row * 29
            cv2.rectangle(photo, (x, y), (x + 27, y + 22), (32, 35, 38), -1)
            cv2.putText(photo, 'A', (x + 8, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (195, 200, 205), 1)
    expected = np.float32(quad if quad is not None else [[65, 225], [600, 238], [585, 590], [82, 580]])
    display = np.full((420, 640, 3), (215, 220, 225) if light else (65, 78, 82), np.uint8)
    cv2.rectangle(display, (0, 0), (639, 20), (110, 120, 125), -1)
    cv2.rectangle(display, (0, 395), (639, 419), (205, 215, 225), -1)
    cv2.rectangle(display, (8, 30), (120, 390), (42, 50, 56), -1)
    ink = (40, 45, 50) if light else (175, 185, 195)
    for y in range(45, 385, 16):
        cv2.putText(display, 'source code 0123456789', (135, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, ink, 1)
    cv2.rectangle(display, (500, 260), (625, 380), (225, 230, 235), -1)
    cv2.putText(display, 'WiFi', (515, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 45, 50), 1)
    transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [639, 0], [639, 419], [0, 419]]), expected)
    warped = cv2.warpPerspective(display, transform, (650, 900))
    mask = cv2.warpPerspective(np.full((420, 640), 255, np.uint8), transform, (650, 900))
    photo[mask > 0] = warped[mask > 0]
    return photo, expected


class SurfaceDetectionTests(unittest.TestCase):
    def test_monochrome_photo_keeps_the_same_display_plane(self):
        photo, expected = laptop_photo()
        result = scan_document(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), output_width=2400, verbose=False)
        np.testing.assert_allclose(result['corners'], expected, atol=18)

    def test_display_is_selected_instead_of_screen_and_keyboard(self):
        for light in (False, True):
            with self.subTest(light=light):
                photo, expected = laptop_photo(light)
                result = scan_document(photo, output_width=2400, verbose=False)
                np.testing.assert_allclose(result['corners'], expected, atol=18)
                self.assertGreater(result['final_width'], result['final_height'])

    def test_rolled_and_dimmed_camera_views_keep_the_display_plane(self):
        for angle, brightness in ((-12, 1.0), (10, 0.6)):
            with self.subTest(angle=angle, brightness=brightness):
                photo, expected = laptop_photo()
                matrix = cv2.getRotationMatrix2D((325, 450), angle, 0.9)
                photo = cv2.warpAffine(photo, matrix, (650, 900), borderMode=cv2.BORDER_REFLECT_101)
                photo = np.clip(photo.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
                expected = cv2.transform(expected[None], matrix)[0]
                result = scan_document(photo, output_width=2400, verbose=False)
                np.testing.assert_allclose(result['corners'], expected, atol=20)

    def test_thin_printed_rules_are_not_a_surface_boundary(self):
        image = np.full((600, 420, 3), 240, np.uint8)
        cv2.rectangle(image, (50, 70), (370, 520), (10, 10, 10), 2)
        for y in range(100, 520, 40):
            cv2.line(image, (50, y), (370, y), (10, 10, 10), 2)
        edges = canny(image)['edge']
        surface, _ = find_supported_surface(image, edges)
        self.assertIsNone(surface)

    def test_large_printed_panel_does_not_replace_an_enclosing_sheet(self):
        image = np.full((900, 650, 3), 30, np.uint8)
        cv2.rectangle(image, (80, 30), (570, 870), (245, 245, 245), -1)
        cv2.rectangle(image, (140, 150), (515, 700), (95, 95, 95), -1)
        edges = canny(image)['edge']
        # Even a missing outer Canny side must not promote the large graphic
        # inside the paper over the whole sheet found by its paper boundary.
        edges[:50] = 0
        result = detect_document(image, edges)
        np.testing.assert_allclose(result['corners'], [[80, 30], [570, 30], [570, 870], [80, 870]], atol=10)

    def test_camera_guidance_and_final_api_both_exclude_keyboard(self):
        import app as web
        import json
        for light in (False, True):
            with self.subTest(light=light):
                photo, expected = laptop_photo(light)
                raw = cv2.imencode('.png', photo)[1].tobytes()
                with web.app.test_client() as client:
                    with client.post('/api/multipage/detect', data={
                        'image': (io.BytesIO(raw), 'laptop.png'),
                    }) as response:
                        self.assertEqual(response.status_code, 200)
                        guide = np.asarray(response.json['corners']) * [649, 899]
                        np.testing.assert_allclose(guide, expected, atol=35)
                        self.assertLess(float(guide[:, 1].max()), 635)
                    with client.post('/api/multipage/scan', data={
                        'image': (io.BytesIO(raw), 'laptop.png'),
                    }) as response:
                        self.assertEqual(response.status_code, 200)
                        corners = np.asarray(json.loads(response.headers['X-Document-Corners'])) * [649, 899]
                        np.testing.assert_allclose(corners, expected, atol=18)
                        result = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_UNCHANGED)
                        self.assertGreater(result.shape[1], result.shape[0])


if __name__ == '__main__':
    unittest.main()
