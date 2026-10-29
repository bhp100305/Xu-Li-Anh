from __future__ import annotations

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 điểm theo thứ tự:
    top-left, top-right, bottom-right, bottom-left.
    """

    if points.shape != (4, 2):
        raise ValueError(
            "order_points yêu cầu đúng 4 điểm."
        )

    points = points.astype(np.float32)

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    point_sum = points.sum(axis=1)

    point_diff = np.diff(
        points,
        axis=1,
    ).reshape(-1)

    # Top-left
    ordered[0] = points[
        np.argmin(point_sum)
    ]

    # Bottom-right
    ordered[2] = points[
        np.argmax(point_sum)
    ]

    # Top-right
    ordered[1] = points[
        np.argmin(point_diff)
    ]

    # Bottom-left
    ordered[3] = points[
        np.argmax(point_diff)
    ]

    return ordered


def prepare_edges(
    edge_image: np.ndarray,
) -> np.ndarray:
    """
    Nối các biên bị đứt để tìm contour dễ hơn.
    """

    edge = np.clip(
        edge_image,
        0,
        255,
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )

    closed = cv2.morphologyEx(
        edge,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    dilated = cv2.dilate(
        closed,
        kernel,
        iterations=1,
    )

    return dilated


def find_document_from_edges(
    edge_image: np.ndarray,
    min_area_ratio: float = 0.08,
) -> np.ndarray | None:
    """
    Tìm tài liệu từ ảnh biên Canny.
    """

    if edge_image.ndim != 2:
        raise ValueError(
            "edge_image phải là ảnh 2 chiều."
        )

    height, width = edge_image.shape

    image_area = height * width

    min_area = (
        image_area
        * min_area_ratio
    )

    processed = prepare_edges(
        edge_image
    )

    contours, _ = cv2.findContours(
        processed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    epsilon_values = [
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.04,
        0.05,
        0.07,
    ]

    for contour in contours[:30]:

        area = cv2.contourArea(
            contour
        )

        if area < min_area:
            continue

        hull = cv2.convexHull(
            contour
        )

        perimeter = cv2.arcLength(
            hull,
            True,
        )

        if perimeter <= 0:
            continue

        for epsilon_ratio in epsilon_values:

            approximation = (
                cv2.approxPolyDP(
                    hull,
                    epsilon_ratio
                    * perimeter,
                    True,
                )
            )

            if len(approximation) != 4:
                continue

            if not cv2.isContourConvex(
                approximation
            ):
                continue

            points = (
                approximation
                .reshape(4, 2)
                .astype(np.float32)
            )

            return order_points(
                points
            )

    return None


def find_document_from_hough(
    edge_image: np.ndarray,
    min_area_ratio: float = 0.08,
) -> np.ndarray | None:
    """Find a document by intersecting long Hough line segments.

    This handles pages whose borders are visible as four separate edges but
    do not form one closed contour. Coordinates are returned in the same
    resized-image coordinate system as ``edge_image``.
    """

    if edge_image.ndim != 2:
        raise ValueError("edge_image must be a 2D image.")

    height, width = edge_image.shape
    diagonal = float(np.hypot(width, height))
    min_dimension = min(height, width)
    edges = np.clip(edge_image, 0, 255).astype(np.uint8)

    # Bridge small gaps before Hough so a printed border can vote as one line.
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(25, int(min_dimension * 0.08)),
        minLineLength=max(24, int(min_dimension * 0.18)),
        maxLineGap=max(12, int(min_dimension * 0.06)),
    )
    if segments is None:
        return None

    # Keep the longest distinct segments; suppress repeated votes for the
    # same physical edge while allowing nearby parallel page sides.
    candidates: list[tuple[np.ndarray, float, float]] = []
    for raw in segments.reshape(-1, 4):
        p1 = raw[:2].astype(np.float64)
        p2 = raw[2:].astype(np.float64)
        length = float(np.linalg.norm(p2 - p1))
        if length < max(24, min_dimension * 0.18):
            continue
        angle = float(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) % np.pi)
        midpoint = (p1 + p2) / 2.0

        duplicate = False
        for old_index, (old_segment, old_angle, old_length) in enumerate(candidates):
            angle_delta = abs(angle - old_angle)
            angle_delta = min(angle_delta, np.pi - angle_delta)
            old_mid = (old_segment[:2] + old_segment[2:]) / 2.0
            normal = np.array([-np.sin(angle), np.cos(angle)])
            offset = abs(float(np.dot(midpoint - old_mid, normal)))
            if angle_delta < np.deg2rad(3.0) and offset < max(8.0, diagonal * 0.012):
                duplicate = True
                if length > old_length:
                    candidates[old_index] = (raw.astype(np.float64), angle, length)
                break
        if not duplicate:
            candidates.append((raw.astype(np.float64), angle, length))

    candidates.sort(key=lambda item: item[2], reverse=True)
    candidates = candidates[:24]
    if len(candidates) < 4:
        return None

    def orientation_delta(angle_a: float, angle_b: float) -> float:
        delta = abs(angle_a - angle_b) % np.pi
        return min(delta, np.pi - delta)

    def mean_orientation(angle_a: float, angle_b: float) -> float:
        # Angles are unoriented (0 and pi describe the same line).
        return float(
            (0.5 * np.arctan2(
                np.sin(2.0 * angle_a) + np.sin(2.0 * angle_b),
                np.cos(2.0 * angle_a) + np.cos(2.0 * angle_b),
            )) % np.pi
        )

    # Potential opposite sides: similar direction, with a meaningful gap.
    side_pairs: list[tuple[int, int, float]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            _, angle_a, _ = candidates[i]
            _, angle_b, _ = candidates[j]
            delta = orientation_delta(angle_a, angle_b)
            if delta > np.deg2rad(18.0):
                continue
            mean_angle = mean_orientation(angle_a, angle_b)
            mid_a = (candidates[i][0][:2] + candidates[i][0][2:]) / 2.0
            mid_b = (candidates[j][0][:2] + candidates[j][0][2:]) / 2.0
            normal = np.array([-np.sin(mean_angle), np.cos(mean_angle)])
            separation = abs(float(np.dot(mid_a - mid_b, normal)))
            if separation >= diagonal * 0.12:
                side_pairs.append((i, j, mean_angle))

    def intersection(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = line_a
        x3, y3, x4, y4 = line_b
        a1, b1, c1 = y2 - y1, x1 - x2, x2 * y1 - x1 * y2
        a2, b2, c2 = y4 - y3, x3 - x4, x4 * y3 - x3 * y4
        determinant = a1 * b2 - a2 * b1
        if abs(determinant) < 1e-6:
            return None
        return np.array(
            [(b1 * c2 - b2 * c1) / determinant,
             (c1 * a2 - c2 * a1) / determinant],
            dtype=np.float32,
        )

    image_area = float(width * height)
    best_score = -1.0
    best_corners: np.ndarray | None = None

    for pair_index, pair_a in enumerate(side_pairs):
        a1, a2, angle_a = pair_a
        for pair_b in side_pairs[pair_index + 1:]:
            b1, b2, angle_b = pair_b
            if len({a1, a2, b1, b2}) != 4:
                continue
            cross_angle = orientation_delta(angle_a, angle_b)
            if not np.deg2rad(25.0) < cross_angle < np.deg2rad(155.0):
                continue

            points = [
                intersection(candidates[a1][0], candidates[b1][0]),
                intersection(candidates[a1][0], candidates[b2][0]),
                intersection(candidates[a2][0], candidates[b2][0]),
                intersection(candidates[a2][0], candidates[b1][0]),
            ]
            if any(point is None for point in points):
                continue
            corners = order_points(np.asarray(points, dtype=np.float32))
            if (
                np.any(corners[:, 0] < -width * 0.04)
                or np.any(corners[:, 0] > width * 1.04)
                or np.any(corners[:, 1] < -height * 0.04)
                or np.any(corners[:, 1] > height * 1.04)
            ):
                continue
            area_ratio = abs(float(cv2.contourArea(corners))) / image_area
            if not min_area_ratio <= area_ratio <= 0.98:
                continue
            contour = corners.astype(np.int32).reshape((-1, 1, 2))
            if not cv2.isContourConvex(contour):
                continue

            score = score_document_corners(edge_image, corners)
            if score > best_score:
                best_score = score
                best_corners = corners

    return best_corners


def score_document_corners(
    edge_image: np.ndarray,
    corners: np.ndarray,
) -> float:
    """Score a four-corner candidate by area, centering, and edge support."""

    height, width = edge_image.shape
    image_area = float(width * height)
    diagonal = float(np.hypot(width, height))
    area_ratio = abs(float(cv2.contourArea(corners.astype(np.float32)))) / image_area
    if not 0.0 < area_ratio <= 0.98:
        return -1.0

    edge_map = cv2.dilate(
        np.clip(edge_image, 0, 255).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    ) > 0
    support = 0.0
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        samples = np.linspace(start, end, 40)[3:-3].astype(np.int32)
        inside = (
            (samples[:, 0] >= 0) & (samples[:, 0] < width)
            & (samples[:, 1] >= 0) & (samples[:, 1] < height)
        )
        if np.any(inside):
            support += float(np.mean(edge_map[samples[inside, 1], samples[inside, 0]]))
    support /= 4.0

    center = np.array([width / 2.0, height / 2.0])
    center_distance = float(np.linalg.norm(corners.mean(axis=0) - center))
    center_score = 1.0 - min(center_distance / diagonal, 1.0)
    return 0.55 * area_ratio + 0.20 * center_score + 0.25 * support


def find_document_from_brightness(
    image: np.ndarray,
    min_area_ratio: float = 0.10,
) -> np.ndarray | None:
    """
    Fallback:
    phát hiện vùng giấy sáng từ ảnh gốc.

    Hữu ích khi một góc giấy bị che hoặc
    Canny không tạo được contour khép kín.
    """

    if image is None:
        return None

    if image.ndim != 3:
        return None

    height, width = image.shape[:2]

    image_area = (
        height * width
    )

    min_area = (
        image_area
        * min_area_ratio
    )

    # ==========================================
    # GRAYSCALE
    # ==========================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # ==========================================
    # BLUR NHẸ
    # ==========================================

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # ==========================================
    # OTSU THRESHOLD
    # ==========================================

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    # ==========================================
    # LÀM KÍN VÙNG GIẤY
    # ==========================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    # ==========================================
    # CONTOURS
    # ==========================================

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    best_score = -1.0
    best_box = None

    image_center = np.array(
        [
            width / 2.0,
            height / 2.0,
        ],
        dtype=np.float32,
    )

    max_center_distance = np.sqrt(
        width ** 2
        + height ** 2
    )

    for contour in contours:

        contour_area = cv2.contourArea(
            contour
        )

        if contour_area < min_area:
            continue

        rect = cv2.minAreaRect(
            contour
        )

        (
            center,
            size,
            _
        ) = rect

        rect_width = size[0]
        rect_height = size[1]

        if (
            rect_width <= 0
            or rect_height <= 0
        ):
            continue

        rect_area = (
            rect_width
            * rect_height
        )

        if rect_area <= 0:
            continue

        # Mức độ giống hình chữ nhật
        rectangularity = min(
            contour_area
            / rect_area,
            1.0,
        )

        center_array = np.array(
            center,
            dtype=np.float32,
        )

        center_distance = np.linalg.norm(
            center_array
            - image_center
        )

        center_score = (
            1.0
            - center_distance
            / max_center_distance
        )

        area_score = min(
            contour_area
            / image_area,
            1.0,
        )

        # Ưu tiên:
        # diện tích lớn
        # gần giữa ảnh
        # hình tương đối chữ nhật
        score = (
            0.50 * area_score
            + 0.30 * rectangularity
            + 0.20 * center_score
        )

        if score > best_score:

            box = cv2.boxPoints(
                rect
            )

            best_box = np.array(
                box,
                dtype=np.float32,
            )

            best_score = score

    if best_box is None:
        return None

    return order_points(
        best_box
    )


def detect_document(
    image: np.ndarray,
    edge_image: np.ndarray,
) -> dict[str, object]:
    """
    Phát hiện tài liệu bằng nhiều chiến lược.

    Ưu tiên:
    1. Canny contour.
    2. Brightness segmentation.
    """

    # ==========================================
    # CÁCH 1: EDGE
    # ==========================================

    corners = find_document_from_edges(
        edge_image
    )

    method = "edge" if corners is not None else "hough"


    # ==========================================
    # CÁCH 2: HOUGH-LINE FALLBACK
    # ==========================================

    hough_corners = find_document_from_hough(
        edge_image
    )

    if hough_corners is not None and (
        corners is None
        or score_document_corners(edge_image, hough_corners)
        > score_document_corners(edge_image, corners) + 0.03
    ):
        corners = hough_corners
        method = "hough"


    # ==========================================
    # CÁCH 3: BRIGHTNESS FALLBACK
    # ==========================================

    if corners is None:

        corners = (
            find_document_from_brightness(
                image
            )
        )

        method = "brightness" if corners is not None else "none"


    preview = draw_document_contour(
        image,
        corners,
    )

    return {
        "corners": corners,
        "preview": preview,
        "method": method,
    }


def draw_document_contour(
    image: np.ndarray,
    corners: np.ndarray | None,
) -> np.ndarray:
    """
    Vẽ vùng tài liệu và 4 góc.
    """

    output = image.copy()

    if corners is None:
        return output

    contour = (
        corners
        .astype(np.int32)
        .reshape(
            (-1, 1, 2)
        )
    )

    # Khung xanh
    cv2.polylines(
        output,
        [contour],
        True,
        (0, 255, 0),
        3,
    )

    labels = [
        "TL",
        "TR",
        "BR",
        "BL",
    ]

    for index, point in enumerate(
        corners
    ):

        x, y = point.astype(int)

        cv2.circle(
            output,
            (x, y),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            output,
            labels[index],
            (
                x + 10,
                y - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    return output
