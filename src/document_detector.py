from __future__ import annotations

from typing import Callable

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

    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != 4:
        raise ValueError("Bốn góc tài liệu phải khác nhau và có tọa độ hợp lệ.")
    # Sort around the center instead of choosing four independent extrema.
    # At ~45 degrees, sum/difference ties can otherwise select a corner twice.
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    if not cv2.isContourConvex(ordered):
        raise ValueError("Bốn góc phải tạo thành tứ giác lồi.")
    # Choose the upper edge, not the smallest independent x+y extremum.
    # In an oblique portrait photo the lower-left corner can have a smaller
    # x+y than the upper-left, which would rotate the scan by 90 degrees.
    edge_mid_y = (ordered[:, 1] + np.roll(ordered[:, 1], -1)) / 2
    start = np.lexsort((ordered[:, 0], ordered[:, 1], edge_mid_y))[0]
    ordered = np.roll(ordered, -int(start), axis=0)

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
    *,
    scorer: Callable[[np.ndarray], float] | None = None,
    candidate_quads: list[np.ndarray] | None = None,
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
    supported_edges = cv2.dilate(np.clip(edge_image, 0, 255).astype(np.uint8),
                                 np.ones((7, 7), np.uint8)) > 0

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
            try:
                corners = order_points(np.asarray(points, dtype=np.float32))
            except ValueError:
                continue
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

            if candidate_quads is not None:
                candidate_quads.append(corners)
            score = (score_document_corners(edge_image, corners, supported_edges)
                     if scorer is None else scorer(corners))
            if score > best_score:
                best_score = score
                best_corners = corners

    return best_corners


def score_document_corners(
    edge_image: np.ndarray,
    corners: np.ndarray,
    edge_map: np.ndarray | None = None,
) -> float:
    """Score a four-corner candidate by area, centering, and edge support."""

    height, width = edge_image.shape
    image_area = float(width * height)
    diagonal = float(np.hypot(width, height))
    area_ratio = abs(float(cv2.contourArea(corners.astype(np.float32)))) / image_area
    if not 0.0 < area_ratio <= 0.98:
        return -1.0

    if edge_map is None:
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


def _paper_boundary_contrast(gray: np.ndarray, corners: np.ndarray) -> int:
    """Count sides separating the paper from its surroundings, not printed lines."""
    center = corners.mean(axis=0)
    distance = max(2.0, float(np.hypot(*gray.shape)) * 0.008)
    supported = 0
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        side = b - a
        normal = np.array([-side[1], side[0]], np.float32)
        length = float(np.linalg.norm(normal))
        if length == 0:
            continue
        normal /= length
        if np.dot(center - (a + b) / 2, normal) < 0:
            normal = -normal
        samples = np.linspace(a, b, 24)[3:-3]
        inside = np.rint(samples + normal * distance).astype(np.int32)
        outside = np.rint(samples - normal * distance).astype(np.int32)
        valid = ((inside[:, 0] >= 0) & (inside[:, 0] < gray.shape[1])
                 & (inside[:, 1] >= 0) & (inside[:, 1] < gray.shape[0])
                 & (outside[:, 0] >= 0) & (outside[:, 0] < gray.shape[1])
                 & (outside[:, 1] >= 0) & (outside[:, 1] < gray.shape[0]))
        if np.count_nonzero(valid) < 6:
            continue
        inside, outside = inside[valid], outside[valid]
        contrast = gray[inside[:, 1], inside[:, 0]].astype(np.float32) - gray[outside[:, 1], outside[:, 0]]
        if float(np.median(contrast)) > 4:
            supported += 1
    return supported


def find_document_from_brightness(
    image: np.ndarray,
    min_area_ratio: float = 0.10,
) -> np.ndarray | None:
    """Find the outer paper quadrilateral while retaining perspective.

    Otsu can merge pale paper with a pale desk. Additional brightness levels
    separate those surfaces. A rotated rectangle must not replace a trapezoid.
    """
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 3:
        return None
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if float(np.percentile(gray, 95)) < 80:
        return None
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    levels = sorted({int(otsu), *(max(0, int(value) - 1)
                    for value in np.percentile(gray, [55, 70, 85, 95]))})
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    best = None
    best_score = -1.0
    image_area = float(height * width)
    for level in levels:
        _, binary = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            area = float(cv2.contourArea(contour))
            if not min_area_ratio <= area / image_area <= 0.985:
                continue
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            for epsilon in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05):
                quad = cv2.approxPolyDP(hull, epsilon * perimeter, True)
                if len(quad) != 4 or not cv2.isContourConvex(quad):
                    continue
                corners = order_points(quad.reshape(4, 2))
                quad_area = float(cv2.contourArea(corners))
                if quad_area <= 0 or min(area, quad_area) / max(area, quad_area) < 0.85:
                    continue
                supported_sides = _paper_boundary_contrast(gray, corners)
                if supported_sides < 3:
                    continue
                center_distance = float(np.linalg.norm(corners.mean(axis=0) - [width / 2, height / 2]))
                score = 0.8 * quad_area / image_area + 0.2 * (1 - center_distance / np.hypot(width, height))
                if score > best_score:
                    best_score, best = score, corners
                break
    if best is not None:
        return best
    # Already-scanned pages can fill the photo. Preserve their whole contents
    # instead of mistaking a printed table inside the page for the paper border.
    # A scanned page may contain shadows and large printed screenshots. A
    # single high brightness cutoff rejects these pages and lets an internal
    # table become the crop. Check spatial coverage as well as brightness.
    paper_reference = float(np.percentile(gray, 90))
    paper_level = max(100.0, paper_reference - 60)
    patch = max(2, min(height, width) // 25)
    corner_patches = (gray[:patch, :patch], gray[:patch, -patch:],
                      gray[-patch:, :patch], gray[-patch:, -patch:])
    corner_levels = np.array([np.median(region) for region in corner_patches])
    tiles = [tile for row in np.array_split(gray, 4)
             for tile in np.array_split(row, 4, axis=1)]
    if (paper_reference >= 160 and np.mean(gray >= paper_level) > 0.70
            and sum(float(np.median(tile)) >= paper_level for tile in tiles) >= 12
            and np.count_nonzero(corner_levels >= paper_level) >= 3
            and float(corner_levels.min()) >= paper_level - 20):
        return np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    return None


def refine_document_region(image: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    """Remove a substantial strip of background from a loose line-based crop.

    Long laptop edges can outvote the paper's shorter/curved sides in Hough.
    Fit foreground/background colour models on a small copy, then require an
    enclosing four-sided paper region before replacing the proposed crop.
    Small differences are left alone to avoid trimming an already good scan.
    """
    height, width = image.shape[:2]
    original_area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    if original_area / (width * height) >= 0.975:
        return None
    ratio = min(1.0, 480.0 / max(height, width))
    small = cv2.resize(image, (max(2, round(width * ratio)), max(2, round(height * ratio))))
    scale = np.float32([small.shape[1] / width, small.shape[0] / height])
    region = np.zeros(small.shape[:2], np.uint8)
    cv2.fillConvexPoly(region, np.rint(corners * scale).astype(np.int32), 255)
    inner = cv2.erode(region, np.ones((9, 9), np.uint8)) > 0
    outer = cv2.dilate(region, np.ones((13, 13), np.uint8)) > 0
    if np.count_nonzero(inner) < 100 or np.count_nonzero(~outer) < 100:
        return None
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    paper_level = float(np.percentile(gray[inner], 65))
    if paper_level < 80:
        return None
    mask = np.where(region, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    mask[~outer] = cv2.GC_BGD
    paper_seed = inner & (gray >= paper_level)
    mask[paper_seed] = cv2.GC_FGD
    # A hand touching the sheet can be joined to the foreground even though
    # its chroma differs strongly from the paper. Compare a/b independently
    # of luminance so shaded paper is retained; dark printed areas are holes
    # inside the paper contour and do not change the final image pixels.
    chroma = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(np.float32)
    paper_chroma = np.median(chroma[paper_seed], axis=0)
    spread = np.median(np.abs(chroma[paper_seed] - paper_chroma), axis=0)
    colour_outlier = np.any(np.abs(chroma - paper_chroma) > np.maximum(20, 4 * spread), axis=2)
    mask[colour_outlier & ~paper_seed] = cv2.GC_BGD
    try:
        cv2.grabCut(small, mask, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for epsilon in (0.01, 0.015, 0.02, 0.03, 0.04):
        quad = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        refined = order_points(quad.reshape(4, 2))
        area = float(cv2.contourArea(refined))
        if area <= 0 or float(cv2.contourArea(contour)) / area < 0.90:
            continue
        # Curled sides can sit inside the straight quadrilateral; two visible
        # paper/background transitions plus the filled-region fit are required.
        if _paper_boundary_contrast(gray, refined) < 2:
            continue
        refined /= scale
        area = float(cv2.contourArea(refined))
        intersection, _ = cv2.intersectConvexConvex(refined, corners.astype(np.float32))
        if 0.55 < area / original_area < 0.90 and intersection / area > 0.96:
            return refined
        break
    return None


def _surface_maps(image: np.ndarray, edge_image: np.ndarray):
    gray = np.clip(image, 0, 255).astype(np.uint8) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    distance = cv2.distanceTransform((edge_image == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return gray, gx, gy, magnitude, distance


def _surface_quality(corners: np.ndarray, maps) -> float:
    """Require four continuous physical boundaries, rather than nearby texture.

    Keyboard keys can support a loose quadrilateral in an average edge count.
    Score the weakest side too, check gradient orientation, and compare the
    surfaces on either side. Thin printed rules have paper on both sides and
    therefore do not qualify just because their edge is strong.
    """
    gray, gx, gy, magnitude, distance = maps
    height, width = gray.shape
    area_ratio = abs(float(cv2.contourArea(corners.astype(np.float32)))) / (height * width)
    if not 0.08 <= area_ratio <= 0.97:
        return 0.0
    center = corners.mean(axis=0)
    offset = max(4.0, np.hypot(height, width) * 0.008)
    supports, contrasts, positive_contrasts = [], [], []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        side = b - a
        length = float(np.linalg.norm(side))
        if length < min(height, width) * 0.12:
            return 0.0
        normal = np.float32([-side[1], side[0]]) / length
        if np.dot(center - (a + b) / 2, normal) < 0:
            normal = -normal
        points = a + np.linspace(0.08, 0.92, 64, dtype=np.float32)[:, None] * side
        probes = np.rint(points[None] + np.arange(-2, 3)[:, None, None] * normal).astype(np.int32)
        x, y = probes[..., 0], probes[..., 1]
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y = x.clip(0, width - 1), y.clip(0, height - 1)
        aligned = np.abs(gx[y, x] * normal[0] + gy[y, x] * normal[1]) / (magnitude[y, x] + 1e-6)
        hits = valid & (distance[y, x] <= 1.5) & (aligned >= 0.85) & (magnitude[y, x] >= 12)
        supports.append(float(np.mean(np.any(hits, axis=0))))

        inside = np.rint(points + normal * offset).astype(np.int32)
        outside = np.rint(points - normal * offset).astype(np.int32)
        valid = ((inside[:, 0] >= 0) & (inside[:, 0] < width)
                 & (inside[:, 1] >= 0) & (inside[:, 1] < height)
                 & (outside[:, 0] >= 0) & (outside[:, 0] < width)
                 & (outside[:, 1] >= 0) & (outside[:, 1] < height))
        if np.count_nonzero(valid) < 40:
            return 0.0
        inside, outside = inside[valid], outside[valid]
        delta = gray[inside[:, 1], inside[:, 0]].astype(np.float32) - gray[outside[:, 1], outside[:, 0]]
        positive = float(np.mean(delta >= 8))
        positive_contrasts.append(positive)
        contrasts.append(max(positive, float(np.mean(delta <= -8))))
    if min(supports) < 0.72 or min(contrasts) < 0.35 or np.mean(contrasts) < 0.65:
        return 0.0
    centering = 1 - min(float(np.linalg.norm(center - [width / 2, height / 2])) / np.hypot(width, height), 1)
    return float(0.40 * np.mean(supports) + 0.30 * min(supports)
                 + 0.15 * np.mean(contrasts) + 0.03 * np.mean(positive_contrasts)
                 + 0.08 * np.sqrt(area_ratio) + 0.04 * centering)


def find_supported_surface(image: np.ndarray, edge_image: np.ndarray, maps=None, hough_candidates=None):
    """Find closed inner frames too, including a display inside a laptop outline."""
    if not np.any(edge_image):
        return None, 0.0
    maps = _surface_maps(image, edge_image) if maps is None else maps
    edges = np.clip(edge_image, 0, 255).astype(np.uint8)
    masks = [edges, cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)),
             cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))]
    best, best_score, seen = None, 0.0, []
    min_area = edge_image.size * 0.08
    for mask in masks:
        # RETR_EXTERNAL alone discards the screen when its bezel is connected
        # to the keyboard. Keep inner closed contours without filling their hull.
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:100]:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                break
            perimeter = cv2.arcLength(contour, True)
            for epsilon in (0.01, 0.015, 0.02, 0.03):
                quad = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(quad) != 4 or not cv2.isContourConvex(quad):
                    continue
                corners = order_points(quad.reshape(4, 2))
                quad_area = float(cv2.contourArea(corners))
                if quad_area <= 0 or min(area, quad_area) / max(area, quad_area) < 0.92:
                    continue
                if any(np.max(np.linalg.norm(corners - old, axis=1)) < 2 for old in seen):
                    break
                seen.append(corners)
                score = _surface_quality(corners, maps)
                if score > best_score:
                    best, best_score = corners, score
                break
    if best_score < 0.94:
        if hough_candidates is None:
            def score_quad(quad):
                score = _surface_quality(quad, maps)
                return score if score >= 0.82 else -1.0
            candidate = find_document_from_hough(edge_image, scorer=score_quad)
            hough_candidates = [] if candidate is None else [candidate]
        for candidate in hough_candidates:
            score = _surface_quality(candidate, maps)
            if score > best_score:
                best, best_score = candidate, score
    return (best, best_score) if best_score >= 0.82 else (None, 0.0)


def _enclosing_paper(image: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> bool:
    """Preserve an established sheet when a printed panel forms an inner frame."""
    outer_area, inner_area = cv2.contourArea(outer), cv2.contourArea(inner)
    if outer_area <= 0 or inner_area >= outer_area * 0.85:
        return False
    intersection, _ = cv2.intersectConvexConvex(outer.astype(np.float32), inner.astype(np.float32))
    if intersection / inner_area < 0.95:
        return False
    gray = np.clip(image, 0, 255).astype(np.uint8) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if _paper_boundary_contrast(gray, outer) < 2:
        return False
    matrix = cv2.getPerspectiveTransform(outer.astype(np.float32),
                                         np.float32([[0, 0], [63, 0], [63, 63], [0, 63]]))
    sample = cv2.warpPerspective(image, matrix, (64, 64))
    gray = sample if sample.ndim == 2 else cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    border = np.concatenate((gray[:6].ravel(), gray[-6:].ravel(),
                             gray[6:-6, :6].ravel(), gray[6:-6, -6:].ravel()))
    return float(np.median(border)) >= 185


def detect_document(
    image: np.ndarray,
    edge_image: np.ndarray,
    preview_only: bool = False,
) -> dict[str, object]:
    """
    Phát hiện tài liệu bằng nhiều chiến lược.

    So sánh contour/Hough với vùng giấy ngoài để tránh chọn bảng in bên trong.
    Xét thêm các khung kín bên trong với bốn cạnh liên tục, như màn hình.
    preview_only bỏ qua bước tách vùng chậm; gợi ý camera vẫn dùng cùng
    cách chấm điểm khung và không thay tọa độ của lần quét ảnh gốc.
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

    hough_candidates = []
    hough_corners = None if preview_only else find_document_from_hough(
        edge_image, candidate_quads=hough_candidates,
    )

    if hough_corners is not None and (
        corners is None
        or score_document_corners(edge_image, hough_corners)
        > score_document_corners(edge_image, corners) + 0.03
    ):
        corners = hough_corners
        method = "hough"


    # ==========================================
    # CÁCH 3: KIỂM TRA KHUNG GIẤY NGOÀI
    # ==========================================

    paper_corners = find_document_from_brightness(image)
    if paper_corners is not None:
        use_paper = corners is None
        if corners is not None:
            paper_area = abs(float(cv2.contourArea(paper_corners)))
            current_area = abs(float(cv2.contourArea(corners)))
            intersection, _ = cv2.intersectConvexConvex(paper_corners, corners.astype(np.float32))
            # A large enclosing paper outline beats a strong printed table.
            use_paper = (current_area > 0 and paper_area > current_area * 1.12
                         and intersection / current_area > 0.90)
            # Also remove background included by a loose bounding rectangle.
            use_paper = use_paper or (paper_area > 0 and current_area > paper_area * 1.12
                                     and paper_area / current_area > 0.75
                                     and intersection / paper_area > 0.95)
        if use_paper:
            corners = paper_corners
            method = "brightness"
    height, width = image.shape[:2]
    full_frame = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    if corners is not None and np.array_equal(corners, full_frame):
        method = "full-frame"
    if method != "full-frame" and np.any(edge_image):
        maps = _surface_maps(image, edge_image)
        surface, quality = find_supported_surface(image, edge_image, maps,
                                                   None if preview_only else hough_candidates)
        if surface is not None:
            current_quality = 0.0 if corners is None else _surface_quality(corners, maps)
            large_enough = corners is None or cv2.contourArea(surface) >= cv2.contourArea(corners) * 0.40
            protected_paper = corners is not None and _enclosing_paper(image, corners, surface)
            if large_enough and not protected_paper and quality > current_quality + 0.08:
                corners, method = surface, "surface"
    if corners is None:
        method = "none"
    elif method not in ("surface", "full-frame") and not preview_only:
        refined = refine_document_region(image, corners)
        if refined is not None:
            corners, method = refined, "paper-region"
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
