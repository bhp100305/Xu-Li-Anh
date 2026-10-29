from __future__ import annotations

import cv2
import numpy as np

from src.document_detector import order_points


def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Tính khoảng cách Euclid giữa hai điểm.

    Args:
        point1: Điểm thứ nhất [x, y].
        point2: Điểm thứ hai [x, y].

    Returns:
        Khoảng cách giữa hai điểm.
    """

    return float(
        np.sqrt(
            (point1[0] - point2[0]) ** 2
            + (point1[1] - point2[1]) ** 2
        )
    )


def get_output_size(
    corners: np.ndarray
) -> tuple[int, int]:
    """
    Tính chiều rộng và chiều cao ảnh sau khi nắn phối cảnh.

    Args:
        corners:
            4 góc theo thứ tự:
            top-left
            top-right
            bottom-right
            bottom-left

    Returns:
        Tuple:
            (width, height)
    """

    if corners.shape != (4, 2):
        raise ValueError(
            "corners phải có kích thước (4, 2)."
        )

    top_left = corners[0]
    top_right = corners[1]
    bottom_right = corners[2]
    bottom_left = corners[3]

    # ==========================================
    # TÍNH CHIỀU RỘNG
    # ==========================================

    width_top = distance(
        top_left,
        top_right
    )

    width_bottom = distance(
        bottom_left,
        bottom_right
    )

    width = int(
        max(
            width_top,
            width_bottom
        )
    )


    # ==========================================
    # TÍNH CHIỀU CAO
    # ==========================================

    height_left = distance(
        top_left,
        bottom_left
    )

    height_right = distance(
        top_right,
        bottom_right
    )

    height = int(
        max(
            height_left,
            height_right
        )
    )

    if width <= 0 or height <= 0:
        raise ValueError(
            "Kích thước ảnh sau transform không hợp lệ."
        )

    return width, height


def perspective_transform(
    image: np.ndarray,
    corners: np.ndarray
) -> np.ndarray:
    """
    Nắn phối cảnh ảnh tài liệu.

    Args:
        image:
            Ảnh BGR đầu vào.

        corners:
            4 góc của tài liệu.

    Returns:
        Ảnh tài liệu sau khi nắn phối cảnh.
    """

    if image is None:
        raise ValueError(
            "Ảnh đầu vào không được None."
        )

    if corners.shape != (4, 2):
        raise ValueError(
            "corners phải gồm đúng 4 điểm."
        )


    # ==========================================
    # BƯỚC 1: SẮP XẾP 4 GÓC
    # ==========================================

    ordered = order_points(
        corners
    )


    # ==========================================
    # BƯỚC 2: TÍNH KÍCH THƯỚC ẢNH ĐẦU RA
    # ==========================================

    width, height = get_output_size(
        ordered
    )


    # ==========================================
    # BƯỚC 3: TỌA ĐỘ ĐÍCH
    # ==========================================

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype=np.float32
    )


    # ==========================================
    # BƯỚC 4: TÍNH MA TRẬN PERSPECTIVE
    # ==========================================

    matrix = cv2.getPerspectiveTransform(
        ordered.astype(np.float32),
        destination
    )


    # ==========================================
    # BƯỚC 5: BIẾN ĐỔI ẢNH
    # ==========================================

    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    return warped


def scan_perspective(
    image: np.ndarray,
    corners: np.ndarray
) -> dict[str, object]:
    """
    Hàm tổng hợp cho bước nắn phối cảnh.

    Args:
        image:
            Ảnh gốc.

        corners:
            4 góc của tài liệu.

    Returns:
        Dictionary gồm:
            corners
            width
            height
            warped
    """

    ordered = order_points(
        corners
    )

    width, height = get_output_size(
        ordered
    )

    warped = perspective_transform(
        image,
        ordered
    )

    return {
        "corners": ordered,
        "width": width,
        "height": height,
        "warped": warped
    }