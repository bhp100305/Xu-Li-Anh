from __future__ import annotations

import time

import cv2
import numpy as np

from src.canny import canny
from src.laplacian import laplacian
from src.log_detector import log_detector
from src.prewitt import prewitt
from src.sobel import sobel
from src.sobel import to_grayscale


def resize_for_comparison(
    image: np.ndarray,
    max_width: int = 700,
) -> np.ndarray:
    """
    Thu nhỏ ảnh để quá trình so sánh thuật toán chạy nhanh hơn.

    Args:
        image:
            Ảnh BGR đầu vào.

        max_width:
            Chiều rộng tối đa.

    Returns:
        Ảnh sau khi resize.
    """

    if image is None:
        raise ValueError(
            "Ảnh đầu vào không được None."
        )

    if image.size == 0:
        raise ValueError(
            "Ảnh đầu vào rỗng."
        )

    height, width = image.shape[:2]

    if width <= max_width:
        return image.copy()

    ratio = max_width / width

    new_height = max(
        1,
        int(round(height * ratio))
    )

    resized = cv2.resize(
        image,
        (max_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized


def normalize_edge(
    image: np.ndarray,
) -> np.ndarray:
    """
    Chuẩn hóa ảnh về uint8 [0, 255].

    Args:
        image:
            Ma trận ảnh.

    Returns:
        Ảnh uint8.
    """

    image_float = image.astype(
        np.float32
    )

    minimum = float(
        np.min(image_float)
    )

    maximum = float(
        np.max(image_float)
    )

    if maximum == minimum:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    normalized = (
        (
            image_float
            - minimum
        )
        / (
            maximum
            - minimum
        )
        * 255.0
    )

    return normalized.astype(
        np.uint8
    )


def run_with_timer(
    function,
    *args,
    **kwargs,
) -> tuple[object, float]:
    """
    Chạy một hàm và đo thời gian thực thi.

    Args:
        function:
            Hàm cần chạy.

        *args:
            Tham số positional.

        **kwargs:
            Tham số keyword.

    Returns:
        Tuple gồm kết quả và thời gian tính bằng giây.
    """

    start_time = time.perf_counter()

    result = function(
        *args,
        **kwargs,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return result, elapsed


def compare_algorithms(
    image: np.ndarray,
    max_width: int = 700,
) -> dict[str, object]:
    """
    So sánh các thuật toán phát hiện biên.

    Các phương pháp:
        - Sobel tự cài
        - Prewitt tự cài
        - Laplacian tự cài
        - LoG tự cài
        - Canny tự cài
        - OpenCV Canny

    Args:
        image:
            Ảnh BGR đầu vào.

        max_width:
            Chiều rộng tối đa để xử lý.

    Returns:
        Dictionary chứa ảnh kết quả
        và thời gian xử lý từng thuật toán.
    """

    # ==========================================
    # RESIZE
    # ==========================================

    working_image = resize_for_comparison(
        image=image,
        max_width=max_width,
    )

    gray = to_grayscale(
        working_image
    )

    gray_uint8 = np.clip(
        gray,
        0,
        255,
    ).astype(
        np.uint8
    )


    # ==========================================
    # SOBEL
    # ==========================================

    print(
        "[1/6] Sobel..."
    )

    sobel_result, sobel_time = run_with_timer(
        sobel,
        working_image,
    )

    sobel_edge = sobel_result[
        "edge"
    ]


    # ==========================================
    # PREWITT
    # ==========================================

    print(
        "[2/6] Prewitt..."
    )

    prewitt_result, prewitt_time = (
        run_with_timer(
            prewitt,
            working_image,
        )
    )

    prewitt_edge = prewitt_result[
        "edge"
    ]


    # ==========================================
    # LAPLACIAN
    # ==========================================

    print(
        "[3/6] Laplacian..."
    )

    laplacian_result, laplacian_time = (
        run_with_timer(
            laplacian,
            working_image,
            kernel_type="4-neighbor",
        )
    )

    laplacian_edge = laplacian_result[
        "edge"
    ]


    # ==========================================
    # LoG
    # ==========================================

    print(
        "[4/6] LoG..."
    )

    log_result, log_time = run_with_timer(
        log_detector,
        working_image,
        gaussian_size=5,
        sigma=1.2,
        zero_threshold=15.0,
    )

    log_edge = log_result[
        "edge"
    ]


    # ==========================================
    # CANNY TỰ CÀI
    # ==========================================

    print(
        "[5/6] Canny tự cài..."
    )

    canny_result, canny_time = (
        run_with_timer(
            canny,
            working_image,
            gaussian_size=5,
            gaussian_sigma=1.4,
            threshold_sigma=0.33,
        )
    )

    canny_edge = canny_result[
        "edge"
    ]


    # ==========================================
    # OPENCV CANNY
    # ==========================================

    print(
        "[6/6] OpenCV Canny..."
    )

    low_threshold = int(
        canny_result[
            "low_threshold"
        ]
    )

    high_threshold = int(
        canny_result[
            "high_threshold"
        ]
    )

    opencv_start = time.perf_counter()

    opencv_edge = cv2.Canny(
        gray_uint8,
        low_threshold,
        high_threshold,
    )

    opencv_time = (
        time.perf_counter()
        - opencv_start
    )


    # ==========================================
    # KẾT QUẢ
    # ==========================================

    results = {
        "sobel": {
            "name": "Sobel",
            "edge": sobel_edge,
            "time": sobel_time,
        },

        "prewitt": {
            "name": "Prewitt",
            "edge": prewitt_edge,
            "time": prewitt_time,
        },

        "laplacian": {
            "name": "Laplacian",
            "edge": laplacian_edge,
            "time": laplacian_time,
        },

        "log": {
            "name": "LoG",
            "edge": log_edge,
            "time": log_time,
        },

        "canny": {
            "name": "Canny tự cài",
            "edge": canny_edge,
            "time": canny_time,
        },

        "opencv_canny": {
            "name": "OpenCV Canny",
            "edge": opencv_edge,
            "time": opencv_time,
        },
    }


    return {
        "original": working_image,
        "gray": gray_uint8,
        "results": results,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "width": working_image.shape[1],
        "height": working_image.shape[0],
    }