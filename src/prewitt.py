import numpy as np

from src.convolution import convolution2d
from src.sobel import to_grayscale, normalize_to_uint8


def prewitt(image):
    """
    Phát hiện biên bằng toán tử Prewitt tự cài đặt.

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào, có thể là ảnh màu hoặc grayscale.

    Returns
    -------
    result : dict
        Chứa các kết quả trung gian:
        gray
        gx
        gy
        magnitude
        direction
        edge
    """

    # ==========================================
    # BƯỚC 1: Chuyển ảnh về grayscale
    # ==========================================

    gray = to_grayscale(image)


    # ==========================================
    # BƯỚC 2: Kernel Prewitt theo trục X
    # ==========================================

    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)


    # ==========================================
    # BƯỚC 3: Kernel Prewitt theo trục Y
    # ==========================================

    kernel_y = np.array([
        [-1, -1, -1],
        [0,   0,  0],
        [1,   1,  1]
    ], dtype=np.float32)


    # ==========================================
    # BƯỚC 4: Tính gradient X và Y
    # ==========================================

    gx = convolution2d(
        gray,
        kernel_x
    )

    gy = convolution2d(
        gray,
        kernel_y
    )


    # ==========================================
    # BƯỚC 5: Tính độ lớn gradient
    # ==========================================

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )


    # ==========================================
    # BƯỚC 6: Tính hướng gradient
    # ==========================================

    direction = np.arctan2(
        gy,
        gx
    )

    direction = np.degrees(
        direction
    )


    # ==========================================
    # BƯỚC 7: Chuẩn hóa về [0,255]
    # ==========================================

    edge = normalize_to_uint8(
        magnitude
    )


    return {
        "gray": gray,
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "direction": direction,
        "edge": edge
    }


def prewitt_edge(image):
    """
    Hàm rút gọn chỉ trả về ảnh biên Prewitt.
    """

    result = prewitt(image)

    return result["edge"]