import numpy as np

from src.convolution import convolution2d


def to_grayscale(image):
    """
    Chuyển ảnh màu BGR sang ảnh xám bằng NumPy.

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào.
        Có thể là ảnh xám H x W
        hoặc ảnh màu H x W x 3.

    Returns
    -------
    gray : np.ndarray
        Ảnh xám float32.
    """

    # Nếu ảnh đã là grayscale
    if image.ndim == 2:
        return image.astype(np.float32)

    # Kiểm tra ảnh màu phải có 3 kênh
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Ảnh đầu vào phải là ảnh xám hoặc ảnh màu BGR 3 kênh."
        )

    # OpenCV đọc ảnh theo thứ tự BGR
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)

    # Công thức chuyển RGB -> grayscale
    gray = (
        0.114 * b
        + 0.587 * g
        + 0.299 * r
    )

    return gray


def normalize_to_uint8(image):
    """
    Chuẩn hóa ma trận float về miền [0, 255].

    Parameters
    ----------
    image : np.ndarray
        Ma trận ảnh kiểu float.

    Returns
    -------
    result : np.ndarray
        Ảnh uint8 trong khoảng [0, 255].
    """

    image = image.astype(np.float32)

    min_value = np.min(image)
    max_value = np.max(image)

    # Tránh chia cho 0 nếu ảnh chỉ có một giá trị
    if max_value == min_value:
        return np.zeros(
            image.shape,
            dtype=np.uint8
        )

    normalized = (
        (image - min_value)
        / (max_value - min_value)
        * 255.0
    )

    return normalized.astype(np.uint8)


def sobel(image):
    """
    Phát hiện biên bằng toán tử Sobel tự cài đặt.

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào, màu hoặc grayscale.

    Returns
    -------
    result : dict
        Chứa:
        gray       : ảnh xám
        gx         : gradient theo trục x
        gy         : gradient theo trục y
        magnitude  : độ lớn gradient
        direction  : hướng gradient theo độ
        edge       : ảnh biên uint8
    """

    # ==========================================
    # BƯỚC 1: Chuyển ảnh sang grayscale
    # ==========================================

    gray = to_grayscale(image)


    # ==========================================
    # BƯỚC 2: Kernel Sobel theo x
    # ==========================================

    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)


    # ==========================================
    # BƯỚC 3: Kernel Sobel theo y
    # ==========================================

    kernel_y = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ], dtype=np.float32)


    # ==========================================
    # BƯỚC 4: Tính gradient theo x và y
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
    # BƯỚC 5: Độ lớn gradient
    # ==========================================

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )


    # ==========================================
    # BƯỚC 6: Hướng gradient
    # ==========================================

    direction = np.arctan2(
        gy,
        gx
    )

    # Đổi radian sang độ
    direction = np.degrees(
        direction
    )


    # ==========================================
    # BƯỚC 7: Chuẩn hóa để hiển thị
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


def sobel_edge(image):
    """
    Hàm rút gọn chỉ trả về ảnh biên Sobel.

    Dùng cho Flask hoặc các module khác.
    """

    result = sobel(image)

    return result["edge"]