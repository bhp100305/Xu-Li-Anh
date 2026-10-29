import numpy as np

from src.convolution import convolution2d
from src.sobel import to_grayscale
from src.log_detector import gaussian_kernel


def non_maximum_suppression(magnitude, direction):
    """
    Làm mảnh biên bằng Non-Maximum Suppression.

    Parameters
    ----------
    magnitude : np.ndarray
        Độ lớn gradient.

    direction : np.ndarray
        Hướng gradient theo độ.

    Returns
    -------
    output : np.ndarray
        Ảnh gradient sau khi làm mảnh.
    """

    height, width = magnitude.shape

    output = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # Đưa góc về miền [0, 180)
    angle = direction.copy()

    angle[angle < 0] += 180

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            current = magnitude[y, x]

            neighbor1 = 0
            neighbor2 = 0

            # Hướng 0 độ
            if (
                0 <= angle[y, x] < 22.5
                or 157.5 <= angle[y, x] <= 180
            ):
                neighbor1 = magnitude[y, x - 1]
                neighbor2 = magnitude[y, x + 1]

            # Hướng 45 độ
            elif 22.5 <= angle[y, x] < 67.5:
                neighbor1 = magnitude[y - 1, x + 1]
                neighbor2 = magnitude[y + 1, x - 1]

            # Hướng 90 độ
            elif 67.5 <= angle[y, x] < 112.5:
                neighbor1 = magnitude[y - 1, x]
                neighbor2 = magnitude[y + 1, x]

            # Hướng 135 độ
            elif 112.5 <= angle[y, x] < 157.5:
                neighbor1 = magnitude[y - 1, x - 1]
                neighbor2 = magnitude[y + 1, x + 1]

            # Chỉ giữ local maximum
            if (
                current >= neighbor1
                and current >= neighbor2
            ):
                output[y, x] = current

    return output


def automatic_thresholds(gray, sigma=0.33):
    """
    Tự động tính ngưỡng thấp và cao dựa trên median ảnh.

    Parameters
    ----------
    gray : np.ndarray
        Ảnh grayscale.

    sigma : float
        Hệ số điều chỉnh ngưỡng.

    Returns
    -------
    low_threshold : float
    high_threshold : float
    """

    median_value = np.median(gray)

    low_threshold = max(
        0,
        (1.0 - sigma) * median_value
    )

    high_threshold = min(
        255,
        (1.0 + sigma) * median_value
    )

    return (
        float(low_threshold),
        float(high_threshold)
    )


def double_threshold(
    image,
    low_threshold,
    high_threshold
):
    """
    Phân loại pixel thành:
    strong edge
    weak edge
    non-edge.
    """

    strong_value = 255
    weak_value = 75

    result = np.zeros(
        image.shape,
        dtype=np.uint8
    )

    strong_y, strong_x = np.where(
        image >= high_threshold
    )

    weak_y, weak_x = np.where(
        (image >= low_threshold)
        & (image < high_threshold)
    )

    result[
        strong_y,
        strong_x
    ] = strong_value

    result[
        weak_y,
        weak_x
    ] = weak_value

    return (
        result,
        weak_value,
        strong_value
    )


def hysteresis(
    image,
    weak_value=75,
    strong_value=255
):
    """
    Giữ weak edge nếu nó nối với strong edge.

    Parameters
    ----------
    image : np.ndarray
        Ảnh sau double threshold.

    Returns
    -------
    result : np.ndarray
        Ảnh biên cuối cùng.
    """

    result = image.copy()

    height, width = result.shape

    changed = True

    while changed:
        changed = False

        for y in range(1, height - 1):
            for x in range(1, width - 1):

                if result[y, x] == weak_value:

                    region = result[
                        y - 1:y + 2,
                        x - 1:x + 2
                    ]

                    if np.any(
                        region == strong_value
                    ):
                        result[y, x] = strong_value
                        changed = True

    # Các weak edge còn lại bị loại
    result[
        result != strong_value
    ] = 0

    return result


def canny(
    image,
    gaussian_size=5,
    gaussian_sigma=1.4,
    threshold_sigma=0.33
):
    """
    Thuật toán Canny tự cài đặt.

    Pipeline:
    1. Gaussian smoothing
    2. Gradient magnitude + direction
    3. Non-Maximum Suppression
    4. Double threshold + hysteresis

    Parameters
    ----------
    image : np.ndarray
        Ảnh màu hoặc grayscale.

    gaussian_size : int
        Kích thước Gaussian kernel.

    gaussian_sigma : float
        Sigma Gaussian.

    threshold_sigma : float
        Hệ số tính ngưỡng tự động.

    Returns
    -------
    result : dict
        gray
        blurred
        gx
        gy
        magnitude
        direction
        nms
        threshold
        edge
        low_threshold
        high_threshold
    """

    # ==========================================
    # BƯỚC 1: GRAYSCALE
    # ==========================================

    gray = to_grayscale(
        image
    )


    # ==========================================
    # BƯỚC 2: GAUSSIAN SMOOTHING
    # ==========================================

    g_kernel = gaussian_kernel(
        size=gaussian_size,
        sigma=gaussian_sigma
    )

    blurred = convolution2d(
        gray,
        g_kernel
    )


    # ==========================================
    # BƯỚC 3: SOBEL GRADIENT
    # ==========================================

    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ], dtype=np.float32)


    gx = convolution2d(
        blurred,
        kernel_x
    )

    gy = convolution2d(
        blurred,
        kernel_y
    )


    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )


    direction = np.degrees(
        np.arctan2(
            gy,
            gx
        )
    )


    # ==========================================
    # BƯỚC 4: NON-MAXIMUM SUPPRESSION
    # ==========================================

    nms = non_maximum_suppression(
        magnitude,
        direction
    )


    # ==========================================
    # BƯỚC 5: NGƯỠNG TỰ ĐỘNG
    # ==========================================

    low_threshold, high_threshold = (
        automatic_thresholds(
            gray,
            sigma=threshold_sigma
        )
    )


    # ==========================================
    # BƯỚC 6: DOUBLE THRESHOLD
    # ==========================================

    threshold_image, weak, strong = (
        double_threshold(
            nms,
            low_threshold,
            high_threshold
        )
    )


    # ==========================================
    # BƯỚC 7: HYSTERESIS
    # ==========================================

    edge = hysteresis(
        threshold_image,
        weak_value=weak,
        strong_value=strong
    )


    return {
        "gray": gray,
        "blurred": blurred,
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "direction": direction,
        "nms": nms,
        "threshold": threshold_image,
        "edge": edge,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold
    }


def canny_edge(
    image,
    gaussian_size=5,
    gaussian_sigma=1.4,
    threshold_sigma=0.33
):
    """
    Hàm rút gọn chỉ trả về ảnh biên Canny.
    """

    result = canny(
        image,
        gaussian_size=gaussian_size,
        gaussian_sigma=gaussian_sigma,
        threshold_sigma=threshold_sigma
    )

    return result["edge"]