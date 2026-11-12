import numpy as np
from collections import deque

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

    current = magnitude[1:-1, 1:-1]
    inner_angle = angle[1:-1, 1:-1]
    # So với hai điểm theo hướng gradient; chỉ giữ cực đại để làm mảnh biên.
    # Trục y của ảnh hướng xuống nên cần chọn đúng cặp điểm chéo.
    directions = (
        (((inner_angle >= 0) & (inner_angle < 22.5))
         | ((inner_angle >= 157.5) & (inner_angle <= 180)),
         magnitude[1:-1, :-2], magnitude[1:-1, 2:]),
        ((inner_angle >= 22.5) & (inner_angle < 67.5),
         magnitude[:-2, :-2], magnitude[2:, 2:]),
        ((inner_angle >= 67.5) & (inner_angle < 112.5),
         magnitude[:-2, 1:-1], magnitude[2:, 1:-1]),
        ((inner_angle >= 112.5) & (inner_angle < 157.5),
         magnitude[:-2, 2:], magnitude[2:, :-2]),
    )
    keep = current >= 0
    for mask, neighbor1, neighbor2 in directions:
        keep[mask] = (current[mask] >= neighbor1[mask]) & (current[mask] >= neighbor2[mask])
    output[1:-1, 1:-1] = np.where(keep, current, 0)

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

    # Dùng trung vị độ sáng để tự chọn hai ngưỡng Canny.
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

    # Chia điểm ảnh thành biên mạnh, biên yếu và nền theo hai ngưỡng.
    strong_y, strong_x = np.where(
        (image > 0) & (image >= high_threshold)
    )

    weak_y, weak_x = np.where(
        (image >= low_threshold)
        & (image > 0)
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

    strong = result == strong_value
    weak = result == weak_value
    # The original loop never promotes border pixels, even when connected.
    if height > 2 and width > 2:
        weak[0, :] = weak[-1, :] = False
        weak[:, 0] = weak[:, -1] = False
        adjacent = np.zeros((height - 2, width - 2), dtype=bool)
        for dy in range(3):
            for dx in range(3):
                if dy != 1 or dx != 1:
                    adjacent |= strong[dy:dy + height - 2, dx:dx + width - 2]
        seeds = np.zeros_like(strong)
        seeds[1:-1, 1:-1] = weak[1:-1, 1:-1] & adjacent
        pending = deque(np.flatnonzero(seeds).tolist())
        strong_flat = strong.reshape(-1)
        weak_flat = weak.reshape(-1)
        strong_flat[seeds.reshape(-1)] = True
        weak_flat[seeds.reshape(-1)] = False
        offsets = (-width - 1, -width, -width + 1, -1, 1, width - 1, width, width + 1)
        # Duyệt 8 láng giềng bằng hàng đợi, giữ biên yếu nối với biên mạnh.
        while pending:
            current = pending.popleft()
            for offset in offsets:
                neighbor = current + offset
                if weak_flat[neighbor]:
                    weak_flat[neighbor] = False
                    strong_flat[neighbor] = True
                    pending.append(neighbor)
    result[strong] = strong_value
    result[~strong] = 0

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
