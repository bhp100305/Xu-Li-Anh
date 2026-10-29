import numpy as np

from src.convolution import convolution2d
from src.sobel import to_grayscale, normalize_to_uint8


def gaussian_kernel(size=5, sigma=1.0):
    """
    Tạo kernel Gaussian 2D.

    Parameters
    ----------
    size : int
        Kích thước kernel, phải là số lẻ.

    sigma : float
        Độ lệch chuẩn của Gaussian.

    Returns
    -------
    kernel : np.ndarray
        Kernel Gaussian đã chuẩn hóa.
    """

    if size % 2 == 0:
        raise ValueError(
            "Kích thước Gaussian kernel phải là số lẻ."
        )

    if sigma <= 0:
        raise ValueError(
            "sigma phải lớn hơn 0."
        )

    half = size // 2

    y, x = np.mgrid[
        -half:half + 1,
        -half:half + 1
    ]

    kernel = np.exp(
        -(x ** 2 + y ** 2)
        / (2 * sigma ** 2)
    )

    kernel = kernel / np.sum(kernel)

    return kernel.astype(np.float32)


def zero_crossing(response, threshold=0.0):
    """
    Phát hiện điểm zero-crossing trên đáp ứng Laplacian.

    Một pixel được xem là biên nếu trong vùng lân cận
    có sự đổi dấu từ âm sang dương hoặc ngược lại.

    Parameters
    ----------
    response : np.ndarray
        Kết quả sau Laplacian.

    threshold : float
        Ngưỡng tối thiểu cho độ chênh giữa min và max
        trong vùng lân cận.

    Returns
    -------
    edge : np.ndarray
        Ảnh nhị phân uint8.
    """

    height, width = response.shape

    edge = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            region = response[
                y - 1:y + 2,
                x - 1:x + 2
            ]

            min_value = np.min(region)
            max_value = np.max(region)

            # Có đổi dấu qua 0
            has_sign_change = (
                min_value < 0
                and max_value > 0
            )

            # Độ chênh đủ lớn để tránh nhiễu nhỏ
            strong_enough = (
                max_value - min_value
                >= threshold
            )

            if has_sign_change and strong_enough:
                edge[y, x] = 255

    return edge


def log_detector(
    image,
    gaussian_size=5,
    sigma=1.0,
    zero_threshold=10.0
):
    """
    Phát hiện biên bằng Laplacian of Gaussian (LoG).

    Pipeline:
    grayscale
    -> Gaussian smoothing
    -> Laplacian
    -> zero crossing

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào.

    gaussian_size : int
        Kích thước kernel Gaussian.

    sigma : float
        Sigma của Gaussian.

    zero_threshold : float
        Ngưỡng zero-crossing.

    Returns
    -------
    result : dict
        gray
        gaussian_kernel
        blurred
        laplacian_response
        laplacian_display
        edge
    """

    # ==========================================
    # BƯỚC 1: Chuyển grayscale
    # ==========================================

    gray = to_grayscale(image)


    # ==========================================
    # BƯỚC 2: Tạo Gaussian kernel
    # ==========================================

    g_kernel = gaussian_kernel(
        size=gaussian_size,
        sigma=sigma
    )


    # ==========================================
    # BƯỚC 3: Làm mượt ảnh
    # ==========================================

    blurred = convolution2d(
        gray,
        g_kernel
    )


    # ==========================================
    # BƯỚC 4: Laplacian
    # ==========================================

    laplacian_kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)


    laplacian_response = convolution2d(
        blurred,
        laplacian_kernel
    )


    # ==========================================
    # BƯỚC 5: Ảnh Laplacian để hiển thị
    # ==========================================

    laplacian_display = normalize_to_uint8(
        np.abs(laplacian_response)
    )


    # ==========================================
    # BƯỚC 6: Zero Crossing
    # ==========================================

    edge = zero_crossing(
        laplacian_response,
        threshold=zero_threshold
    )


    return {
        "gray": gray,
        "gaussian_kernel": g_kernel,
        "blurred": blurred,
        "laplacian_response": laplacian_response,
        "laplacian_display": laplacian_display,
        "edge": edge
    }


def log_edge(
    image,
    gaussian_size=5,
    sigma=1.0,
    zero_threshold=10.0
):
    """
    Hàm rút gọn chỉ trả về ảnh biên LoG.
    """

    result = log_detector(
        image,
        gaussian_size=gaussian_size,
        sigma=sigma,
        zero_threshold=zero_threshold
    )

    return result["edge"]