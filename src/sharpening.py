import numpy as np

from src.convolution import convolution2d
from src.log_detector import gaussian_kernel
from src.sobel import to_grayscale


def clip_uint8(image: np.ndarray) -> np.ndarray:
    """
    Giới hạn giá trị ảnh về miền [0, 255]
    và chuyển sang kiểu uint8.

    Args:
        image: Ảnh đầu vào kiểu float.

    Returns:
        Ảnh uint8 trong miền [0, 255].
    """

    return np.clip(
        image,
        0,
        255
    ).astype(np.uint8)


def unsharp_mask(
    image: np.ndarray,
    gaussian_size: int = 5,
    sigma: float = 1.0,
    amount: float = 1.0
) -> dict[str, np.ndarray]:
    """
    Làm nét ảnh bằng Unsharp Masking.

    Công thức:
        mask = f - f_blur
        g = f + amount * mask

    Khi amount = 1:
        Unsharp Masking chuẩn.

    Args:
        image:
            Ảnh màu BGR hoặc ảnh grayscale.

        gaussian_size:
            Kích thước Gaussian kernel.

        sigma:
            Độ lệch chuẩn Gaussian.

        amount:
            Cường độ làm nét.

    Returns:
        Dictionary chứa:
            gray
            blurred
            mask
            sharpened
    """

    # ==========================================
    # BƯỚC 1: GRAYSCALE
    # ==========================================

    gray = to_grayscale(
        image
    )


    # ==========================================
    # BƯỚC 2: TẠO GAUSSIAN KERNEL
    # ==========================================

    kernel = gaussian_kernel(
        size=gaussian_size,
        sigma=sigma
    )


    # ==========================================
    # BƯỚC 3: LÀM MỜ ẢNH
    # ==========================================

    blurred = convolution2d(
        gray,
        kernel
    )


    # ==========================================
    # BƯỚC 4: TẠO UNSHARP MASK
    # ==========================================

    mask = (
        gray
        - blurred
    )


    # ==========================================
    # BƯỚC 5: CỘNG MASK VÀO ẢNH GỐC
    # ==========================================

    sharpened_float = (
        gray
        + amount * mask
    )


    # ==========================================
    # BƯỚC 6: ĐƯA VỀ [0, 255]
    # ==========================================

    sharpened = clip_uint8(
        sharpened_float
    )


    return {
        "gray": gray,
        "blurred": blurred,
        "mask": mask,
        "sharpened": sharpened
    }


def high_boost(
    image: np.ndarray,
    k: float = 1.5,
    gaussian_size: int = 5,
    sigma: float = 1.0
) -> dict[str, np.ndarray]:
    """
    Làm nét ảnh bằng High-Boost Filtering.

    Công thức:
        mask = f - f_blur
        g = f + k * mask

    Với:
        k > 1

    Args:
        image:
            Ảnh đầu vào.

        k:
            Hệ số khuếch đại chi tiết.

        gaussian_size:
            Kích thước Gaussian kernel.

        sigma:
            Sigma của Gaussian.

    Returns:
        Dictionary chứa:
            gray
            blurred
            mask
            sharpened
    """

    if k <= 1:
        raise ValueError(
            "High-Boost yêu cầu k > 1."
        )

    return unsharp_mask(
        image=image,
        gaussian_size=gaussian_size,
        sigma=sigma,
        amount=k
    )


def sharpen_image(
    image: np.ndarray,
    method: str = "unsharp",
    strength: float = 1.0
) -> np.ndarray:
    """
    Hàm tiện ích để dùng trong Flask.

    Args:
        image:
            Ảnh đầu vào.

        method:
            'unsharp' hoặc 'highboost'.

        strength:
            Hệ số làm nét.

    Returns:
        Ảnh sau khi làm nét.
    """

    method = method.lower()

    if method == "unsharp":

        result = unsharp_mask(
            image=image,
            amount=strength
        )

    elif method == "highboost":

        if strength <= 1:
            strength = 1.5

        result = high_boost(
            image=image,
            k=strength
        )

    else:
        raise ValueError(
            "method phải là 'unsharp' hoặc 'highboost'."
        )

    return result["sharpened"]