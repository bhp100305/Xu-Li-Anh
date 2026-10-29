import numpy as np

from src.convolution import convolution2d
from src.sobel import to_grayscale, normalize_to_uint8


def laplacian(image, kernel_type="4-neighbor"):
    """
    Phát hiện biên bằng toán tử Laplacian tự cài đặt.

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào, màu hoặc grayscale.

    kernel_type : str
        "4-neighbor" hoặc "8-neighbor".

    Returns
    -------
    result : dict
        gray
        response
        edge
    """

    # ==========================================
    # BƯỚC 1: Chuyển ảnh sang grayscale
    # ==========================================

    gray = to_grayscale(image)


    # ==========================================
    # BƯỚC 2: Chọn kernel Laplacian
    # ==========================================

    if kernel_type == "4-neighbor":

        kernel = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=np.float32)

    elif kernel_type == "8-neighbor":

        kernel = np.array([
            [1,  1, 1],
            [1, -8, 1],
            [1,  1, 1]
        ], dtype=np.float32)

    else:
        raise ValueError(
            "kernel_type phải là '4-neighbor' hoặc '8-neighbor'."
        )


    # ==========================================
    # BƯỚC 3: Tích chập ảnh với kernel
    # ==========================================

    response = convolution2d(
        gray,
        kernel
    )


    # ==========================================
    # BƯỚC 4: Lấy trị tuyệt đối
    # ==========================================

    absolute_response = np.abs(
        response
    )


    # ==========================================
    # BƯỚC 5: Chuẩn hóa về [0, 255]
    # ==========================================

    edge = normalize_to_uint8(
        absolute_response
    )


    return {
        "gray": gray,
        "response": response,
        "edge": edge
    }


def laplacian_edge(image, kernel_type="4-neighbor"):
    """
    Hàm rút gọn chỉ trả về ảnh biên Laplacian.
    """

    result = laplacian(
        image,
        kernel_type
    )

    return result["edge"]