import numpy as np


def convolution2d(image, kernel, padding="reflect"):
    """
    Thực hiện phép tích chập 2D giữa ảnh xám và kernel.

    Parameters
    ----------
    image : np.ndarray
        Ảnh xám 2 chiều, kích thước H x W.

    kernel : np.ndarray
        Ma trận kernel, ví dụ 3x3 hoặc 5x5.

    padding : str
        Phương pháp xử lý biên.
        Mặc định là "reflect".

    Returns
    -------
    output : np.ndarray
        Ảnh kết quả kiểu float32,
        có cùng kích thước với ảnh đầu vào.
    """

    # Kiểm tra ảnh phải là ảnh xám 2 chiều
    if image.ndim != 2:
        raise ValueError(
            "convolution2d chỉ nhận ảnh xám 2 chiều."
        )

    # Kiểm tra kernel phải là ma trận 2 chiều
    if kernel.ndim != 2:
        raise ValueError(
            "Kernel phải là ma trận 2 chiều."
        )

    # Kernel nên có kích thước lẻ
    kh, kw = kernel.shape

    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(
            "Kernel phải có kích thước lẻ, ví dụ 3x3 hoặc 5x5."
        )

    # Chuyển sang float32 để tránh tràn số uint8
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    # Tính số pixel cần padding
    pad_h = kh // 2
    pad_w = kw // 2

    # Padding ảnh
    padded = np.pad(
        image,
        (
            (pad_h, pad_h),
            (pad_w, pad_w)
        ),
        mode=padding
    )

    # Tạo ảnh đầu ra
    output = np.zeros(
        image.shape,
        dtype=np.float32
    )

    height, width = image.shape

    # Quét qua từng pixel
    for y in range(height):
        for x in range(width):

            # Lấy vùng ảnh có kích thước bằng kernel
            region = padded[
                y:y + kh,
                x:x + kw
            ]

            # Nhân từng phần tử và cộng lại
            output[y, x] = np.sum(
                region * kernel
            )

    return output