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
    # Lật kernel 180 độ để thực hiện đúng phép tích chập.
    kernel = np.flip(kernel, axis=(0, 1))

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

    if height == 0 or width == 0:
        return output

    # Trượt kernel trên ảnh; chia theo khối để hạn chế bộ nhớ tạm.
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    rows_per_block = max(1, (16 * 1024 * 1024) // (width * kh * kw * image.itemsize))
    for start in range(0, height, rows_per_block):
        stop = min(start + rows_per_block, height)
        products = np.empty((stop - start, width, kh, kw), dtype=np.float32)
        np.multiply(windows[start:stop], kernel, out=products)
        output[start:stop] = products.reshape(stop - start, width, kh * kw).sum(axis=-1)

    return output
