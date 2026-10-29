import cv2

from src.sharpening import (
    unsharp_mask,
    high_boost
)


IMAGE_PATH = "test_image.jpg"


# ==========================================
# ĐỌC ẢNH
# ==========================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Không tìm thấy ảnh: {IMAGE_PATH}"
    )


# ==========================================
# UNSHARP MASKING
# ==========================================

unsharp_result = unsharp_mask(
    image=image,
    gaussian_size=5,
    sigma=1.0,
    amount=1.0
)


# ==========================================
# HIGH BOOST
# ==========================================

highboost_result = high_boost(
    image=image,
    k=1.8,
    gaussian_size=5,
    sigma=1.0
)


gray = unsharp_result["gray"]

blurred = unsharp_result["blurred"]

mask = unsharp_result["mask"]

unsharp_image = unsharp_result[
    "sharpened"
]

highboost_image = highboost_result[
    "sharpened"
]


# ==========================================
# CHUYỂN MASK SANG DẠNG HIỂN THỊ
# ==========================================

mask_display = cv2.normalize(
    mask,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype("uint8")


# ==========================================
# HIỂN THỊ
# ==========================================

cv2.imshow(
    "1 - Anh goc",
    image
)

cv2.imshow(
    "2 - Grayscale",
    gray.astype("uint8")
)

cv2.imshow(
    "3 - Gaussian Blur",
    blurred.astype("uint8")
)

cv2.imshow(
    "4 - Unsharp Mask",
    mask_display
)

cv2.imshow(
    "5 - Unsharp Result",
    unsharp_image
)

cv2.imshow(
    "6 - High Boost Result",
    highboost_image
)


# ==========================================
# LƯU FILE
# ==========================================

cv2.imwrite(
    "unsharp_result.jpg",
    unsharp_image
)

cv2.imwrite(
    "highboost_result.jpg",
    highboost_image
)


print("Sharpening chạy thành công.")

print(
    "Kích thước ảnh:",
    image.shape
)

print(
    "Đã lưu: unsharp_result.jpg"
)

print(
    "Đã lưu: highboost_result.jpg"
)


cv2.waitKey(0)

cv2.destroyAllWindows()