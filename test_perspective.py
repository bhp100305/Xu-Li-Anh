import cv2

from src.canny import canny_edge
from src.document_detector import detect_document
from src.perspective import scan_perspective


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
# TẠO EDGE
# ==========================================

edge = canny_edge(
    image,
    gaussian_size=5,
    gaussian_sigma=1.4,
    threshold_sigma=0.33
)


# ==========================================
# TÌM 4 GÓC TÀI LIỆU
# ==========================================

detection = detect_document(
    image,
    edge
)

corners = detection[
    "corners"
]

preview = detection[
    "preview"
]


if corners is None:
    raise RuntimeError(
        "Không tìm thấy 4 góc của tài liệu."
    )


# ==========================================
# NẮN PHỐI CẢNH
# ==========================================

result = scan_perspective(
    image,
    corners
)

warped = result[
    "warped"
]


# ==========================================
# IN THÔNG TIN
# ==========================================

print(
    "Perspective Transform chạy thành công."
)

print(
    "4 góc tài liệu:"
)

print(
    result["corners"]
)

print(
    "Chiều rộng:",
    result["width"]
)

print(
    "Chiều cao:",
    result["height"]
)


# ==========================================
# LƯU ẢNH
# ==========================================

cv2.imwrite(
    "perspective_detected.jpg",
    preview
)

cv2.imwrite(
    "perspective_result.jpg",
    warped
)


print(
    "Đã lưu: perspective_detected.jpg"
)

print(
    "Đã lưu: perspective_result.jpg"
)