import cv2

from src.canny import canny_edge
from src.document_detector import detect_document


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
# TẠO EDGE BẰNG CANNY TỰ VIẾT
# ==========================================

edge = canny_edge(
    image,
    gaussian_size=5,
    gaussian_sigma=1.4,
    threshold_sigma=0.33
)


# ==========================================
# PHÁT HIỆN DOCUMENT
# ==========================================

result = detect_document(
    image,
    edge
)

corners = result[
    "corners"
]

preview = result[
    "preview"
]


# ==========================================
# IN KẾT QUẢ
# ==========================================

if corners is None:

    print(
        "Không tìm thấy trang giấy."
    )

else:

    print(
        "Phát hiện trang giấy thành công."
    )

    print(
        "4 góc:"
    )

    print(
        corners
    )


# ==========================================
# HIỂN THỊ
# ==========================================

cv2.imshow(
    "1 - Anh goc",
    image
)

cv2.imshow(
    "2 - Canny Edge",
    edge
)

cv2.imshow(
    "3 - Document Detection",
    preview
)


# ==========================================
# LƯU
# ==========================================

cv2.imwrite(
    "document_detected.jpg",
    preview
)


print(
    "Đã lưu: document_detected.jpg"
)

cv2.waitKey(3000)
cv2.destroyAllWindows()