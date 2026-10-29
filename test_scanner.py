import time

import cv2

from src.scanner import scan_document


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


print("=" * 50)
print("BẮT ĐẦU SCAN TÀI LIỆU")
print("=" * 50)


start_time = time.perf_counter()


# ==========================================
# CHẠY PIPELINE
# ==========================================

try:

    result = scan_document(
        image=image,

        sharpening_method="unsharp",

        sharpening_strength=1.0,

        # Canny chỉ chạy trên ảnh rộng tối đa 800px
        detection_width=800,

        # Sharpen ảnh rộng tối đa 1000px
        output_width=1000,

        verbose=True
    )

except RuntimeError as error:

    print()
    print("Scan thất bại:")
    print(error)

    raise SystemExit(1)


end_time = time.perf_counter()

elapsed = end_time - start_time


# ==========================================
# LẤY KẾT QUẢ
# ==========================================

edge = result[
    "edge"
]

preview = result[
    "detection_preview"
]

warped = result[
    "warped"
]

scanned = result[
    "sharpened"
]


# ==========================================
# THÔNG TIN
# ==========================================

print()
print("=" * 50)
print("SCAN THÀNH CÔNG")
print("=" * 50)

print(
    "Low threshold:",
    result["low_threshold"]
)

print(
    "High threshold:",
    result["high_threshold"]
)

print()

print(
    "4 góc trên ảnh gốc:"
)

print(
    result["corners"]
)

print()

print(
    "Kích thước perspective gốc:",
    f'{result["width"]} x {result["height"]}'
)

print(
    "Kích thước ảnh cuối:",
    f'{result["final_width"]} x {result["final_height"]}'
)

print()

print(
    f"Thời gian xử lý: {elapsed:.2f} giây"
)


# ==========================================
# LƯU KẾT QUẢ
# ==========================================

cv2.imwrite(
    "scanner_1_edge.jpg",
    edge
)

cv2.imwrite(
    "scanner_2_detection.jpg",
    preview
)

cv2.imwrite(
    "scanner_3_perspective.jpg",
    warped
)

cv2.imwrite(
    "scanner_4_final.jpg",
    scanned
)


print()
print("Đã lưu:")

print(
    "scanner_1_edge.jpg"
)

print(
    "scanner_2_detection.jpg"
)

print(
    "scanner_3_perspective.jpg"
)

print(
    "scanner_4_final.jpg"
)