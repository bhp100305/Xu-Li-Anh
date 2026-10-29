import cv2

from src.sobel import sobel


# ==============================
# ĐƯỜNG DẪN ẢNH TEST
# ==============================

IMAGE_PATH = "test_image.jpg"


# ==============================
# ĐỌC ẢNH
# ==============================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Không tìm thấy ảnh: {IMAGE_PATH}"
    )


# ==============================
# CHẠY SOBEL TỰ CÀI ĐẶT
# ==============================

result = sobel(image)


# ==============================
# LẤY KẾT QUẢ
# ==============================

gray = result["gray"]
gx = result["gx"]
gy = result["gy"]
magnitude = result["magnitude"]
edge = result["edge"]


# ==============================
# CHUYỂN GX, GY SANG ẢNH HIỂN THỊ
# ==============================

gx_display = cv2.convertScaleAbs(gx)
gy_display = cv2.convertScaleAbs(gy)


# ==============================
# HIỂN THỊ
# ==============================

cv2.imshow(
    "1 - Anh goc",
    image
)

cv2.imshow(
    "2 - Grayscale",
    gray.astype("uint8")
)

cv2.imshow(
    "3 - Sobel Gx",
    gx_display
)

cv2.imshow(
    "4 - Sobel Gy",
    gy_display
)

cv2.imshow(
    "5 - Sobel Magnitude",
    edge
)


print("Sobel chạy thành công.")
print("Kích thước ảnh:", image.shape)
print("Kích thước edge:", edge.shape)


# ==============================
# LƯU KẾT QUẢ
# ==============================

cv2.imwrite(
    "sobel_result.jpg",
    edge
)


print(
    "Đã lưu kết quả: sobel_result.jpg"
)


# ==============================
# CHỜ NHẤN PHÍM
# ==============================

cv2.waitKey(0)

cv2.destroyAllWindows()