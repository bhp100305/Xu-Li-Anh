import cv2

from src.laplacian import laplacian


IMAGE_PATH = "test_image.jpg"


image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Không tìm thấy ảnh: {IMAGE_PATH}"
    )


# ==========================================
# LAPLACIAN 4-NEIGHBOR
# ==========================================

result_4 = laplacian(
    image,
    kernel_type="4-neighbor"
)


# ==========================================
# LAPLACIAN 8-NEIGHBOR
# ==========================================

result_8 = laplacian(
    image,
    kernel_type="8-neighbor"
)


gray = result_4["gray"]

edge_4 = result_4["edge"]

edge_8 = result_8["edge"]


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
    "3 - Laplacian 4-neighbor",
    edge_4
)

cv2.imshow(
    "4 - Laplacian 8-neighbor",
    edge_8
)


# ==========================================
# LƯU KẾT QUẢ
# ==========================================

cv2.imwrite(
    "laplacian_4_result.jpg",
    edge_4
)

cv2.imwrite(
    "laplacian_8_result.jpg",
    edge_8
)


print("Laplacian chạy thành công.")

print(
    "Kích thước ảnh:",
    image.shape
)

print(
    "Kích thước edge 4-neighbor:",
    edge_4.shape
)

print(
    "Kích thước edge 8-neighbor:",
    edge_8.shape
)

print(
    "Đã lưu: laplacian_4_result.jpg"
)

print(
    "Đã lưu: laplacian_8_result.jpg"
)


cv2.waitKey(0)

cv2.destroyAllWindows()