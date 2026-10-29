import cv2

from src.log_detector import log_detector


IMAGE_PATH = "test_image.jpg"


image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Không tìm thấy ảnh: {IMAGE_PATH}"
    )


result = log_detector(
    image,
    gaussian_size=5,
    sigma=1.2,
    zero_threshold=15.0
)


gray = result["gray"]
blurred = result["blurred"]
laplacian_display = result["laplacian_display"]
edge = result["edge"]


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
    "4 - Laplacian sau Gaussian",
    laplacian_display
)

cv2.imshow(
    "5 - LoG Zero Crossing",
    edge
)


cv2.imwrite(
    "log_result.jpg",
    edge
)


print("LoG chạy thành công.")
print("Kích thước ảnh:", image.shape)
print("Kích thước edge:", edge.shape)

print(
    "Gaussian kernel:"
)

print(
    result["gaussian_kernel"]
)

print(
    "Đã lưu: log_result.jpg"
)


cv2.waitKey(0)
cv2.destroyAllWindows()