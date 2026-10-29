import cv2

from src.canny import canny


IMAGE_PATH = "test_image.jpg"


image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Không tìm thấy ảnh: {IMAGE_PATH}"
    )


result = canny(
    image,
    gaussian_size=5,
    gaussian_sigma=1.4,
    threshold_sigma=0.33
)


gray = result["gray"]

blurred = result["blurred"]

magnitude = result["magnitude"]

nms = result["nms"]

threshold_image = result["threshold"]

edge = result["edge"]


# Chuyển về uint8 để hiển thị
magnitude_display = cv2.normalize(
    magnitude,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype("uint8")


nms_display = cv2.normalize(
    nms,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype("uint8")


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
    "4 - Gradient Magnitude",
    magnitude_display
)

cv2.imshow(
    "5 - Non Maximum Suppression",
    nms_display
)

cv2.imshow(
    "6 - Double Threshold",
    threshold_image
)

cv2.imshow(
    "7 - Canny Final",
    edge
)


cv2.imwrite(
    "canny_result.jpg",
    edge
)


print("Canny chạy thành công.")

print(
    "Low threshold:",
    result["low_threshold"]
)

print(
    "High threshold:",
    result["high_threshold"]
)

print(
    "Kích thước edge:",
    edge.shape
)

print(
    "Đã lưu: canny_result.jpg"
)


cv2.waitKey(0)

cv2.destroyAllWindows()