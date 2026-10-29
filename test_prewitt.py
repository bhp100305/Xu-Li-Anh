import cv2

from src.prewitt import prewitt


IMAGE_PATH = "test_image.jpg"


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Không tìm thấy ảnh: {IMAGE_PATH}"
    )


result = prewitt(image)


gray = result["gray"]
gx = result["gx"]
gy = result["gy"]
edge = result["edge"]


gx_display = cv2.convertScaleAbs(gx)
gy_display = cv2.convertScaleAbs(gy)


cv2.imshow(
    "1 - Anh goc",
    image
)

cv2.imshow(
    "2 - Grayscale",
    gray.astype("uint8")
)

cv2.imshow(
    "3 - Prewitt Gx",
    gx_display
)

cv2.imshow(
    "4 - Prewitt Gy",
    gy_display
)

cv2.imshow(
    "5 - Prewitt Magnitude",
    edge
)


cv2.imwrite(
    "prewitt_result.jpg",
    edge
)


print("Prewitt chạy thành công.")
print("Kích thước ảnh:", image.shape)
print("Kích thước edge:", edge.shape)
print("Đã lưu: prewitt_result.jpg")


cv2.waitKey(0)
cv2.destroyAllWindows()