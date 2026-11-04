# XLA Đề 5 – Luồng thuật toán, code và giải thích

Nguồn: các file hiện có trong src/ của thư mục dự án, đọc ngày 05/11/2026. Code dưới đây được trích từ chính mã nguồn; bỏ comment/docstring và định dạng lại bằng AST để dễ đọc, giữ nguyên câu lệnh. Đây là tài liệu đối chiếu mã nguồn cục bộ, chưa phải xác nhận ZIP hay toàn bộ trang con Confluence. Kết nối Confluence báo lỗi khi đọc lại và tìm trang con trong lượt này.

## Cách đọc
Mỗi mục gồm luồng, giải thích từng bước và code tương ứng. Các hàm phụ được trình bày cùng thuật toán sử dụng chúng; import xem tại file nguồn.

## 1. Tiền xử lý và tích chập

Nguồn: `src/convolution.py`.

**Luồng:** Ảnh xám 2D → kiểm tra kernel lẻ → float32 → lật kernel hai chiều → padding reflect → quét cửa sổ → nhân và cộng → ảnh float32 cùng kích thước.

1. Kiểm tra ảnh và kernel đều là ma trận 2D; kernel chẵn gây ValueError.
2. Chuyển float32 để phép trừ và gradient không bị tràn uint8.
3. np.flip(kernel, axis=(0, 1)) thực hiện tích chập toán học. Nếu không lật thì là tương quan.
4. Padding reflect cung cấp dữ liệu cho cửa sổ ở mép ảnh.
5. Mỗi pixel đầu ra bằng tổng region * kernel. Chi phí xấp xỉ H×W×kh×kw.
Đầu ra này giữ giá trị âm; chỉ chuẩn hóa khi cần hiển thị.

### Code `convolution2d` – dòng 4

```python
def convolution2d(image, kernel, padding='reflect'):
    if image.ndim != 2:
        raise ValueError('convolution2d chỉ nhận ảnh xám 2 chiều.')
    if kernel.ndim != 2:
        raise ValueError('Kernel phải là ma trận 2 chiều.')
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError('Kernel phải có kích thước lẻ, ví dụ 3x3 hoặc 5x5.')
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)
    kernel = np.flip(kernel, axis=(0, 1))
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=padding)
    output = np.zeros(image.shape, dtype=np.float32)
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            region = padded[y:y + kh, x:x + kw]
            output[y, x] = np.sum(region * kernel)
    return output
```

## 2. Sobel

Nguồn: `src/sobel.py`.

**Luồng:** BGR/xám → grayscale → tích chập Kx và Ky → Gx, Gy → magnitude và direction → chuẩn hóa ảnh biên.

1. Grayscale = 0.114B + 0.587G + 0.299R; ảnh xám được chuyển float32.
2. Hai kernel Sobel có trọng số 2 ở hàng/cột giữa để kết hợp sai phân với làm trơn theo phương vuông góc.
3. magnitude = sqrt(Gx² + Gy²) biểu diễn cường độ thay đổi; direction = degrees(arctan2(Gy, Gx)).
4. normalize_to_uint8 ánh xạ min–max về 0–255; ma trận hằng được trả thành ảnh 0.
Đầu ra gồm gray, gx, gy, magnitude, direction, edge. edge là ảnh cường độ biên, chưa phải ảnh nhị phân.
Do tích chập có lật kernel, Gx/Gy ngược dấu so với phép tương quan dùng cùng kernel; magnitude không đổi.

### Code `to_grayscale` – dòng 6

```python
def to_grayscale(image):
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Ảnh đầu vào phải là ảnh xám hoặc ảnh màu BGR 3 kênh.')
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return gray
```

### Code `normalize_to_uint8` – dòng 48

```python
def normalize_to_uint8(image):
    image = image.astype(np.float32)
    min_value = np.min(image)
    max_value = np.max(image)
    if max_value == min_value:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = (image - min_value) / (max_value - min_value) * 255.0
    return normalized.astype(np.uint8)
```

### Code `sobel` – dòng 84

```python
def sobel(image):
    gray = to_grayscale(image)
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolution2d(gray, kernel_x)
    gy = convolution2d(gray, kernel_y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)
    direction = np.degrees(direction)
    edge = normalize_to_uint8(magnitude)
    return {'gray': gray, 'gx': gx, 'gy': gy, 'magnitude': magnitude, 'direction': direction, 'edge': edge}
```

## 3. Prewitt

Nguồn: `src/prewitt.py`.

**Luồng:** BGR/xám → grayscale → hai kernel Prewitt → Gx, Gy → độ lớn/hướng → ảnh hiển thị.

1. Cùng cấu trúc với Sobel nhưng trọng số bên mỗi phía đều là 1.
2. Tích chập hai hướng tạo hai thành phần gradient.
3. Lấy căn tổng bình phương để kết hợp hướng; arctan2 giữ thông tin hướng gradient.
4. Chuẩn hóa magnitude về uint8 để hiển thị. Không có Gaussian, NMS hoặc hysteresis trong hàm này.
Đầu ra có cùng các khóa với Sobel, thuận tiện so sánh.

### Code `prewitt` – dòng 7

```python
def prewitt(image):
    gray = to_grayscale(image)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = convolution2d(gray, kernel_x)
    gy = convolution2d(gray, kernel_y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)
    direction = np.degrees(direction)
    edge = normalize_to_uint8(magnitude)
    return {'gray': gray, 'gx': gx, 'gy': gy, 'magnitude': magnitude, 'direction': direction, 'edge': edge}
```

## 4. Laplacian

Nguồn: `src/laplacian.py`.

**Luồng:** Ảnh → grayscale → chọn 4 hoặc 8 láng giềng → tích chập đạo hàm bậc hai → trị tuyệt đối → chuẩn hóa.

1. Kernel 4 láng giềng có tâm -4, bốn ô kề bằng 1; kernel 8 láng giềng có tâm -8 và tám ô bằng 1.
2. response giữ dấu của đáp ứng đạo hàm bậc hai.
3. abs(response) bỏ dấu để thể hiện độ mạnh; chuẩn hóa thành edge 0–255.
4. kernel_type không hợp lệ gây ValueError.
Hàm này không tự làm mượt và không tìm zero-crossing. Không mô tả nó như luồng LoG; nhiễu cũng có thể tạo đáp ứng lớn.

### Code `laplacian` – dòng 7

```python
def laplacian(image, kernel_type='4-neighbor'):
    gray = to_grayscale(image)
    if kernel_type == '4-neighbor':
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    elif kernel_type == '8-neighbor':
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    else:
        raise ValueError("kernel_type phải là '4-neighbor' hoặc '8-neighbor'.")
    response = convolution2d(gray, kernel)
    absolute_response = np.abs(response)
    edge = normalize_to_uint8(absolute_response)
    return {'gray': gray, 'response': response, 'edge': edge}
```

## 5. Laplacian of Gaussian (LoG)

Nguồn: `src/log_detector.py`.

**Luồng:** Grayscale → Gaussian → Laplacian 4 láng giềng → kiểm tra đổi dấu trong cửa sổ 3×3 → biên nhị phân.

1. gaussian_kernel tạo exp(-(x²+y²)/(2σ²)) và chia tổng để tổng kernel bằng 1; mặc định size=5, sigma=1.0.
2. Tích chập Gaussian làm giảm dao động nhỏ trước khi đạo hàm bậc hai.
3. Tích chập Laplacian tạo laplacian_response có cả dấu âm và dương.
4. zero_crossing xét từng vùng 3×3: min < 0 và max > 0, đồng thời max-min >= zero_threshold (mặc định 10).
5. Đủ hai điều kiện thì pixel được gán 255, còn lại 0. Hàng/cột ngoài cùng không được duyệt nên giữ 0.
laplacian_display chỉ để xem; phát hiện đổi dấu dùng đáp ứng có dấu, không dùng ảnh abs đã chuẩn hóa.

### Code `gaussian_kernel` – dòng 7

```python
def gaussian_kernel(size=5, sigma=1.0):
    if size % 2 == 0:
        raise ValueError('Kích thước Gaussian kernel phải là số lẻ.')
    if sigma <= 0:
        raise ValueError('sigma phải lớn hơn 0.')
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)
```

### Code `zero_crossing` – dòng 52

```python
def zero_crossing(response, threshold=0.0):
    height, width = response.shape
    edge = np.zeros((height, width), dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            region = response[y - 1:y + 2, x - 1:x + 2]
            min_value = np.min(region)
            max_value = np.max(region)
            has_sign_change = min_value < 0 and max_value > 0
            strong_enough = max_value - min_value >= threshold
            if has_sign_change and strong_enough:
                edge[y, x] = 255
    return edge
```

### Code `log_detector` – dòng 110

```python
def log_detector(image, gaussian_size=5, sigma=1.0, zero_threshold=10.0):
    gray = to_grayscale(image)
    g_kernel = gaussian_kernel(size=gaussian_size, sigma=sigma)
    blurred = convolution2d(gray, g_kernel)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    laplacian_response = convolution2d(blurred, laplacian_kernel)
    laplacian_display = normalize_to_uint8(np.abs(laplacian_response))
    edge = zero_crossing(laplacian_response, threshold=zero_threshold)
    return {'gray': gray, 'gaussian_kernel': g_kernel, 'blurred': blurred, 'laplacian_response': laplacian_response, 'laplacian_display': laplacian_display, 'edge': edge}
```

## 6. Canny

Nguồn: `src/canny.py`.

**Luồng:** Grayscale → Gaussian → Sobel → magnitude/direction → NMS → ngưỡng kép → hysteresis → biên 0/255.

1. Mặc định Gaussian 5×5, sigma=1.4; làm trơn rồi tính Sobel trên ảnh blurred.
2. NMS đưa góc âm cộng 180, phân thành bốn hướng 0/45/90/135. Mỗi pixel chỉ giữ nếu không nhỏ hơn hai hàng xóm theo cặp hướng được code chọn. NMS để mép ngoài bằng 0.
3. Ngưỡng lấy từ median ảnh gray, không từ NMS: low=max(0,(1-s)median), high=min(255,(1+s)median), s mặc định 0.33.
4. Pixel >0 và >= high là strong=255; pixel >0, >=low và <high là weak=75; còn lại 0. Điều kiện >0 ngăn ảnh gradient 0 trở thành biên khi median=0.
5. Hysteresis lặp đến khi không còn thay đổi: weak có strong trong vùng 3×3 được nâng thành strong. Nhờ lặp, giữ được chuỗi weak nối gián tiếp tới strong; cuối cùng loại mọi giá trị khác strong.
Đầu ra giữ các ảnh trung gian gray, blurred, gx, gy, magnitude, direction, nms, threshold, edge và hai ngưỡng.
Lưu ý đối chiếu: code NMS dùng cặp trên-phải/dưới-trái cho nhóm 45°, cặp trên-trái/dưới-phải cho 135°. Cần kiểm chứng quy ước góc với cạnh chéo trước khi khẳng định đúng mọi hướng; kiểm thử ảnh hằng chưa chứng minh điều đó.

### Code `automatic_thresholds` – dòng 79

```python
def automatic_thresholds(gray, sigma=0.33):
    median_value = np.median(gray)
    low_threshold = max(0, (1.0 - sigma) * median_value)
    high_threshold = min(255, (1.0 + sigma) * median_value)
    return (float(low_threshold), float(high_threshold))
```

### Code `non_maximum_suppression` – dòng 8

```python
def non_maximum_suppression(magnitude, direction):
    height, width = magnitude.shape
    output = np.zeros((height, width), dtype=np.float32)
    angle = direction.copy()
    angle[angle < 0] += 180
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current = magnitude[y, x]
            neighbor1 = 0
            neighbor2 = 0
            if 0 <= angle[y, x] < 22.5 or 157.5 <= angle[y, x] <= 180:
                neighbor1 = magnitude[y, x - 1]
                neighbor2 = magnitude[y, x + 1]
            elif 22.5 <= angle[y, x] < 67.5:
                neighbor1 = magnitude[y - 1, x + 1]
                neighbor2 = magnitude[y + 1, x - 1]
            elif 67.5 <= angle[y, x] < 112.5:
                neighbor1 = magnitude[y - 1, x]
                neighbor2 = magnitude[y + 1, x]
            elif 112.5 <= angle[y, x] < 157.5:
                neighbor1 = magnitude[y - 1, x - 1]
                neighbor2 = magnitude[y + 1, x + 1]
            if current >= neighbor1 and current >= neighbor2:
                output[y, x] = current
    return output
```

### Code `double_threshold` – dòng 115

```python
def double_threshold(image, low_threshold, high_threshold):
    strong_value = 255
    weak_value = 75
    result = np.zeros(image.shape, dtype=np.uint8)
    strong_y, strong_x = np.where((image > 0) & (image >= high_threshold))
    weak_y, weak_x = np.where((image >= low_threshold) & (image > 0) & (image < high_threshold))
    result[strong_y, strong_x] = strong_value
    result[weak_y, weak_x] = weak_value
    return (result, weak_value, strong_value)
```

### Code `hysteresis` – dòng 162

```python
def hysteresis(image, weak_value=75, strong_value=255):
    result = image.copy()
    height, width = result.shape
    changed = True
    while changed:
        changed = False
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if result[y, x] == weak_value:
                    region = result[y - 1:y + 2, x - 1:x + 2]
                    if np.any(region == strong_value):
                        result[y, x] = strong_value
                        changed = True
    result[result != strong_value] = 0
    return result
```

### Code `canny` – dòng 214

```python
def canny(image, gaussian_size=5, gaussian_sigma=1.4, threshold_sigma=0.33):
    gray = to_grayscale(image)
    g_kernel = gaussian_kernel(size=gaussian_size, sigma=gaussian_sigma)
    blurred = convolution2d(gray, g_kernel)
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolution2d(blurred, kernel_x)
    gy = convolution2d(blurred, kernel_y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.degrees(np.arctan2(gy, gx))
    nms = non_maximum_suppression(magnitude, direction)
    low_threshold, high_threshold = automatic_thresholds(gray, sigma=threshold_sigma)
    threshold_image, weak, strong = double_threshold(nms, low_threshold, high_threshold)
    edge = hysteresis(threshold_image, weak_value=weak, strong_value=strong)
    return {'gray': gray, 'blurred': blurred, 'gx': gx, 'gy': gy, 'magnitude': magnitude, 'direction': direction, 'nms': nms, 'threshold': threshold_image, 'edge': edge, 'low_threshold': low_threshold, 'high_threshold': high_threshold}
```

## 7. Unsharp Masking và High-boost

Nguồn: `src/sharpening.py`.

**Luồng:** Ảnh → grayscale → Gaussian → mask = gray-blurred → gray + hệ số×mask → clip 0–255.

1. Gaussian tách phần thay đổi chậm thành blurred.
2. mask giữ chi tiết tần số cao; vùng phẳng có mask gần 0.
3. Unsharp cộng amount×mask vào gray; amount mặc định 1.0.
4. High-boost yêu cầu k>1 rồi gọi lại unsharp_mask(amount=k), mặc định k=1.5. Đây là công thức thực tế của dự án.
5. clip_uint8 cắt giá trị vượt miền, khác với kéo giãn min–max của ảnh biên.
Đầu ra sharpened là ảnh xám, kể cả khi đầu vào BGR. Hàm tiện ích sharpen_image tự đổi strength<=1 thành 1.5 khi chọn highboost.

### Code `clip_uint8` – dòng 8

```python
def clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)
```

### Code `unsharp_mask` – dòng 27

```python
def unsharp_mask(image: np.ndarray, gaussian_size: int=5, sigma: float=1.0, amount: float=1.0) -> dict[str, np.ndarray]:
    gray = to_grayscale(image)
    kernel = gaussian_kernel(size=gaussian_size, sigma=sigma)
    blurred = convolution2d(gray, kernel)
    mask = gray - blurred
    sharpened_float = gray + amount * mask
    sharpened = clip_uint8(sharpened_float)
    return {'gray': gray, 'blurred': blurred, 'mask': mask, 'sharpened': sharpened}
```

### Code `high_boost` – dòng 130

```python
def high_boost(image: np.ndarray, k: float=1.5, gaussian_size: int=5, sigma: float=1.0) -> dict[str, np.ndarray]:
    if k <= 1:
        raise ValueError('High-Boost yêu cầu k > 1.')
    return unsharp_mask(image=image, gaussian_size=gaussian_size, sigma=sigma, amount=k)
```

### Code `sharpen_image` – dòng 180

```python
def sharpen_image(image: np.ndarray, method: str='unsharp', strength: float=1.0) -> np.ndarray:
    method = method.lower()
    if method == 'unsharp':
        result = unsharp_mask(image=image, amount=strength)
    elif method == 'highboost':
        if strength <= 1:
            strength = 1.5
        result = high_boost(image=image, k=strength)
    else:
        raise ValueError("method phải là 'unsharp' hoặc 'highboost'.")
    return result['sharpened']
```

## 8. Phát hiện trang giấy: contour, Hough và độ sáng

Nguồn: `src/document_detector.py`.

**Luồng:** Biên Canny → ứng viên contour + ứng viên Hough → so điểm → nếu chưa có góc thì tìm vùng sáng → 4 góc hoặc None.

1. Nhánh contour: closing kernel 5×5 hai lần rồi dilate một lần; tìm contour ngoài, sắp diện tích giảm dần và xét tối đa 30 contour.
2. Bỏ contour dưới 8% diện tích ảnh. Lấy convex hull, thử epsilon từ 0.01 đến 0.07 lần chu vi; trả tứ giác lồi đầu tiên phù hợp.
3. Hough luôn được chạy, kể cả khi contour thành công. Nối khe nhỏ 3×3 rồi HoughLinesP; lọc đoạn ngắn/trùng, giữ tối đa 24 ứng viên dài.
4. Ghép hai cặp cạnh đối gần cùng hướng, có khoảng cách đủ lớn; giao các đường để tạo bốn góc. Lọc theo biên ảnh, diện tích và tính lồi.
5. Điểm chung = 0.55×tỷ lệ diện tích + 0.20×độ gần tâm + 0.25×mức hỗ trợ bởi biên. Hough thay contour khi contour không có hoặc điểm Hough lớn hơn điểm contour + 0.03.
6. Chỉ khi chưa có góc mới chạy brightness: BGR→gray bằng OpenCV→GaussianBlur→Otsu→closing 9×9 hai lần→contour. Lọc diện tích dưới 10%, lấy minAreaRect và chấm 0.50×diện tích + 0.30×độ chữ nhật + 0.20×gần tâm.
7. order_points dùng min/max(x+y) và min/max(y-x), trả TL, TR, BR, BL. Đây là heuristic có thể trùng điểm ở một số cấu hình; code chưa có xác thực tổng quát cho mọi tứ giác.
Kết quả gồm corners, preview, method thuộc edge/hough/brightness/none. Brightness dùng hình chữ nhật xoay bao ngoài, không bảo đảm phục hồi chính xác phối cảnh khi góc giấy bị che.
Phần này dùng OpenCV cho morphology, contour, Hough và Otsu; không được mô tả toàn bộ scanner là NumPy tự cài.

### Code `prepare_edges` – dòng 55

```python
def prepare_edges(edge_image: np.ndarray) -> np.ndarray:
    edge = np.clip(edge_image, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated
```

### Code `find_document_from_edges` – dòng 89

```python
def find_document_from_edges(edge_image: np.ndarray, min_area_ratio: float=0.08) -> np.ndarray | None:
    if edge_image.ndim != 2:
        raise ValueError('edge_image phải là ảnh 2 chiều.')
    height, width = edge_image.shape
    image_area = height * width
    min_area = image_area * min_area_ratio
    processed = prepare_edges(edge_image)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    epsilon_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07]
    for contour in contours[:30]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        if perimeter <= 0:
            continue
        for epsilon_ratio in epsilon_values:
            approximation = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
            if len(approximation) != 4:
                continue
            if not cv2.isContourConvex(approximation):
                continue
            points = approximation.reshape(4, 2).astype(np.float32)
            return order_points(points)
    return None
```

### Code `find_document_from_hough` – dòng 194

```python
def find_document_from_hough(edge_image: np.ndarray, min_area_ratio: float=0.08) -> np.ndarray | None:
    if edge_image.ndim != 2:
        raise ValueError('edge_image must be a 2D image.')
    height, width = edge_image.shape
    diagonal = float(np.hypot(width, height))
    min_dimension = min(height, width)
    edges = np.clip(edge_image, 0, 255).astype(np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    segments = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=max(25, int(min_dimension * 0.08)), minLineLength=max(24, int(min_dimension * 0.18)), maxLineGap=max(12, int(min_dimension * 0.06)))
    if segments is None:
        return None
    candidates: list[tuple[np.ndarray, float, float]] = []
    for raw in segments.reshape(-1, 4):
        p1 = raw[:2].astype(np.float64)
        p2 = raw[2:].astype(np.float64)
        length = float(np.linalg.norm(p2 - p1))
        if length < max(24, min_dimension * 0.18):
            continue
        angle = float(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) % np.pi)
        midpoint = (p1 + p2) / 2.0
        duplicate = False
        for old_index, (old_segment, old_angle, old_length) in enumerate(candidates):
            angle_delta = abs(angle - old_angle)
            angle_delta = min(angle_delta, np.pi - angle_delta)
            old_mid = (old_segment[:2] + old_segment[2:]) / 2.0
            normal = np.array([-np.sin(angle), np.cos(angle)])
            offset = abs(float(np.dot(midpoint - old_mid, normal)))
            if angle_delta < np.deg2rad(3.0) and offset < max(8.0, diagonal * 0.012):
                duplicate = True
                if length > old_length:
                    candidates[old_index] = (raw.astype(np.float64), angle, length)
                break
        if not duplicate:
            candidates.append((raw.astype(np.float64), angle, length))
    candidates.sort(key=lambda item: item[2], reverse=True)
    candidates = candidates[:24]
    if len(candidates) < 4:
        return None

    def orientation_delta(angle_a: float, angle_b: float) -> float:
        delta = abs(angle_a - angle_b) % np.pi
        return min(delta, np.pi - delta)

    def mean_orientation(angle_a: float, angle_b: float) -> float:
        return float(0.5 * np.arctan2(np.sin(2.0 * angle_a) + np.sin(2.0 * angle_b), np.cos(2.0 * angle_a) + np.cos(2.0 * angle_b)) % np.pi)
    side_pairs: list[tuple[int, int, float]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            _, angle_a, _ = candidates[i]
            _, angle_b, _ = candidates[j]
            delta = orientation_delta(angle_a, angle_b)
            if delta > np.deg2rad(18.0):
                continue
            mean_angle = mean_orientation(angle_a, angle_b)
            mid_a = (candidates[i][0][:2] + candidates[i][0][2:]) / 2.0
            mid_b = (candidates[j][0][:2] + candidates[j][0][2:]) / 2.0
            normal = np.array([-np.sin(mean_angle), np.cos(mean_angle)])
            separation = abs(float(np.dot(mid_a - mid_b, normal)))
            if separation >= diagonal * 0.12:
                side_pairs.append((i, j, mean_angle))

    def intersection(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = line_a
        x3, y3, x4, y4 = line_b
        a1, b1, c1 = (y2 - y1, x1 - x2, x2 * y1 - x1 * y2)
        a2, b2, c2 = (y4 - y3, x3 - x4, x4 * y3 - x3 * y4)
        determinant = a1 * b2 - a2 * b1
        if abs(determinant) < 1e-06:
            return None
        return np.array([(b1 * c2 - b2 * c1) / determinant, (c1 * a2 - c2 * a1) / determinant], dtype=np.float32)
    image_area = float(width * height)
    best_score = -1.0
    best_corners: np.ndarray | None = None
    for pair_index, pair_a in enumerate(side_pairs):
        a1, a2, angle_a = pair_a
        for pair_b in side_pairs[pair_index + 1:]:
            b1, b2, angle_b = pair_b
            if len({a1, a2, b1, b2}) != 4:
                continue
            cross_angle = orientation_delta(angle_a, angle_b)
            if not np.deg2rad(25.0) < cross_angle < np.deg2rad(155.0):
                continue
            points = [intersection(candidates[a1][0], candidates[b1][0]), intersection(candidates[a1][0], candidates[b2][0]), intersection(candidates[a2][0], candidates[b2][0]), intersection(candidates[a2][0], candidates[b1][0])]
            if any((point is None for point in points)):
                continue
            corners = order_points(np.asarray(points, dtype=np.float32))
            if np.any(corners[:, 0] < -width * 0.04) or np.any(corners[:, 0] > width * 1.04) or np.any(corners[:, 1] < -height * 0.04) or np.any(corners[:, 1] > height * 1.04):
                continue
            area_ratio = abs(float(cv2.contourArea(corners))) / image_area
            if not min_area_ratio <= area_ratio <= 0.98:
                continue
            contour = corners.astype(np.int32).reshape((-1, 1, 2))
            if not cv2.isContourConvex(contour):
                continue
            score = score_document_corners(edge_image, corners)
            if score > best_score:
                best_score = score
                best_corners = corners
    return best_corners
```

### Code `score_document_corners` – dòng 352

```python
def score_document_corners(edge_image: np.ndarray, corners: np.ndarray) -> float:
    height, width = edge_image.shape
    image_area = float(width * height)
    diagonal = float(np.hypot(width, height))
    area_ratio = abs(float(cv2.contourArea(corners.astype(np.float32)))) / image_area
    if not 0.0 < area_ratio <= 0.98:
        return -1.0
    edge_map = cv2.dilate(np.clip(edge_image, 0, 255).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))) > 0
    support = 0.0
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        samples = np.linspace(start, end, 40)[3:-3].astype(np.int32)
        inside = (samples[:, 0] >= 0) & (samples[:, 0] < width) & (samples[:, 1] >= 0) & (samples[:, 1] < height)
        if np.any(inside):
            support += float(np.mean(edge_map[samples[inside, 1], samples[inside, 0]]))
    support /= 4.0
    center = np.array([width / 2.0, height / 2.0])
    center_distance = float(np.linalg.norm(corners.mean(axis=0) - center))
    center_score = 1.0 - min(center_distance / diagonal, 1.0)
    return 0.55 * area_ratio + 0.2 * center_score + 0.25 * support
```

### Code `find_document_from_brightness` – dòng 386

```python
def find_document_from_brightness(image: np.ndarray, min_area_ratio: float=0.1) -> np.ndarray | None:
    if image is None:
        return None
    if image.ndim != 3:
        return None
    height, width = image.shape[:2]
    image_area = height * width
    min_area = image_area * min_area_ratio
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best_score = -1.0
    best_box = None
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    max_center_distance = np.sqrt(width ** 2 + height ** 2)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        center, size, _ = rect
        rect_width = size[0]
        rect_height = size[1]
        if rect_width <= 0 or rect_height <= 0:
            continue
        rect_area = rect_width * rect_height
        if rect_area <= 0:
            continue
        rectangularity = min(contour_area / rect_area, 1.0)
        center_array = np.array(center, dtype=np.float32)
        center_distance = np.linalg.norm(center_array - image_center)
        center_score = 1.0 - center_distance / max_center_distance
        area_score = min(contour_area / image_area, 1.0)
        score = 0.5 * area_score + 0.3 * rectangularity + 0.2 * center_score
        if score > best_score:
            box = cv2.boxPoints(rect)
            best_box = np.array(box, dtype=np.float32)
            best_score = score
    if best_box is None:
        return None
    return order_points(best_box)
```

### Code `order_points` – dòng 7

```python
def order_points(points: np.ndarray) -> np.ndarray:
    if points.shape != (4, 2):
        raise ValueError('order_points yêu cầu đúng 4 điểm.')
    points = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]
    return ordered
```

### Code `detect_document` – dòng 587

```python
def detect_document(image: np.ndarray, edge_image: np.ndarray) -> dict[str, object]:
    corners = find_document_from_edges(edge_image)
    method = 'edge' if corners is not None else 'hough'
    hough_corners = find_document_from_hough(edge_image)
    if hough_corners is not None and (corners is None or score_document_corners(edge_image, hough_corners) > score_document_corners(edge_image, corners) + 0.03):
        corners = hough_corners
        method = 'hough'
    if corners is None:
        corners = find_document_from_brightness(image)
        method = 'brightness' if corners is not None else 'none'
    preview = draw_document_contour(image, corners)
    return {'corners': corners, 'preview': preview, 'method': method}
```

## 9. Biến đổi phối cảnh

Nguồn: `src/perspective.py`.

**Luồng:** 4 góc ảnh gốc → sắp TL/TR/BR/BL → tính width/height → bốn góc hình chữ nhật đích → homography → warp.

1. Chiều rộng lấy max(độ dài cạnh trên, cạnh dưới), chiều cao lấy max(cạnh trái, cạnh phải), rồi ép int.
2. Đích là (0,0), (W-1,0), (W-1,H-1), (0,H-1).
3. cv2.getPerspectiveTransform tìm ma trận 3×3 ánh xạ tứ giác nguồn sang hình chữ nhật.
4. cv2.warpPerspective tái lấy mẫu ảnh trên lưới đích.
5. scan_perspective trả corners, width, height, warped. Kích thước không dương gây ValueError.
Nắn phối cảnh thực hiện trên ảnh gốc để giữ chi tiết; không nắn trực tiếp ảnh Canny.

### Code `distance` – dòng 9

```python
def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return float(np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))
```

### Code `get_output_size` – dòng 29

```python
def get_output_size(corners: np.ndarray) -> tuple[int, int]:
    if corners.shape != (4, 2):
        raise ValueError('corners phải có kích thước (4, 2).')
    top_left = corners[0]
    top_right = corners[1]
    bottom_right = corners[2]
    bottom_left = corners[3]
    width_top = distance(top_left, top_right)
    width_bottom = distance(bottom_left, bottom_right)
    width = int(max(width_top, width_bottom))
    height_left = distance(top_left, bottom_left)
    height_right = distance(top_right, bottom_right)
    height = int(max(height_left, height_right))
    if width <= 0 or height <= 0:
        raise ValueError('Kích thước ảnh sau transform không hợp lệ.')
    return (width, height)
```

### Code `perspective_transform` – dòng 109

```python
def perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError('Ảnh đầu vào không được None.')
    if corners.shape != (4, 2):
        raise ValueError('corners phải gồm đúng 4 điểm.')
    ordered = order_points(corners)
    width, height = get_output_size(ordered)
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped
```

### Code `scan_perspective` – dòng 194

```python
def scan_perspective(image: np.ndarray, corners: np.ndarray) -> dict[str, object]:
    ordered = order_points(corners)
    width, height = get_output_size(ordered)
    warped = perspective_transform(image, ordered)
    return {'corners': ordered, 'width': width, 'height': height, 'warped': warped}
```

## 10. Document Scanner tổng hợp

Nguồn: `src/scanner.py`.

**Luồng:** Kiểm tra ảnh → resize phát hiện → Canny → detect_document → đủ góc? → scale về ảnh gốc → clip tọa độ → perspective → resize đầu ra → sharpening.

1. Từ chối None/ảnh rỗng và giới hạn chiều rộng không dương.
2. detection_width mặc định 800; ảnh nhỏ hơn giữ nguyên. Lưu scale_x=Wgốc/Wnhỏ và scale_y=Hgốc/Hnhỏ riêng do chiều cao đã làm tròn.
3. Chạy Canny tự cài trên ảnh nhỏ; chuyển edge và detection_image cho detect_document.
4. Không có góc thì RuntimeError; góc phải là ndarray (4,2). Hiện chưa có bước chỉnh góc thủ công trong scan_document.
5. Nhân x/y theo tỷ lệ tương ứng, giới hạn tọa độ trong ảnh gốc, rồi scan_perspective trên ảnh gốc.
6. Giới hạn chiều rộng warped về output_width mặc định 1000 trước tích chập làm nét để giảm thời gian.
7. Chọn unsharp hoặc highboost; highboost với strength<=1 đổi thành 1.5; tên khác gây ValueError.
8. Trả ảnh trung gian, góc, detection_method, warped_original, warped và sharpened. width/height là kích thước phối cảnh trước resize; final_width/final_height là kích thước ảnh cuối.
Docstring scanner còn mô tả hai chiến lược cũ; thân hàm detect_document thực tế có thêm Hough và phép so điểm.

### Code `resize_for_detection` – dòng 18

```python
def resize_for_detection(image: np.ndarray, max_width: int=800) -> tuple[np.ndarray, float, float]:
    if image is None:
        raise ValueError('Ảnh đầu vào không được None.')
    if image.size == 0:
        raise ValueError('Ảnh đầu vào rỗng.')
    if max_width <= 0:
        raise ValueError('max_width phải lớn hơn 0.')
    height, width = image.shape[:2]
    if width <= max_width:
        return (image.copy(), 1.0, 1.0)
    ratio = max_width / width
    new_width = max_width
    new_height = max(1, int(round(height * ratio)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    scale_x = width / new_width
    scale_y = height / new_height
    return (resized, scale_x, scale_y)
```

### Code `scale_corners_to_original` – dòng 181

```python
def scale_corners_to_original(corners: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    if corners.shape != (4, 2):
        raise ValueError('corners phải có kích thước (4, 2).')
    original_corners = corners.astype(np.float32).copy()
    original_corners[:, 0] *= scale_x
    original_corners[:, 1] *= scale_y
    return original_corners
```

### Code `resize_output` – dòng 112

```python
def resize_output(image: np.ndarray, max_width: int=1000) -> np.ndarray:
    if image is None:
        raise ValueError('Ảnh đầu vào không được None.')
    if image.size == 0:
        raise ValueError('Ảnh đầu vào rỗng.')
    if max_width <= 0:
        raise ValueError('max_width phải lớn hơn 0.')
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy()
    ratio = max_width / width
    new_height = max(1, int(round(height * ratio)))
    resized = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    return resized
```

### Code `scan_document` – dòng 228

```python
def scan_document(image: np.ndarray, sharpening_method: str='unsharp', sharpening_strength: float=1.0, gaussian_size: int=5, gaussian_sigma: float=1.4, threshold_sigma: float=0.33, detection_width: int=800, output_width: int=1000, verbose: bool=True) -> dict[str, object]:
    if image is None:
        raise ValueError('Ảnh đầu vào không được None.')
    if image.size == 0:
        raise ValueError('Ảnh đầu vào rỗng.')
    if detection_width <= 0:
        raise ValueError('detection_width phải lớn hơn 0.')
    if output_width <= 0:
        raise ValueError('output_width phải lớn hơn 0.')
    if verbose:
        print()
        print('[1/5] Thu nhỏ ảnh để phát hiện tài liệu...')
    detection_image, scale_x, scale_y = resize_for_detection(image=image, max_width=detection_width)
    if verbose:
        print('      Ảnh gốc:', image.shape)
        print('      Ảnh detection:', detection_image.shape)
    if verbose:
        print('[2/5] Đang chạy Canny...')
    canny_result = canny(image=detection_image, gaussian_size=gaussian_size, gaussian_sigma=gaussian_sigma, threshold_sigma=threshold_sigma)
    edge = canny_result['edge']
    if not isinstance(edge, np.ndarray):
        raise RuntimeError('Canny không trả về ảnh biên hợp lệ.')
    if verbose:
        print('[3/5] Đang tìm vùng tài liệu...')
    detection_result = detect_document(image=detection_image, edge_image=edge)
    resized_corners = detection_result['corners']
    detection_method = str(detection_result.get('method', 'unknown'))
    if resized_corners is None:
        raise RuntimeError('Không phát hiện được vùng tài liệu.')
    if not isinstance(resized_corners, np.ndarray):
        raise RuntimeError('Tọa độ tài liệu không hợp lệ.')
    if resized_corners.shape != (4, 2):
        raise RuntimeError('Không xác định được đủ 4 góc tài liệu.')
    if verbose:
        print('      Phương pháp:', detection_method)
        print('      4 góc trên ảnh nhỏ:')
        print(resized_corners)
    original_corners = scale_corners_to_original(corners=resized_corners, scale_x=scale_x, scale_y=scale_y)
    original_height, original_width = image.shape[:2]
    original_corners[:, 0] = np.clip(original_corners[:, 0], 0, original_width - 1)
    original_corners[:, 1] = np.clip(original_corners[:, 1], 0, original_height - 1)
    detection_preview = draw_document_contour(image=image, corners=original_corners)
    if verbose:
        print('[4/5] Đang nắn phối cảnh...')
    perspective_result = scan_perspective(image=image, corners=original_corners)
    warped_original = perspective_result['warped']
    if not isinstance(warped_original, np.ndarray):
        raise RuntimeError('Perspective Transform thất bại.')
    warped = resize_output(image=warped_original, max_width=output_width)
    if verbose:
        print('[5/5] Đang làm nét ảnh...')
    method = sharpening_method.strip().lower()
    if method == 'unsharp':
        sharpening_result = unsharp_mask(image=warped, amount=sharpening_strength)
    elif method == 'highboost':
        if sharpening_strength <= 1:
            sharpening_strength = 1.5
        sharpening_result = high_boost(image=warped, k=sharpening_strength)
    else:
        raise ValueError("sharpening_method phải là 'unsharp' hoặc 'highboost'.")
    scanned = sharpening_result['sharpened']
    if not isinstance(scanned, np.ndarray):
        raise RuntimeError('Sharpening không trả về ảnh hợp lệ.')
    if verbose:
        print('      Kích thước perspective:', warped_original.shape)
        print('      Kích thước ảnh cuối:', scanned.shape)
        print('Hoàn thành pipeline scan.')
    return {'original': image, 'detection_image': detection_image, 'gray': canny_result['gray'], 'blurred': canny_result['blurred'], 'magnitude': canny_result['magnitude'], 'nms': canny_result['nms'], 'edge': edge, 'low_threshold': canny_result['low_threshold'], 'high_threshold': canny_result['high_threshold'], 'corners_detection': resized_corners, 'corners': original_corners, 'detection_method': detection_method, 'detection_preview': detection_preview, 'warped_original': warped_original, 'warped': warped, 'width': perspective_result['width'], 'height': perspective_result['height'], 'sharpened': scanned, 'final_width': scanned.shape[1], 'final_height': scanned.shape[0]}
```

## 11. So sánh thuật toán

Nguồn: `src/comparison.py`.

**Luồng:** Resize chung → chạy 5 thuật toán tự cài → lấy ngưỡng Canny → cv2.Canny → gom ảnh và thời gian.

1. Resize tối đa 700 pixel chiều rộng và giữ tỷ lệ để các thuật toán dùng cùng working_image.
2. Chạy Sobel, Prewitt, Laplacian 4-neighbor, LoG (5,1.2,15), Canny (5,1.4,0.33). run_with_timer dùng perf_counter đo thời gian từng lời gọi.
3. cv2.Canny chạy trên gray_uint8 với hai ngưỡng của Canny tự cài đã ép int.
4. Trả results chứa name, edge, time; kèm ảnh gốc làm việc và các ngưỡng.
Giới hạn: Canny tự cài có Gaussian riêng, cv2.Canny ở đây nhận gray chưa làm mượt tương ứng; chuẩn độ lớn gradient cũng khác mặc định. Vì thế đây là so sánh hiển thị/thời gian của cấu hình hiện tại, chưa phải phép kiểm chứng tương đương từng pixel.
Hàm chưa tính Precision/Recall/F-measure, chưa thêm ba mức nhiễu, chưa thống kê trung bình ± độ lệch chuẩn hay tỷ lệ thành công trên 30 tài liệu.

### Code `resize_for_comparison` – dòng 16

```python
def resize_for_comparison(image: np.ndarray, max_width: int=700) -> np.ndarray:
    if image is None:
        raise ValueError('Ảnh đầu vào không được None.')
    if image.size == 0:
        raise ValueError('Ảnh đầu vào rỗng.')
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy()
    ratio = max_width / width
    new_height = max(1, int(round(height * ratio)))
    resized = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    return resized
```

### Code `run_with_timer` – dòng 114

```python
def run_with_timer(function, *args, **kwargs) -> tuple[object, float]:
    start_time = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start_time
    return (result, elapsed)
```

### Code `compare_algorithms` – dòng 151

```python
def compare_algorithms(image: np.ndarray, max_width: int=700) -> dict[str, object]:
    working_image = resize_for_comparison(image=image, max_width=max_width)
    gray = to_grayscale(working_image)
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    print('[1/6] Sobel...')
    sobel_result, sobel_time = run_with_timer(sobel, working_image)
    sobel_edge = sobel_result['edge']
    print('[2/6] Prewitt...')
    prewitt_result, prewitt_time = run_with_timer(prewitt, working_image)
    prewitt_edge = prewitt_result['edge']
    print('[3/6] Laplacian...')
    laplacian_result, laplacian_time = run_with_timer(laplacian, working_image, kernel_type='4-neighbor')
    laplacian_edge = laplacian_result['edge']
    print('[4/6] LoG...')
    log_result, log_time = run_with_timer(log_detector, working_image, gaussian_size=5, sigma=1.2, zero_threshold=15.0)
    log_edge = log_result['edge']
    print('[5/6] Canny tự cài...')
    canny_result, canny_time = run_with_timer(canny, working_image, gaussian_size=5, gaussian_sigma=1.4, threshold_sigma=0.33)
    canny_edge = canny_result['edge']
    print('[6/6] OpenCV Canny...')
    low_threshold = int(canny_result['low_threshold'])
    high_threshold = int(canny_result['high_threshold'])
    opencv_start = time.perf_counter()
    opencv_edge = cv2.Canny(gray_uint8, low_threshold, high_threshold)
    opencv_time = time.perf_counter() - opencv_start
    results = {'sobel': {'name': 'Sobel', 'edge': sobel_edge, 'time': sobel_time}, 'prewitt': {'name': 'Prewitt', 'edge': prewitt_edge, 'time': prewitt_time}, 'laplacian': {'name': 'Laplacian', 'edge': laplacian_edge, 'time': laplacian_time}, 'log': {'name': 'LoG', 'edge': log_edge, 'time': log_time}, 'canny': {'name': 'Canny tự cài', 'edge': canny_edge, 'time': canny_time}, 'opencv_canny': {'name': 'OpenCV Canny', 'edge': opencv_edge, 'time': opencv_time}}
    return {'original': working_image, 'gray': gray_uint8, 'results': results, 'low_threshold': low_threshold, 'high_threshold': high_threshold, 'width': working_image.shape[1], 'height': working_image.shape[0]}
```

## Trạng thái kiểm chứng
Đã đối chiếu tĩnh từng luồng với thân hàm thực tế. Chạy `.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -v`: 14/14 kiểm thử có sẵn đạt, gồm 8 kiểm thử thuật toán và 6 kiểm thử web. Lượt chạy mặc định trước đó lỗi do console cp1252 không in được tiếng Việt; chạy với UTF-8 đạt. Đây chưa phải kiểm chứng đủ mọi hướng biên hay dữ liệu thực nghiệm. Không suy ra kết quả BSDS500 hoặc tỷ lệ thành công trên 30 ảnh từ các kiểm thử này.
