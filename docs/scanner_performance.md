# Giữ ảnh chụp và tối ưu tốc độ scanner

## Hành vi giao diện

Khi bấm chụp, canvas giữ nguyên khung hình lấy từ video. Giao diện chuyển sang trạng thái `processing`, ẩn video và hiển thị canvas cho đến khi API scan trả kết quả. Khi thành công, bản quét được thêm vào danh sách và video hiện lại trên cùng camera stream. Khi lỗi, ảnh chụp được giữ lại để tải về, thử lại hoặc chụp lại.

Các nút chụp và chuyển camera bị khóa khi đang xử lý. Khung nhận diện trực tiếp tạm dừng và không được dùng để thay thế vùng cắt trong pipeline scan.

## Tối ưu giữ nguyên thuật toán

- `convolution2d`: thay vòng lặp từng pixel bằng sliding windows NumPy. Chia theo khối với bộ đệm tích khoảng 16 MiB; giữ nguyên float32, kernel lật hai chiều, padding và thứ tự cộng từng cửa sổ.
- Non-maximum suppression: so sánh các mảng láng giềng theo đúng các nhóm góc, quy ước đường chéo và điều kiện `>=` cũ. Biên ngoài vẫn bằng 0.
- Hysteresis: hàng đợi lan truyền từ strong tới weak liên thông 8 hướng; mỗi weak được nâng tối đa một lần. Giữ nguyên quy tắc không nâng weak ở hàng/cột ngoài cùng.
- Detector contour/Hough/brightness, xếp điểm, nắn phối cảnh, hệ số Gaussian và làm nét giữ nguyên. Không thay Canny tự cài bằng `cv2.Canny` cho pipeline scan.

## Số đo tại máy phát triển

Ảnh `test_image.jpg`: 1920×2560 BGR; đầu ra 1600×2120 xám. Cấu hình `detection_width=800`, `output_width=1600`, `verbose=False`, OpenCV một luồng. Các số dưới đây đo không bật profiler.

| Lần chạy | Thời gian xử lý |
| --- | ---: |
| Tham chiếu từ commit `8853341`, một lần | 43,897 giây |
| Bản tối ưu, lần 1 | 0,586 giây |
| Bản tối ưu, lần 2 | 0,589 giây |
| Bản tối ưu, lần 3 | 0,582 giây |
| Trung vị bản tối ưu | 0,586 giây |
| API cục bộ, gồm đọc JPEG và mã hóa PNG | 1,035 giây |

Đối chiếu trực tiếp với bản tham chiếu xác nhận **bốn góc, ảnh biên, NMS, ảnh nắn phối cảnh và ảnh làm nét trùng từng phần tử**. Ảnh PNG qua API giải mã ra cũng trùng từng pixel với ảnh làm nét tham chiếu.

SHA-256 của các pixel ảnh làm nét trong cả hai bản:

```text
6bb7f8fdc97fd698dcf4a8fedf773af79663dc631b3daf3e5c804c4792b3d10c
```

Đây là số đo trên một ảnh và một máy; chưa đo trọn luồng trên điện thoại/ngrok. Thời gian tải ảnh lên và tải PNG về phụ thuộc mạng, không nằm trong thời gian xử lý 0,586 giây.

## Chạy lại

PowerShell, từ thư mục dự án:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/benchmark_scanner.py --reference-commit 8853341 --save "$env:TEMP\docuscan-reference.npz"
.\.venv\Scripts\python.exe -X utf8 scripts/benchmark_scanner.py --compare "$env:TEMP\docuscan-reference.npz" --repeat 3
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
node --test tests/test_camera_flow.cjs
```

Kết quả kiểm thử sau thay đổi: 27 kiểm thử Python và 19 kiểm thử JavaScript đạt. Các kiểm thử mới so sánh với phép tính vô hướng cũ, gồm kernel bất đối xứng, nhiều chế độ padding, nhiều khối, góc NMS biên, plateau, chuỗi weak dài và weak ở mép ảnh. Kiểm thử giao diện mô phỏng API bị chậm và camera đổi hướng để xác nhận canvas giữ nguyên ảnh đến khi scan hoàn tất.

## Cập nhật chất lượng ảnh chụp

Sau khi tối ưu tốc độ, bổ sung các thay đổi để giữ chi tiết chữ:

- Camera trực tiếp yêu cầu độ phân giải ưu tiên 2560×1920, dùng `ideal` để thiết bị tự chọn mức hỗ trợ. Chỉ thử lại cấu hình cơ bản khi trình duyệt báo `OverconstrainedError`.
- Yêu cầu lấy nét liên tục khi camera công bố hỗ trợ; nếu điều khiển lấy nét bị từ chối, vẫn cho chụp bình thường.
- JPEG từ canvas dùng chất lượng 0,98 thay vì 0,92.
- Nút HD mở file input với `capture=environment` để người dùng chụp bằng giao diện camera của điện thoại. JPEG/PNG được gửi nguyên bản, không thu nhỏ hay mã hóa lại ở trình duyệt; các định dạng khác chỉ được đổi sang JPEG nếu trình duyệt đọc được.
- API multipage giữ `detection_width=800` và cùng detector/cách nắn giấy. Giới hạn chiều rộng đầu ra tăng từ 1600 lên 2400; ảnh nhỏ không bị phóng lớn. Vì thay đổi giới hạn đầu ra, không so hash ảnh chất lượng mới với hash của cấu hình 1600 ở bảng lịch sử phía trên.

Đo lại API trên cùng ảnh 1920×2560: 0,952 giây, đầu ra 1826×2419, PNG 2.298.189 byte. Đây vẫn là đo trên máy cục bộ, chưa tính truyền ảnh điện thoại/ngrok. Kiểm thử ảnh giấy độ phân giải cao xác nhận bốn góc không đổi trong khi giữ được chiều rộng ảnh lớn hơn mức 1600 cũ. Tổng cộng 28 kiểm thử Python và 25 kiểm thử JavaScript đạt.

Các thay đổi này áp dụng cho ảnh chụp mới; không phục hồi được chi tiết vốn đã thiếu trong ảnh chụp mờ. Chất lượng lấy nét và độ phân giải thực nhận vẫn phụ thuộc thiết bị. Nút HD cho phép chọn đường chụp ảnh thay cho khung hình video khi cần.

Tài liệu API tham khảo: [ràng buộc camera](https://developer.mozilla.org/en-US/docs/Web/API/Media_Capture_and_Streams_API/Constraints), [thuộc tính focusMode](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints), [camera capture trên file input](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Attributes/capture).
