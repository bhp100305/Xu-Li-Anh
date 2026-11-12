
import io
import json
import cv2
import numpy as np

from PIL import Image, ImageOps
from flask import (
    Blueprint, jsonify, render_template,
    request, send_file
)

from src.scanner import scan_document, resize_output
from src.document_detector import detect_document
from src.perspective import perspective_transform
from src.sharpening import unsharp_mask

multipage_bp = Blueprint("multipage", __name__)


@multipage_bp.get("/multipage")
def multipage_page():
    return render_template("multipage.html")


def read_uploaded_image():
    file = request.files.get("image")
    if not file:
        raise ValueError("Chưa chọn ảnh")
    raw = file.read(12 * 1024 * 1024 + 1)
    if len(raw) > 12 * 1024 * 1024:
        raise ValueError("Ảnh vượt quá 12 MB")
    # Giới hạn số pixel trước khi giải mã để tránh tốn quá nhiều bộ nhớ.
    with Image.open(io.BytesIO(raw)) as source:
        if source.width * source.height > 20000000:
            raise ValueError("Ảnh vượt quá 20 triệu pixel")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Ảnh không hợp lệ")
    return image


@multipage_bp.post("/api/multipage/detect")
def detect_page():
    """Lightweight camera guidance; final processing is an explicit scan request."""
    try:
        image = read_uploaded_image()
        height, width = image.shape[:2]
        # Thu nhỏ khung hình để gợi ý mép giấy nhanh trên camera.
        ratio = min(1.0, 480.0 / max(height, width))
        small = cv2.resize(image, (max(1, round(width * ratio)), max(1, round(height * ratio))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
        edges = cv2.Canny(blurred, 40, 120)
        corners = detect_document(small, edges, preview_only=True)["corners"]
        if corners is None:
            return jsonify(corners=None)
        corners = corners / np.array([small.shape[1] - 1, small.shape[0] - 1]).clip(1)
        corners = np.clip(corners, 0, 1)
        if len(np.unique(corners, axis=0)) != 4 or not cv2.isContourConvex(corners.astype(np.float32)):
            return jsonify(corners=None)
        return jsonify(corners=corners.tolist())
    except (ValueError, OSError, Image.DecompressionBombError):
        return jsonify(error="Không thể đọc ảnh để nhận diện trang giấy"), 400


@multipage_bp.post("/api/multipage/scan")
def scan_page():
    """Scan một ảnh và trả về PNG đã xử lý."""
    file = request.files.get("image")
    if not file:
        return jsonify(error="Chưa chọn ảnh"), 400

    raw = file.read(12 * 1024 * 1024 + 1)
    if len(raw) > 12 * 1024 * 1024:
        return jsonify(error="Ảnh quá lớn"), 413

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(error="Ảnh không hợp lệ"), 400

    try:
        # Quét trên ảnh gốc; khung gợi ý camera không quyết định vùng cắt.
        result = scan_document(
            image=image,
            detection_width=800,
            output_width=2400,
            verbose=False,
        )
        scanned = result["sharpened"]

        success, encoded = cv2.imencode(
            ".png", scanned
        )

        if not success:
            raise RuntimeError("Không thể mã hóa ảnh")

        response = send_file(
            io.BytesIO(encoded.tobytes()),
            mimetype="image/png"
        )
        if "corners" in result:
            height, width = image.shape[:2]
            normalized = result["corners"] / np.float32([max(1, width - 1), max(1, height - 1)])
            response.headers["X-Document-Corners"] = json.dumps(normalized.tolist())
        return response

    except (ValueError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 422


@multipage_bp.post("/api/multipage/rectify")
def rectify_page():
    """Apply only the four corners explicitly confirmed in the crop editor.

    The live camera overlay never calls this route. Work on original pixels,
    avoiding another round of detection or repeated resampling of a scan.
    """
    try:
        image = read_uploaded_image()
        corners = np.asarray(json.loads(request.form.get("corners", "null")), dtype=np.float32)
        if (corners.shape != (4, 2) or not np.isfinite(corners).all()
                or np.any(corners < 0) or np.any(corners > 1)):
            raise ValueError("Cần chọn đủ bốn góc trong ảnh gốc")
        # Loại khung tự cắt, trùng góc hoặc quá nhỏ trước khi nắn ảnh.
        sides = np.roll(corners, -1, axis=0) - corners
        cross = sides[:, 0] * np.roll(sides[:, 1], -1) - sides[:, 1] * np.roll(sides[:, 0], -1)
        if np.any(cross <= 0) or cv2.contourArea(corners) < 0.01:
            raise ValueError("Bốn góc phải tạo thành khung giấy không giao nhau")
        height, width = image.shape[:2]
        # Đổi góc người dùng chọn từ tỉ lệ 0–1 sang tọa độ pixel ảnh gốc.
        pixels = corners * np.float32([width - 1, height - 1])
        warped = perspective_transform(image, pixels, preserve_order=True)
        scanned = unsharp_mask(resize_output(warped, max_width=2400))["sharpened"]
        success, encoded = cv2.imencode(".png", scanned)
        if not success:
            raise RuntimeError("Không thể mã hóa ảnh")
        response = send_file(io.BytesIO(encoded.tobytes()), mimetype="image/png")
        response.headers["X-Document-Corners"] = json.dumps(corners.tolist())
        return response
    except (ValueError, TypeError, OSError, Image.DecompressionBombError, cv2.error) as exc:
        return jsonify(error=str(exc) or "Không thể đọc ảnh gốc"), 400
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 422


@multipage_bp.post("/api/multipage/pdf")
def create_pdf():
    """Gộp các ảnh đã scan thành PDF nhiều trang."""
    files = request.files.getlist("pages")

    if not 1 <= len(files) <= 15:
        return jsonify(
            error="Vui lòng chọn từ 1 đến 15 trang"
        ), 400

    images = []

    try:
        for file in files:
            raw = file.read(8 * 1024 * 1024 + 1)

            if len(raw) > 8 * 1024 * 1024:
                return jsonify(
                    error="Một trang vượt quá 8 MB"
                ), 413

            with Image.open(io.BytesIO(raw)) as source:
                if source.width * source.height > 20000000:
                    return jsonify(
                        error="Kích thước ảnh quá lớn"
                    ), 400

                page = ImageOps.exif_transpose(source)
                images.append(page.convert("RGB"))

        pdf_buffer = io.BytesIO()

        # Ghép các ảnh theo thứ tự người dùng sắp xếp thành PDF nhiều trang.
        images[0].save(
            pdf_buffer,
            format="PDF",
            save_all=True,
            append_images=images[1:],
            resolution=150.0
        )

        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="document_scan.pdf"
        )

    except Exception:
        return jsonify(
            error="Không thể tạo PDF từ các trang"
        ), 400

    finally:
        for image in images:
            image.close()
