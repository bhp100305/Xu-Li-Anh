
import io
import cv2
import numpy as np

from PIL import Image, ImageOps
from flask import (
    Blueprint, jsonify, render_template,
    request, send_file
)

from src.scanner import scan_document
from src.document_detector import find_document_from_edges, find_document_from_brightness

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
    # Check dimensions before OpenCV allocates the decoded pixel buffer.
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
        ratio = min(1.0, 480.0 / max(height, width))
        small = cv2.resize(image, (max(1, round(width * ratio)), max(1, round(height * ratio))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
        edges = cv2.Canny(blurred, 40, 120)
        corners = find_document_from_edges(edges)
        if corners is None:
            corners = find_document_from_brightness(small)
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
        # Keep the original detector, coordinate scaling, perspective and
        # sharpening pipeline. Camera overlay coordinates are guidance only.
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

        return send_file(
            io.BytesIO(encoded.tobytes()),
            mimetype="image/png"
        )

    except (ValueError, RuntimeError) as exc:
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
