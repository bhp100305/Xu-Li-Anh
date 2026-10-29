from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, render_template, request
from werkzeug.datastructures import FileStorage

from src.comparison import compare_algorithms
from src.scanner import scan_document


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"

UPLOAD_FOLDER = STATIC_DIR / "uploads"

RESULT_FOLDER = STATIC_DIR / "results"


UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

# Tối đa 15 MB / file
app.config["MAX_CONTENT_LENGTH"] = (
    15 * 1024 * 1024
)


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def allowed_file(
    filename: str,
) -> bool:
    """
    Kiểm tra file upload có phải định dạng ảnh được hỗ trợ.

    Args:
        filename:
            Tên file upload.

    Returns:
        True nếu hợp lệ.
    """

    extension = Path(
        filename
    ).suffix.lower()

    return (
        extension
        in ALLOWED_EXTENSIONS
    )


def create_job_id() -> str:
    """
    Tạo ID riêng cho mỗi lần xử lý.

    Returns:
        Chuỗi ID 12 ký tự.
    """

    return uuid.uuid4().hex[:12]


def save_uploaded_file(
    file: FileStorage,
    job_id: str,
    prefix: str,
) -> tuple[Path, str]:
    """
    Lưu file upload vào static/uploads.

    Args:
        file:
            FileStorage từ Flask.

        job_id:
            ID của lần xử lý.

        prefix:
            Tiền tố tên file.

    Returns:
        Tuple:
            đường dẫn tuyệt đối
            đường dẫn tương đối cho Flask static
    """

    if not file.filename:
        raise ValueError(
            "Tên file không hợp lệ."
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    filename = (
        f"{job_id}_{prefix}{extension}"
    )

    file_path = (
        UPLOAD_FOLDER
        / filename
    )

    file.save(
        str(file_path)
    )

    relative_path = (
        f"uploads/{filename}"
    )

    return (
        file_path,
        relative_path,
    )


def read_image(
    path: Path,
) -> np.ndarray:
    """
    Đọc ảnh bằng OpenCV.

    Args:
        path:
            Đường dẫn ảnh.

    Returns:
        Ảnh BGR NumPy.

    Raises:
        ValueError:
            Nếu không đọc được ảnh.
    """

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise ValueError(
            "Không thể đọc file ảnh."
        )

    return image


def save_image(
    image: np.ndarray,
    filename: str,
) -> str:
    """
    Lưu một ảnh vào static/results.

    Args:
        image:
            Ma trận ảnh.

        filename:
            Tên file cần lưu.

    Returns:
        Đường dẫn tương đối dùng cho Flask static.

    Raises:
        RuntimeError:
            Nếu OpenCV không lưu được ảnh.
    """

    file_path = (
        RESULT_FOLDER
        / filename
    )

    success = cv2.imwrite(
        str(file_path),
        image,
    )

    if not success:
        raise RuntimeError(
            f"Không thể lưu ảnh {filename}."
        )

    return (
        f"results/{filename}"
    )


def validate_uploaded_image() -> FileStorage:
    """
    Kiểm tra file ảnh từ request.

    Returns:
        FileStorage hợp lệ.

    Raises:
        ValueError:
            Nếu request không chứa ảnh hợp lệ.
    """

    if "image" not in request.files:
        raise ValueError(
            "Không nhận được file ảnh."
        )

    file = request.files["image"]

    if not file.filename:
        raise ValueError(
            "Bạn chưa chọn ảnh."
        )

    if not allowed_file(
        file.filename
    ):
        raise ValueError(
            "Định dạng không hỗ trợ. "
            "Chỉ chấp nhận JPG, JPEG, PNG, BMP hoặc WEBP."
        )

    return file


def get_float_form_value(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Lấy một giá trị float từ HTML form.

    Giá trị được giới hạn trong khoảng min-max.

    Args:
        name:
            Tên field.

        default:
            Giá trị mặc định.

        minimum:
            Giá trị nhỏ nhất.

        maximum:
            Giá trị lớn nhất.

    Returns:
        Giá trị float hợp lệ.
    """

    try:
        value = float(
            request.form.get(
                name,
                str(default),
            )
        )

    except ValueError:
        value = default

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


# ============================================================
# HOME PAGE
# ============================================================


@app.route("/", methods=["GET", "POST"])
def index() -> str | tuple[str, int]:
    """
    Trang chính Document Scanner.
    """

    if request.method == "POST":
        action = request.form.get("action", "scan")
        return scan() if action == "scan" else compare()
    return render_template("index.html", active_tab="scan")


# ============================================================
# DOCUMENT SCANNER
# ============================================================


def scan() -> tuple[str, int] | str:
    """
    Nhận ảnh upload và chạy toàn bộ pipeline scan.

    Pipeline:
        upload
        -> Canny
        -> document detection
        -> perspective transform
        -> sharpening
        -> result page
    """

    try:

        # ----------------------------------------------------
        # 1. VALIDATE FILE
        # ----------------------------------------------------

        file = validate_uploaded_image()


        # ----------------------------------------------------
        # 2. JOB ID
        # ----------------------------------------------------

        job_id = create_job_id()


        # ----------------------------------------------------
        # 3. SAVE ORIGINAL IMAGE
        # ----------------------------------------------------

        (
            upload_path,
            original_static_path,
        ) = save_uploaded_file(
            file=file,
            job_id=job_id,
            prefix="original",
        )


        # ----------------------------------------------------
        # 4. READ IMAGE
        # ----------------------------------------------------

        image = read_image(
            upload_path
        )


        # ----------------------------------------------------
        # 5. FORM SETTINGS
        # ----------------------------------------------------

        sharpening_method = (
            request.form.get(
                "sharpening_method",
                "unsharp",
            )
            .strip()
            .lower()
        )


        if sharpening_method not in {
            "unsharp",
            "highboost",
        }:
            sharpening_method = (
                "unsharp"
            )


        sharpening_strength = (
            get_float_form_value(
                name="sharpening_strength",
                default=1.0,
                minimum=0.5,
                maximum=3.0,
            )
        )


        # High Boost đúng nghĩa cần k > 1
        if (
            sharpening_method
            == "highboost"
            and sharpening_strength <= 1
        ):
            sharpening_strength = 1.5


        # ----------------------------------------------------
        # 6. RUN SCANNER
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("DOCUMENT SCANNER")
        print("=" * 60)

        start_time = (
            time.perf_counter()
        )


        result = scan_document(
            image=image,

            sharpening_method=(
                sharpening_method
            ),

            sharpening_strength=(
                sharpening_strength
            ),

            detection_width=800,

            output_width=1000,

            verbose=True,
        )


        processing_time = (
            time.perf_counter()
            - start_time
        )


        # ----------------------------------------------------
        # 7. GET RESULTS
        # ----------------------------------------------------

        edge = result[
            "edge"
        ]

        detection = result[
            "detection_preview"
        ]

        perspective = result[
            "warped"
        ]

        final_image = result[
            "sharpened"
        ]


        if not all(
            isinstance(
                item,
                np.ndarray,
            )
            for item in (
                edge,
                detection,
                perspective,
                final_image,
            )
        ):
            raise RuntimeError(
                "Pipeline trả về dữ liệu ảnh không hợp lệ."
            )


        # ----------------------------------------------------
        # 8. SAVE RESULTS
        # ----------------------------------------------------

        edge_path = save_image(
            edge,
            f"{job_id}_edge.jpg",
        )


        detection_path = save_image(
            detection,
            f"{job_id}_detection.jpg",
        )


        perspective_path = save_image(
            perspective,
            f"{job_id}_perspective.jpg",
        )


        final_path = save_image(
            final_image,
            f"{job_id}_final.jpg",
        )


        # ----------------------------------------------------
        # 9. CORNERS
        # ----------------------------------------------------

        corners_raw = result[
            "corners"
        ]


        corners: list[
            list[float]
        ] = []


        if isinstance(
            corners_raw,
            np.ndarray,
        ):

            corners = (
                corners_raw
                .round(2)
                .tolist()
            )


        # ----------------------------------------------------
        # 10. RESULT PAGE
        # ----------------------------------------------------

        print()
        print(
            "Scan hoàn thành sau "
            f"{processing_time:.2f} giây"
        )


        return render_template(
            "index.html",

            original_image=(
                original_static_path
            ),

            edge_image=(
                edge_path
            ),

            detection_image=(
                detection_path
            ),

            perspective_image=(
                perspective_path
            ),

            final_image=(
                final_path
            ),

            low_threshold=round(
                float(
                    result[
                        "low_threshold"
                    ]
                ),
                2,
            ),

            high_threshold=round(
                float(
                    result[
                        "high_threshold"
                    ]
                ),
                2,
            ),

            corners=corners,

            width=result[
                "final_width"
            ],

            height=result[
                "final_height"
            ],

            processing_time=round(
                processing_time,
                2,
            ),

            sharpening_method=(
                sharpening_method
            ),

            sharpening_strength=(
                sharpening_strength
            ),
        )


    except (
        ValueError,
        RuntimeError,
    ) as error:

        print(
            "Scanner error:",
            error,
        )

        return (
            render_template(
                "index.html",
                error=str(error),
            ),
            400,
        )


# ============================================================
# ALGORITHM COMPARISON PAGE
# ============================================================


def compare() -> tuple[str, int] | str:
    """
    Trang so sánh các thuật toán phát hiện biên.

    GET:
        Hiển thị form upload.

    POST:
        Chạy:
        - Sobel
        - Prewitt
        - Laplacian
        - LoG
        - Canny tự cài
        - cv2.Canny
    """

    # ========================================================
    # GET
    # ========================================================

    if request.method == "GET":

        return render_template(
            "index.html",
            comparison=False,
                active_tab="compare",
        )


    # ========================================================
    # POST
    # ========================================================

    try:

        # ----------------------------------------------------
        # 1. VALIDATE
        # ----------------------------------------------------

        file = validate_uploaded_image()


        # ----------------------------------------------------
        # 2. JOB ID
        # ----------------------------------------------------

        job_id = create_job_id()


        # ----------------------------------------------------
        # 3. SAVE UPLOAD
        # ----------------------------------------------------

        (
            upload_path,
            _,
        ) = save_uploaded_file(
            file=file,
            job_id=job_id,
            prefix="compare_upload",
        )


        # ----------------------------------------------------
        # 4. READ
        # ----------------------------------------------------

        image = read_image(
            upload_path
        )


        # ----------------------------------------------------
        # 5. RUN COMPARISON
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print(
            "EDGE DETECTOR COMPARISON"
        )
        print("=" * 60)


        comparison_start = (
            time.perf_counter()
        )


        comparison_result = (
            compare_algorithms(
                image=image,
                max_width=700,
            )
        )


        comparison_total_time = (
            time.perf_counter()
            - comparison_start
        )


        # ----------------------------------------------------
        # 6. SAVE WORKING ORIGINAL
        # ----------------------------------------------------

        comparison_original = (
            comparison_result[
                "original"
            ]
        )


        if not isinstance(
            comparison_original,
            np.ndarray,
        ):
            raise RuntimeError(
                "Ảnh so sánh không hợp lệ."
            )


        original_compare_path = (
            save_image(
                comparison_original,
                (
                    f"{job_id}"
                    "_compare_original.jpg"
                ),
            )
        )


        # ----------------------------------------------------
        # 7. SAVE EACH ALGORITHM RESULT
        # ----------------------------------------------------

        raw_results = (
            comparison_result[
                "results"
            ]
        )


        if not isinstance(
            raw_results,
            dict,
        ):
            raise RuntimeError(
                "Kết quả comparison không hợp lệ."
            )


        template_results: list[
            dict[str, Any]
        ] = []


        for (
            algorithm_key,
            algorithm_data,
        ) in raw_results.items():

            if not isinstance(
                algorithm_data,
                dict,
            ):
                continue


            edge_image = (
                algorithm_data.get(
                    "edge"
                )
            )


            if not isinstance(
                edge_image,
                np.ndarray,
            ):
                continue


            result_filename = (
                f"{job_id}_"
                f"{algorithm_key}.jpg"
            )


            static_path = save_image(
                edge_image,
                result_filename,
            )


            algorithm_time = float(
                algorithm_data.get(
                    "time",
                    0.0,
                )
            )


            template_results.append(
                {
                    "key": (
                        algorithm_key
                    ),

                    "name": (
                        algorithm_data.get(
                            "name",
                            algorithm_key,
                        )
                    ),

                    "image": (
                        static_path
                    ),

                    "time": round(
                        algorithm_time,
                        4,
                    ),
                }
            )


        # ----------------------------------------------------
        # 8. RENDER RESULT
        # ----------------------------------------------------

        print()
        print(
            "Comparison hoàn thành sau "
            f"{comparison_total_time:.2f} giây"
        )


        return render_template(
            "index.html",

            comparison=True,
            active_tab="compare",

            original_image=(
                original_compare_path
            ),

            results=(
                template_results
            ),

            low_threshold=round(
                float(
                    comparison_result[
                        "low_threshold"
                    ]
                ),
                2,
            ),

            high_threshold=round(
                float(
                    comparison_result[
                        "high_threshold"
                    ]
                ),
                2,
            ),

            width=(
                comparison_result[
                    "width"
                ]
            ),

            height=(
                comparison_result[
                    "height"
                ]
            ),

            total_time=round(
                comparison_total_time,
                2,
            ),
        )


    except (
        ValueError,
        RuntimeError,
    ) as error:

        print(
            "Comparison error:",
            error,
        )

        return (
            render_template(
                "index.html",

                comparison=False,
                active_tab="compare",

                error=str(error),
            ),
            400,
        )


# ============================================================
# FILE TOO LARGE
# ============================================================


@app.errorhandler(413)
def file_too_large(
    error: Exception,
) -> tuple[str, int]:
    """
    Xử lý file vượt quá 15 MB.
    """

    message = (
        "Ảnh quá lớn. "
        "Kích thước tối đa là 15 MB."
    )


    if request.path.startswith(
        "/compare"
    ):

        return (
            render_template(
                "index.html",

                comparison=False,
                active_tab="compare",

                error=message,
            ),
            413,
        )


    return (
        render_template(
            "index.html",
            error=message,
        ),
        413,
    )


# ============================================================
# RUN APP
# ============================================================


if __name__ == "__main__":

    print()
    print("=" * 60)

    print(
        "DOCUMENT SCANNER WEB"
    )

    print("=" * 60)

    print(
        "Scanner:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    print()
    print("=" * 60)
    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )



