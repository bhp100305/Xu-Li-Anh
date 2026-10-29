from __future__ import annotations

import cv2
import numpy as np

from src.canny import canny
from src.document_detector import (
    detect_document,
    draw_document_contour,
)
from src.perspective import scan_perspective
from src.sharpening import (
    high_boost,
    unsharp_mask,
)


def resize_for_detection(
    image: np.ndarray,
    max_width: int = 800,
) -> tuple[np.ndarray, float, float]:
    """
    Thu nhỏ ảnh trước khi chạy Canny và phát hiện tài liệu.

    Việc phát hiện được thực hiện trên ảnh nhỏ để tăng tốc.
    Sau khi tìm được 4 góc, tọa độ sẽ được scale
    trở lại ảnh gốc.

    Args:
        image:
            Ảnh BGR đầu vào.

        max_width:
            Chiều rộng tối đa của ảnh dùng để detection.

    Returns:
        Tuple gồm:
            - Ảnh resize.
            - scale_x để chuyển tọa độ về ảnh gốc.
            - scale_y để chuyển tọa độ về ảnh gốc.
    """

    if image is None:
        raise ValueError(
            "Ảnh đầu vào không được None."
        )

    if image.size == 0:
        raise ValueError(
            "Ảnh đầu vào rỗng."
        )

    if max_width <= 0:
        raise ValueError(
            "max_width phải lớn hơn 0."
        )

    height, width = image.shape[:2]

    # Không cần resize nếu ảnh đã đủ nhỏ
    if width <= max_width:

        return (
            image.copy(),
            1.0,
            1.0,
        )

    ratio = (
        max_width
        / width
    )

    new_width = max_width

    new_height = max(
        1,
        int(
            round(
                height * ratio
            )
        ),
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # Tỷ lệ từ ảnh nhỏ về ảnh gốc
    scale_x = (
        width
        / new_width
    )

    scale_y = (
        height
        / new_height
    )

    return (
        resized,
        scale_x,
        scale_y,
    )


def resize_output(
    image: np.ndarray,
    max_width: int = 1000,
) -> np.ndarray:
    """
    Giảm kích thước ảnh trước bước sharpening.

    Sharpening của project dùng convolution tự cài,
    nên giới hạn kích thước giúp giảm đáng kể
    thời gian xử lý.

    Args:
        image:
            Ảnh sau perspective transform.

        max_width:
            Chiều rộng tối đa.

    Returns:
        Ảnh được resize nếu cần.
    """

    if image is None:
        raise ValueError(
            "Ảnh đầu vào không được None."
        )

    if image.size == 0:
        raise ValueError(
            "Ảnh đầu vào rỗng."
        )

    if max_width <= 0:
        raise ValueError(
            "max_width phải lớn hơn 0."
        )

    height, width = image.shape[:2]

    if width <= max_width:

        return image.copy()

    ratio = (
        max_width
        / width
    )

    new_height = max(
        1,
        int(
            round(
                height * ratio
            )
        ),
    )

    resized = cv2.resize(
        image,
        (
            max_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return resized


def scale_corners_to_original(
    corners: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """
    Chuyển tọa độ 4 góc từ ảnh resize về ảnh gốc.

    Args:
        corners:
            Ma trận 4x2 chứa tọa độ trên ảnh nhỏ.

        scale_x:
            Hệ số scale theo trục X.

        scale_y:
            Hệ số scale theo trục Y.

    Returns:
        Ma trận 4x2 tọa độ trên ảnh gốc.
    """

    if corners.shape != (4, 2):

        raise ValueError(
            "corners phải có kích thước (4, 2)."
        )

    original_corners = (
        corners
        .astype(
            np.float32
        )
        .copy()
    )

    original_corners[:, 0] *= (
        scale_x
    )

    original_corners[:, 1] *= (
        scale_y
    )

    return original_corners


def scan_document(
    image: np.ndarray,
    sharpening_method: str = "unsharp",
    sharpening_strength: float = 1.0,
    gaussian_size: int = 5,
    gaussian_sigma: float = 1.4,
    threshold_sigma: float = 0.33,
    detection_width: int = 800,
    output_width: int = 1000,
    verbose: bool = True,
) -> dict[str, object]:
    """
    Thực hiện toàn bộ pipeline scan tài liệu.

    Pipeline:
        1. Resize ảnh để detection.
        2. Canny tự cài.
        3. Phát hiện tài liệu.
           - Ưu tiên contour từ Canny.
           - Nếu thất bại, fallback sang vùng giấy sáng.
        4. Scale 4 góc về ảnh gốc.
        5. Perspective Transform trên ảnh gốc.
        6. Resize ảnh perspective nếu quá lớn.
        7. Làm nét bằng Unsharp hoặc High Boost.

    Args:
        image:
            Ảnh BGR đầu vào.

        sharpening_method:
            "unsharp" hoặc "highboost".

        sharpening_strength:
            Cường độ làm nét.

        gaussian_size:
            Kích thước Gaussian kernel trong Canny.

        gaussian_sigma:
            Sigma Gaussian trong Canny.

        threshold_sigma:
            Hệ số tính threshold tự động.

        detection_width:
            Chiều rộng tối đa ảnh dùng để detection.

        output_width:
            Chiều rộng tối đa trước sharpening.

        verbose:
            Có in tiến trình ra terminal hay không.

    Returns:
        Dictionary chứa toàn bộ kết quả
        trung gian và ảnh cuối.

    Raises:
        ValueError:
            Nếu input hoặc tham số không hợp lệ.

        RuntimeError:
            Nếu không phát hiện được tài liệu.
    """

    # ========================================================
    # VALIDATE
    # ========================================================

    if image is None:

        raise ValueError(
            "Ảnh đầu vào không được None."
        )

    if image.size == 0:

        raise ValueError(
            "Ảnh đầu vào rỗng."
        )

    if detection_width <= 0:

        raise ValueError(
            "detection_width phải lớn hơn 0."
        )

    if output_width <= 0:

        raise ValueError(
            "output_width phải lớn hơn 0."
        )


    # ========================================================
    # STEP 1: RESIZE CHO DETECTION
    # ========================================================

    if verbose:

        print()

        print(
            "[1/5] Thu nhỏ ảnh để phát hiện tài liệu..."
        )


    (
        detection_image,
        scale_x,
        scale_y,
    ) = resize_for_detection(
        image=image,
        max_width=detection_width,
    )


    if verbose:

        print(
            "      Ảnh gốc:",
            image.shape,
        )

        print(
            "      Ảnh detection:",
            detection_image.shape,
        )


    # ========================================================
    # STEP 2: CANNY TỰ CÀI
    # ========================================================

    if verbose:

        print(
            "[2/5] Đang chạy Canny..."
        )


    canny_result = canny(
        image=detection_image,
        gaussian_size=gaussian_size,
        gaussian_sigma=gaussian_sigma,
        threshold_sigma=threshold_sigma,
    )


    edge = canny_result[
        "edge"
    ]


    if not isinstance(
        edge,
        np.ndarray,
    ):

        raise RuntimeError(
            "Canny không trả về ảnh biên hợp lệ."
        )


    # ========================================================
    # STEP 3: DOCUMENT DETECTION
    # ========================================================

    if verbose:

        print(
            "[3/5] Đang tìm vùng tài liệu..."
        )


    detection_result = detect_document(
        image=detection_image,
        edge_image=edge,
    )


    resized_corners = detection_result[
        "corners"
    ]


    detection_method = str(
        detection_result.get(
            "method",
            "unknown",
        )
    )


    if resized_corners is None:

        raise RuntimeError(
            "Không phát hiện được vùng tài liệu."
        )


    if not isinstance(
        resized_corners,
        np.ndarray,
    ):

        raise RuntimeError(
            "Tọa độ tài liệu không hợp lệ."
        )


    if resized_corners.shape != (
        4,
        2,
    ):

        raise RuntimeError(
            "Không xác định được đủ 4 góc tài liệu."
        )


    if verbose:

        print(
            "      Phương pháp:",
            detection_method,
        )

        print(
            "      4 góc trên ảnh nhỏ:"
        )

        print(
            resized_corners
        )


    # ========================================================
    # SCALE 4 GÓC VỀ ẢNH GỐC
    # ========================================================

    original_corners = (
        scale_corners_to_original(
            corners=resized_corners,
            scale_x=scale_x,
            scale_y=scale_y,
        )
    )


    # Tránh tọa độ vượt ra ngoài ảnh
    original_height, original_width = (
        image.shape[:2]
    )


    original_corners[:, 0] = np.clip(
        original_corners[:, 0],
        0,
        original_width - 1,
    )


    original_corners[:, 1] = np.clip(
        original_corners[:, 1],
        0,
        original_height - 1,
    )


    # Preview trên ảnh gốc
    detection_preview = (
        draw_document_contour(
            image=image,
            corners=original_corners,
        )
    )


    # ========================================================
    # STEP 4: PERSPECTIVE TRANSFORM
    # ========================================================

    if verbose:

        print(
            "[4/5] Đang nắn phối cảnh..."
        )


    perspective_result = (
        scan_perspective(
            image=image,
            corners=original_corners,
        )
    )


    warped_original = (
        perspective_result[
            "warped"
        ]
    )


    if not isinstance(
        warped_original,
        np.ndarray,
    ):

        raise RuntimeError(
            "Perspective Transform thất bại."
        )


    # ========================================================
    # RESIZE TRƯỚC SHARPEN
    # ========================================================

    warped = resize_output(
        image=warped_original,
        max_width=output_width,
    )


    # ========================================================
    # STEP 5: SHARPENING
    # ========================================================

    if verbose:

        print(
            "[5/5] Đang làm nét ảnh..."
        )


    method = (
        sharpening_method
        .strip()
        .lower()
    )


    if method == "unsharp":

        sharpening_result = (
            unsharp_mask(
                image=warped,
                amount=sharpening_strength,
            )
        )


    elif method == "highboost":

        # High Boost yêu cầu k > 1
        if sharpening_strength <= 1:

            sharpening_strength = (
                1.5
            )


        sharpening_result = (
            high_boost(
                image=warped,
                k=sharpening_strength,
            )
        )


    else:

        raise ValueError(
            "sharpening_method phải là "
            "'unsharp' hoặc 'highboost'."
        )


    scanned = sharpening_result[
        "sharpened"
    ]


    if not isinstance(
        scanned,
        np.ndarray,
    ):

        raise RuntimeError(
            "Sharpening không trả về ảnh hợp lệ."
        )


    # ========================================================
    # SUCCESS LOG
    # ========================================================

    if verbose:

        print(
            "      Kích thước perspective:",
            warped_original.shape,
        )

        print(
            "      Kích thước ảnh cuối:",
            scanned.shape,
        )

        print(
            "Hoàn thành pipeline scan."
        )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        # -------------------------------
        # INPUT
        # -------------------------------

        "original": image,

        "detection_image": (
            detection_image
        ),


        # -------------------------------
        # CANNY
        # -------------------------------

        "gray": canny_result[
            "gray"
        ],

        "blurred": canny_result[
            "blurred"
        ],

        "magnitude": canny_result[
            "magnitude"
        ],

        "nms": canny_result[
            "nms"
        ],

        "edge": edge,

        "low_threshold": (
            canny_result[
                "low_threshold"
            ]
        ),

        "high_threshold": (
            canny_result[
                "high_threshold"
            ]
        ),


        # -------------------------------
        # DOCUMENT DETECTION
        # -------------------------------

        "corners_detection": (
            resized_corners
        ),

        "corners": (
            original_corners
        ),

        "detection_method": (
            detection_method
        ),

        "detection_preview": (
            detection_preview
        ),


        # -------------------------------
        # PERSPECTIVE
        # -------------------------------

        "warped_original": (
            warped_original
        ),

        "warped": warped,

        "width": (
            perspective_result[
                "width"
            ]
        ),

        "height": (
            perspective_result[
                "height"
            ]
        ),


        # -------------------------------
        # FINAL RESULT
        # -------------------------------

        "sharpened": scanned,

        "final_width": (
            scanned.shape[1]
        ),

        "final_height": (
            scanned.shape[0]
        ),
    }


def quick_scan(
    image: np.ndarray,
) -> np.ndarray:
    """
    Phiên bản rút gọn của scanner.

    Nhận ảnh BGR và trả trực tiếp ảnh scan cuối.

    Args:
        image:
            Ảnh BGR đầu vào.

    Returns:
        Ảnh scan hoàn chỉnh.
    """

    result = scan_document(
        image=image,
        verbose=False,
    )


    scanned = result[
        "sharpened"
    ]


    if not isinstance(
        scanned,
        np.ndarray,
    ):

        raise RuntimeError(
            "Kết quả scan không hợp lệ."
        )


    return scanned