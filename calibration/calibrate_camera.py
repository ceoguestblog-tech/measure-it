from pathlib import Path
import cv2
import numpy as np

from calibration.config import (
    PROJECT_ROOT,
    IMAGE_DIR,
    CHECKERBOARD_SIZE,
    SQUARE_SIZE_MM,
    SUPPORTED_FORMATS,
    MIN_CALIBRATION_IMAGES,
)

OUTPUT_FILE = PROJECT_ROOT / "calibration" / "camera_calibration.npz"


def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            CHECKERBOARD_SIZE,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            return True, corners.astype(np.float32)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD_SIZE,
        None
    )

    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        return True, corners

    return False, None


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Images folder missing: {IMAGE_DIR}")

    columns, rows = CHECKERBOARD_SIZE

    object_template = np.zeros(
        (rows * columns, 3),
        dtype=np.float32
    )

    object_template[:, :2] = np.mgrid[
        0:columns, 0:rows
    ].T.reshape(-1, 2)

    object_template *= SQUARE_SIZE_MM

    object_points = []
    image_points = []

    image_size = None
    detected_count = 0

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"UNREADABLE: {image_path.name}")
            continue

        current_size = (image.shape[1], image.shape[0])

        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(f"SKIPPED: Different image size: {image_path.name}")
            continue

        found, corners = detect_corners(image)

        if found:
            object_points.append(object_template.copy())
            image_points.append(corners)
            detected_count += 1
            print(f"DETECTED: {image_path.name}")
        else:
            print(f"NOT DETECTED: {image_path.name}")

    print("\n--- CALIBRATION INPUT ---")
    print(f"Total images: {len(image_paths)}")
    print(f"Detected images: {detected_count}")

    if detected_count < MIN_CALIBRATION_IMAGES:
        print(
            f"ERROR: Need at least {MIN_CALIBRATION_IMAGES} "
            "detected images."
        )
        return

    rms_error, camera_matrix, distortion, rvecs, tvecs = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )
    )

    total_error = 0

    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            distortion
        )

        original_points = np.asarray(image_points[i]).reshape(-1, 2)
        projected_points = np.asarray(projected).reshape(-1, 2)

        error = np.linalg.norm(
            original_points - projected_points
        ) / len(projected_points)

        total_error += error

    mean_error = total_error / len(object_points)

    np.savez(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        image_width=image_size[0],
        image_height=image_size[1],
        checkerboard_size=CHECKERBOARD_SIZE,
        square_size_mm=SQUARE_SIZE_MM,
        rms_error=rms_error,
        mean_reprojection_error=mean_error
    )

    print("\n--- CALIBRATION RESULTS ---")
    print("Camera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(distortion)
    print(f"\nRMS error: {rms_error:.6f}")
    print(f"Mean reprojection error: {mean_error:.6f}")
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

