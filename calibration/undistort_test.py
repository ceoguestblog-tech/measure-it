from pathlib import Path
import cv2
import numpy as np

from calibration.config import IMAGE_DIR, CHECKERBOARD_SIZE, SUPPORTED_FORMATS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = PROJECT_ROOT / "calibration" / "camera_calibration.npz"
OUTPUT_FILE = PROJECT_ROOT / "calibration" / "undistorted_preview.jpg"


def main():
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError("Camera calibration file not found.")

    data = np.load(CALIBRATION_FILE)

    camera_matrix = data["camera_matrix"]
    distortion = data["distortion_coefficients"]

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
    )

    selected_image = None

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            None
        )

        if found:
            selected_image = image
            print(f"Selected image: {path.name}")
            break

    if selected_image is None:
        raise RuntimeError("No checkerboard image found.")

    height, width = selected_image.shape[:2]

    calibrated_width = int(data["image_width"])
    calibrated_height = int(data["image_height"])

    if (width, height) != (calibrated_width, calibrated_height):
        raise ValueError("Image dimensions do not match calibration.")

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (width, height),
        1,
        (width, height)
    )

    undistorted = cv2.undistort(
        selected_image,
        camera_matrix,
        distortion,
        None,
        new_camera_matrix
    )

    x, y, w, h = roi

    if w > 0 and h > 0:
        undistorted = undistorted[y:y+h, x:x+w]

    success = cv2.imwrite(str(OUTPUT_FILE), undistorted)

    if not success:
        raise RuntimeError("Could not save undistorted image.")

    print("\n--- UNDISTORTION RESULTS ---")
    print(f"Original dimensions: {width} x {height}")
    print(f"Corrected dimensions: {undistorted.shape[1]} x {undistorted.shape[0]}")
    print(f"Saved: {OUTPUT_FILE}")

    cv2.imshow("Original Image", selected_image)
    cv2.imshow("Undistorted Image", undistorted)

    print("\nPress any key to close image windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
