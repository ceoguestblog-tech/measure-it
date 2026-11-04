from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CALIBRATION_FILE = PROJECT_ROOT / "calibration" / "camera_calibration.npz"
IMAGE_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR = PROJECT_ROOT / "measurement" / "undistorted"

def undistort_image(image_path):
    image_path = Path(image_path)

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError("Camera calibration file not found.")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    calibration = np.load(CALIBRATION_FILE)

    camera_matrix = calibration["camera_matrix"]
    distortion = calibration["distortion_coefficients"]

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read input image.")

    height, width = image.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (width, height),
        1,
        (width, height)
    )

    corrected = cv2.undistort(
        image,
        camera_matrix,
        distortion,
        None,
        new_camera_matrix
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x, y, w, h = roi

    if w > 0 and h > 0:
        corrected = corrected[y:y+h, x:x+w]

    output_path = OUTPUT_DIR / f"{image_path.stem}_undistorted.jpg"

    cv2.imwrite(str(output_path), corrected)

    print("Original image:", image_path.name)
    print("Corrected dimensions:", corrected.shape[1], "x", corrected.shape[0])
    print("Saved:", output_path)

    return output_path


if __name__ == "__main__":
    images = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.jpeg"))

    if not images:
        raise FileNotFoundError("No JPG images found in images folder.")

    undistort_image(images[0])
