from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "images"

CHECKERBOARD_SIZE = (7, 7)


def detect_checkerboard(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return False, None, "IMAGE ERROR"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # First attempt: advanced SB detector
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )

        found, corners = cv2.findChessboardCornersSB(
            gray,
            CHECKERBOARD_SIZE,
            flags=flags
        )

        if found:
            return True, corners, "ADVANCED SB"

    # Second attempt: standard detector
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

        return True, corners, "STANDARD"

    return False, None, "NOT DETECTED"


def main():
    image_paths = sorted(IMAGE_DIR.iterdir())

    total = 0
    detected = 0
    missed = []

    for image_path in image_paths:
        if image_path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }:
            continue

        total += 1
        found, corners, method = detect_checkerboard(image_path)

        if found:
            detected += 1
            print(f"DETECTED ({method}): {image_path.name}")
        else:
            missed.append(image_path.name)
            print(f"NOT DETECTED: {image_path.name}")

    print("\n--- ADVANCED DETECTION RESULTS ---")
    print(f"Total images: {total}")
    print(f"Detected: {detected}")
    print(f"Not detected: {len(missed)}")

    if missed:
        print("\nMissed images:")
        for filename in missed:
            print(filename)


if __name__ == "__main__":
    main()
