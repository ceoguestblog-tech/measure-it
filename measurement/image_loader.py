from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "images"

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def get_project_images():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGE_DIR}")

    return sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS
    )


def validate_images():
    image_paths = get_project_images()
    valid_images = []
    invalid_images = []

    for path in image_paths:
        try:
            image = load_image(path)
            height, width = image.shape[:2]

            valid_images.append({
                "filename": path.name,
                "width": width,
                "height": height
            })

        except (ValueError, FileNotFoundError):
            invalid_images.append(path.name)

    return valid_images, invalid_images


if __name__ == "__main__":
    valid, invalid = validate_images()

    print(f"Total valid images: {len(valid)}")
    print(f"Total invalid images: {len(invalid)}")

    for item in valid:
        print(
            f"{item['filename']} | "
            f"{item['width']} x {item['height']}"
        )

    if invalid:
        print("\nInvalid images:")
        for filename in invalid:
            print(filename)
