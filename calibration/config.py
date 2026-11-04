from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Checkerboard internal corners: columns, rows
CHECKERBOARD_SIZE = (7, 7)

# Temporary assumed square size (actual size must be measured)
SQUARE_SIZE_MM = 25.0

# Calibration images directory
IMAGE_DIR = PROJECT_ROOT / "images"

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}

# Minimum number of detected calibration images
MIN_CALIBRATION_IMAGES = 10
