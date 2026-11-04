# Plan

## Object

I will measure a calculator using a phone camera.

## Workflow

1. Capture at least 20 checkerboard calibration images.
2. Calibrate the camera with OpenCV.
3. Capture at least 70 calculator images.
4. Undistort the images.
5. Label calculator masks.
6. Create a 70/20/10 train-validation-test split.
7. Train a segmentation model.
8. Detect a reference object.
9. Convert pixels to millimetres.
10. Compare predictions with ruler measurements.

## Objectives

- Keep mean measurement error below 3 mm.
- Achieve at least 0.80 mask IoU on the test set.

## Deliberately skipped

If time runs out, I will prioritise camera calibration, dataset creation, segmentation, and measurement.

## Decision record

I will use OpenCV and a permitted segmentation model. I will not use Ultralytics YOLO or Roboflow.