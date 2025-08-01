import cv2
import os
from typing import List, Tuple
from ultralytics import YOLO

from src.annotator import annotate_image
from src.detectors.candlestick_detector import detect_patterns
from src.image_parser import ImageParser

def main():
    """
    Main function to run the full hybrid chart pattern detection pipeline.
    This uses both a rule-based engine for candlesticks and a YOLOv8 model
    for chart patterns.
    """
    # --- Configuration ---
    input_image_path = "chart.png"
    output_image_path = "annotated_chart_hybrid.png"
    yolo_model_path = "/home/jules/.pyenv/runs/detect/train2/weights/best.pt"

    # --- Final list for all annotations ---
    patterns_for_annotation = []

    # --- 1. Rule-Based Candlestick Detection ---
    print("--- Running Rule-Based Candlestick Detector ---")
    try:
        parser = ImageParser(input_image_path)
        candles, bboxes = parser.extract_candles_with_bboxes()
        print(f"Successfully extracted {len(candles)} candles from the image.")

        candles_with_patterns = detect_patterns(candles)
        for i, candle in enumerate(candles_with_patterns):
            if candle.pattern:
                print(f"  - Found Candlestick Pattern: '{candle.pattern}' at index {i}")
                bbox = bboxes[i]
                is_bullish = candle.is_bullish
                patterns_for_annotation.append((*bbox, candle.pattern, is_bullish))
    except Exception as e:
        print(f"An error occurred during rule-based detection: {e}")

    # --- 2. YOLOv8-Based Chart Pattern Detection ---
    print("\n--- Running YOLOv8 Chart Pattern Detector ---")
    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLOv8 model not found at '{yolo_model_path}'. Skipping.")
    else:
        try:
            model = YOLO(yolo_model_path)
            results = model.predict(source=input_image_path, conf=0.25)

            names = model.names
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    cls_id = int(box.cls[0])
                    label = names[cls_id]
                    print(f"  - Found Chart Pattern: '{label}' with confidence {box.conf[0]:.2f}")
                    # For chart patterns, we'll default to a neutral (green) color
                    patterns_for_annotation.append((*bbox, label, True))
        except Exception as e:
            print(f"An error occurred during YOLOv8 detection: {e}")

    # --- 3. Annotate and Save the Image ---
    if not patterns_for_annotation:
        print("\nNo patterns of any type were detected.")
    else:
        print("\nAnnotating image with all detected patterns...")
        original_image = cv2.imread(input_image_path)
        annotated_image = annotate_image(original_image, patterns_for_annotation)

        cv2.imwrite(output_image_path, annotated_image)
        print(f"Annotation complete. Annotated image saved to '{output_image_path}'.")


if __name__ == "__main__":
    main()
