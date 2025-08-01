import cv2
import os
from typing import List, Tuple

from src.annotator import annotate_image
from src.detectors.candlestick_detector import detect_patterns
from src.image_parser import ImageParser

def main():
    """
    Main function to run the full image-based chart pattern detection pipeline.
    """
    # --- Configuration ---
    input_image_path = "chart.png"
    output_image_path = "annotated_chart_from_image.png"

    # --- 1. Parse the Image to Extract OHLC Data ---
    print("Parsing chart image to extract candle data...")
    try:
        parser = ImageParser(input_image_path)
        # We need both the Candle objects and their original bounding boxes for annotation
        candles_with_data, candle_bboxes = parser.extract_candles_with_bboxes()
        print(f"Successfully extracted {len(candles_with_data)} candles from the image.")
    except Exception as e:
        print(f"An error occurred during image parsing: {e}")
        return

    # --- 2. Detect Patterns in the Extracted Data ---
    print("Detecting patterns in extracted data...")
    candles_with_patterns = detect_patterns(candles_with_data)

    # --- 3. Prepare Annotations ---
    patterns_for_annotation = []
    for i, candle in enumerate(candles_with_patterns):
        if candle.pattern:
            print(f"  - Found '{candle.pattern}' pattern at index {candle.index}")
            # Use the actual bounding box from segmentation
            bbox = candle_bboxes[i]
            is_bullish = candle.is_bullish
            patterns_for_annotation.append((*bbox, candle.pattern, is_bullish))

    # --- 4. Annotate and Save the Image ---
    if not patterns_for_annotation:
        print("No patterns detected to annotate.")
    else:
        print("Annotating image with detected patterns...")
        # Load the original image again for annotation
        original_image = cv2.imread(input_image_path)
        annotated_image = annotate_image(original_image, patterns_for_annotation)

        cv2.imwrite(output_image_path, annotated_image)
        print(f"Annotation complete. Annotated image saved to '{output_image_path}'.")


if __name__ == "__main__":
    main()
