import cv2
import os
from typing import List, Tuple

from src.annotator import annotate_image
from src.detectors.candlestick_detector import detect_patterns
from src.sample_data import get_sample_candles
from src.data_structures import Candle

def map_candle_to_bbox(candle_index: int) -> Tuple[int, int, int, int]:
    """
    Maps a candle's index to a bounding box on the chart.png image.

    This is a temporary function for visualization purposes. The coordinates
    are estimated based on the visual layout of the specific 'chart.png' image.

    Args:
        candle_index: The index of the candle in the sequence.

    Returns:
        A tuple representing the bounding box: (x, y, w, h).
    """
    # Magic numbers based on visual inspection of chart.png
    start_x = 50  # The x-coordinate of the first candle
    candle_width = 20  # The approximate width of a single candle
    spacing = 10  # The space between candles

    # Calculate the top-left x coordinate
    x = start_x + candle_index * (candle_width + spacing)

    # Use fixed y, w, h for simplicity for now
    y = 100
    w = candle_width
    h = 200

    return (x, y, w, h)

def main():
    """
    Main function to run the candlestick pattern detection and annotation pipeline.
    """
    # --- Configuration ---
    input_image_path = "chart.png"
    output_image_path = "annotated_chart_with_rules.png"

    # --- Check for Input Image ---
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at '{input_image_path}'.")
        return

    # --- Load Image ---
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Failed to load image from '{input_image_path}'.")
        return

    # --- Get and Process Data ---
    print("Loading sample candlestick data...")
    candles = get_sample_candles()

    print("Detecting patterns...")
    candles_with_patterns = detect_patterns(candles)

    # --- Prepare Annotations ---
    patterns_for_annotation = []
    for candle in candles_with_patterns:
        if candle.pattern:
            print(f"  - Found '{candle.pattern}' pattern at index {candle.index}")
            bbox = map_candle_to_bbox(candle.index)
            is_bullish = candle.is_bullish  # Color based on the candle's nature
            patterns_for_annotation.append((*bbox, candle.pattern, is_bullish))

    # --- Annotate Image ---
    if not patterns_for_annotation:
        print("No patterns detected to annotate.")
    else:
        print("Annotating image with detected patterns...")
        annotated_image = annotate_image(image, patterns_for_annotation)

        # --- Save Annotated Image ---
        cv2.imwrite(output_image_path, annotated_image)
        print(f"Annotation complete. Annotated image saved to '{output_image_path}'.")


if __name__ == "__main__":
    main()
