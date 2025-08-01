import cv2
import numpy as np
from src.annotator import annotate_image
import os

def main():
    """
    Main function to demonstrate the chart annotation process.
    """
    # --- Configuration ---
    # The input image path. This file needs to be provided by the user.
    input_image_path = "chart.png"

    # The output path for the annotated image.
    output_image_path = "annotated_chart.png"

    # --- Check for Input Image ---
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at '{input_image_path}'.")
        print("Please provide a chart screenshot named 'chart.png' in the root directory.")
        return

    # --- Load Image ---
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Failed to load image from '{input_image_path}'.")
        return

    # --- Sample Pattern Data ---
    # This is a hardcoded list of patterns for demonstration purposes.
    # In the full application, this data will come from the pattern detection model.
    # Each pattern is a tuple: (x, y, w, h, label, is_bullish)
    patterns = [
        (100, 200, 150, 60, "Bull Flag", True),      # A sample bullish pattern
        (300, 100, 120, 80, "Head and Shoulders", False), # A sample bearish pattern
        (500, 300, 25, 25, "Doji", True),           # A sample candlestick pattern
    ]

    # --- Annotate Image ---
    print("Annotating image with detected patterns...")
    annotated_image = annotate_image(image, patterns)

    # --- Save Annotated Image ---
    cv2.imwrite(output_image_path, annotated_image)
    print(f"Annotation complete. Annotated image saved to '{output_image_path}'.")


if __name__ == "__main__":
    main()
