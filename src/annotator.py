import cv2
import numpy as np
from typing import List, Tuple

def annotate_image(
    image: np.ndarray,
    patterns: List[Tuple[int, int, int, int, str, bool]]
) -> np.ndarray:
    """
    Annotates an image with bounding boxes and labels for detected patterns.

    Args:
        image: The input image as a NumPy array.
        patterns: A list of detected patterns. Each pattern is a tuple containing:
                  (x, y, w, h, label, is_bullish).
                  - (x, y): Top-left corner of the bounding box.
                  - w: Width of the bounding box.
                  - h: Height of the bounding box.
                  - label: The text label for the pattern.
                  - is_bullish: True for bullish (green), False for bearish (red).

    Returns:
        The annotated image as a NumPy array.
    """
    annotated_img = image.copy()

    for (x, y, w, h, label, is_bullish) in patterns:
        # Set color: green for bullish, red for bearish
        color = (0, 255, 0) if is_bullish else (0, 0, 255)

        # Draw the bounding box rectangle
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)

        # Add the text label
        cv2.putText(annotated_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated_img
