import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import List, Tuple

from src.data_structures import Candle

class ImageParser:
    """
    A class to parse a stock chart image and extract OHLC data.

    This class encapsulates all the logic for image processing, including
    candle segmentation, OCR of the price axis, and mapping pixel data to
    structured Candle objects.
    """

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        self.height, self.width, _ = self.image.shape
        self.price_scale = None # To be populated by OCR

    def _segment_candles(self) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        Segments the candles from the image using color masking.

        Returns:
            A list of tuples, where each tuple contains:
            - A bounding box (x, y, w, h) for a candle.
            - The color of the candle ('green' or 'red').
        """
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Define HSV color ranges for green and red
        # These ranges may need tuning for different chart styles
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for green and red colors
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Combine the masks
        combined_mask = cv2.bitwise_or(green_mask, red_mask)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candle_data = []
        for contour in contours:
            # Ignore very small contours that are likely noise
            if cv2.contourArea(contour) < 10:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Determine the color of the candle by checking the masks
            # We check the center of the contour's bounding box
            cx, cy = x + w // 2, y + h // 2
            if green_mask[cy, cx] > 0:
                color = 'green'
            elif red_mask[cy, cx] > 0:
                color = 'red'
            else:
                continue # Should not happen if contour is from the mask

            candle_data.append(((x, y, w, h), color))

        # Sort candles by their x-coordinate to ensure they are in chronological order
        candle_data.sort(key=lambda data: data[0][0])

        return candle_data

    def extract_candles_with_bboxes(self) -> Tuple[List[Candle], List[Tuple[int, int, int, int]]]:
        """
        Main method to orchestrate the full extraction pipeline.

        Returns:
            A tuple containing:
            - A list of Candle objects with OHLC data.
            - A list of corresponding bounding boxes (x, y, w, h).
        """
        # 1. Segment candles from the image
        segmented_candles_data = self._segment_candles()

        # 2. Perform OCR on the Y-axis to get the price scale
        self._extract_price_scale()

        # 3. Map pixel data to OHLC data
        if not segmented_candles_data or self.price_scale is None:
            print("Cannot proceed with data mapping due to missing segmentation or price scale.")
            return [], []

        candles = []
        bboxes = []
        for i, (bbox, color) in enumerate(segmented_candles_data):
            x, y, w, h = bbox

            # Use the price scale to convert pixel values to prices
            high_price = self._map_pixel_to_price(y)
            low_price = self._map_pixel_to_price(y + h)

            # Find the body of the candle to determine open and close
            candle_roi = self.image[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(candle_roi, cv2.COLOR_BGR2GRAY)
            _, binary_roi = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY)
            horizontal_projection = np.sum(binary_roi, axis=1)
            body_threshold = np.max(horizontal_projection) * 0.5
            body_indices = np.where(horizontal_projection > body_threshold)[0]

            if len(body_indices) == 0:
                continue

            body_top_y_local = body_indices[0]
            body_bottom_y_local = body_indices[-1]
            body_top_y_global = y + body_top_y_local
            body_bottom_y_global = y + body_bottom_y_local

            price1 = self._map_pixel_to_price(body_top_y_global)
            price2 = self._map_pixel_to_price(body_bottom_y_global)

            if color == 'green':
                open_price, close_price = price2, price1
            else:
                open_price, close_price = price1, price2

            candles.append(
                Candle(
                    index=i, open=open_price, high=high_price,
                    low=low_price, close=close_price
                )
            )
            bboxes.append(bbox)

        return candles, bboxes

    def _map_pixel_to_price(self, y_pixel: int) -> float:
        """
        Converts a Y-pixel coordinate to a price using the calculated scale.
        """
        if self.price_scale is None:
            raise ValueError("Price scale has not been extracted yet.")

        # The price is calculated relative to the reference point.
        # We must account for the fact that pixel Y increases downwards, while price increases upwards.
        price = self.price_scale['ref_price'] - \
            (y_pixel - self.price_scale['ref_pixel']) / self.price_scale['pixels_per_dollar']

        return round(price, 2) # Round to 2 decimal places for typical prices

    def _extract_price_scale(self):
        """
        Performs OCR on the Y-axis to build a pixel-to-price mapping.
        """
        # --- 1. Crop the Y-axis ---
        # These coordinates are estimates for a typical TradingView chart.
        # We assume the price scale is in the right-most part of the image.
        axis_x_start = self.width - 100
        y_axis_roi = self.image[0:self.height, axis_x_start:self.width]

        # --- 2. Pre-process the ROI for OCR ---
        gray_axis = cv2.cvtColor(y_axis_roi, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to get a binary image.
        # THRESH_OTSU automatically determines the optimal threshold value.
        _, binary_axis = cv2.threshold(gray_axis, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # --- 3. Run Tesseract OCR ---
        # Use Page Segmentation Mode (PSM) 6, which assumes a single uniform block of text.
        custom_config = r'--oem 3 --psm 6'
        ocr_data = pytesseract.image_to_data(
            binary_axis,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        # --- 4. Parse OCR Data ---
        price_points = []
        n_boxes = len(ocr_data['level'])
        for i in range(n_boxes):
            # We only care about text with high confidence
            if int(ocr_data['conf'][i]) > 70:
                text = ocr_data['text'][i].strip()
                try:
                    price = float(text)
                    # Get the y-coordinate of the center of the text
                    y_pixel = ocr_data['top'][i] + ocr_data['height'][i] // 2
                    price_points.append((y_pixel, price))
                except ValueError:
                    # The text was not a valid number, so we ignore it.
                    continue

        print(f"OCR found {len(price_points)} potential price points.")
        if len(price_points) < 2:
            print("Error: Not enough data points from OCR to create a price scale.")
            return

        # --- 5. Build the Price Scale Mapping ---
        # Sort points by pixel value to avoid issues with calculation
        price_points.sort()

        # Use the two most separated points for a more stable calculation
        p1_pixel, p1_price = price_points[0]
        p2_pixel, p2_price = price_points[-1]

        # Calculate the pixels per dollar ratio
        pixel_diff = abs(p1_pixel - p2_pixel)
        price_diff = abs(p1_price - p2_price)

        if pixel_diff == 0 or price_diff == 0:
            print("Error: Cannot create price scale from OCR data (zero difference).")
            return

        pixels_per_dollar = pixel_diff / price_diff

        # We can now create a mapping function. For now, store the scale components.
        # The reference point is p1.
        self.price_scale = {
            'ref_pixel': p1_pixel,
            'ref_price': p1_price,
            'pixels_per_dollar': pixels_per_dollar
        }
        print(f"Successfully created price scale: {self.price_scale}")


# Example usage (for testing during development)
if __name__ == '__main__':
    parser = ImageParser('chart.png')
    # This will now run the full pipeline as implemented so far
    candles = parser.extract_candles()
    print(f"Successfully extracted {len(candles)} candles.")

    # Print the first 5 candles to inspect the data
    for i in range(min(5, len(candles))):
        print(candles[i])
