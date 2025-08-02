import cv2
import os
from ultralytics import YOLO
from src.annotator import annotate_image

def main():
    """
    Main function to demonstrate the final, improved chart pattern detection
    using the retrained YOLOv8 model.
    """
    # --- Configuration ---
    input_image_path = "dataset/images/train/AAPL.png"
    output_image_path = "final_annotated_chart.png"
    # Path to the latest and best trained model
    yolo_model_path = "/home/jules/.pyenv/runs/detect/train6/weights/best.pt"

    print(f"--- Running Final Model on {input_image_path} ---")

    if not os.path.exists(yolo_model_path):
        print(f"Error: Final YOLOv8 model not found at '{yolo_model_path}'.")
        return

    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at '{input_image_path}'.")
        return

    # --- Run YOLOv8 Inference ---
    patterns_for_annotation = []
    try:
        model = YOLO(yolo_model_path)
        # Use a higher confidence threshold as the model should be more accurate
        results = model.predict(source=input_image_path, conf=0.5)

        names = model.names
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                cls_id = int(box.cls[0])
                label = names[cls_id]
                conf = float(box.conf[0])
                print(f"  - Found Pattern: '{label}' with confidence {conf:.2f}")
                # Color bullish patterns green, bearish red
                is_bullish = "Bullish" in label or "Doji" in label or "Hammer" in label
                patterns_for_annotation.append((*bbox, label, is_bullish))
    except Exception as e:
        print(f"An error occurred during YOLOv8 detection: {e}")

    # --- Annotate and Save the Image ---
    if not patterns_for_annotation:
        print("\nNo high-confidence patterns were detected by the final model.")
    else:
        print(f"\nFound {len(patterns_for_annotation)} patterns. Annotating image...")
        original_image = cv2.imread(input_image_path)
        annotated_image = annotate_image(original_image, patterns_for_annotation)

        cv2.imwrite(output_image_path, annotated_image)
        print(f"Final annotation complete. Image saved to '{output_image_path}'.")


if __name__ == "__main__":
    main()
