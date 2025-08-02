import cv2
import os
from ultralytics import YOLO
from src.annotator import annotate_image

def main():
    """
    This script is for demonstration purposes. It runs the first
    proof-of-concept YOLO model on the original chart.png to show
    a visual output of a detected pattern.
    """
    # --- Configuration ---
    input_image_path = "dataset/images/train/AAPL.png"
    output_image_path = "final_annotated_image.png"
    # Path to the FINAL model
    yolo_model_path = "/home/jules/.pyenv/runs/detect/train7/weights/best.pt"

    print(f"--- Running Proof-of-Concept Model on {input_image_path} ---")

    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLOv8 model not found at '{yolo_model_path}'.")
        return

    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at '{input_image_path}'.")
        return

    # --- Run YOLOv8 Inference ---
    patterns_for_annotation = []
    try:
        model = YOLO(yolo_model_path)
        # Use a low confidence threshold to ensure we see the result
        results = model.predict(source=input_image_path, conf=0.1)

        names = model.names
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                cls_id = int(box.cls[0])
                label = names[cls_id]
                conf = float(box.conf[0])
                print(f"  - Found Pattern: '{label}' with confidence {conf:.2f}")
                patterns_for_annotation.append((*bbox, label, True))
    except Exception as e:
        print(f"An error occurred during YOLOv8 detection: {e}")

    # --- Annotate and Save the Image ---
    if not patterns_for_annotation:
        print("\nNo patterns were detected.")
    else:
        print(f"\nFound {len(patterns_for_annotation)} patterns. Annotating image...")
        original_image = cv2.imread(input_image_path)
        annotated_image = annotate_image(original_image, patterns_for_annotation)

        cv2.imwrite(output_image_path, annotated_image)
        print(f"Final annotation complete. Image saved to '{output_image_path}'.")


if __name__ == "__main__":
    main()
