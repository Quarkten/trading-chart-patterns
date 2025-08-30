import os
import cv2
from src.yolo_inference import YoloInference
from src.annotator import Annotator

# Define paths
project_root = os.path.abspath(os.path.dirname(__file__))

# For initial testing, use a sample image from your dataset
sample_image_name = "AAPL.png" # You can change this to any image in your dataset/images/train/
sample_image_path = os.path.join(project_root, 'dataset', 'images', 'train', sample_image_name)
output_annotated_image_path = os.path.join(project_root, 'annotated_chart_with_yolo_detections.png')

if __name__ == "__main__":
    print("Starting chart pattern detection process...")

    # 1. Initialize YOLO Inference
    try:
        yolo_detector = YoloInference()
    except FileNotFoundError as e:
        print(f"Error initializing YOLO model: {e}")
        print("Please ensure 'yolov8s.pt' is in the project root directory.")
        exit()

    # 2. Check if sample image exists
    if not os.path.exists(sample_image_path):
        print(f"Error: Sample image not found at {sample_image_path}")
        print("Please ensure you have images in your 'dataset/images/train/' directory.")
        exit()

    print(f"Processing image: {sample_image_path}")

    # 3. Perform YOLO Inference
    detected_patterns = yolo_detector.predict(sample_image_path)

    if detected_patterns:
        print("Detected Patterns:")
        for pattern in detected_patterns:
            print(f"  - {pattern['class_name']} (Confidence: {pattern['confidence']:.2f}) at {pattern['box']}")
    else:
        print("No patterns detected.")

    # 4. Annotate the image
    annotator = Annotator()
    annotated_image = annotator.annotate_image(sample_image_path, detected_patterns)

    # 5. Save the annotated image
    cv2.imwrite(output_annotated_image_path, annotated_image)
    print(f"Annotated image saved to: {output_annotated_image_path}")

    print("Process completed.")
