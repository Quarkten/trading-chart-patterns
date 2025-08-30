import os
from ultralytics import YOLO
import cv2

class YoloInference:
    def __init__(self, model_path='models/foduucom_yolov8/model.pt'):
        """
        Initializes the YOLOv8 model for inference.
        :param model_path: Path to the YOLOv8 model weights (e.g., 'yolov8s.pt').
        """
        # Resolve the absolute path to the model weights
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.model_path = os.path.join(project_root, model_path)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
            
        self.model = YOLO(self.model_path)
        print(f"YOLOv8 model loaded from: {self.model_path}")

    def predict(self, image_path):
        """
        Performs inference on a given image.
        :param image_path: Path to the input image.
        :return: A list of dictionaries, each representing a detected object.
                 Example: [{'box': [x1, y1, x2, y2], 'confidence': 0.95, 'class_id': 0, 'class_name': 'pattern_name'}]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        results = self.model(image_path)
        
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = round(box.conf[0].item(), 2)
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id] # Get class name from model
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        return detections

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    # You'll need a sample image for this to run.
    # For initial testing, you can use one of the images from dataset/images/train/
    
    # Assuming you have an image like 'dataset/images/train/AAPL.png'
    # Adjust this path to a valid image in your setup for testing.
    sample_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'images', 'train', 'AAPL.png'))
    
    if os.path.exists(sample_image_path):
        print(f"Attempting to run inference on: {sample_image_path}")
        yolo_detector = YoloInference()
        detected_patterns = yolo_detector.predict(sample_image_path)
        print("Detected Patterns:")
        for pattern in detected_patterns:
            print(pattern)
    else:
        print(f"Sample image not found at {sample_image_path}. Please provide a valid image path for testing.")