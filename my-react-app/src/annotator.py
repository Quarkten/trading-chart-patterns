import cv2
import numpy as np

class Annotator:
    def __init__(self):
        pass

    def annotate_image(self, image_path, detections):
        """
        Annotates an image with bounding boxes and labels from YOLO detections.
        :param image_path: Path to the original image.
        :param detections: A list of dictionaries, each representing a detected object,
                           as returned by YoloInference.predict().
        :return: The annotated image (NumPy array).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or could not be read at: {image_path}")

        # Create a copy to draw on
        annotated_image = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['box']
            class_name = det['class_name']
            confidence = det['confidence']

            # Draw bounding box
            color = (0, 255, 0)  # Green color for bounding box
            thickness = 2
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)

            # Prepare label text
            label = f"{class_name} ({confidence:.2f})"

            # Get text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Draw background rectangle for text
            cv2.rectangle(annotated_image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)

            # Put text on the image
            cv2.putText(annotated_image, label, (x1, y1 - baseline), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        return annotated_image

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    # This requires a sample image and some dummy detections
    import os
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sample_image_path = os.path.join(project_root, 'dataset', 'images', 'train', 'AAPL.png')

    if os.path.exists(sample_image_path):
        print(f"Attempting to annotate: {sample_image_path}")
        dummy_detections = [
            {'box': [50, 50, 150, 150], 'confidence': 0.85, 'class_id': 0, 'class_name': 'DummyPattern1'},
            {'box': [200, 200, 300, 300], 'confidence': 0.70, 'class_id': 1, 'class_name': 'DummyPattern2'}
        ]
        
        annotator = Annotator()
        annotated_img = annotator.annotate_image(sample_image_path, dummy_detections)
        
        output_test_path = os.path.join(project_root, 'annotated_test_image.png')
        cv2.imwrite(output_test_path, annotated_img)
        print(f"Test annotated image saved to: {output_test_path}")
    else:
        print(f"Sample image not found at {sample_image_path}. Cannot run direct annotator test.")