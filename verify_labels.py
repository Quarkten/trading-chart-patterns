import cv2
import os
import yaml

def denormalize_yolo_bbox(img_width, img_height, yolo_bbox):
    """
    Converts a YOLO bounding box back to pixel coordinates (xmin, ymin, xmax, ymax).
    """
    x_center, y_center, width, height = yolo_bbox

    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    xmin = int(x_center_px - (width_px / 2))
    ymin = int(y_center_px - (height_px / 2))
    xmax = int(x_center_px + (width_px / 2))
    ymax = int(y_center_px + (height_px / 2))

    return xmin, ymin, xmax, ymax

def main():
    """
    Reads a sample image and its YOLO label file and draws the bounding
    boxes to visually verify that the labels are correct.
    """
    print("--- Verifying Dataset Labels ---")

    # --- Configuration ---
    dataset_yaml_path = "dataset/data.yaml"
    # Let's check the labels for the AAPL image
    image_name = "AAPL.png"
    image_path = os.path.join("dataset/images/train", image_name)
    label_path = os.path.join("dataset/labels/train", os.path.splitext(image_name)[0] + ".txt")

    # --- Load Image and Class Names ---
    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print(f"Image or label for {image_name} not found. Exiting.")
        return

    with open(dataset_yaml_path, 'r') as f:
        class_names = yaml.safe_load(f)['names']

    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape

    # --- Read Labels and Draw Boxes ---
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            yolo_bbox = list(map(float, parts[1:]))

            # De-normalize the bounding box
            xmin, ymin, xmax, ymax = denormalize_yolo_bbox(img_width, img_height, yolo_bbox)

            # Draw the rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Add the class name label
            label = class_names[class_id]
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Save the Debug Image ---
    output_path = f"debug_labels_{os.path.splitext(image_name)[0]}.png"
    cv2.imwrite(output_path, image)
    print(f"Verification image saved to '{output_path}'. Please inspect it.")

if __name__ == "__main__":
    main()
