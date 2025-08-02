import os
import shutil
import yaml

def convert_to_yolo_format(img_width, img_height, bbox):
    """
    Converts a bounding box from (xmin, ymin, xmax, ymax) to YOLO's
    normalized (x_center, y_center, width, height) format.
    """
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

def main():
    """
    Prepares a small, proof-of-concept dataset for YOLOv8 training.
    """
    print("Starting dataset preparation...")

    # --- Configuration ---
    IMG_WIDTH, IMG_HEIGHT = 1200, 800 # Approximate dimensions of chart.png

    # Define the class names and their corresponding integer IDs
    CLASSES = {
        "triangle": 0,
        "flag": 1
    }

    # Define some hypothetical bounding boxes in pixel coordinates (xmin, ymin, xmax, ymax)
    # These are made up for demonstration purposes on chart.png
    annotations = {
        "chart.png": [
            {"class": "triangle", "bbox": [150, 250, 500, 550]},
            {"class": "flag", "bbox": [600, 150, 800, 350]}
        ]
    }

    # --- Directory Setup ---
    # The directories are already created by a previous step, but we ensure they exist
    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)

    # --- Process Images and Labels ---
    for img_filename, bboxes in annotations.items():
        # Copy the image file
        source_img_path = img_filename
        dest_img_path = os.path.join("dataset/images/train", img_filename)
        if not os.path.exists(source_img_path):
            print(f"Error: Source image '{source_img_path}' not found. Skipping.")
            continue
        shutil.copy(source_img_path, dest_img_path)
        print(f"Copied '{source_img_path}' to '{dest_img_path}'")

        # Create the corresponding label file
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join("dataset/labels/train", label_filename)

        with open(label_path, "w") as f:
            for ann in bboxes:
                class_id = CLASSES[ann["class"]]
                yolo_bbox = convert_to_yolo_format(IMG_WIDTH, IMG_HEIGHT, ann["bbox"])
                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
        print(f"Created label file at '{label_path}'")

    # --- Create data.yaml file ---
    yaml_data = {
        "train": "../dataset/images/train/",
        "val": "../dataset/images/train/", # Using train set for validation in this simple case
        "nc": len(CLASSES),
        "names": list(CLASSES.keys())
    }

    yaml_path = "dataset/data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"Created dataset YAML file at '{yaml_path}'")

    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
