from ultralytics import YOLO

def main():
    """
    Trains a YOLOv8 model on the custom chart pattern dataset.
    """
    print("Starting YOLOv8 model training...")

    # Load a pretrained YOLOv8s model
    # 'yolov8s.pt' is a small, fast model suitable for initial training.
    # The library will automatically download these weights if not present.
    model = YOLO('yolov8s.pt')

    # Train the model on our custom dataset
    # The 'data' argument points to our YAML file which describes the dataset.
    results = model.train(
        data='dataset/data.yaml',
        epochs=50,          # Increased epochs for a larger dataset
        batch=4,            # A small batch size to fit within memory
        imgsz=640,          # Resize images to 640x640 for training
        patience=0,         # Disable early stopping for this demo
        plots=True          # Generate and save plots of training progress
    )

    print("Training complete.")
    print("Model and training results are saved in the 'runs/' directory.")
    print(f"Final model path: {results.save_dir}")

if __name__ == '__main__':
    main()
