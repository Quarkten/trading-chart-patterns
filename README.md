# Reinforcement Learning-Powered Stock Chart Pattern Detection

This project uses a combination of computer vision and deep learning, specifically the YOLOv8 model, to detect candlestick patterns in stock chart images. The goal is to create a system that can automatically identify and label common patterns like "Doji," "Hammer," "Bullish Engulfing," and "Bearish Engulfing."

## Project Overview

The project is structured as a pipeline that includes the following key components:

1.  **Data Generation**: A script (`build_dataset.py`) that fetches historical stock data from the [Polygon.io](https://polygon.io/) API, generates candlestick charts using Matplotlib, and automatically creates labeled datasets in the YOLO format.
2.  **Model Training**: A script (`train_yolo.py`) that uses the generated dataset to train a YOLOv8 object detection model. The model learns to identify and locate the different candlestick patterns on the charts.
3.  **Inference and Annotation**: A main script (`main.py`) that loads the trained model and runs it on new chart images to detect patterns. The script then uses an annotator module to draw bounding boxes and labels on the image, providing a visual representation of the detected patterns.

## Key Technologies

*   **Python**: The primary programming language used for the project.
*   **YOLOv8**: A state-of-the-art, real-time object detection model used for identifying the candlestick patterns.
*   **PyTorch**: The deep learning framework used by YOLOv8.
*   **OpenCV**: A computer vision library used for image processing and annotation.
*   **Matplotlib**: A plotting library used to generate the candlestick charts for the training dataset.
*   **Polygon.io**: A financial data platform used to source the historical stock data.

## File Descriptions

*   `build_dataset.py`: Fetches stock data, generates charts, and creates YOLO labels.
*   `train_yolo.py`: Trains the YOLOv8 model on the custom dataset.
*   `main.py`: Runs inference with the trained model and annotates chart images.
*   `prepare_dataset.py`: Prepares a small, proof-of-concept dataset (used for initial testing).
*   `verify_labels.py`: A utility script to visually inspect the generated labels.
*   `src/`: This directory contains the core modules of the project:
    *   `annotator.py`: Contains the function for drawing annotations on images.
    *   `data_structures.py`: Defines the `Candle` data class.
    *   `detectors/candlestick_detector.py`: Contains the logic for detecting candlestick patterns.
    *   `image_parser.py`: Parses chart images to extract candle data (work in progress).
    *   `sample_data.py`: Provides sample data for testing.

## How to Run

### 1. Set up the Environment

```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset

To generate the dataset, you will need a Polygon.io API key. Set the `POLYGON_API_KEY` environment variable and then run the following command:

```bash
python build_dataset.py
```

### 3. Train the Model

Once the dataset is generated, you can train the YOLOv8 model by running:

```bash
python train_yolo.py
```

### 4. Run Inference

To run inference on a new chart image, you can use the `main.py` script. Make sure to update the `input_image_path` and `yolo_model_path` variables in the script.

```bash
python main.py
```

## Future Work

*   **Improve Model Accuracy**: The current model is trained on a small dataset and has limited accuracy. The next step is to expand the dataset and fine-tune the model to improve its performance.
*   **Add More Patterns**: The candlestick detector can be extended to identify a wider range of patterns.
*   **Real-Time Analysis**: The ultimate goal is to create a system that can perform real-time analysis of stock charts. This will involve integrating the model with a live data feed and optimizing the inference process for speed.
*   **Reinforcement Learning**: The project name mentions reinforcement learning, which is a planned future addition. The idea is to use RL to optimize trading strategies based on the detected patterns.
