# Stock Chart Pattern Detector

Welcome to the Stock Chart Pattern Detector! This tool is designed to help you automatically identify common candlestick patterns in stock chart images. Whether you're a seasoned trader or just starting out, this project can help you to spot trends and make more informed decisions.

## What Does It Do?

At its core, this project uses a powerful computer vision model called YOLOv8 to "look" at stock chart images and identify candlestick patterns. It can recognize patterns like:

*   **Doji**: Indicates indecision in the market.
*   **Hammer**: A bullish reversal pattern.
*   **Bullish Engulfing**: A strong bullish reversal pattern.
*   **Bearish Engulfing**: A strong bearish reversal pattern.
*   And many more!

The project is a pipeline that consists of three main parts:

1.  **Data Generation**: We fetch real stock data, create candlestick charts, and automatically label the patterns on them. This creates a dataset that we can use to teach our model.
2.  **Model Training**: We take the dataset we generated and use it to train the YOLOv8 model. This is like teaching the model what each pattern looks like.
3.  **Pattern Detection**: Once the model is trained, we can give it a new stock chart image, and it will draw boxes around the patterns it recognizes, telling you what they are.

## Getting Started

Ready to give it a try? Here's how to get up and running.

### Step 1: Set Up Your Environment

First, you'll need to install the necessary Python libraries. We've included a `requirements.txt` file to make this easy. Just open your terminal or command prompt and run the following command:

```bash
pip install -r requirements.txt
```

### Step 2: Get a Polygon.io API Key

To get the stock data, we use a service called [Polygon.io](https://polygon.io/). You'll need to sign up for a free account and get an API key. Once you have your key, you'll need to set it as an environment variable.

*   **On macOS or Linux**:
    ```bash
    export POLYGON_API_KEY="YOUR_API_KEY"
    ```
*   **On Windows**:
    ```bash
    set POLYGON_API_KEY="YOUR_API_KEY"
    ```

    (Replace `"YOUR_API_KEY"` with the key you got from Polygon.io)

### Step 3: Generate the Dataset

Now it's time to create the dataset that we'll use to train our model. This script will fetch data for several stocks, generate charts, and create the labels for them.

```bash
python build_dataset.py
```

This might take a few minutes, as it needs to fetch a lot of data and generate the images.

### Step 4: Train the Model

Once you have the dataset, you can train the model. This is the most important step, as it's where the model learns to recognize the patterns.

```bash
python train_yolo.py
```

This process will also take some time, depending on your computer's hardware. You'll see a lot of output in your terminal as the model trains.

### Step 5: Detect Patterns in a Chart

Now for the fun part! You can use the `main.py` script to detect patterns in a stock chart image. We've included a sample image (`AAPL.png`) in the `dataset/images/train` directory that you can use to test it out.

```bash
python main.py
```

This will run the model on the `AAPL.png` image and save a new image called `debug_labels_AAPL.png` with the detected patterns highlighted.

## Understanding the Output

When you run the `main.py` script, it will create a new image file with bounding boxes drawn around the patterns it has detected. Each box will have a label indicating the name of the pattern.

*   **Green boxes** indicate bullish patterns (patterns that suggest the price might go up).
*   **Red boxes** indicate bearish patterns (patterns that suggest the price might go down).

## What's Next?

This project is a great starting point, but there's always room for improvement. Here are a few ideas for how you could take it to the next level:

*   **Add More Patterns**: You can extend the `src/detectors/candlestick_detector.py` file to recognize even more patterns.
*   **Use Your Own Charts**: You can modify the `main.py` script to run on your own stock chart images.
*   **Improve the Model**: You can try to improve the model's accuracy by experimenting with different training settings in the `train_yolo.py` script.

We hope you enjoy using the Stock Chart Pattern Detector! If you have any questions or ideas for improvement, please feel free to contribute to the project.
