import os
from polygon import RESTClient
from datetime import date, timedelta
import matplotlib.pyplot as plt
from src.data_structures import Candle
from src.detectors.candlestick_detector import detect_patterns

def get_polygon_client() -> RESTClient:
    """
    Initializes and returns the Polygon.io REST client.

    It retrieves the API key from the environment variable `POLYGON_API_KEY`.
    A fallback key is provided for development, but it's recommended to use your own.

    Returns:
        A Polygon.io RESTClient instance.

    Raises:
        ValueError: If the API key is not set in the environment variables.
    """
    api_key = os.environ.get("POLYGON_API_KEY", "XiI2wrBA3GwL01JoeV7npRhE4VWCEmTk")  # Fallback for dev
    if not api_key:
        raise ValueError("Please set the POLYGON_API_KEY environment variable.")
    return RESTClient(api_key)


def fetch_stock_data(client: RESTClient, ticker: str, start_date: str, end_date: str) -> list:
    """
    Fetches historical daily OHLC data for a given stock ticker from Polygon.io.

    Args:
        client: The Polygon.io REST client.
        ticker: The stock ticker symbol (e.g., "AAPL").
        start_date: The start date in "YYYY-MM-DD" format.
        end_date: The end date in "YYYY-MM-DD" format.

    Returns:
        A list of aggregate data points (candles).
    """
    try:
        aggs = client.get_aggs(
            ticker=ticker, multiplier=1, timespan="day",
            from_=start_date, to=end_date, limit=50000
        )
        print(f"Successfully fetched {len(aggs)} data points for {ticker}.")
        return aggs
    except Exception as e:
        print(f"Could not fetch data for {ticker}: {e}")
        return []


def generate_chart_and_labels(ticker: str, aggs: list):
    """
    Generates a candlestick chart image and a corresponding YOLO label file.

    This function performs several key steps:
    1.  Converts the raw aggregate data into `Candle` objects.
    2.  Detects candlestick patterns using the `detect_patterns` function.
    3.  Generates a Matplotlib chart of the candlestick data.
    4.  Calculates precise bounding boxes for the detected patterns.
    5.  Saves the chart as a PNG image and the labels as a TXT file in the YOLO format.

    Args:
        ticker: The stock ticker symbol.
        aggs: A list of aggregate data points from Polygon.io.
    """
    # 1. Convert data and detect patterns
    candles = [Candle(i, agg.open, agg.high, agg.low, agg.close) for i, agg in enumerate(aggs)]
    candles_with_patterns = detect_patterns(candles)
    patterns_to_label = [c for c in candles_with_patterns if c.pattern]

    if not patterns_to_label:
        print(f"No high-confidence patterns found for {ticker}. Skipping image generation.")
        return

    # 2. Generate Chart Image
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.set_title(f"{ticker} Stock Chart")
    ax.set_facecolor('lightgray')

    candle_artists = []
    for candle in candles:
        color = 'green' if candle.is_bullish else 'red'
        wick = ax.plot([candle.index, candle.index], [candle.low, candle.high], color=color, linewidth=1)[0]
        body = plt.Rectangle((candle.index - 0.4, candle.open), 0.8, candle.close - candle.open, color=color)
        ax.add_patch(body)
        candle_artists.append({'body': body, 'wick': wick})

    ax.autoscale_view()

    # 3. Create Labels using Precise Bounding Boxes from Artists
    fig.canvas.draw()

    label_path = f"dataset/labels/train/{ticker}.txt"
    img_width, img_height = fig.canvas.get_width_height()

    with open(label_path, 'w') as f:
        for candle in patterns_to_label:
            artists = candle_artists[candle.index]
            body_bbox = artists['body'].get_window_extent()
            wick_bbox = artists['wick'].get_window_extent()

            x_min_pixel = min(body_bbox.x0, wick_bbox.x0)
            x_max_pixel = max(body_bbox.x1, wick_bbox.x1)
            y_min_pixel = min(body_bbox.y0, wick_bbox.y0)
            y_max_pixel = max(body_bbox.y1, wick_bbox.y1)

            y_top_pixel_inv = img_height - y_max_pixel
            y_bottom_pixel_inv = img_height - y_min_pixel

            x_center_yolo = ((x_min_pixel + x_max_pixel) / 2) / img_width
            y_center_yolo = ((y_top_pixel_inv + y_bottom_pixel_inv) / 2) / img_height
            width_yolo = (x_max_pixel - x_min_pixel) / img_width
            height_yolo = (y_bottom_pixel_inv - y_top_pixel_inv) / img_height

            # Map pattern names to class IDs
            class_map = {
                "Doji": 0, "Hammer": 1, "Bullish Engulfing": 2, "Bearish Engulfing": 3,
                "Morning Star": 4, "Evening Star": 5, "Three White Soldiers": 6, "Three Black Crows": 7
            }
            class_id = class_map.get(candle.pattern)
            if class_id is not None:
                f.write(f"{class_id} {x_center_yolo} {y_center_yolo} {width_yolo} {height_yolo}\n")

    print(f"Saved {len(patterns_to_label)} labels for {ticker} to {label_path}")

    img_path = f"dataset/images/train/{ticker}.png"
    fig.savefig(img_path)
    plt.close(fig)
    print(f"Saved chart for {ticker} to {img_path}")

def main():
    """
    Main function to orchestrate the dataset generation process.
    """
    print("--- Starting High-Quality Dataset Generation ---")

    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "JPM", "V", "JNJ", "WMT"]
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 5)

    client = get_polygon_client()

    import time
    for ticker in tickers:
        data = fetch_stock_data(client, ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if data:
            generate_chart_and_labels(ticker, data)
        # Add a delay to respect API rate limits (free tier is 5 calls/min)
        print("Waiting 15 seconds to respect API rate limits...")
        time.sleep(15)

    print("\n--- Dataset Generation Complete ---")

if __name__ == "__main__":
    main()
