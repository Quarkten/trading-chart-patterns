import os
from polygon import RESTClient
from datetime import date, timedelta

def get_polygon_client():
    """
    Initializes and returns the Polygon.io REST client.

    Reads the API key from the 'POLYGON_API_KEY' environment variable.
    """
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("Please set the POLYGON_API_KEY environment variable.")
    return RESTClient(api_key)

def fetch_stock_data(client: RESTClient, ticker: str, start_date: str, end_date: str):
    """
    Fetches historical daily OHLC data for a given stock ticker.
    """
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
        )
        print(f"Successfully fetched {len(aggs)} data points for {ticker}.")
        return aggs
    except Exception as e:
        print(f"Could not fetch data for {ticker}: {e}")
        return []

def main():
    """
    Main function to orchestrate the dataset generation process.
    """
    print("--- Starting Dataset Generation ---")

    # --- Configuration ---
    tickers = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
        "NVDA", "JPM", "V", "JNJ", "WMT"
    ] # A larger, more diverse list of tickers
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 2) # Fetch 2 years of data

    # --- Initialize API Client ---
    try:
        client = get_polygon_client()
    except ValueError as e:
        print(e)
        # As a fallback for the development environment, I'll use the key provided.
        # In a real application, this should not be hardcoded.
        print("Falling back to hardcoded API key for development.")
        client = RESTClient("XiI2wrBA3GwL01JoeV7npRhE4VWCEmTk")

    # --- Fetch Data for Each Ticker ---
    all_data = {}
    for ticker in tickers:
        data = fetch_stock_data(
            client,
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        if data:
            all_data[ticker] = data
            # Print first 2 data points as a sample
            for agg in data[:2]:
                print(f"  - {date.fromtimestamp(agg.timestamp / 1000)}: O={agg.open}, H={agg.high}, L={agg.low}, C={agg.close}")

    # --- Generate and Label Images ---
    print("\n--- Generating and Labeling Chart Images ---")
    for ticker, data in all_data.items():
        generate_chart_and_labels(ticker, data)

    print("\n--- Dataset Generation Complete ---")


def generate_chart_and_labels(ticker: str, aggs: list):
    """
    Generates a chart image from OHLC data and creates a corresponding
    YOLO label file for any detected patterns.
    """
    import matplotlib.pyplot as plt
    from src.data_structures import Candle
    from src.detectors.candlestick_detector import detect_patterns

    # --- 1. Convert data and detect patterns ---
    candles = []
    for i, agg in enumerate(aggs):
        candles.append(Candle(index=i, open=agg.open, high=agg.high, low=agg.low, close=agg.close))

    candles_with_patterns = detect_patterns(candles)
    patterns_to_label = [c for c in candles_with_patterns if c.pattern]

    if not patterns_to_label:
        print(f"No patterns found for {ticker}. Skipping image generation.")
        return

    # --- 2. Generate Chart Image ---
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.set_title(f"{ticker} Stock Chart")
    ax.set_facecolor('lightgray')

    # Store the artist objects for each candle (body and wick)
    candle_artists = []

    for candle in candles:
        color = 'green' if candle.is_bullish else 'red'
        # Draw the wick (high-low line)
        wick = ax.plot([candle.index, candle.index], [candle.low, candle.high], color=color, linewidth=1)[0]
        # Draw the body (open-close rectangle)
        body = plt.Rectangle((candle.index - 0.4, candle.open), 0.8, candle.close - candle.open, color=color)
        ax.add_patch(body)
        candle_artists.append({'body': body, 'wick': wick})

    ax.autoscale_view()

    # --- 3. Create Labels using Precise Bounding Boxes from Artists ---
    fig.canvas.draw()

    label_path = f"dataset/labels/train/{ticker}.txt"
    img_width, img_height = fig.canvas.get_width_height()

    with open(label_path, 'w') as f:
        for candle in patterns_to_label:
            # Get the artists for the candle with the pattern
            artists = candle_artists[candle.index]
            body_bbox = artists['body'].get_window_extent()
            wick_bbox = artists['wick'].get_window_extent()

            # The full bounding box of the candle is the union of the body and wick boxes
            x_min_pixel = min(body_bbox.x0, wick_bbox.x0)
            x_max_pixel = max(body_bbox.x1, wick_bbox.x1)
            y_min_pixel = min(body_bbox.y0, wick_bbox.y0)
            y_max_pixel = max(body_bbox.y1, wick_bbox.y1)

            # The y-coordinates from matplotlib are from the bottom-left,
            # so we need to invert them for top-left origin used in images.
            y_top_pixel_inverted = img_height - y_max_pixel
            y_bottom_pixel_inverted = img_height - y_min_pixel

            # Convert to YOLO format (x_center, y_center, width, height) normalized
            x_center_yolo = ((x_min_pixel + x_max_pixel) / 2) / img_width
            y_center_yolo = ((y_top_pixel_inverted + y_bottom_pixel_inverted) / 2) / img_height
            width_yolo = (x_max_pixel - x_min_pixel) / img_width
            height_yolo = (y_bottom_pixel_inverted - y_top_pixel_inverted) / img_height

            # Get class ID
            if candle.pattern == "Doji": class_id = 0
            elif candle.pattern == "Hammer": class_id = 1
            elif candle.pattern == "Bullish Engulfing": class_id = 2
            elif candle.pattern == "Bearish Engulfing": class_id = 3
            else: continue

            f.write(f"{class_id} {x_center_yolo} {y_center_yolo} {width_yolo} {height_yolo}\n")

    print(f"Saved labels for {ticker} to {label_path}")

    # Now, save the image file
    img_path = f"dataset/images/train/{ticker}.png"
    fig.savefig(img_path)
    plt.close(fig) # Close the figure to free up memory
    print(f"Saved chart for {ticker} to {img_path}")



if __name__ == "__main__":
    main()
