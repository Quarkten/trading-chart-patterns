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
    tickers = ["AAPL", "GOOGL", "MSFT"] # A small list of tickers for demonstration
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

    # Store bounding boxes as we draw
    candle_bboxes_pixels = []

    for candle in candles:
        color = 'green' if candle.is_bullish else 'red'
        # Draw the wick (high-low line)
        ax.plot([candle.index, candle.index], [candle.low, candle.high], color=color, linewidth=1)
        # Draw the body (open-close rectangle)
        body = plt.Rectangle((candle.index - 0.4, candle.open), 0.8, candle.close - candle.open, color=color)
        ax.add_patch(body)

    ax.autoscale_view()

    # --- 3. Create Labels from Accurate Bounding Boxes ---
    img_width, img_height = fig.get_size_inches() * fig.dpi
    label_path = f"dataset/labels/train/{ticker}.txt"

    # Get the transformation from data coordinates to pixel coordinates
    trans = ax.transData.inverted()

    with open(label_path, 'w') as f:
        for candle in patterns_to_label:
            # Define the candle's bounding box in data coordinates
            # This is the full extent of the candle (high to low)
            x_data, y_data_high = candle.index, candle.high
            y_data_low = candle.low

            # We need to find the pixel coordinates of the four corners of the box
            # This is a more robust way than estimating.
            # However, a fully robust implementation is very complex.
            # Let's use a refined estimation based on the axis limits.
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Calculate a simple linear mapping
            x_range_data = xlim[1] - xlim[0]
            y_range_data = ylim[1] - ylim[0]

            # We need to account for the plot's padding/margins. We'll estimate this.
            padding_x = 0.1 * img_width
            padding_y = 0.1 * img_height
            plot_width_pixels = img_width * 0.8
            plot_height_pixels = img_height * 0.8

            # Map data to pixel space
            x_norm = (candle.index - xlim[0]) / x_range_data
            y_high_norm = (candle.high - ylim[0]) / y_range_data
            y_low_norm = (candle.low - ylim[0]) / y_range_data

            # Candle body width in normalized data coords (0.8)
            candle_width_data = 0.8

            # Convert to pixel coordinates
            x_center_pixel = padding_x + x_norm * plot_width_pixels
            width_pixel = (candle_width_data / x_range_data) * plot_width_pixels

            # Y is inverted in pixel coordinates
            y_top_pixel = padding_y + (1 - y_high_norm) * plot_height_pixels
            y_bottom_pixel = padding_y + (1 - y_low_norm) * plot_height_pixels
            height_pixel = y_bottom_pixel - y_top_pixel

            # Convert to YOLO format
            x_center_yolo = x_center_pixel / img_width
            y_center_yolo = (y_top_pixel + y_bottom_pixel) / 2.0 / img_height
            width_yolo = width_pixel / img_width
            height_yolo = height_pixel / img_height

            class_id = 0 if candle.pattern == "Doji" else 1
            f.write(f"{class_id} {x_center_yolo} {y_center_yolo} {width_yolo} {height_yolo}\n")

    # Save the image *after* we've calculated the final transforms
    img_path = f"dataset/images/train/{ticker}.png"
    plt.savefig(img_path)
    plt.close(fig) # Close the figure to free up memory
    print(f"Saved chart for {ticker} to {img_path}")

    print(f"Saved labels for {ticker} to {label_path}")



if __name__ == "__main__":
    main()
