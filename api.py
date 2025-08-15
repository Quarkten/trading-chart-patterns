from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from src.data_fetcher import DataFetcher
from datetime import datetime, timedelta

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from src.data_fetcher import DataFetcher
from datetime import datetime, timedelta
import threading
import asyncio
from src.yolo_inference import YoloInference
import base64
import io
from PIL import Image
import os

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}) # Enable CORS for specific origin
    socketio = SocketIO(app, cors_allowed_origins="http://localhost:5173")

    data_fetcher = DataFetcher(socketio=socketio)
    yolo_detector = YoloInference()

    @socketio.on('connect')
    def test_connect():
        print('Client connected')
        emit('my response', {'data': 'Connected'})

    @socketio.on('start_stream')
    def handle_start_stream(data):
        symbols = ['SPY'] # Enforce SPY as the only symbol for real-time stream
        if symbols:
            print(f"Starting stream for: {symbols}")
            def run_async_in_thread():
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Run the coroutine in this new event loop
                    loop.run_until_complete(data_fetcher.start_realtime_stream(symbols))
                finally:
                    loop.close() # Close the loop when done
            thread = threading.Thread(target=run_async_in_thread)
            thread.daemon = True
            thread.start()
        else:
            print("No symbols provided for streaming.")

    @app.route('/candlestick_data', methods=['GET'])
    def get_candlestick_data():
        symbol = request.args.get('symbol', default='SPY', type=str).upper()
        timeframe_value = request.args.get('timeframe_value', default=1, type=int)
        timeframe_unit = request.args.get('timeframe_unit', default='day', type=str)
        from_date_str = request.args.get('from_date', type=str)
        to_date_str = request.args.get('to_date', type=str)

        if not from_date_str or not to_date_str:
            # Default to last 2 years if dates are not provided
            to_date = datetime.now()
            from_date = to_date - timedelta(days=365 * 2)
            from_date_str = from_date.strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d')

        print(f"Fetching data for symbol={symbol}, multiplier={timeframe_value}, timespan={timeframe_unit}, from={from_date_str}, to={to_date_str}")
        df = data_fetcher.get_candlestick_data(
            symbol=symbol,
            multiplier=timeframe_value,
            timespan=timeframe_unit,
            from_date=from_date_str,
            to_date=to_date_str
        )

        print(f"DataFrame empty: {df.empty}")
        if df.empty:
            return jsonify({"error": "No data available for the given parameters. Check symbol, timeframe, and date range. Also ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are set and valid."}), 404

        # Format data for lightweight-charts
        chart_data = []
        for index, row in df.iterrows():
            chart_data.append({
                "time": row['timestamp'], # Lightweight-charts expects Unix timestamp in seconds
                "open": row['open'],
                "high": row['high'],
                "low": row['low'],
                "close": row['close'],
                "volume": row['volume']
            })
        return jsonify(chart_data)

    @app.route('/detect_patterns', methods=['POST'])
    def detect_patterns():
        try:
            data = request.get_json()
            img_data_b64 = data['image']
            print(f"Type of img_data_b64: {type(img_data_b64)}")
            print(f"First 50 chars of img_data_b64: {img_data_b64[:50]}")
            # Decode base64 image
            # Remove the "data:image/png;base64," prefix
            img_data_b64_stripped = img_data_b64.split(',')[1]
            img_bytes = base64.b64decode(img_data_b64_stripped)
            img = Image.open(io.BytesIO(img_bytes))

            # Save image temporarily for YOLO inference
            temp_image_path = "temp_chart_for_yolo.png"
            img.save(temp_image_path)

            detected_patterns = yolo_detector.predict(temp_image_path)

            # Clean up temporary image
            os.remove(temp_image_path)

            return jsonify(detected_patterns)

        except Exception as e:
            print(f"Error during pattern detection: {e}")
            return jsonify({"error": str(e)}), 500
    
    return app, socketio

if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, debug=True, port=5000)
