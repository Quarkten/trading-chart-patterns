# Real-Time Trading Chart with Pattern Detection

This project combines a real-time, interactive financial chart with a deep learning backend for automatic candlestick pattern detection. The frontend is built with React and Lightweight Charts, providing a responsive and feature-rich user interface. The backend is a Python server that serves historical and real-time market data and uses a YOLOv8 model to identify common chart patterns from images.

## Project Overview

The project is structured into two main components:

1.  **React Frontend**: An interactive charting application built with Vite, React, and Tailwind CSS. It uses the Lightweight Charts library for high-performance financial visualizations and connects to the backend for data.
2.  **Python Backend**: A Flask and Socket.IO server that provides historical data (via Alpaca) and real-time data (via Finnhub). It also exposes an endpoint that uses a pre-trained YOLOv8 model to detect candlestick patterns in chart images sent from the frontend.

## Key Features

*   **Interactive Candlestick Chart**: A high-performance chart that displays historical and real-time market data.
*   **Dynamic Background Color**: The chart's background color automatically changes to reflect the current trading session (Pre-market, Regular Hours, After-Hours), providing at-a-glance context.
*   **Real-Time Data Streaming**: Uses a WebSocket connection to Finnhub for live market data, ensuring the chart is always up-to-date.
*   **AI-Powered Pattern Detection**: Leverages a YOLOv8 model to identify common candlestick patterns directly from a snapshot of the chart.
*   **Customizable Timeframes**: Users can switch between various timeframes, from minutes to years.

## Key Technologies

*   **React**: A JavaScript library for building the user interface.
*   **Vite**: A fast build tool and development server.
*   **Tailwind CSS**: A utility-first CSS framework for styling.
*   **Lightweight Charts**: A high-performance financial charting library.
*   **Python**: Powers the backend server and AI model.
*   **Flask & Socket.IO**: Used to create the backend server and manage WebSocket connections.
*   **YOLOv8**: A state-of-the-art object detection model for pattern recognition.
*   **Alpaca & Finnhub**: Financial data APIs used for historical and real-time data, respectively.

## Getting Started

This project requires running both the frontend application and the backend server simultaneously.

### Prerequisites

*   Node.js and npm
*   Python and pip
*   API keys for Alpaca and Finnhub

### Backend Setup

1.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up environment variables**:
    Create a `.env` file in the project root and add your API keys:
    ```
    ALPACA_API_KEY="YOUR_ALPACA_KEY"
    ALPACA_SECRET_KEY="YOUR_ALPACA_SECRET"
    FINNHUB_API_KEY="YOUR_FINNHUB_KEY"
    ```

3.  **Download the YOLO model**:
    The required model is not included in the repository. Run the download script:
    ```bash
    python download_model.py
    ```

4.  **Run the backend server**:
    ```bash
    python api.py
    ```
    The server will start on `http://localhost:5000`.

### Frontend Setup

1.  **Navigate to the React app directory**:
    ```bash
    cd my-react-app
    ```

2.  **Install Node.js dependencies**:
    ```bash
    npm install
    ```

3.  **Run the development server**:
    ```bash
    npm run dev
    ```
    This will open the application in your browser, typically at `http://localhost:5173`.

## Future Work

*   **Improve Model Accuracy**: Expand the dataset and fine-tune the model to improve its performance and add more patterns.
*   **Real-Time Analysis**: Enhance the system to perform real-time pattern detection on the incoming data stream.
*   **Reinforcement Learning**: The original goal of the project was to use reinforcement learning to optimize trading strategies based on the detected patterns. This remains a key area for future development.
*   **Add More Indicators**: Integrate common technical indicators like Moving Averages, RSI, and MACD into the chart.