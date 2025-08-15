import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.data_fetcher import DataFetcher
from alpaca.data.live import StockDataStream
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import json
import websockets

# Mock environment variables
@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_alpaca_key',
        'ALPACA_SECRET_KEY': 'test_alpaca_secret',
        'FINNHUB_API_KEY': 'test_finnhub_key'
    }):
        yield

# Fixture for DataFetcher instance
@pytest.fixture
def data_fetcher_instance():
    mock_socketio = MagicMock()
    fetcher = DataFetcher(socketio=mock_socketio)
    return fetcher

@pytest.mark.asyncio
async def test_alpaca_successful_connection(data_fetcher_instance):
    fetcher = data_fetcher_instance
    
    # Mock Alpaca StockDataStream
    with patch('src.data_fetcher.StockDataStream') as MockStockDataStream:
        mock_alpaca_client = MockStockDataStream.return_value
        mock_alpaca_client.run = AsyncMock(return_value=None) # Simulate successful run

        # Call the method under test
        symbols = ["SPY"]
        await fetcher.start_realtime_stream(symbols)

        # Assertions
        MockStockDataStream.assert_called_once_with('test_alpaca_key', 'test_alpaca_secret')
        mock_alpaca_client.subscribe_bars.assert_called_once_with(fetcher._alpaca_bar_data_handler, *symbols)
        mock_alpaca_client.run.assert_awaited_once()
        # The stream should exit gracefully, so should_run should become False
        # This requires a slight modification to the DataFetcher to ensure it exits the loop
        # after a successful run. For now, we'll assume it does and check the call counts.
        # The `start_realtime_stream` method has `return` after successful run, so it exits.
        assert fetcher.should_run is True # should_run remains True as the method returns

@pytest.mark.asyncio
async def test_alpaca_fallback_to_finnhub(data_fetcher_instance):
    fetcher = data_fetcher_instance
    
    # Mock Alpaca StockDataStream to raise ValueError multiple times
    with patch('src.data_fetcher.StockDataStream') as MockStockDataStream:
        mock_alpaca_client = MockStockDataStream.return_value
        mock_alpaca_client.run = AsyncMock(side_effect=ValueError("connection limit exceeded"))

        # Mock websockets.connect for Finnhub
        with patch('src.data_fetcher.websockets.connect') as mock_finnhub_connect:
            mock_websocket = AsyncMock()
            mock_finnhub_connect.return_value.__aenter__.return_value = mock_websocket
            mock_websocket.recv.side_effect = [
                json.dumps({"type": "trade", "data": [{"s": "SPY", "p": 100.0, "t": 1678886400000, "v": 1000}]}),
                websockets.exceptions.ConnectionClosedOK(rcvd=1000, sent=1000, rcvd_then_sent=True) # Simulate graceful shutdown after receiving one message
            ]

            symbols = ["SPY"]
            await fetcher.start_realtime_stream(symbols)

            # Assertions for Alpaca attempts
            assert MockStockDataStream.call_count == fetcher.MAX_ALPACA_RETRIES
            assert mock_alpaca_client.run.call_count == fetcher.MAX_ALPACA_RETRIES

            # Assertions for Finnhub connection
            mock_finnhub_connect.assert_called_once_with(f"wss://ws.finnhub.io?token=test_finnhub_key")
            mock_websocket.send.assert_called_once_with(json.dumps({"type": "subscribe", "symbol": "SPY"}))
            mock_websocket.recv.assert_awaited() # Ensure messages were received

            # Verify socketio emit for Finnhub data
            expected_finnhub_bar = {
                "time": 1678886400,
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000
            }
            fetcher.socketio.emit.assert_called_with('new_bar', expected_finnhub_bar)

@pytest.mark.asyncio
async def test_stop_realtime_stream(data_fetcher_instance):
    fetcher = data_fetcher_instance
    
    # Mock Alpaca StockDataStream
    with patch('src.data_fetcher.StockDataStream') as MockStockDataStream:
        mock_alpaca_client = MockStockDataStream.return_value
        mock_alpaca_client.close = AsyncMock()

        # Simulate stream running
        fetcher.should_run = True
        fetcher.websocket_client = mock_alpaca_client # Assign the mocked client

        await fetcher.stop_realtime_stream()

        assert fetcher.should_run is False
        mock_alpaca_client.close.assert_awaited_once()

# Test _alpaca_bar_data_handler
@pytest.mark.asyncio
async def test_alpaca_bar_data_handler(data_fetcher_instance):
    fetcher = data_fetcher_instance
    
    # Mock an Alpaca bar object
    mock_bar = MagicMock()
    mock_bar.timestamp.timestamp.return_value = 1678886400.0
    mock_bar.open = 100.0
    mock_bar.high = 101.0
    mock_bar.low = 99.0
    mock_bar.close = 100.5
    mock_bar.volume = 1000

    await fetcher._alpaca_bar_data_handler(mock_bar)

    expected_formatted_bar = {
        "time": 1678886400,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000
    }
    fetcher.socketio.emit.assert_called_once_with('new_bar', expected_formatted_bar)

# Test _bar_data_handler (unified)
@pytest.mark.asyncio
async def test_unified_bar_data_handler(data_fetcher_instance):
    fetcher = data_fetcher_instance
    
    test_bar_data = {
        "time": 1678886400,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000
    }

    await fetcher._bar_data_handler(test_bar_data)

    fetcher.socketio.emit.assert_called_once_with('new_bar', test_bar_data)
