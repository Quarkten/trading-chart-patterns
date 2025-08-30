import React, { useState, useEffect, useMemo, useRef } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import { io } from 'socket.io-client';
import { BarChart3, TrendingUp, TrendingDown, Volume2, Settings, Eye } from 'lucide-react';

// Function to aggregate data based on timeframe
const aggregateData = (data, timeframe) => {
    if (!data || data.length === 0) return [];

    const aggregated = [];
    let currentAggregate = null;
    let periodStart = null;

    let intervalMs;
    const unit = timeframe.slice(-1);
    const value = parseInt(timeframe.slice(0, -1));

    if (unit === 'h') {
        intervalMs = value * 60 * 60 * 1000; // hours to milliseconds
    } else if (unit === 'M') {
        // For months, we aggregate daily data
        // This is a simplification, as months have variable days
        // We'll group by month and year
        intervalMs = null; // Special handling for months
    } else if (unit === 'Y') {
        // For years, we aggregate daily data
        intervalMs = null; // Special handling for years
    } else {
        return data; // Should not happen if logic is correct
    }

    for (const item of data) {
        const itemTime = item.time * 1000; // Convert to milliseconds

        if (intervalMs) { // For fixed intervals like hours
            if (currentAggregate === null || itemTime >= periodStart + intervalMs) {
                if (currentAggregate) {
                    aggregated.push(currentAggregate);
                }
                periodStart = Math.floor(itemTime / intervalMs) * intervalMs;
                currentAggregate = {
                    time: periodStart / 1000, // Convert back to seconds
                    open: item.open,
                    high: item.high,
                    low: item.low,
                    close: item.close,
                    volume: item.volume,
                };
            } else {
                currentAggregate.high = Math.max(currentAggregate.high, item.high);
                currentAggregate.low = Math.min(currentAggregate.low, item.low);
                currentAggregate.close = item.close;
                currentAggregate.volume += item.volume;
            }
        } else if (unit === 'M') { // For months
            const date = new Date(itemTime);
            const month = date.getMonth();
            const year = date.getFullYear();

            if (currentAggregate === null || currentAggregate.month !== month || currentAggregate.year !== year) {
                if (currentAggregate) {
                    aggregated.push({
                        time: new Date(currentAggregate.year, currentAggregate.month, 1).getTime() / 1000,
                        open: currentAggregate.open,
                        high: currentAggregate.high,
                        low: currentAggregate.low,
                        close: currentAggregate.close,
                        volume: currentAggregate.volume,
                    });
                }
                currentAggregate = {
                    time: item.time,
                    open: item.open,
                    high: item.high,
                    low: item.low,
                    close: item.close,
                    volume: item.volume,
                    month: month,
                    year: year,
                };
            } else {
                currentAggregate.high = Math.max(currentAggregate.high, item.high);
                currentAggregate.low = Math.min(currentAggregate.low, item.low);
                currentAggregate.close = item.close;
                currentAggregate.volume += item.volume;
            }
        } else if (unit === 'Y') { // For years
            const date = new Date(itemTime);
            const year = date.getFullYear();

            if (currentAggregate === null || currentAggregate.year !== year) {
                if (currentAggregate) {
                    aggregated.push({
                        time: new Date(currentAggregate.year, 0, 1).getTime() / 1000,
                        open: currentAggregate.open,
                        high: currentAggregate.high,
                        low: currentAggregate.low,
                        close: currentAggregate.close,
                        volume: currentAggregate.volume,
                    });
                }
                currentAggregate = {
                    time: item.time,
                    open: item.open,
                    high: item.high,
                    low: item.low,
                    close: item.close,
                    volume: item.volume,
                    year: year,
                };
            } else {
                currentAggregate.high = Math.max(currentAggregate.high, item.high);
                currentAggregate.low = Math.min(currentAggregate.low, item.low);
                currentAggregate.close = item.close;
                currentAggregate.volume += item.volume;
            }
        }
    }

    if (currentAggregate) {
        if (intervalMs) {
            aggregated.push(currentAggregate);
        } else if (unit === 'M') {
            aggregated.push({
                time: new Date(currentAggregate.year, currentAggregate.month, 1).getTime() / 1000,
                open: currentAggregate.open,
                high: currentAggregate.high,
                low: currentAggregate.low,
                close: currentAggregate.close,
                volume: currentAggregate.volume,
            });
        } else if (unit === 'Y') {
            aggregated.push({
                time: new Date(currentAggregate.year, 0, 1).getTime() / 1000,
                open: currentAggregate.open,
                high: currentAggregate.high,
                low: currentAggregate.low,
                close: currentAggregate.close,
                volume: currentAggregate.volume,
            });
        }
    }
    return aggregated;
};

// Helper function to determine background color based on trading session
const getBackgroundColorForSession = (unixTimestamp) => {
    if (!unixTimestamp) {
        return '#131722'; // Default color
    }
    // Assuming timestamps are UTC, convert to ET for market hours
    const date = new Date(unixTimestamp * 1000);
    
    // Get UTC time and apply a -4 hour offset for EDT
    const utcHour = date.getUTCHours();
    const utcMinutes = date.getUTCMinutes();
    
    // Calculate total minutes from midnight in UTC
    const totalUtcMinutes = utcHour * 60 + utcMinutes;
    
    // Apply offset for ET (EDT is UTC-4, so 4 * 60 = 240 minutes)
    // Use modulo to handle day wrapping
    const totalEtMinutes = (totalUtcMinutes - (4 * 60) + 1440) % 1440;

    // Define market hours in minutes from midnight ET
    const marketOpen = 9 * 60 + 30; // 9:30 AM
    const marketClose = 16 * 60;     // 4:00 PM
    const afterHoursEnd = 20 * 60;   // 8:00 PM
    const preMarketStart = 4 * 60;   // 4:00 AM

    // Regular trading hours
    if (totalEtMinutes >= marketOpen && totalEtMinutes < marketClose) {
        return '#1e2a3a'; // Dark navy blue
    }
    // After-hours
    else if (totalEtMinutes >= marketClose && totalEtMinutes < afterHoursEnd) {
        return '#2a3a4a'; // Lighter navy
    }
    // Pre-market
    else if (totalEtMinutes >= preMarketStart && totalEtMinutes < marketOpen) {
        return '#3a2e1e'; // Brown-yellowish
    }
    // Default for overnight
    return '#131722';
};

const TradingChart = () => {
    const [selectedSymbol, setSelectedSymbol] = useState('SPY');
    const [timeframe, setTimeframe] = useState('15m');
    const [detectedPatterns, setDetectedPatterns] = useState([]);

    const chartContainerRef = useRef();
    const chartRef = useRef(); // Use useRef for the chart instance
    const candlestickSeriesRef = useRef();
    const volumeSeriesRef = useRef();

    const [chartData, setChartData] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const currentCandle = useMemo(() => chartData[chartData.length - 1], [chartData]);
    const previousCandle = useMemo(() => chartData[Math.max(0, chartData.length - 2)], [chartData]);
    const priceChange = useMemo(() => currentCandle ? currentCandle.close - previousCandle.close : 0, [currentCandle, previousCandle]);
    const priceChangePercent = useMemo(() => previousCandle ? (priceChange / previousCandle.close) * 100 : 0, [priceChange, previousCandle]);

    // Function to detect patterns
    const detectPatterns = async () => {
        if (!chartRef.current) return;

        try {
            // Get chart as image
            const canvas = await chartRef.current.takeScreenshot();
            const imageData = canvas.toDataURL('image/png');

            const response = await fetch('http://localhost:5000/detect_patterns', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const patterns = await response.json();
            setDetectedPatterns(patterns);
            console.log("Detected Patterns:", patterns);

        } catch (e) {
            console.error("Error detecting patterns:", e);
        }
    };

    // Effect for fetching data
    useEffect(() => {
        const fetchData = async () => {
            setIsLoading(true);
            setError(null);
            try {
                let baseMultiplier, baseTimespan;
                const now = new Date();
                let fromDate = new Date();
                let toDate = now;

                // Determine multiplier and timespan based on selected timeframe
                let aggregate = false;

                if (timeframe.endsWith('m')) {
                    baseMultiplier = parseInt(timeframe.slice(0, -1));
                    baseTimespan = 'minute';
                    fromDate.setDate(now.getDate() - 7); // Last 7 days for minutes
                } else if (timeframe.endsWith('h')) {
                    baseMultiplier = parseInt(timeframe.slice(0, -1));
                    baseTimespan = 'hour';
                    fromDate.setDate(now.getDate() - 30); // Last 30 days for hours
                    if (parseInt(timeframe.slice(0, -1)) > 1) {
                        aggregate = true;
                    }
                } else if (timeframe.endsWith('d')) {
                    baseMultiplier = parseInt(timeframe.slice(0, -1));
                    baseTimespan = 'day';
                    fromDate.setFullYear(now.getFullYear() - 1); // Last 1 year for days
                } else if (timeframe.endsWith('M')) { // For months
                    baseMultiplier = 1; // Fetch daily data for month aggregation
                    baseTimespan = 'day';
                    fromDate.setFullYear(now.getFullYear() - 5); // Last 5 years for months
                    aggregate = true;
                } else if (timeframe.endsWith('Y')) { // For years
                    baseMultiplier = 1; // Fetch daily data for year aggregation
                    baseTimespan = 'day';
                    fromDate.setFullYear(now.getFullYear() - 10); // Last 10 years for years
                    aggregate = true;
                } else {
                    // Default to 1 day if timeframe is not recognized
                    baseMultiplier = 1;
                    baseTimespan = 'day';
                    fromDate.setFullYear(now.getFullYear() - 1);
                }

                // Format dates for API
                const fromDateStr = fromDate.toISOString().split('T')[0];
                const toDateStr = toDate.toISOString().split('T')[0];

                const response = await fetch(`http://localhost:5000/candlestick_data?symbol=${selectedSymbol}&timeframe_value=${baseMultiplier}&timeframe_unit=${baseTimespan}&from_date=${fromDateStr}&to_date=${toDateStr}`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                let data = await response.json();

                if (aggregate) {
                    data = aggregateData(data, timeframe);
                }
                setChartData(data);
            } catch (e) {
                console.error("Error fetching data:", e);
                setError(e.message);
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();
    }, [selectedSymbol, timeframe]);

    // Effect for Socket.IO connection and real-time data
    useEffect(() => {
        const socket = io('http://localhost:5000');

        socket.on('connect', () => {
            console.log('Connected to Socket.IO server');
            socket.emit('start_stream', { symbols: [selectedSymbol] });
        });

        socket.on('new_bar', (bar) => {
            console.log('Received new bar:', bar);
            // Update chart data with the new bar
            // Lightweight-charts expects data to be sorted by time
            // For real-time, we usually append or update the last bar
            setChartData(prevData => {
                const lastBar = prevData[prevData.length - 1];
                if (lastBar && lastBar.time === bar.time) {
                    // Update existing bar if it's the same time (e.g., partial bar update)
                    return [...prevData.slice(0, -1), bar];
                } else {
                    // Add new bar
                    return [...prevData, bar];
                }
            });
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from Socket.IO server');
        });

        // Clean up on component unmount or symbol change
        return () => {
            socket.disconnect();
        };
    }, [selectedSymbol]);

    // Effect for chart initialization and resizing
    useEffect(() => {
        if (!chartContainerRef.current) {
            return; // Exit if container is not available
        }

        // Create chart only once
        if (!chartRef.current) {
            console.log("Creating chart instance and series...");
            const chart = createChart(chartContainerRef.current, {
                width: chartContainerRef.current.clientWidth,
                height: 500,
                layout: {
                    backgroundColor: '#131722',
                    textColor: 'rgba(255, 255, 255, 0.9)',
                },
                grid: {
                    vertLines: { color: '#2A2E39' },
                    horzLines: { color: '#2A2E39' },
                },
                crosshair: {
                    mode: CrosshairMode.Normal,
                },
                rightPriceScale: {
                    borderColor: '#485158',
                },
                timeScale: {
                    borderColor: '#485158',
                },
            });
            chartRef.current = chart;

            candlestickSeriesRef.current = chart.addCandlestickSeries({
                upColor: '#10B981',
                downColor: '#EF4444',
                borderDownColor: '#EF4444',
                borderUpColor: '#10B981',
                wickDownColor: '#EF4444',
                wickUpColor: '#10B981',
            });

            volumeSeriesRef.current = chart.addHistogramSeries({
                color: '#26a69a',
                priceFormat: {
                    type: 'volume',
                },
                priceScaleId: 'volume_scale',
                scaleMargins: {
                    top: 0.8,
                    bottom: 0,
                },
            });
            chart.priceScale('volume_scale').applyOptions({
                scaleMargins: {
                    top: 0.8,
                    bottom: 0,
                },
            });

            const handleResize = () => {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            };

            window.addEventListener('resize', handleResize);

            return () => {
                window.removeEventListener('resize', handleResize);
                if (chartRef.current) {
                    chartRef.current.remove();
                    chartRef.current = null;
                    candlestickSeriesRef.current = null;
                    volumeSeriesRef.current = null;
                }
            };
        }
    }, [chartContainerRef]);

    // Effect for updating chart data when chartData changes
    useEffect(() => {
        if (candlestickSeriesRef.current && volumeSeriesRef.current && chartData.length > 0) {
            candlestickSeriesRef.current.setData(chartData);
            const volumeData = chartData.map(d => ({
                time: d.time,
                value: d.volume,
                color: d.close >= d.open ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)',
            }));
            volumeSeriesRef.current.setData(volumeData);

            // Directly calculate and apply background color
            const lastCandle = chartData[chartData.length - 1];
            if (chartRef.current && lastCandle) {
                const newBackgroundColor = getBackgroundColorForSession(lastCandle.time);
                chartRef.current.applyOptions({
                    layout: {
                        backgroundColor: newBackgroundColor,
                    },
                });
            }

            if (chartRef.current) {
                chartRef.current.timeScale().fitContent();
            }
        }
    }, [chartData]);

    // Effect for overlaying detected patterns
    useEffect(() => {
        if (!chartRef.current || detectedPatterns.length === 0) {
            return;
        }

        // Clear previous overlays (if any)
        // Lightweight-charts doesn't have a direct way to remove shapes by ID
        // A common approach is to manage a separate series for overlays
        // For simplicity, we'll just re-draw everything or manage a dedicated overlay series
        // For now, let's assume we add new series for each pattern, which is not ideal
        // A better approach would be to use custom drawings or a single series with transparent data

        const overlaySeries = [];

        detectedPatterns.forEach(pattern => {
            const { box, class_name } = pattern;
            const [x1_px, y1_px, x2_px, y2_px] = box; // Pixel coordinates from YOLO

            // Get chart dimensions and visible range
            const chartWidth = chartContainerRef.current.clientWidth;
            const chartHeight = 500; // Fixed height of the chart container

            const timeScale = chartRef.current.timeScale();
            const priceScale = chartRef.current.priceScale();

            // Map pixel coordinates to time and price values
            // This is an approximation and might need fine-tuning based on chart padding/margins
            const timeStart = timeScale.coordinateToTime(x1_px);
            const timeEnd = timeScale.coordinateToTime(x2_px);
            const priceTop = priceScale.coordinateToPrice(y1_px);
            const priceBottom = priceScale.coordinateToPrice(y2_px);

            // Add a new line series to draw the rectangle
            // This is a simplified representation, a custom renderer would be better
            const patternLineSeries = chartRef.current.addLineSeries({
                color: 'rgba(255, 0, 0, 0.5)',
                lineWidth: 2,
                lineStyle: 0, // LineStyle.Solid
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });

            // Draw the rectangle using 4 data points
            const rectangleData = [
                { time: timeStart, value: priceTop },
                { time: timeEnd, value: priceTop },
                { time: timeEnd, value: priceBottom },
                { time: timeStart, value: priceBottom },
                { time: timeStart, value: priceTop }, // Close the rectangle
            ];
            patternLineSeries.setData(rectangleData);

            // Store the series to remove it later
            overlaySeries.push(patternLineSeries);

            // Add text annotation (requires custom drawing or HTML overlay)
            // For now, we'll just log it
            console.log(`Pattern: ${class_name} at [${timeStart}, ${priceTop}] to [${timeEnd}, ${priceBottom}]`);
        });

        // Cleanup function to remove overlays when detectedPatterns changes or component unmounts
        return () => {
            overlaySeries.forEach(series => {
                chartRef.current.removeSeries(series);
            });
        };

    }, [detectedPatterns, chartRef.current]);

    return (
        <div className="bg-gray-900 text-white p-4 font-sans antialiased">
            {/* Header */}
            <div className="flex items-center justify-between mb-4 bg-gray-800 p-3 rounded-lg shadow-md">
                <div className="flex items-center space-x-4">
                    <select value={selectedSymbol} onChange={(e) => setSelectedSymbol(e.target.value)} className="bg-gray-700 text-white px-3 py-1.5 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500" >
                        <option value="SPY">SPY</option>
                        <option value="AAPL">AAPL</option>
                        <option value="TSLA">TSLA</option>
                        <option value="MSFT">MSFT</option>
                    </select>
                    <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)} className="bg-gray-700 text-white px-3 py-1.5 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500" >
                        <option value="1m">1m</option>
                        <option value="5m">5m</option>
                        <option value="15m">15m</option>
                        <option value="1h">1h</option>
                        <option value="2h">2h</option>
                        <option value="3h">3h</option>
                        <option value="4h">4h</option>
                        <option value="1d">1d</option>
                        <option value="1M">1M</option>
                        <option value="3M">3M</option>
                        <option value="6M">6M</option>
                        <option value="1Y">1Y</option>
                    </select>
                </div>
                <div className="flex items-center space-x-6">
                    <button className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors">
                        <BarChart3 size={18} />
                        <span>Indicators</span>
                    </button>
                    <button onClick={detectPatterns} className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors">
                        <Eye size={18} />
                        <span>Detect Patterns</span>
                    </button>
                    <button className="text-gray-300 hover:text-white transition-colors">
                        <Settings size={20} />
                    </button>
                </div>
            </div>

            {/* Price Info */}
            <div className="mb-4 bg-gray-800 p-4 rounded-lg shadow-md">
                <div className="flex items-center space-x-8">
                    <div>
                        <span className="text-gray-400 text-sm">Symbol</span>
                        <p className="text-2xl font-bold">{selectedSymbol}</p>
                    </div>
                    <div>
                        <span className="text-gray-400 text-sm">Price</span>
                        <p className="text-2xl font-bold">{currentCandle ? `$${currentCandle.close.toFixed(2)}` : 'N/A'}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                        {priceChange >= 0 ? <TrendingUp size={20} className="text-green-500" /> : <TrendingDown size={20} className="text-red-500" />}
                        <div>
                            <span className="text-gray-400 text-sm">Change</span>
                            <p className={`font-semibold ${priceChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                {currentCandle ? `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)} (${priceChangePercent.toFixed(2)}%)` : 'N/A'}
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center space-x-2">
                        <Volume2 size={20} className="text-gray-400" />
                        <div>
                            <span className="text-gray-400 text-sm">Volume</span>
                            <p className="font-semibold">{currentCandle ? currentCandle.volume.toLocaleString() : 'N/A'}</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Chart */}
            <div className="bg-gray-800 rounded-lg shadow-md" ref={chartContainerRef} style={{ height: '500px' }}>
                {isLoading && <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75 text-white text-lg">Loading Chart Data...</div>}
                {error && <div className="absolute inset-0 flex items-center justify-center bg-red-900 bg-opacity-75 text-white text-lg">Error: {error}</div>}
                {!isLoading && !error && chartData.length === 0 && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75 text-white text-lg">No data available for the selected period.</div>
                )}
            </div>

            {/* Controls (removed play button and slider) */}
            <div className="mt-4 flex items-center justify-between mb-4 bg-gray-800 p-3 rounded-lg shadow-md">
                <div className="flex items-center space-x-4 text-sm text-gray-400">
                    <span>Data Points: {chartData.length}</span>
                </div>
            </div>
        </div>
    );
};

export default TradingChart;