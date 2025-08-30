import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class InteractiveChartGenerator:
    def __init__(self):
        pass

    def generate_candlestick_chart(self, df: pd.DataFrame, title: str = "Candlestick Chart") -> go.Figure:
        """
        Generates an interactive candlestick chart with a separate volume subplot using Plotly.

        :param df: Pandas DataFrame with OHLCV data (must have 'open', 'high', 'low', 'close', 'volume' columns and a datetime index).
        :param title: Title of the chart.
        :return: Plotly Figure object.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close', and 'volume' columns.")

        # Create a figure with two subplots, sharing the x-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('', 'Volume'),
            row_width=[0.2, 0.7]  # Relative height of the subplots
        )

        # Add Candlestick chart to the first row
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Candlesticks'
            ),
            row=1,
            col=1
        )

        # Add Volume bar chart to the second row
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color='#666666'
            ),
            row=2,
            col=1
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=24, color='#E0E0E0')),
            xaxis_rangeslider_visible=False,  # Hide the range slider on the main chart
            template="plotly_dark",
            paper_bgcolor='#1E2026',
            plot_bgcolor='#1E2026',
            font=dict(color='#E0E0E0'),
            height=800,
            dragmode='pan',
            margin=dict(t=60, b=60, l=60, r=60),
            hovermode="x unified",
            bargap=0.1,
            uirevision="some_value"
        )

        # Update y-axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Remove the subplot titles which are not needed as we have y-axis titles
        fig.update_annotations(dict(text="", visible=False))


        return fig

if __name__ == '__main__':
    # Example Usage:
    # Create some dummy data for demonstration
    data = {
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
        'open': [100, 102, 105, 103, 106],
        'high': [103, 106, 107, 105, 108],
        'low': [99, 101, 103, 102, 104],
        'close': [102, 105, 103, 106, 105],
        'volume': [1000, 1200, 800, 1500, 1100]
    }
    dummy_df = pd.DataFrame(data).set_index('timestamp')

    chart_generator = InteractiveChartGenerator()
    fig = chart_generator.generate_candlestick_chart(dummy_df, title="Dummy Stock Chart")
    fig.show()
