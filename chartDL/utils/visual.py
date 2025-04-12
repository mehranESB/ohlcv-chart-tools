import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from . import time


def plot_bar_chart(
    df: pd.DataFrame,
    color: str = "blue",
    title: str = None,
    ax=None,
    linewidth=1,
    x_as_axis: bool = False,
):
    """
    Plots a candlestick-style bar chart representing market data (Open, High, Low, Close) for each timestamp.

    Args:
        df (pd.DataFrame): DataFrame containing market data with columns:
                           ['TimeStamp', 'Open', 'High', 'Low', 'Close'].
        color (str): Color of the bars in the chart. Default is 'blue'.
        title (str): Title of the plot. Default is 'Candlestick Chart'.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new plot will be created.
        linewidth (float): Line width for the bars. Default is 1.
        x_as_axis(bool): indicate that use 'X' column value for x axes in plot.

    Returns:
        ax: The axes object with the plot for further customization or saving.
    """
    # Ensure the DataFrame has the required columns
    required_columns = {"TimeStamp", "Open", "High", "Low", "Close"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The DataFrame must contain the columns: {required_columns}")

    # Create a new figure and axes if 'ax' is not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    # Get time delta for bar width
    timeframe = time.get_timeframe(df)
    delta = time.get_X_delta(df) if x_as_axis else time.get_time_delta(timeframe)

    # Loop through each row of the DataFrame to plot OHLC data as bars
    for i, row in enumerate(df.itertuples()):

        # x axes index
        x_index = (row.X - delta / 2) if x_as_axis else row.TimeStamp

        # Plot the opening price line (left half of bar)
        ax.plot(
            [x_index, x_index + delta / 2],
            [row.Open, row.Open],
            color=color,
            linewidth=linewidth,
            label=timeframe if i == 0 else "",
        )  # Label only once with timeframe

        # Plot the closing price line (right half of bar)
        ax.plot(
            [x_index + delta / 2, x_index + delta],
            [row.Close, row.Close],
            color=color,
            linewidth=linewidth,
        )

        # Plot the high-low range as a vertical line through the center of the bar
        ax.plot(
            [x_index + delta / 2, x_index + delta / 2],
            [row.High, row.Low],
            color=color,
            linewidth=linewidth,
        )

    # Labeling and formatting
    if title is not None:
        ax.set_title(title)  # Title for the plot
    ax.set_xlabel("Time")  # X-axis label
    ax.set_ylabel("Price")  # Y-axis label

    # Grid for better readability
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    # Tight layout for better plot appearance
    ax.axis("tight")

    return ax  # Return the axes object


def plot_line_indicator(
    df: pd.DataFrame,
    column: str,
    color: str = "blue",
    ax=None,
    linewidth=1,
    x_as_axis: bool = False,
):
    """
    Plots a line chart for a specified indicator column over time.

    Args:
        df (pd.DataFrame): DataFrame containing the timestamp and indicator data.
                           Must contain 'TimeStamp' and the specified 'column'.
        column (str): The column name for the indicator to plot.
        color (str): Line color for the plot. Default is 'blue'.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new plot will be created.
        linewidth (float): Line width of the indicator plot. Default is 1.
        x_as_axis(bool): indicate that use 'X' column value for x axes in plot.

    Returns:
        ax: The axes object with the line indicator plot.
    """
    # Ensure the DataFrame has the required column
    if column not in df.columns:
        raise ValueError(f"The DataFrame must contain the column: '{column}'")

    # Create a new figure and axes if 'ax' is not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    # Get time delta for horizontal alignment (assuming time.get_timeframe and time.get_time_delta are valid functions)
    timeframe = time.get_timeframe(df)
    delta = time.get_X_delta(df) if x_as_axis else time.get_time_delta(timeframe)

    # X and Y data for plotting the indicator
    x_data = (
        df["X"] if x_as_axis else df["TimeStamp"] + (delta / 2)
    )  # Adjusted to center on bar
    y_data = df[column]

    # Plot the indicator as a line chart
    ax.plot(x_data, y_data, color=color, linewidth=linewidth, label=column)

    # Tight layout for neat display
    ax.axis("tight")

    return ax  # Return the axes object


def plot_candlestick_chart(
    df: pd.DataFrame,
    color: str = "black",
    title: str = None,
    ax=None,
    linewidth=1,
    x_as_axis: bool = False,
):
    """
    Plots a candlestick chart representing market data (Open, High, Low, Close) for each timestamp.

    Args:
        df (pd.DataFrame): DataFrame containing market data with columns:
                           ['TimeStamp', 'Open', 'High', 'Low', 'Close'].
        bullish_color (str): Color for bullish (price up) candles. Default is 'green'.
        bearish_color (str): Color for bearish (price down) candles. Default is 'red'.
        title (str): Title of the plot. Default is 'Candlestick Chart'.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new plot will be created.
        linewidth (float): Line width for candlestick edges. Default is 1.
        x_as_axis (bool): Whether to use 'X' column for x-axis instead of 'TimeStamp'. Default is False.

    Returns:
        ax: The axes object with the plot for further customization or saving.
    """
    # Ensure the DataFrame has the required columns
    required_columns = {"TimeStamp", "Open", "High", "Low", "Close"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The DataFrame must contain the columns: {required_columns}")

    # Create a new figure and axes if 'ax' is not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    # Get time delta for candlestick width
    timeframe = time.get_timeframe(df)
    delta = time.get_X_delta(df) if x_as_axis else time.get_time_delta(timeframe)

    # Loop through each row to plot candlesticks
    for i, row in enumerate(df.itertuples()):
        # Determine x-coordinate (center of candlestick)
        x_index = row.X if x_as_axis else (row.TimeStamp + delta / 2)

        # Plot the high-low line (wick/shadow)
        ax.plot(
            [x_index, x_index], [row.Low, row.High], color=color, linewidth=linewidth
        )

        # Determine body color based on Open and Close values
        body_color = "#f2f2f2" if row.Close >= row.Open else color

        # Plot the candlestick body as a rectangle
        body_height = abs(row.Close - row.Open)
        lower_bound = min(row.Open, row.Close)  # Bottom of the rectangle
        rect = Rectangle(
            (x_index - delta / 2, lower_bound),  # Bottom-left corner of the rectangle
            width=delta,  # Width of the rectangle
            height=body_height,  # Height of the rectangle (Open to Close difference)
            facecolor=body_color,  # Fill color (bullish or bearish)
            edgecolor=color,  # Border color
            linewidth=linewidth,  # Border line width
            alpha=1.0,  # Transparency
        )
        ax.add_patch(rect)

    # Labeling and formatting
    if title is not None:
        ax.set_title(title)  # Title for the plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")

    # Grid for better readability
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    # Tight layout for neat display
    ax.axis("tight")

    return ax
