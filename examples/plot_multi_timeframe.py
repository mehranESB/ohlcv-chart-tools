from pathlib import Path
from chartDL.utils import visual
from chartDL.dataset import MultiDataset

# create mutli time frame dataset
dataset_path = Path("./DATA/multi_view/EURUSD-1h")
dataset = MultiDataset(dataset_path, 20)

# pick a sample and convert into dataframe
data = dataset[3]

# plot each time frame saperately
ax2 = visual.plot_bar_chart(
    data[2], color="#FBD288", linewidth=1.5, title="chart in '1h' timeframe"
)
ax2 = visual.plot_line_indicator(
    data[2], column="ema_10", color="#FBD288", linewidth=1, ax=ax2
)

ax1 = visual.plot_bar_chart(
    data[1], color="#FF9C73", linewidth=1.5, title="chart in '30m' timeframe"
)
ax1 = visual.plot_line_indicator(
    data[1], column="ema_10", color="#FF9C73", linewidth=1, ax=ax1
)

ax0 = visual.plot_bar_chart(
    data[0], color="#FF4545", linewidth=1.5, title="chart in '15m' timeframe"
)
ax0 = visual.plot_line_indicator(
    data[0], column="ema_10", color="#FF4545", linewidth=1, ax=ax0
)


# plot all time frame togather
ax = visual.plot_bar_chart(data[2], color="#FBD288", linewidth=6, ax=None)
ax = visual.plot_line_indicator(
    data[2], column="ema_10", color="#FBD288", linewidth=1, ax=ax
)
ax = visual.plot_bar_chart(data[1], color="#FF9C73", linewidth=3, ax=ax)
ax = visual.plot_line_indicator(
    data[1], column="ema_10", color="#FF9C73", linewidth=1, ax=ax
)
ax = visual.plot_bar_chart(
    data[0],
    color="#FF4545",
    linewidth=1.5,
    ax=ax,
    title="multi-view in ('15m', '30m', '1h')",
)
ax = visual.plot_line_indicator(
    data[0], column="ema_10", color="#FF4545", linewidth=1, ax=ax
)


# add legend
ax.legend()
