"""Shared figure style (consistent look for slides and the report)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#0b0b0b"          # observed data
INK_2 = "#52514e"        # secondary text
GRID = "#e4e3df"
PRIOR = "#c3c2b7"        # the 100 provided cases (recessive)
BLUE = "#2a78d6"         # model / forecast (series 1)
ORANGE = "#eb6834"       # comparison series (series 2)
AQUA = "#1baf7a"         # series 3
BLUE_LIGHT = "#9ec5f4"   # uncertainty band
SURFACE = "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0,
        "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path.relative_to(path.parents[2])}")


def year_axis(ax, step=2):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
