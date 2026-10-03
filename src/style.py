"""One consistent look for every figure in the report, the app and the slides."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1B2A41"       # dark navy for text and main series
PAPER = "#FAF8F4"     # warm off-white background
GRID = "#E4E0D8"
MUTED = "#6B7280"
ACCENT = "#E07A2E"    # orange: highlights, warnings
BLUE = "#3A6EA5"

GROUP_COLORS = {
    "Violent Crime": "#C8433A",
    "Vehicle Crime": "#3A6EA5",
    "Burglary & Theft": "#E9A23B",
    "Fraud & Identity": "#1F8A84",
    "Vandalism": "#7A5C99",
    "Other / Public Order": "#9CA3AF",
}


def apply():
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
