from __future__ import annotations

import os

import matplotlib.pyplot as plt


def setup_style():
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", font="sans-serif")
    except ImportError:
        pass
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "axes.titlesize": 12, "axes.labelsize": 11})


def save_figure(fig_dir, name: str, show: bool = True):
    os.makedirs(fig_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{name}.png"))
    plt.savefig(os.path.join(fig_dir, f"{name}.pdf"))
    if show:
        plt.show()
    else:
        plt.close()
