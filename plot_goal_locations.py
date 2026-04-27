"""
Plot goal locations from a goal_loc_images.pkl file.

Usage:
    python plot_goals.py --path /path/to/goal_loc_images.pkl
    python plot_goals.py --path /path/to/goal_loc_images.pkl --mode 3d
    python plot_goals.py --path /path/to/goal_loc_images.pkl --axes xz
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_goals(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        goal_data = pickle.load(f)

    if not isinstance(goal_data, (list, tuple)) or len(goal_data) == 0:
        raise ValueError("Pickle file must contain a non-empty list of goal dicts.")

    positions = []
    for g in goal_data:
        if "position" not in g:
            raise KeyError(f"Goal entry missing 'position' key: {g}")
        positions.append(np.asarray(g["position"], dtype=np.float32))

    return np.stack(positions)  # shape (N, D)


def plot_2d(positions: np.ndarray, axes: str = "xz", save_path: str = None):
    axis_map = {"x": 0, "y": 1, "z": 2}
    a0, a1 = axis_map[axes[0]], axis_map[axes[1]]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")

    scatter = ax.scatter(
        positions[:, a0],
        positions[:, a1],
        c=np.arange(len(positions)),
        cmap="plasma",
        s=40,
        alpha=0.85,
        edgecolors="none",
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Goal index", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel(axes[0].upper(), color="white", fontsize=12)
    ax.set_ylabel(axes[1].upper(), color="white", fontsize=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.set_title(
        f"Goal Locations — {axes[0].upper()}/{axes[1].upper()} plane  (N={len(positions)})",
        color="white",
        fontsize=13,
        pad=12,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_3d(positions: np.ndarray, save_path: str = None):
    if positions.shape[1] < 3:
        raise ValueError("3D plot requires positions with at least 3 dimensions.")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    sc = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=np.arange(len(positions)),
        cmap="plasma",
        s=30,
        alpha=0.85,
        depthshade=True,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Goal index", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("X", color="white", labelpad=8)
    ax.set_ylabel("Y", color="white", labelpad=8)
    ax.set_zlabel("Z", color="white", labelpad=8)
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333")
    ax.yaxis.pane.set_edgecolor("#333")
    ax.zaxis.pane.set_edgecolor("#333")

    ax.set_title(
        f"Goal Locations — 3D  (N={len(positions)})",
        color="white",
        fontsize=13,
        pad=15,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot goal locations from a pickle file.")
    parser.add_argument("--path", required=True, help="Path to goal_loc_images.pkl")
    parser.add_argument(
        "--mode",
        choices=["2d", "3d"],
        default="2d",
        help="Plot mode: '2d' (default) or '3d'",
    )
    parser.add_argument(
        "--axes",
        default="xz",
        help="Which two axes to plot in 2D mode (e.g. 'xz', 'xy', 'yz'). Default: xz",
    )
    parser.add_argument("--save", default=None, help="Optional path to save the figure (e.g. goals.png)")
    args = parser.parse_args()

    positions = load_goals(args.path)
    print(f"Loaded {len(positions)} goals, position shape: {positions.shape}")
    print(f"  X range: [{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]")
    if positions.shape[1] > 1:
        print(f"  Y range: [{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")
    if positions.shape[1] > 2:
        print(f"  Z range: [{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")

    if args.mode == "3d":
        plot_3d(positions, save_path=args.save)
    else:
        plot_2d(positions, axes=args.axes, save_path=args.save)


if __name__ == "__main__":
    main()