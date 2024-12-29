#!/usr/bin/env python3
"""
plot_board.py
Usage:
    python plot_board.py game_X.pkl

Description:
    Loads a saved board dictionary (with NumPy arrays) and visualizes
    2D slices or channels as a heatmap in matplotlib.

    Example:
      python plot_board.py board.pkl
"""

import pickle
import numpy as np
import sys

import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5 installed
import matplotlib.pyplot as plt


def visualize_board_dict(board_dict):
    """
    Loops over each item in 'board_dict'. If it's a NumPy array with
    dimensionality >= 2, we visualize it.

    - 2D (H,W): directly plot as heatmap.
    - 3D (B,H,W): plot the slice arr[0]
    - 4D (B,C,H,W): plot the slice arr[0, 0]
    - 5D (B,C,H,W,Channels): plot each channel in the last dimension
      as a separate figure.
    """
    for key, arr in board_dict.items():
        if not (isinstance(arr, np.ndarray) and arr.ndim >= 2):
            print(f"{key} is not a multi-dimensional NumPy array, skipping plot.")
            continue

        print(f"Plotting {key} with shape {arr.shape}")

        if arr.ndim == 2:
            # Straight 2D array
            data_2d = arr
            plt.figure()
            plt.imshow(data_2d, cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.title(f"{key} (2D)")

        elif arr.ndim == 3:
            # e.g. (B, H, W) – pick the first index
            data_2d = arr[0]
            plt.figure()
            plt.imshow(data_2d, cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.title(f"{key} (3D, showing arr[0])")

        elif arr.ndim == 4:
            # e.g. (B, C, H, W) – pick arr[0, 0]
            data_2d = arr[0, 0]
            plt.figure()
            plt.imshow(data_2d, cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.title(f"{key} (4D, showing arr[0,0])")

        elif arr.ndim == 5:
            # e.g. (B, C, H, W, Channels=17)
            # Plot each channel in the last dimension
            B, C, H, W, Channels = arr.shape
            # We'll just grab [0,0,:,:,:] for the first two dims
            # and then make one plot for each of the last dimension channels
            for ch in range(Channels):
                data_2d = arr[0, 0, :, :, ch]
                plt.figure()
                plt.imshow(data_2d, cmap="viridis", aspect="auto")
                plt.colorbar()
                plt.title(f"{key} (5D, channel={ch})")

        # Add other elif cases if you expect shapes beyond 5D, etc.

    # Finally, show all the figures
    plt.show()


def load_and_plot(filename):
    """
    Load board dictionary from 'filename' and call visualize_board_dict.
    """
    with open(filename, "rb") as f:
        board_dict = pickle.load(f)
    print(f"Loaded board dict from {filename}")
    visualize_board_dict(board_dict)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_board.py <path_to_board_file.pkl>")
        sys.exit(1)
    board_file = sys.argv[1]
    load_and_plot(board_file)
