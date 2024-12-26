#!/usr/bin/env python3
"""
plot_board.py
Usage:
    python plot_board.py game_X.pkl

Description:
    Loads a saved board dictionary (with NumPy arrays) and visualizes
    each 2D slice as a heatmap in matplotlib.

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
    Loops over each item in 'board_dict'. If it's a NumPy array of shape
    >= 2D, we show a heatmap for the first 2D slice.
    """
    for key, arr in board_dict.items():
        if isinstance(arr, np.ndarray) and arr.ndim >= 2:
            print(f"Plotting {key} with shape {arr.shape}")

            # If arr is exactly 2D, we can just show it
            if arr.ndim == 2:
                data_2d = arr
            elif arr.ndim == 3:
                # Example: (B, H, W) – pick the first "batch" or "channel"
                data_2d = arr[0]
            else:
                # For 4D or more, pick [0,0] to get a 2D slice
                data_2d = arr[0, 0]

            plt.figure()
            plt.imshow(data_2d, cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.title(f"{key} (slice)")
        else:
            # If it's not a NumPy array or it's 1D, skip or print
            print(f"{key} is not a multi-dimensional NumPy array (shape?), skipping plot.")

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
