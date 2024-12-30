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
import torch

def visualize(action_tensors_dict,
              action_strings_dict,
              actions_taken_dict,
              game_state_dict,
              monobeat_train_loop_agent_output_dict,
              monobeat_train_loop_env_output_dict,
              pos_to_unit_dict,
              unprocessed_actions_dict,
              early_actions_in_wrapper_dict):
    """
    Loops over each item in 'board_dict'. If it's a NumPy array with
    dimensionality >= 2, we visualize it.

    - 2D (H,W): directly plot as heatmap.
    - 3D (B,H,W): plot the slice arr[0]
    - 4D (B,C,H,W): plot the slice arr[0, 0]
    - 5D (B,C,H,W,Channels): plot each channel in the last dimension
      as a separate figure.
    """
    for key, arr in action_tensors_dict.items():
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



def load_and_plot(action_tensors,
                  action_strings,
                  actions_taken,
                  game_state,
                  monobeat_train_loop_agent_output,
                  monobeat_train_loop_env_output,
                  pos_to_unit,
                  unprocessed_actions,
                  early_actions_in_wrapper):
    """
    Load board dictionary from 'filename' and call visualize_board_dict.
    """
    with open(action_tensors, "rb") as f:
        action_tensors_dict = pickle.load(f)
    with open(action_strings, "rb") as f:
        action_strings_dict = pickle.load(f)
    with open(actions_taken, "rb") as f:
        actions_taken_dict = pickle.load(f)
    with open(game_state, "rb") as f:
        game_state_dict = pickle.load(f)
    with open(monobeat_train_loop_agent_output, "rb") as f:
        monobeat_train_loop_agent_output_dict = pickle.load(f)
    with open(monobeat_train_loop_env_output, "rb") as f:
        monobeat_train_loop_env_output_dict = pickle.load(f)
    with open(pos_to_unit, "rb") as f:
        pos_to_unit_dict = pickle.load(f)
    with open(unprocessed_actions, "rb") as f:
        unprocessed_actions_dict = pickle.load(f)
    with open(early_actions_in_wrapper, "rb") as f:
        early_actions_in_wrapper_dict = pickle.load(f)


    print(f"Loaded action_tensors dict from {action_tensors}")
    visualize(action_tensors_dict,
              action_strings_dict,
              actions_taken_dict,
              game_state_dict,
              monobeat_train_loop_agent_output_dict,
              monobeat_train_loop_env_output_dict,
              pos_to_unit_dict,
              unprocessed_actions_dict,
              early_actions_in_wrapper_dict)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_board.py "
              "<path_to_action_tensors_file.pkl> "
              "<path_to_action_strings_file.pkl> "
              "<path_to_actions_taken_file.pkl> "
              "<path_to_game_state_file.pkl> "
              "<path_to_monobeat_train_loop_agent_output.pkl> "
              "<path_to_monobeat_train_loop_env_output.pkl> "
              "<path_to_pos_to_unit_dict.pkl> "
              "<path_to_unprocessed_actions_file.pkl>"
              "<path_to_early_actions_in_wrapper.pkl>")
        sys.exit(1)
    action_tensors = sys.argv[1]
    action_strings = sys.argv[2]
    actions_taken = sys.argv[3]
    game_state = sys.argv[4]
    monobeat_train_loop_agent_output = sys.argv[5]
    monobeat_train_loop_env_output = sys.argv[6]
    pos_to_unit_dict = sys.argv[7]
    unprocessed_actions = sys.argv[8]
    early_actions_in_wrapper = sys.argv[9]
    load_and_plot(action_tensors,
                  action_strings,
                  actions_taken,
                  game_state,
                  monobeat_train_loop_agent_output,
                  monobeat_train_loop_env_output,
                  pos_to_unit_dict,
                  unprocessed_actions,
                  early_actions_in_wrapper)

