# cerberus_replays Directory Overview

## Purpose
This directory contains tools and utilities for processing and analyzing game replays from the Lux AI competition. It helps in understanding agent behavior, visualizing game states, and extracting insights from match results.

## Key Files
- `cerberus_viz.py`: Visualization tool for game states
  - Generates heatmaps of unit positions and resource distribution
  - Creates time series plots of game metrics
  - Helps analyze strategic decisions and positioning

- `process_cerberus_replays.py`: Replay analysis utilities
  - Extracts game results and statistics
  - Processes replay data for analysis
  - Helps identify patterns in agent behavior

## Role in RL Approach
The replay analysis tools were crucial for:
1. Understanding Agent Behavior
   - Visualizing learned strategies
   - Identifying successful and failed approaches
   - Analyzing positional play and resource management

2. Training Improvement
   - Identifying scenarios where the agent struggled
   - Understanding the impact of different reward structures
   - Validating the effectiveness of training phases

3. Strategic Analysis
   - Heat maps showing territorial control
   - Resource collection and usage patterns
   - City placement and expansion strategies

This directory's tools helped validate and refine the training approach by providing insights into the agent's learned behaviors and strategic decisions.
