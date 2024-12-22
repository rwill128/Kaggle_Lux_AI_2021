"""
Main entry point for the Lux AI agent. This script handles the interaction between
the reinforcement learning agent and the Kaggle game environment.

The script implements a game loop that:
1. Reads observations from stdin (game state updates)
2. Processes these into a structured observation format
3. Feeds observations to the RL agent
4. Returns the agent's actions back to the environment

The RL agent (imported from lux_ai.rl_agent.rl_agent) uses a sophisticated
neural network with 24 residual blocks and squeeze-excitation layers to process
the game state and output actions for all units simultaneously.
"""

from contextlib import redirect_stdout
import io
# Silence "Loading environment football failed: No module named 'gfootball'" message
with redirect_stdout(io.StringIO()):
    import kaggle_environments
from typing import Dict
from lux_ai.rl_agent.rl_agent import agent
# from lux_ai.handcrafted_agents.needs_name_v0 import agent


if __name__ == "__main__":

    def read_input():
        """
        Reads a single line of input from stdin containing game state updates.
        
        The input format follows the Lux AI competition specification, with each line
        containing either a game state update or special commands like "D_DONE".
        
        Returns:
            str: A line of input from stdin
            
        Raises:
            SystemExit: If EOF is reached, indicating the game has ended
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)
    step = 0

    class Observation(Dict[str, any]):
        """
        A structured representation of the game state used by the RL agent.
        
        This class inherits from Dict to maintain compatibility with the Kaggle environment
        while adding custom attributes needed for the RL agent. It accumulates game state
        updates and tracks important information like the current step and player ID.
        
        Args:
            player (int, optional): The player ID (0 or 1). Defaults to 0.
            
        Attributes:
            player (int): The player ID this observation is for
            updates (list): List of game state update strings from the environment
            step (int): Current game step number
            remainingOverageTime (float): Remaining computation time buffer
        """
        def __init__(self, player=0):
            self.player = player
            # Initialize these via the dict interface instead for compatibility
            # self.updates = []
            # self.step = 0
    observation = Observation()
    observation["updates"] = []
    observation["step"] = 0
    observation["remainingOverageTime"] = 60.
    player_id = 0
    # Main game loop - continuously process inputs until the game ends
    while True:
        # Read and accumulate game state updates
        inputs = read_input()
        observation["updates"].append(inputs)
        
        # Special handling for first step to identify player ID
        if step == 0:
            player_id = int(observation["updates"][0])
            observation.player = player_id
            """
            # fixes bug where updates array is shared, but the first update is agent dependent actually
            observation["updates"][0] = f"{observation.player}"
            """
            
        # When D_DONE is received, get actions from RL agent and respond
        if inputs == "D_DONE":
            # Get actions from RL agent - the agent processes the observation through
            # its neural network to generate actions for all units simultaneously
            actions = agent(observation, None)
            
            # Reset for next step
            observation["updates"] = []
            step += 1
            observation["step"] = step
            
            # Return actions to environment
            print(",".join(actions))
            print("D_FINISH")
