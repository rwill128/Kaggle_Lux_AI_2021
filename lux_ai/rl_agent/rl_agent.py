"""
Core reinforcement learning agent implementation for the Lux AI competition.
This module implements a sophisticated RL agent that uses a neural network to control
multiple units simultaneously in the Lux AI game environment.

Key Features:
1. Single neural network controlling all units (workers, carts, and city tiles)
2. Data augmentation for improved generalization
3. Sophisticated collision detection and action resolution
4. Integration with IMPALA distributed training framework
5. Support for teacher model knowledge distillation

The agent uses a 24-block ResNet with squeeze-excitation layers to process the game
state and output actions for all units simultaneously. The network is trained using
the IMPALA algorithm with V-trace for off-policy correction and UPGO/TD-lambda losses.
"""

import numpy as np
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from typing import *
import yaml

from . import data_augmentation
from ..lux_gym import create_reward_space, LuxEnv, wrappers
from ..lux_gym.act_spaces import ACTION_MEANINGS
from ..utils import DEBUG_MESSAGE, RUNTIME_DEBUG_MESSAGE, LOCAL_EVAL
from ..utility_constants import MAX_RESEARCH, DN_CYCLE_LEN, MAX_BOARD_SIZE
from ..nns import create_model, models
from ..utils import flags_to_namespace, Stopwatch

from ..lux.game import Game
from ..lux.game_constants import GAME_CONSTANTS
from ..lux.game_objects import CityTile, Unit
from ..lux import annotate

MODEL_CONFIG_PATH = Path(__file__).parent / "config.yaml"
RL_AGENT_CONFIG_PATH = Path(__file__).parent / "rl_agent_config.yaml"
CHECKPOINT_PATH, = list(Path(__file__).parent.glob('*.pt'))
AGENT = None

os.environ["OMP_NUM_THREADS"] = "1"


def pos_to_loc(pos: Tuple[int, int], board_dims: Tuple[int, int] = MAX_BOARD_SIZE) -> int:
    """
    Converts a 2D board position to a 1D location index.
    
    This function is used to map 2D game board coordinates to flat indices for
    efficient storage and lookup in dictionaries/arrays.
    
    Args:
        pos (Tuple[int, int]): (x, y) coordinates on the game board
        board_dims (Tuple[int, int], optional): Board dimensions. Defaults to MAX_BOARD_SIZE.
        
    Returns:
        int: Flattened 1D index corresponding to the input position
    """
    return pos[0] * board_dims[1] + pos[1]


class RLAgent:
    """
    Reinforcement learning agent for the Lux AI competition.
    
    This class implements a sophisticated RL agent that uses a neural network to
    control multiple units simultaneously. The agent processes game state observations
    through a 24-block ResNet with squeeze-excitation layers to generate actions
    for all units.
    
    Key Features:
    1. Data augmentation for improved generalization
    2. Collision detection to prevent illegal moves
    3. Support for both worker and cart units
    4. City tile management and research prioritization
    5. Optional teacher model knowledge distillation
    
    The agent is trained using the IMPALA algorithm with V-trace for off-policy
    correction and UPGO/TD-lambda losses for stable learning.
    """
    
    def __init__(self, obs, conf):
        """
        Initializes the RL agent with model and environment configurations.
        
        Args:
            obs: Initial game observation containing player state and updates
            conf: Game configuration parameters
            
        The initialization process:
        1. Loads model and agent configurations
        2. Sets up the environment with proper wrappers
        3. Loads the trained neural network model
        4. Initializes data augmentation pipeline
        5. Sets up game state tracking
        """
        with open(MODEL_CONFIG_PATH, 'r') as f:
            self.model_flags = flags_to_namespace(yaml.safe_load(f))
        with open(RL_AGENT_CONFIG_PATH, 'r') as f:
            self.agent_flags = SimpleNamespace(**yaml.safe_load(f))
        if torch.cuda.is_available():
            if self.agent_flags.device == "player_id":
                device_id = f"cuda:{min(obs.player, torch.cuda.device_count() - 1)}"
            else:
                device_id = self.agent_flags.device
        else:
            device_id = "cpu"
        self.device = torch.device(device_id)

        # Build the env used to convert observations for the model
        env = LuxEnv(
            act_space=self.model_flags.act_space(),
            obs_space=self.model_flags.obs_space(),
            configuration=conf,
            run_game_automatically=False
        )
        reward_space = create_reward_space(self.model_flags)
        env = wrappers.RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env)
        env = wrappers.PadFixedShapeEnv(env)
        env = wrappers.VecEnv([env])
        # We'll move the data onto the target device if necessary after preprocessing
        env = wrappers.PytorchEnv(env, torch.device("cpu"))
        env = wrappers.DictEnv(env)
        self.env = env
        self.env.reset(observation_updates=obs["updates"], force=True)
        self.action_placeholder = {
            key: torch.zeros(space.shape)
            for key, space in self.unwrapped_env.action_space.get_action_space().spaces.items()
        }

        # Load the model
        self.model = create_model(self.model_flags, self.device)
        checkpoint_states = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint_states["model_state_dict"])
        self.model.eval()

        # Load the data augmenters
        self.data_augmentations = []
        for da_factory in self.agent_flags.data_augmentations:
            da = data_augmentation.__dict__[da_factory](game_state=self.game_state)
            if not isinstance(da, data_augmentation.DataAugmenter):
                raise ValueError(f"Unrecognized data augmentation '{da}' created by: {da_factory}")
            self.data_augmentations.append(da)

        # Various utility properties
        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]
        self.my_city_tile_mat = np.zeros(MAX_BOARD_SIZE, dtype=bool)
        # NB: loc = pos[0] * n_cols + pos[1]
        self.loc_to_actionable_city_tiles = {}
        self.loc_to_actionable_workers = {}
        self.loc_to_actionable_carts = {}

        # Logging
        self.stopwatch = Stopwatch()

    def __call__(self, obs, conf, raw_model_output: bool = False):
        """
        Processes game observations and returns actions for all units.
        
        This is the main inference loop that:
        1. Processes new observations
        2. Applies data augmentation
        3. Runs neural network inference
        4. Resolves potential unit collisions
        5. Returns final actions
        
        Args:
            obs: Current game observation
            conf: Game configuration
            raw_model_output (bool): If True, returns raw network outputs for debugging
            
        Returns:
            list: List of actions for all units, or raw model outputs if raw_model_output=True
            
        The process uses the trained neural network to generate actions, then applies
        sophisticated collision detection to ensure all actions are legal and optimal.
        """
        self.stopwatch.reset()

        self.stopwatch.start("Observation processing")
        self.preprocess(obs, conf)
        env_output = self.get_env_output()
        relevant_env_output_augmented = {
            "obs": self.augment_data(env_output["obs"], is_policy=False),
            "info": {
                "input_mask": self.augment_data(env_output["info"]["input_mask"].unsqueeze(1),
                                                is_policy=False).squeeze(1),
                "available_actions_mask": self.augment_data(env_output["info"]["available_actions_mask"],
                                                            is_policy=True),
            },
        }

        self.stopwatch.stop().start("Model inference")
        with torch.no_grad():
            agent_output_augmented = self.model.select_best_actions(relevant_env_output_augmented)
            agent_output = {
                "policy_logits": self.aggregate_augmented_predictions(agent_output_augmented["policy_logits"]),
                "baseline": agent_output_augmented["baseline"].mean(dim=0, keepdim=True).cpu()
            }
            agent_output["actions"] = {
                key: models.DictActor.logits_to_actions(
                    torch.flatten(val, start_dim=0, end_dim=-2),
                    sample=False,
                    actions_per_square=None
                ).view(*val.shape[:-1], -1)
                for key, val in agent_output["policy_logits"].items()
            }
        # Used for debugging and visualization
        if raw_model_output:
            return agent_output

        self.stopwatch.stop().start("Collision detection")
        if self.agent_flags.use_collision_detection:
            actions = self.resolve_collision_detection(obs, agent_output)
        else:
            actions, _ = self.unwrapped_env.process_actions({
                key: value.squeeze(0).numpy() for key, value in agent_output["actions"].items()
            })
            actions = actions[obs.player]
        self.stopwatch.stop()

        if LOCAL_EVAL:
            # Add transfer annotations locally
            actions.extend(self.get_transfer_annotations(actions))

        value = agent_output["baseline"].squeeze().numpy()[obs.player]
        value_msg = f"Turn: {self.game_state.turn} - Predicted value: {value:.2f}"
        timing_msg = f"{str(self.stopwatch)}"
        overage_time_msg = f"Remaining overage time: {obs['remainingOverageTime']:.2f}"

        actions.append(annotate.sidetext(value_msg))
        DEBUG_MESSAGE(" - ".join([value_msg, timing_msg, overage_time_msg]))
        return actions

    def preprocess(self, obs, conf) -> NoReturn:
        """
        Prepares the game state for neural network processing.
        
        This method:
        1. Updates internal game state
        2. Identifies actionable units and city tiles
        3. Updates unit location mappings
        4. Manages data augmentation based on remaining computation time
        
        Args:
            obs: Current game observation
            conf: Game configuration
            
        The preprocessing ensures efficient lookup of units and city tiles during
        action selection and collision detection.
        """
        # Do not call manual_step on the first turn, or you will be off-by-1 turn the entire game
        if obs["step"] > 0:
            self.unwrapped_env.manual_step(obs["updates"])
            # need to update turn with obs, otherwise things get messed up if
            # you give the agent obs out of strict order
            self.game_state.turn = obs["step"]
            # I use this in the visualisation code, so need it to be set correctly
            self.game_state.id = obs["player"]

        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]

        self.my_city_tile_mat[:] = False
        self.loc_to_actionable_city_tiles: Dict[int, CityTile] = {}
        self.loc_to_actionable_workers: Dict[int, List[Unit]] = {}
        self.loc_to_actionable_carts: Dict[int, List[Unit]] = {}
        for unit in self.me.units:
            if unit.can_act():
                if unit.is_worker():
                    dictionary = self.loc_to_actionable_workers
                elif unit.is_cart():
                    dictionary = self.loc_to_actionable_carts
                else:
                    DEBUG_MESSAGE(f"Unrecognized unit type: {unit}")
                    continue
                dictionary.setdefault(pos_to_loc(unit.pos.astuple()), []).append(unit)
        for city_tile in self.me.city_tiles:
            self.my_city_tile_mat[city_tile.pos.x, city_tile.pos.y] = True
            if city_tile.can_act():
                self.loc_to_actionable_city_tiles[pos_to_loc(city_tile.pos.astuple())] = city_tile

        # Remove data augmentations if there are fewer overage seconds than 2x the number of data augmentations
        while max(obs["remainingOverageTime"], 0.) < len(self.data_augmentations) * 2:
            DEBUG_MESSAGE(f"Removing data augmentation: {self.data_augmentations[-1]}")
            del self.data_augmentations[-1]

    def get_env_output(self) -> Dict:
        """
        Gets the current environment state after preprocessing.
        
        Returns:
            Dict: Environment output containing observations and available action masks
            
        This method steps the environment with placeholder actions to get the
        current state representation without actually taking any actions.
        """
        return self.env.step(self.action_placeholder)

    def augment_data(
            self,
            data: Union[torch.Tensor, Dict[str, torch.Tensor]],
            is_policy: bool
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Applies data augmentation to improve model generalization.
        
        This method applies multiple transformations to the input data (like rotations
        and reflections) to help the model learn invariant features. The augmentations
        are applied differently for policy outputs vs state observations.
        
        Args:
            data: Input tensor or dictionary of tensors to augment
            is_policy: Whether the data represents policy outputs (True) or
                      observations (False)
            
        Returns:
            Augmented data with all transformations applied and concatenated
            
        The augmentations help the model learn that game states are equivalent
        under various transformations, improving generalization.
        """
        """
        Applies and concatenates all augmented observations into a single tensor/dict of tensors and moves the tensor
        to the correct device for inference.
        """
        if isinstance(data, dict):
            augmented_data = [data] + [augmentation.apply(data, inverse=False, is_policy=is_policy)
                                       for augmentation in self.data_augmentations]
            return {
                key: torch.cat([d[key] for d in augmented_data], dim=0).to(device=self.device)
                for key in data.keys()
            }
        else:
            augmented_data = [data] + [augmentation.op(data, inverse=False, is_policy=is_policy)
                                       for augmentation in self.data_augmentations]
            return torch.cat(augmented_data, dim=0).to(device=self.device)

    def aggregate_augmented_predictions(self, policy: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Combines predictions from multiple data augmentations.
        
        This method:
        1. Moves predictions to CPU
        2. Applies inverse transformations to align predictions
        3. Averages aligned predictions for each action
        
        Args:
            policy: Dictionary of policy logits from the neural network
            
        Returns:
            Dictionary of averaged policy logits after inverse transformations
            
        The aggregation ensures that predictions from different augmented views
        are properly aligned before being combined, improving prediction stability.
        """
        """
        Moves the predictions to the cpu, applies the inverse of all augmentations,
        and then returns the mean prediction for each available action.
        """
        policy = {key: val.cpu() for key, val in policy.items()}
        if len(self.data_augmentations) == 0:
            return policy

        policy_reoriented = [{key: val[0].unsqueeze(0) for key, val in policy.items()}]
        for i, augmentation in enumerate(self.data_augmentations):
            augmented_policy = {key: val[i + 1].unsqueeze(0) for key, val in policy.items()}
            policy_reoriented.append(augmentation.apply(augmented_policy, inverse=True, is_policy=True))
        return {
            key: torch.cat([d[key] for d in policy_reoriented], dim=0).mean(dim=0, keepdim=True)
            for key in policy.keys()
        }

    def resolve_collision_detection(self, obs, agent_output) -> List[str]:
        """
        Resolves potential collisions between unit actions using a priority system.
        
        This sophisticated collision detection system:
        1. Prioritizes actions based on their log probabilities
        2. Ensures city tiles don't exceed unit/research caps
        3. Prevents units from moving to the same square
        4. Handles edge cases like map boundaries
        
        Args:
            obs: Current game observation
            agent_output: Raw neural network outputs including policy logits
            
        Returns:
            List[str]: Final list of collision-free actions for all units
            
        The method uses a priority queue based on action probabilities to resolve
        conflicts, ensuring that high-priority actions are executed first while
        maintaining game rules and constraints.
        """
        # Get log_probs for all of my actions
        flat_log_probs = {
            key: torch.flatten(
                F.log_softmax(val.squeeze(0).squeeze(0), dim=-1),
                start_dim=-3,
                end_dim=-2
            )
            for key, val in agent_output["policy_logits"].items()
        }
        my_flat_log_probs = {
            key: val[obs.player] for key, val in flat_log_probs.items()
        }
        my_flat_actions = {
            key: torch.flatten(
                val.squeeze(0).squeeze(0)[obs.player],
                start_dim=-3,
                end_dim=-2
            )
            for key, val in agent_output["actions"].items()
        }
        # Use actions with highest prob/log_prob as highest priority
        city_tile_priorities = torch.argsort(my_flat_log_probs["city_tile"].max(dim=-1)[0], dim=-1, descending=True)

        # First handle city tile actions, ensuring the unit cap and research cap is not exceeded
        units_to_build = max(self.me.city_tile_count - len(self.me.units), 0)
        research_remaining = max(MAX_RESEARCH - self.me.research_points, 0)
        for loc in city_tile_priorities:
            loc = loc.item()
            actions = my_flat_actions["city_tile"][loc]
            if self.loc_to_actionable_city_tiles.get(loc, None) is not None:
                for i, act in enumerate(actions):
                    illegal_action = False
                    action_meaning = ACTION_MEANINGS["city_tile"][act]
                    # Check that it is allowed to build carts
                    if action_meaning == "BUILD_CART" and not self.agent_flags.can_build_carts:
                        illegal_action = True
                    # Check that the city will not build more units than the unit cap
                    elif action_meaning.startswith("BUILD_"):
                        if units_to_build > 0:
                            units_to_build -= 1
                        else:
                            illegal_action = True
                    # Check that the city will not research more than the research cap
                    elif action_meaning == "RESEARCH":
                        if research_remaining > 0:
                            research_remaining -= 1
                        else:
                            illegal_action = True
                    # Ban no-ops after the first night until research is complete
                    # This might prevent games like this from happening:
                    # https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-episode-26458475
                    elif (
                            action_meaning == "NO-OP" and
                            self.game_state.turn >= DN_CYCLE_LEN and
                            research_remaining > 0 and
                            self.agent_flags.must_research
                    ):
                        illegal_action = True
                    # Ban all non-unit-creating actions on the final step
                    if self.game_state.turn >= GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] - 1:
                        if action_meaning == "BUILD_CART":
                            illegal_action = False
                        else:
                            illegal_action = True
                    if illegal_action:
                        my_flat_log_probs["city_tile"][loc, act] = float("-inf")
                    else:
                        break

        # Then handle unit actions, ensuring that no units try to move to the same square
        occupied_squares = np.zeros(MAX_BOARD_SIZE, dtype=bool)
        max_loc_val = MAX_BOARD_SIZE[0] * MAX_BOARD_SIZE[1]
        combined_unit_log_probs = torch.cat(
            [my_flat_log_probs["worker"].max(dim=-1)[0], my_flat_log_probs["cart"].max(dim=-1)[0]],
            dim=-1
        )
        unit_priorities = torch.argsort(combined_unit_log_probs, dim=-1, descending=True)
        for loc in unit_priorities:
            loc = loc.item()
            if loc >= max_loc_val:
                unit_type = "cart"
                actionable_dict = self.loc_to_actionable_carts
            else:
                unit_type = "worker"
                actionable_dict = self.loc_to_actionable_workers
            loc = loc % max_loc_val
            actions = my_flat_actions[unit_type][loc]
            actionable_list = actionable_dict.get(loc, None)
            if actionable_list is not None:
                acted_count = 0
                for i, act in enumerate(actions):
                    illegal_action = False
                    action_meaning = ACTION_MEANINGS[unit_type][act]
                    if action_meaning.startswith("MOVE_"):
                        direction = action_meaning.split("_")[1]
                        new_pos = actionable_list[acted_count].pos.translate(direction, 1)
                    else:
                        new_pos = actionable_list[acted_count].pos

                    # Check that the new position is a legal square
                    if (
                            new_pos.x < 0 or new_pos.x >= self.game_state.map_width or
                            new_pos.y < 0 or new_pos.y >= self.game_state.map_height
                    ):
                        illegal_action = True
                    # Check that the new position does not conflict with another unit's new position
                    elif occupied_squares[new_pos.x, new_pos.y] and not self.my_city_tile_mat[new_pos.x, new_pos.y]:
                        illegal_action = True
                    else:
                        occupied_squares[new_pos.x, new_pos.y] = True

                    if illegal_action:
                        my_flat_log_probs[unit_type][loc, act] = float("-inf")
                    else:
                        acted_count += 1

                    if acted_count >= len(actionable_list):
                        break

        # Finally, get new actions from the modified log_probs
        actions_tensors = {
            key: val.view(1, *val.shape[:-2], *MAX_BOARD_SIZE, -1).argsort(dim=-1, descending=True)
            for key, val in flat_log_probs.items()
        }
        actions, _ = self.unwrapped_env.process_actions({
            key: value.numpy() for key, value in actions_tensors.items()
        })
        actions = actions[obs.player]
        return actions

    def get_transfer_annotations(self, actions: List[str]) -> List[str]:
        """
        Generates visualization annotations for resource transfers between units.
        
        This method creates visual indicators (lines and X marks) to show resource
        transfers between units in the game visualization, helping with debugging
        and understanding agent behavior.
        
        Args:
            actions: List of action strings to process for transfer annotations
            
        Returns:
            List of annotation commands for visualizing transfers
            
        The annotations help visualize resource movement between units, which is
        crucial for understanding the agent's resource management strategy.
        """
        annotations = []
        for act in actions:
            act_split = act.split(" ")
            if act_split[0] == "t":
                unit_from = self.me.get_unit_by_id(act_split[1])
                unit_to = self.me.get_unit_by_id(act_split[2])
                if unit_from is None or unit_to is None:
                    DEBUG_MESSAGE(f"Unrecognized transfer: {act}")
                    continue
                annotations.append(annotate.line(unit_from.pos.x, unit_from.pos.y, unit_to.pos.x, unit_to.pos.y))
                annotations.append(annotate.x(unit_to.pos.x, unit_to.pos.y))
        return annotations

    @property
    def unwrapped_env(self) -> LuxEnv:
        """
        Provides access to the underlying Lux environment without wrappers.
        
        Returns:
            LuxEnv: The base environment instance without any wrappers
            
        This property is used to access low-level environment functionality
        like action processing and game state management.
        """
        return self.env.unwrapped[0]

    @property
    def game_state(self) -> Game:
        """
        Provides access to the current game state.
        
        Returns:
            Game: Current game state instance containing all game information
            
        This property gives direct access to the game state for checking unit
        positions, resources, and other game-specific information.
        """
        return self.unwrapped_env.game_state

    # Helper functions for debugging
    def set_to_turn_and_call(self, turn: int, *args, **kwargs):
        """
        Helper method for debugging specific game turns.
        
        Args:
            turn: Game turn to set before calling the agent
            *args: Additional arguments to pass to __call__
            **kwargs: Additional keyword arguments to pass to __call__
            
        Returns:
            The result of calling the agent at the specified turn
            
        This method is primarily used for debugging and analyzing agent
        behavior at specific points in the game.
        """
        self.game_state.turn = max(turn - 1, 0)
        return self(*args, **kwargs)


def agent(obs, conf) -> List[str]:
    """
    Main entry point for the Lux AI agent, implementing the competition interface.
    
    This function:
    1. Creates the RLAgent instance if needed (first call)
    2. Processes the current observation
    3. Returns actions for all units
    
    Args:
        obs: Current game observation from the environment
        conf: Game configuration parameters
        
    Returns:
        List[str]: List of actions for all units in the required format
        
    The function maintains a single global agent instance across turns to preserve
    state and improve efficiency.
    """
    global AGENT
    if AGENT is None:
        AGENT = RLAgent(obs, conf)
    return AGENT(obs, conf)
