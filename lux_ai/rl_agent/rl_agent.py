import pickle

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
    return pos[0] * board_dims[1] + pos[1]


class RLAgent:
    """
    Reinforcement Learning agent for the Lux AI competition.
    
    This agent:
    1. Processes game observations through a custom gym environment
    2. Uses a trained neural network for action selection
    3. Handles data augmentation for improved robustness
    4. Implements collision detection and action validation
    
    Architecture:
    - Environment Wrappers:
        - RewardSpaceWrapper: Custom reward computation
        - PadFixedShapeEnv: Consistent tensor dimensions
        - VecEnv: Parallel environment processing
        - PytorchEnv: Tensor conversion
        - DictEnv: Standardized outputs
        
    - Neural Network:
        - Policy network for action selection
        - Value network for state evaluation
        - Supports multiple data augmentations
        
    - Game Logic:
        - Manages city tiles, workers, and carts
        - Handles resource collection and city building
        - Implements research and unit production
        
    Args:
        obs: Initial game observation
        conf: Game configuration parameters
    """
    def __init__(self, obs, conf):
        """
        Initialize the RL agent with game configuration and model setup.
        
        Process:
        1. Load model and agent configurations
        2. Set up GPU/CPU device placement
        3. Initialize environment wrappers
        4. Load pre-trained model weights
        5. Set up data augmentation pipeline
        6. Initialize game state tracking
        
        Args:
            obs: Initial game observation containing:
                - player: Current player ID
                - step: Current game step
                - updates: Game state updates
            conf: Game configuration with:
                - episode length
                - map dimensions
                - resource parameters
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
        Process game observation and select actions using the trained model.
        
        Pipeline:
        1. Preprocess observation and update game state
        2. Apply data augmentations for robust predictions
        3. Run model inference to get action probabilities
        4. Resolve collisions and validate actions
        5. Add debugging annotations if in local evaluation
        
        Args:
            obs: Current game observation
            conf: Game configuration
            raw_model_output: If True, return model predictions instead of actions
            
        Returns:
            List[str]: Valid game actions for current player
            or Dict: Raw model outputs if raw_model_output=True
            
        Performance:
            - Tracks timing for observation processing, inference, collision detection
            - Adapts data augmentation based on remaining computation time
        """

        def save_obs(obs, filename="obs.pkl"):
            """
            Save the entire observation dict (with all array values printed) to a pickle file.
            Also turn off print truncation for debugging prints.
            """
            # Turn off the default truncation in printing:
            np.set_printoptions(threshold=np.inf)

            # (Optional) If you just want to print the entire array in the console right now:
            # print(obs)

            # Now pickle and store the full data
            with open(filename, "wb") as f:
                pickle.dump(obs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved obs to {filename}")


        self.stopwatch.reset()

        self.stopwatch.start("Observation processing")
        self.preprocess(obs, conf)
        env_output = self.get_env_output()

        # save_obs(env_output, "env_output.pkl")

        relevant_env_output_augmented = {
            "obs": self.augment_data(env_output["obs"], is_policy=False),
            "info": {
                "input_mask": self.augment_data(env_output["info"]["input_mask"].unsqueeze(1),
                                                is_policy=False).squeeze(1),
                "available_actions_mask": self.augment_data(env_output["info"]["available_actions_mask"],
                                                            is_policy=True),
            },
        }

        # save_obs(relevant_env_output_augmented, "env_output_augmented.pkl")

        self.stopwatch.stop().start("Model inference")
        with torch.no_grad():
            agent_output_augmented = self.model.select_best_actions(relevant_env_output_augmented)

            # save_obs(agent_output_augmented, "agent_output_augmented.pkl")

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

            # save_obs(agent_output, "agent_output.pkl")

        # Used for debugging and visualization
        if raw_model_output:
            return agent_output

        self.stopwatch.stop().start("Collision detection")
        if self.agent_flags.use_collision_detection:
            actions = self.resolve_collision_detection(obs, agent_output)
            # save_obs(actions, "actions_collision_detection.pkl")
        else:
            items_ = {key: value.squeeze(0).numpy() for key, value in agent_output["actions"].items()}
            # save_obs(items_, "actions_going_into_process_actions.pkl")
            actions, _ = self.unwrapped_env.process_actions(items_)
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
        Update game state and prepare for action selection.
        
        Key Functions:
        1. Game State Management:
           - Updates game state with new observations
           - Synchronizes turn count and player IDs
           
        2. Unit Tracking:
           - Maps actionable units to their locations
           - Separates workers and carts
           - Tracks city tile positions and states
           
        3. Performance Optimization: 
           - Removes data augmentations if low on computation time
           - Maintains dictionaries for O(1) unit lookup
        
        Args:
            obs: Current game observation with updates
            conf: Game configuration parameters
            
        Important:
            Do not call manual_step on the first turn to avoid
            off-by-1 turn desynchronization throughout the game
        """
        # Do not call manual_step on the first turn, or you will be off-by-1 turn the entire game
        if obs["step"] > 0:
            self.unwrapped_env.manual_step(obs["updates"])
            # need to update turn with obs, otherwise things get messed up if
            # you give the agent obs out of strict order
            self.game_state.turn = obs["step"]
            # I use this in the visualisation code, so need it to be set correctly
            # self.game_state.id = obs["player"]

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
        return self.env.step(self.action_placeholder)

    def augment_data(
            self,
            data: Union[torch.Tensor, Dict[str, torch.Tensor]],
            is_policy: bool
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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
        Validate and modify actions to prevent illegal moves and collisions.
        
        This method implements a priority-based system to:
        1. Process city tile actions:
           - Respect unit production limits
           - Manage research points
           - Handle end-game restrictions
           
        2. Process unit actions:
           - Prevent unit collisions
           - Validate movement boundaries
           - Allow city tile stacking
           
        Strategy:
        - Uses log probabilities to prioritize actions
        - Modifies illegal actions by setting probability to -inf
        - Maintains game rules and unit constraints
        
        Args:
            obs: Current game observation
            agent_output: Model predictions containing:
                - policy_logits: Action probabilities
                - actions: Selected actions
                
        Returns:
            List[str]: Valid actions for the current player
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
        Generate visual annotations for resource transfers between units.
        
        Creates:
        1. Lines connecting units involved in transfers
        2. X markers indicating transfer destinations
        
        Args:
            actions: List of game actions to process
            
        Returns:
            List[str]: Annotation commands for visualization
            
        Note:
            Only processes transfer actions ('t' commands)
            Used primarily for local evaluation and debugging
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
        Access the base LuxEnv instance without wrappers.
        
        Returns:
            LuxEnv: Unwrapped environment for direct game state access
        """
        return self.env.unwrapped[0]

    @property
    def game_state(self) -> Game:
        """
        Access the current game state.
        
        Returns:
            Game: Current game state containing:
                - Map information
                - Player states
                - Unit positions
                - Resource locations
        """
        return self.unwrapped_env.game_state

    # Helper functions for debugging
    def set_to_turn_and_call(self, turn: int, *args, **kwargs):
        """
        Debug helper to simulate agent at specific game turn.
        
        Args:
            turn: Target game turn to simulate
            *args: Arguments to pass to __call__
            **kwargs: Keyword arguments to pass to __call__
            
        Returns:
            Same as __call__ method
            
        Note:
            Primarily used for debugging and testing
        """
        self.game_state.turn = max(turn - 1, 0)
        return self(*args, **kwargs)


def agent(obs, conf) -> List[str]:
    """
    Entry point for the Lux AI agent.
    
    Creates or reuses a global RLAgent instance and processes
    the current game observation.
    
    Args:
        obs: Game observation from environment
        conf: Game configuration parameters
        
    Returns:
        List[str]: Valid actions for current game state
    """
    global AGENT
    if AGENT is None:
        AGENT = RLAgent(obs, conf)
    return AGENT(obs, conf)
