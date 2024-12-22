"""
Main training script implementing distributed reinforcement learning using the IMPALA algorithm.
This script coordinates multiple actor processes and a learner process to enable efficient
parallel training of the Lux AI agent.

The training setup uses:
1. Multiple actor processes running game environments in parallel
2. A learner process updating the neural network using gathered experiences
3. IMPALA's V-trace algorithm for off-policy correction
4. UPGO and TD-lambda losses for stable learning
5. Optional teacher model distillation for knowledge transfer

The script manages configuration through Hydra, handles checkpointing, and integrates with
Weights & Biases for experiment tracking. The distributed architecture enables efficient
training on personal hardware through careful process and memory management.
"""

from contextlib import redirect_stdout
import io
# Silence "Loading environment football failed: No module named 'gfootball'" message
with redirect_stdout(io.StringIO()):
    import kaggle_environments

import hydra
import logging
import os
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from torch import multiprocessing as mp
import wandb

from lux_ai.utils import flags_to_namespace
from lux_ai.torchbeast.monobeast import train


os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def get_default_flags(flags: DictConfig) -> DictConfig:
    """
    Sets up default training configuration parameters, ensuring all necessary settings exist.
    
    The configuration covers several key areas:
    1. Environment parameters (seeds, buffers, observation spaces)
    2. Training parameters (precision, discounting, gradient clipping)
    3. Model parameters (architecture choices, embedding options)
    4. Experiment management (checkpointing, logging)
    
    Args:
        flags (DictConfig): Initial configuration from Hydra
        
    Returns:
        DictConfig: Complete configuration with all defaults set
        
    Key Parameters:
        num_buffers: Sized to handle both actor count and batch size requirements
        use_mixed_precision: Enables faster training through FP16
        discounting: Controls future reward weighting (0.999 for long-term planning)
        clip_grads: Prevents explosive gradients during training
    """
    flags = OmegaConf.to_container(flags)
    # Environment parameters
    flags.setdefault("seed", None)  # Random seed for reproducibility
    flags.setdefault("num_buffers", max(2 * flags["num_actors"], flags["batch_size"] // flags["n_actor_envs"]))  # Ensure enough buffers for both actors and batches
    flags.setdefault("obs_space_kwargs", {})  # Observation space customization
    flags.setdefault("reward_space_kwargs", {})  # Reward space customization

    # Training parameters - crucial for learning stability
    flags.setdefault("use_mixed_precision", True)  # Enable FP16 for faster training
    flags.setdefault("discounting", 0.999)  # High discount factor for long-term planning
    flags.setdefault("reduction", "mean")  # Loss reduction method
    flags.setdefault("clip_grads", 10.)  # Prevent gradient explosions
    flags.setdefault("checkpoint_freq", 10.)  # Save progress regularly
    flags.setdefault("num_learner_threads", 1)  # Single learner thread for stability
    flags.setdefault("use_teacher", False)  # Optional knowledge distillation
    flags.setdefault("teacher_baseline_cost", flags.get("teacher_kl_cost", 0.) / 2.)  # Teacher model influence

    # Model architecture parameters
    flags.setdefault("use_index_select", True)  # More efficient embedding lookup
    if flags.get("use_index_select"):
        logging.info("index_select disables padding_index and is equivalent to using a learnable pad embedding.")  # Important embedding behavior note

    # Experiment continuity parameters
    flags.setdefault("load_dir", None)  # Directory containing previous run
    flags.setdefault("checkpoint_file", None)  # Specific checkpoint to load
    flags.setdefault("weights_only", False)  # Load only model weights vs full state
    flags.setdefault("n_value_warmup_batches", 0)  # Value network pre-training

    # Experiment tracking and debugging
    flags.setdefault("disable_wandb", False)  # Weights & Biases integration
    flags.setdefault("debug", False)  # Enable debug logging

    return OmegaConf.create(flags)


@hydra.main(config_path="conf", config_name="resume_config")
def main(flags: DictConfig):
    """
    Main entry point for training. Handles configuration management and initiates the
    distributed training process using the IMPALA algorithm.
    
    The function manages several key aspects:
    1. Configuration loading and merging (CLI, saved configs, defaults)
    2. Experiment tracking setup through Weights & Biases
    3. Multi-process training coordination
    4. Checkpoint management for experiment continuity
    
    The training process uses multiple parallel actors feeding experiences to a central
    learner, implementing IMPALA's V-trace algorithm for off-policy correction along
    with UPGO and TD-lambda losses for stable learning.
    
    Args:
        flags (DictConfig): Initial configuration from Hydra CLI
    """
    # Load CLI configuration first to allow overrides
    cli_conf = OmegaConf.from_cli()
    if Path("config.yaml").exists():
        new_flags = OmegaConf.load("config.yaml")
        flags = OmegaConf.merge(new_flags, cli_conf)

    if flags.get("load_dir", None) and not flags.get("weights_only", False):
        # this ignores the local config.yaml and replaces it completely with saved one
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_steps=N before and want to increase it
        logging.info("Loading existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load(Path(flags.load_dir) / "config.yaml")
        # Overwrite some parameters
        new_flags = OmegaConf.merge(new_flags, flags)
        flags = OmegaConf.merge(new_flags, cli_conf)

    flags = get_default_flags(flags)
    logging.info(OmegaConf.to_yaml(flags, resolve=True))
    OmegaConf.save(flags, "config.yaml")
    if not flags.disable_wandb:
        wandb.init(
            config=vars(flags),
            project=flags.project,
            entity=flags.entity,
            group=flags.group,
            name=flags.name,
        )

    flags = flags_to_namespace(OmegaConf.to_container(flags))
    mp.set_sharing_strategy(flags.sharing_strategy)
    train(flags)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
