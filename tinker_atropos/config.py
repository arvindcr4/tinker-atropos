import random
import string
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def generate_run_suffix() -> str:
    """Generate a random 4-character suffix for unique wandb run names."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=4))


class TinkerAtroposConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TINKER_ATROPOS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    learning_rate: float = 4e-5

    # Training hyperparameters
    num_steps: int = 100
    batch_size: int = 128  # Number of datums per training batch
    group_size: int = 16  # Number of rollouts per question
    max_token_env_length: int = 256  # Max tokens for environment rollouts
    max_token_trainer_length: int = 2048  # Max tokens for trainer processing
    max_num_workers: int = 24  # Max parallel workers for rollout generation
    max_batches_offpolicy: int = 3  # Max batches before data is considered too stale

    # Wandb configuration
    use_wandb: bool = True
    wandb_project: str = "atropos-tinker"
    wandb_group: Optional[str] = None
    wandb_run_name: str = "atropos-tinker-run"
    wandb_run_suffix: str = Field(default_factory=generate_run_suffix)

    # API endpoints
    atropos_api_url: str = "http://localhost:8000"  # Atropos rollout server
    inference_api_url: str = "http://localhost:8001"  # Unified trainer inference endpoint

    # Checkpointing
    checkpoint_dir: str = "./temp/"
    save_checkpoint_interval: int = 0

    # Evaluation
    steps_per_eval: int = 100
    num_requests_for_eval: int = 256

    # Debug options
    ensure_scores_not_the_same: bool = False

    def to_dict(self):
        return self.model_dump()
