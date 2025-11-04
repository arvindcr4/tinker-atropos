import random
import string
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def generate_run_suffix() -> str:
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

    num_steps: int = 100
    batch_size: int = 128
    group_size: int = 16
    max_token_env_length: int = 256
    max_token_trainer_length: int = 2048
    max_num_workers: int = 24
    max_batches_offpolicy: int = 3

    use_wandb: bool = True
    wandb_project: str = "atropos-tinker"
    wandb_group: Optional[str] = None
    wandb_run_name: str = "atropos-tinker-run"
    wandb_run_suffix: str = Field(default_factory=generate_run_suffix)

    atropos_api_url: str = "http://localhost:8000"
    inference_api_url: str = "http://localhost:8001"

    checkpoint_dir: str = "./temp/"
    save_checkpoint_interval: int = 0

    steps_per_eval: int = 100
    num_requests_for_eval: int = 256

    ensure_scores_not_the_same: bool = False

    def to_dict(self):
        return self.model_dump()
