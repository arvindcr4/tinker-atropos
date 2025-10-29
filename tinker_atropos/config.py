import random
import string
from dataclasses import dataclass, field
from typing import Optional


def generate_run_suffix() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=4))


@dataclass
class TinkerAtroposConfig:
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
    wandb_group: str = "atropos-tinker-group"
    wandb_run_name: str = "atropos-tinker-run"
    wandb_run_suffix: str = field(default_factory=generate_run_suffix)

    atropos_api_url: str = "http://localhost:8000"
    inference_api_url: str = "http://localhost:8001"

    checkpoint_dir: str = "./temp/"
    save_checkpoint_interval: int = 0

    steps_per_eval: int = 100
    num_requests_for_eval: int = 256

    ensure_scores_not_the_same: bool = False

    def to_dict(self):
        return {
            "base_model": self.base_model,
            "lora_rank": self.lora_rank,
            "learning_rate": self.learning_rate,
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
            "group_size": self.group_size,
            "max_token_length": self.max_token_length,
            "max_num_workers": self.max_num_workers,
            "max_batches_offpolicy": self.max_batches_offpolicy,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
            "atropos_api_url": self.atropos_api_url,
            "inference_api_url": self.inference_api_url,
            "checkpoint_dir": self.checkpoint_dir,
            "save_checkpoint_interval": self.save_checkpoint_interval,
            "steps_per_eval": self.steps_per_eval,
            "num_requests_for_eval": self.num_requests_for_eval,
            "ensure_scores_not_the_same": self.ensure_scores_not_the_same,
        }


# Global config instance
_config: Optional[TinkerAtroposConfig] = None


def get_config() -> TinkerAtroposConfig:
    global _config
    if _config is None:
        _config = TinkerAtroposConfig()
    return _config


def set_config(config: TinkerAtroposConfig):
    global _config
    _config = config
