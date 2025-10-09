"""Main training loop for Tinker-Atropos integration."""
import asyncio
import json
import os
import time
import torch
import requests
import uuid
from typing import Dict, List, Tuple, Optional
import tinker
from tinker.types import AdamParams

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

WANDB_GROUP = ""
WANDB_PROJECT = ""


class TrainingConfig(BaseModel):
    lr: float = Field(1e-5, description="Learning rate for the optimizer")
    training_steps: int = Field(10, description="Number of training steps")
    batch_size: int = Field(2, description="Batch size for training (will be handled by get_data)")
    seq_len: int = Field(2048, description="Sequence length for training")
    gradient_accumulation_steps: int = Field(
        32, description="Number of gradient accumulation steps"
    )

    # Wandb configuration
    use_wandb: bool = Field(False, description="Whether to use Weights & Biases for logging")
    wandb_project: Optional[str] = Field(None, description="Wandb project name")
    wandb_group: Optional[str] = Field(None, description="Wandb group name")


class TinkerAtroposTrainer:
    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
        lora_rank: int = 32,
        learning_rate: float = 2e-5,
        atropos_api_url: str = "http://localhost:8000",
        inference_service_url: str = "http://localhost:8001",
        num_steps: int = 100,
        train_config: TrainingConfig = None,
    ):
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.atropos_api_url = atropos_api_url
        self.inference_service_url = inference_service_url
        self.num_steps = num_steps

        self.service_client = None
        self.training_client = None
        self.trainer_id = None
        self.train_config = train_config
        self.batches = []  # Store batches across training steps

    async def setup(self):
        print("Setting up Tinker trainer...")

        # Initialize Tinker clients
        print(f"Creating training client for {self.base_model}...")
        self.service_client = tinker.ServiceClient()
        self.training_client = await self.service_client.create_lora_training_client_async(
            base_model=self.base_model,
            rank=self.lora_rank,
        )
        print("Training client created")

        # Save initial weights and update inference service
        print("Saving initial weights...")
        initial_path = self.training_client.save_weights_for_sampler(name="step_0").result().path
        await self._update_inference_weights(initial_path, step=0)
        print(f"Initial weights saved: {initial_path}")

        # Register with Atropos API
        print("Registering with Atropos API...")
        self.trainer_id = await self._register_trainer(self.train_config)
        print(f"Registered as trainer: {self.trainer_id}")

    async def _register_trainer(self, config: TrainingConfig) -> str:
        requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": WANDB_GROUP,
                "wandb_project": WANDB_PROJECT,
                "batch_size": config.batch_size * config.gradient_accumulation_steps,
                "max_token_len": config.seq_len,
                "starting_step": 0,
                "num_steps": self.num_steps,
            },
            timeout=10,
        )

        # Dummy UUID for now
        return uuid.uuid4()

    async def _update_inference_weights(self, model_path: str, step: int):
        url = f"{self.inference_service_url}/internal/update_weights"
        response = requests.post(
            url,
            json={"model_path": model_path, "step": step},
        )
        response.raise_for_status()
        print(f"Updated inference service with weights from step {step}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def _get_batch(self) -> Dict:
        data = requests.get(f"{self.atropos_api_url}/batch", timeout=10).json()
        return data

    def _get_data(self) -> List[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
        from tinker_atropos.training.data_processing import pad_data_to_good_offset

        batches = []
        while True:
            data = self._get_batch()
            if data["batch"] is not None:
                # Save the batch for debugging
                with open("temp.json", "w", encoding="utf-8") as f:
                    json.dump(data, f)
                # In case the inference runs ahead of the training, we loop until we don't have any more data
                batches.append(pad_data_to_good_offset(data, self.train_config.batch_size))
            elif len(batches) > 0:
                # Return the batches
                return batches
            else:
                time.sleep(1)

    async def train_step(self, step: int):
        print(f"\n{'='*60}")
        print(f"Step {step}/{self.num_steps}")
        print(f"{'='*60}")

        step_start = time.time()
        metrics = {"step": step}

        # Get batches if we don't have any
        if len(self.batches) == 0:
            print("Fetching batches from Atropos...")
            self.batches = self._get_data()
            print(f"Got {len(self.batches)} batch groups")

        # Pop a batch group (contains token_batches, label_batches, advantage_batches)
        token_batches, label_batches, advantage_batches = self.batches.pop(0)
        print(f"Processing batch group with {len(token_batches)} mini-batches")

        total_loss = 0

        # Process each mini-batch in the group
        for i, (tokens, labels, advantages) in enumerate(
            zip(token_batches, label_batches, advantage_batches)
        ):
            print(f"  Mini-batch {i+1}/{len(token_batches)}: tokens shape {tokens.shape}")

            # TODO: Convert tokens/labels/advantages to Tinker format
            # For now, this is a placeholder - we need to figure out how to:
            # 1. Convert torch tensors to Tinker's Datum format
            # 2. Pass advantages to the loss function
            # This will need additional work to integrate with Tinker's API

            # Placeholder for actual training code
            # data = convert_to_tinker_format(tokens, labels, advantages)
            # fwd_bwd_result = await self.training_client.forward_backward_async(...)

            print("  TODO: Implement Tinker training for this mini-batch")

        print("Running optimizer step...")
        adam_params = AdamParams(learning_rate=self.learning_rate)
        await self.training_client.optim_step_async(adam_params)
        print("Optimizer step complete")

        print("Saving checkpoint...")
        new_path = (
            self.training_client.save_weights_for_sampler(name=f"step_{step+1}").result().path
        )
        await self._update_inference_weights(new_path, step=step + 1)
        print(f"Checkpoint saved: {new_path}")

        step_time = time.time() - step_start
        metrics["step_time"] = step_time
        metrics["learning_rate"] = self.learning_rate
        metrics["total_loss"] = total_loss

        print(f"\nStep {step} metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    async def run(self):
        print("\n" + "=" * 60)
        print("Starting Tinker-Atropos Training")
        print("=" * 60 + "\n")

        await self.setup()

        for step in range(self.num_steps):
            try:
                await self.train_step(step)
            except Exception as e:
                print(f"Error in step {step}: {e}")
                import traceback

                traceback.print_exc()
                break

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60 + "\n")


async def main():
    training_config = TrainingConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        training_steps=20,
        use_wandb=True,  # Set to True to enable logging
        wandb_project="grpo-tinker-trainer-test",  # Replace with your project name
    )

    # Get configuration from environment or use defaults
    trainer = TinkerAtroposTrainer(
        lora_rank=int(os.getenv("LORA_RANK", "32")),
        learning_rate=float(os.getenv("LEARNING_RATE", "2e-5")),
        num_steps=int(os.getenv("NUM_STEPS", "10")),
        train_config=training_config,
    )

    await trainer.run()


if __name__ == "__main__":
    asyncio.run(main())
