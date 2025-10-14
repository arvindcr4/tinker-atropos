import asyncio
import os
import time
import requests
from typing import Dict, Any, List
import tinker
from tinker.types import AdamParams

import math
import numpy as np
import torch

from tenacity import retry, stop_after_attempt, wait_exponential

WANDB_GROUP = ""
WANDB_PROJECT = "grpo-tinker-example-test-3"


class TinkerAtroposTrainer:
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        lora_rank: int = 32,
        learning_rate: float = 4e-5,
        atropos_api_url: str = "http://localhost:8000",
        inference_service_url: str = "http://localhost:8001",
        num_steps: int = 100,
    ):
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.atropos_api_url = atropos_api_url
        self.inference_service_url = inference_service_url
        self.num_steps = num_steps

        # Will be initialized in setup()
        self.service_client = None
        self.training_client = None
        self.trainer_id = None
        self.batches = []
        self.group_mean_rewards = []

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
        self.trainer_id = await self._register_trainer()
        print(f"Registered as trainer: {self.trainer_id}")

    async def _register_trainer(self) -> str:
        url = f"{self.atropos_api_url}/register"

        payload = {
            "wandb_group": WANDB_GROUP,
            "wandb_project": WANDB_PROJECT,
            "batch_size": 128,
            "max_token_len": 2048,
            "starting_step": 0,
            "checkpoint_dir": "./temp/",
            "save_checkpoint_interval": 0,
            "num_steps": self.num_steps,
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        trainer_id = result.get("uuid")
        print(f"Registered with Atropos API, trainer_id: {trainer_id}")

        return trainer_id

    async def _update_inference_weights(self, model_path: str, step: int):
        url = f"{self.inference_service_url}/internal/update_weights"
        response = requests.post(
            url,
            json={"model_path": model_path, "step": step},
        )
        response.raise_for_status()
        print(f"Updated inference service with weights from step {step}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def get_batch(self):
        data = requests.get("http://localhost:8000/batch", timeout=10).json()
        return data

    def pad_data_to_good_offset(
        self, data: Dict[str, Any]
    ) -> tuple[List[tinker.Datum], List[float]]:
        batch = data["batch"]

        # Find max token length and pad to good multiple
        max_token_len = max([max([len(x) for x in item["tokens"]]) for item in batch])
        good_multiple = 64

        if (max_token_len - 1) % good_multiple != 0:
            max_token_len = math.ceil((max_token_len - 1) / good_multiple) * good_multiple
            token_setup_len = max_token_len + 1
        else:
            token_setup_len = max_token_len
            max_token_len = max_token_len - 1

        datums = []
        group_mean_rewards = []
        skipped_count = 0

        for item in batch:
            scores = np.array(item["scores"])
            original_mean = scores.mean()

            advantages = scores - original_mean if len(scores) > 1 else scores.copy()

            # Skip if all advantages are zero
            if len(scores) > 1 and np.all(advantages == 0.0):
                skipped_count += 1
                continue

            group_mean_rewards.append(original_mean)

            if item.get("overrides") is not None:
                for i in range(len(item["overrides"])):
                    if item["overrides"][i].get("set_advantage_to_zero", False):
                        advantages[i] = 0.0

            # Process each trajectory in the group
            for i in range(len(item["tokens"])):
                tokens = item["tokens"][i]
                masks = item["masks"][i]
                trajectory_logprobs = item["ref_logprobs"][i]
                advantage = advantages[i]

                generation_start_idx = next(
                    (idx for idx, mask in enumerate(masks) if mask != -100), len(masks)
                )

                # Pad tokens to good multiple first (before shifting)
                padding_needed = max(0, token_setup_len - len(tokens))
                if padding_needed > 0:
                    padded_tokens = tokens + [0] * padding_needed
                else:
                    padded_tokens = tokens

                ob_len = generation_start_idx

                input_tokens = padded_tokens[:-1]
                target_tokens = padded_tokens[1:]

                all_logprobs = [0.0] * ob_len + trajectory_logprobs

                # Pad logprobs to match input_tokens length
                if len(all_logprobs) < len(input_tokens):
                    all_logprobs = all_logprobs + [0.0] * (len(input_tokens) - len(all_logprobs))

                all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)

                # Verify all arrays have the same length
                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(all_logprobs)
                    == len(all_advantages)
                ), (
                    f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, "
                    f"len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"
                )

                # Create Datum for Tinker training (matching rl_loop.py structure)
                datum = tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData.from_torch(
                            torch.tensor(target_tokens, dtype=torch.int64)
                        ),
                        "logprobs": tinker.TensorData.from_torch(
                            torch.tensor(all_logprobs, dtype=torch.float32)
                        ),
                        "advantages": tinker.TensorData.from_torch(
                            torch.tensor(all_advantages, dtype=torch.float32)
                        ),
                    },
                )
                datums.append(datum)

        if skipped_count > 0:
            print(f"Skipped {skipped_count} groups with zero advantages")

        return datums, group_mean_rewards

    def get_data(self) -> List[tinker.Datum]:
        import time
        import json

        while True:
            data = self.get_batch()

            if data.get("batch") is not None:
                # Save the batch for debugging
                with open("temp.json", "w", encoding="utf-8") as f:
                    json.dump(data, f)

                # Convert to Datums and return immediately
                datums, group_mean_rewards = self.pad_data_to_good_offset(data)
                self.group_mean_rewards = group_mean_rewards
                return datums
            else:
                # Wait for data
                time.sleep(1)

    async def train_step(self, step: int) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Step {step}/{self.num_steps}")
        print(f"{'='*60}")

        step_start = time.time()
        metrics = {"step": step}

        # Fetch data from Atropos if not already available
        if len(self.batches) == 0:
            print("Fetching data from Atropos...")
            self.batches = self.get_data()
            print(f"Got {len(self.batches)} Datum objects")

        data = self.batches
        self.batches = []  # Clear after using

        print(f"Processing {len(data)} trajectories")

        print("Running forward-backward pass...")
        fwd_bwd_future = await self.training_client.forward_backward_async(
            data, loss_fn="importance_sampling"
        )

        print("Running optimizer step...")
        adam_params = AdamParams(learning_rate=self.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
        optim_future = await self.training_client.optim_step_async(adam_params)

        fwd_bwd_result = await fwd_bwd_future.result_async()
        optim_result = await optim_future.result_async()

        print("Optimizer step complete")
        print(f"Forward-backward result: {fwd_bwd_result.metrics}")
        print(f"Optimizer result: {optim_result}")

        # Save checkpoint and update inference service
        print("Saving checkpoint...")
        new_path = (
            self.training_client.save_weights_for_sampler(name=f"step_{step+1}").result().path
        )
        await self._update_inference_weights(new_path, step=step + 1)
        print(f"Checkpoint saved: {new_path}")

        # Collect metrics
        step_time = time.time() - step_start
        metrics["step_time"] = step_time
        metrics["learning_rate"] = self.learning_rate

        # Log reward metrics (mean of group means)
        if hasattr(self, "group_mean_rewards") and self.group_mean_rewards:
            metrics["reward/mean"] = np.mean(self.group_mean_rewards)
            print(f"\nReward/mean (mean of group means): {metrics['reward/mean']:.4f}")
            print(f"   Based on {len(self.group_mean_rewards)} groups")

        return metrics

    async def run(self):
        print("\n" + "=" * 60)
        print("Starting Tinker-Atropos Training")
        print("=" * 60 + "\n")

        await self.setup()

        for step in range(self.num_steps):
            try:
                metrics = await self.train_step(step)
                print(f"\nStep {step} metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"Error in step {step}: {e}")
                import traceback

                traceback.print_exc()
                break

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60 + "\n")


async def main():
    trainer = TinkerAtroposTrainer(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        lora_rank=int(os.getenv("LORA_RANK", "32")),
        learning_rate=float(os.getenv("LEARNING_RATE", "4e-5")),
        num_steps=10,
    )

    await trainer.run()


if __name__ == "__main__":
    asyncio.run(main())
