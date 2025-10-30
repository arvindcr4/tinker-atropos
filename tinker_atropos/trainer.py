import asyncio
import os
import time
import numpy as np
import torch
import random
from typing import Dict, Any, List

import tinker
from tinker.types import AdamParams, ModelInput, SamplingParams
import wandb
import requests
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_exponential

from tinker_atropos.types import (
    GenerateRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
)


class TinkerAtroposTrainer:
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        lora_rank: int = 32,
        learning_rate: float = 4e-5,
        atropos_api_url: str = "http://localhost:8000",
        num_steps: int = 100,
    ):
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.atropos_api_url = atropos_api_url
        self.num_steps = num_steps

        self.service_client = None
        self.training_client = None
        self.current_sampling_client = None

        self.tokenizer = None

        self.trainer_id = None
        self.group_mean_rewards = []
        self.use_wandb = True
        self.wandb_project = "atropos-tinker"
        self.wandb_group = "tinker_logging_group"  # "".join(random.choices(string.ascii_letters + string.digits, k=8))

    async def setup(self):
        print("Setting up Tinker-Atropos Trainer...")

        print(f"Creating ServiceClient for {self.base_model}...")
        self.service_client = tinker.ServiceClient()

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        print(f"Loaded tokenizer for {self.base_model}")

        print("Creating training client...")
        self.training_client = await self.service_client.create_lora_training_client_async(
            base_model=self.base_model,
            rank=self.lora_rank,
        )
        print("Training client created")

        print("Saving initial weights...")
        initial_path = self.training_client.save_weights_for_sampler(name="step_0").result().path
        self.current_sampling_client = self.service_client.create_sampling_client(
            model_path=initial_path
        )
        print(f"Initial sampling client created: {initial_path}")

        print("Registering with Atropos API...")
        self.trainer_id = await self._register_trainer()
        print(f"Registered as trainer: {self.trainer_id}")

        if self.use_wandb:
            try:
                wandb.init(
                    project=self.wandb_project, group=self.wandb_group, name="wandb_test_name"
                )
                print(f"Wandb initialized: {wandb.run.name}")
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                self.use_wandb = False

    async def _register_trainer(self) -> str:
        url = f"{self.atropos_api_url}/register"

        payload = {
            "wandb_group": self.wandb_group,
            "wandb_project": self.wandb_project,
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
        return result.get("uuid")

    async def generate_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: List[str] = None,
    ) -> tuple[list, list, list, list]:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        model_input = ModelInput.from_ints(prompt_tokens)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop if stop else [],
        )

        result = await self.current_sampling_client.sample_async(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=n,
        )

        output_tokens_list = []
        output_logprobs_list = []
        finish_reasons_list = []

        for sequence in result.sequences:
            output_tokens_list.append(sequence.tokens)
            output_logprobs_list.append(sequence.logprobs if sequence.logprobs else [])
            finish_reasons_list.append("stop")  # TODO: get actual finish reason

        return prompt_tokens, output_tokens_list, output_logprobs_list, finish_reasons_list

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def get_batch(self):
        data = requests.get(f"{self.atropos_api_url}/batch", timeout=10).json()
        return data

    def pad_data_to_good_offset(
        self, data: Dict[str, Any]
    ) -> tuple[List[tinker.Datum], List[float]]:
        batch = data["batch"]

        datums = []
        group_mean_rewards = []
        all_reference_logprobs = []
        all_advantages = []
        skipped_count = 0

        for item in batch:
            scores = np.array(item["scores"])
            original_mean = np.mean(scores)
            advantages = scores - original_mean

            if len(scores) > 1 and np.all(advantages == 0.0):
                skipped_count += 1
                continue

            group_mean_rewards.append(original_mean)

            if item.get("overrides") is not None:
                for i in range(len(item["overrides"])):
                    if item["overrides"][i].get("set_advantage_to_zero", False):
                        advantages[i] = 0.0

            for i in range(len(item["tokens"])):
                tokens = item["tokens"][i]
                trajectory_logprobs = item["inference_logprobs"][i]
                advantage = advantages[i]

                all_advantages.append(advantage)

                # ManagedServer provides full aligned logprobs (already masked with 1.0 for prompt)
                # For next-token prediction: input is tokens[:-1], target is tokens[1:]
                input_tokens = tokens[:-1]
                target_tokens = tokens[1:]

                # Shift logprobs to align with targets (tokens[1:])
                # logprobs[i] corresponds to the probability of tokens[i]
                # We want logprobs for target_tokens = tokens[1:], so use trajectory_logprobs[1:]
                all_logprobs = trajectory_logprobs[:-1]  # Remove last logprob to match input length

                # Advantages: use same advantage for all generated tokens, 0.0 for prompt tokens
                # trajectory_logprobs has 1.0 for prompt tokens, actual values for generated tokens
                all_advantages_padded = [0.0 if lp == 1.0 else advantage for lp in all_logprobs]

                all_reference_logprobs.extend(all_logprobs)

                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(all_logprobs)
                    == len(all_advantages_padded)
                ), f"Length mismatch: input={len(input_tokens)}, target={len(target_tokens)}, logprobs={len(all_logprobs)}, advantages={len(all_advantages_padded)}"

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
                            torch.tensor(all_advantages_padded, dtype=torch.float32)
                        ),
                    },
                )
                datums.append(datum)

        if all_reference_logprobs:
            logprob_array = np.array(all_reference_logprobs)
            logprob_array_nonzero = logprob_array[logprob_array != 0.0]
            if len(logprob_array_nonzero) > 0:
                self.logprob_stats = {
                    "logprobs/mean": float(np.mean(logprob_array_nonzero)),
                    "logprobs/std": float(np.std(logprob_array_nonzero)),
                    "logprobs/min": float(np.min(logprob_array_nonzero)),
                    "logprobs/p50": float(np.percentile(logprob_array_nonzero, 50)),
                }
            else:
                self.logprob_stats = {}
        else:
            self.logprob_stats = {}

        if all_advantages:
            advantages_array = np.array(all_advantages)
            if np.std(advantages_array) > 1e-6:
                self.advantage_stats = {
                    "advantages/mean": float(np.mean(advantages_array)),
                    "advantages/std": float(np.std(advantages_array)),
                    "advantages/sum": float(np.sum(advantages_array)),
                }
            else:
                self.advantage_stats = {}
        else:
            self.advantage_stats = {}

        if skipped_count > 0:
            print(f"Skipped {skipped_count} groups with zero advantages")

        return datums, group_mean_rewards

    def get_data(self) -> List[tinker.Datum]:
        import time
        import json

        while True:
            data = self.get_batch()

            if data.get("batch") is not None:
                with open("temp.json", "w", encoding="utf-8") as f:
                    json.dump(data, f)

                datums, group_mean_rewards = self.pad_data_to_good_offset(data)
                self.group_mean_rewards = group_mean_rewards
                return datums
            else:
                time.sleep(1)

    async def train_step(self, step: int) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Step {step}/{self.num_steps}")
        print(f"{'='*60}")

        step_start = time.time()
        metrics = {"step": step}

        print("Fetching data from Atropos...")
        data = self.get_data()
        print(f"Got {len(data)} Datum objects")

        print("Running forward-backward pass...")
        fwd_bwd_result = await self.training_client.forward_backward_async(
            data, loss_fn="importance_sampling"
        )

        print("Running optimizer step...")
        adam_params = AdamParams(learning_rate=self.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
        optim_result = await self.training_client.optim_step_async(adam_params)

        fwd_bwd_result = await fwd_bwd_result.result_async()
        optim_result = await optim_result.result_async()

        print(f"Loss: {fwd_bwd_result.metrics['loss:sum']}")

        training_logprobs_all = []
        for datum, output in zip(data, fwd_bwd_result.loss_fn_outputs):
            training_logprobs = output["logprobs"].to_torch()
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            mask = advantages != 0.0
            training_lp_masked = training_logprobs[mask]
            training_logprobs_all.extend(training_lp_masked.cpu().numpy().tolist())

        if training_logprobs_all:
            training_lp_array = np.array(training_logprobs_all)
            self.training_logprob_stats = {
                "logprobs/mean_training": float(np.mean(training_lp_array)),
                "logprobs/std_training": float(np.std(training_lp_array)),
                "logprobs/min_training": float(np.min(training_lp_array)),
                "logprobs/p50_training": float(np.percentile(training_lp_array, 50)),
            }

            if hasattr(self, "logprob_stats") and "logprobs/mean" in self.logprob_stats:
                ref_mean = self.logprob_stats["logprobs/mean"]
                train_mean = float(np.mean(training_lp_array))
                self.training_logprob_stats["logprobs/diff"] = ref_mean - train_mean
        else:
            self.training_logprob_stats = {}

        print("Saving checkpoint and updating sampling client...")
        new_path = (
            self.training_client.save_weights_for_sampler(name=f"step_{step+1}").result().path
        )
        self.current_sampling_client = self.service_client.create_sampling_client(
            model_path=new_path
        )
        print(f"Sampling client updated: {new_path}")

        step_time = time.time() - step_start
        metrics["step_time"] = step_time
        metrics["learning_rate"] = self.learning_rate

        if self.group_mean_rewards:
            metrics["reward/mean"] = np.mean(self.group_mean_rewards)
            print(f"Reward/mean: {metrics['reward/mean']:.4f}")

        if self.use_wandb:
            wandb_metrics = {
                "train/loss": fwd_bwd_result.metrics["loss:sum"],
                "train/learning_rate": self.learning_rate,
                "reward/mean": metrics["reward/mean"],
            }

            if hasattr(self, "logprob_stats"):
                wandb_metrics.update(self.logprob_stats)
            if hasattr(self, "training_logprob_stats"):
                wandb_metrics.update(self.training_logprob_stats)
            if hasattr(self, "advantage_stats"):
                wandb_metrics.update(self.advantage_stats)

            wandb.log(wandb_metrics, step=step + 1)

        return metrics

    async def run(self):
        print("\n" + "=" * 60)
        print("Starting Tinker-Atropos Training")
        print("=" * 60 + "\n")

        await self.setup()

        for step in range(self.num_steps):
            try:
                metrics = await self.train_step(step)
                print(f"\nStep {step} complete - Loss: {metrics.get('loss', 'N/A')}")
            except Exception as e:
                print(f"Error in step {step}: {e}")
                import traceback

                traceback.print_exc()
                break

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60 + "\n")


trainer: TinkerAtroposTrainer | None = None

app = FastAPI(title="Tinker-Atropos Service")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "trainer_initialized": trainer is not None,
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """
    OpenAI-compatible completions endpoint.
    Called by SGLang server wrapper for regular completions (non-chat).
    """
    if trainer is None:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    try:
        # Handle single prompt (string) or batch (list of strings)
        if isinstance(request.prompt, str):
            prompts = [request.prompt]
        else:
            prompts = request.prompt

        all_choices = []
        choice_index = 0

        for prompt in prompts:
            # Tokenize prompt
            prompt_tokens = trainer.tokenizer.encode(prompt, add_special_tokens=False)
            model_input = ModelInput.from_ints(prompt_tokens)

            # Generate using Tinker sampling client
            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop if request.stop else [],
            )

            result = await trainer.current_sampling_client.sample_async(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=request.n,
            )

            # Format choices
            for sequence in result.sequences:
                output_text = trainer.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
                all_choices.append(
                    {
                        "text": output_text,
                        "index": choice_index,
                        "finish_reason": "stop",
                    }
                )
                choice_index += 1

        return CompletionResponse(
            id=f"cmpl-{random.randint(0, 999999)}",
            choices=all_choices,
            created=int(time.time()),
            model=trainer.base_model,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Completion failed: {str(e)}")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Called by SGLang server wrapper for regular chat completions.
    """
    if trainer is None:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    try:
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # Apply chat template and tokenize
        prompt_text = trainer.tokenizer.apply_chat_template(
            messages_dict, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = trainer.tokenizer.encode(prompt_text, add_special_tokens=False)
        model_input = ModelInput.from_ints(prompt_tokens)

        # Generate using Tinker sampling client
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop if request.stop else [],
        )

        result = await trainer.current_sampling_client.sample_async(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=request.n,
        )

        # Format as OpenAI response
        choices = []
        for i, sequence in enumerate(result.sequences):
            output_text = trainer.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            choices.append(
                {
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                    },
                    "index": i,
                    "finish_reason": "stop",
                }
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{random.randint(0, 999999)}",
            choices=choices,
            created=int(time.time()),
            model=trainer.base_model,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    SGLang-compatible /generate endpoint.
    Called by ManagedServer with tokenized input_ids.
    """
    if trainer is None:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    try:
        # Extract input_ids (ManagedServer sends tokenized input)
        if request.input_ids is None:
            raise HTTPException(status_code=400, detail="input_ids is required")

        prompt_tokens = request.input_ids

        # Extract sampling params
        sampling_params = request.sampling_params or {}
        n = sampling_params.get("n", 1)
        max_tokens = sampling_params.get("max_new_tokens", 256)
        temperature = sampling_params.get("temperature", 0.7)
        stop = sampling_params.get("stop", [])

        # Generate using Tinker sampling client
        model_input = ModelInput.from_ints(prompt_tokens)
        tinker_sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop if isinstance(stop, list) else [stop],
        )

        result = await trainer.current_sampling_client.sample_async(
            prompt=model_input,
            sampling_params=tinker_sampling_params,
            num_samples=n,
        )

        # Process results - format for SGLang compatibility
        # SGLang wrapper expects: if not isinstance(results, list): results = [results]
        # So for n=1, return single dict. For n>1, return list of dicts.

        if n == 1:
            sequence = result.sequences[0]
            output_tokens = sequence.tokens
            output_logprobs = sequence.logprobs if sequence.logprobs else []
            output_text = trainer.tokenizer.decode(output_tokens, skip_special_tokens=True)

            # Format logprobs as SGLang expects: [(logprob, token_id, text), ...]
            output_token_logprobs = []
            for token_id, logprob in zip(output_tokens, output_logprobs):
                token_text = trainer.tokenizer.decode([token_id])
                output_token_logprobs.append((logprob, token_id, token_text))

            return {
                "text": output_text,
                "meta_info": {
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": len(output_tokens),
                    "finish_reason": "stop",
                    "output_token_logprobs": output_token_logprobs,
                },
            }
        else:
            # Multiple completions - return list of dicts
            results = []
            for sequence in result.sequences:
                output_tokens = sequence.tokens
                output_logprobs = sequence.logprobs if sequence.logprobs else []
                output_text = trainer.tokenizer.decode(output_tokens, skip_special_tokens=True)

                # Format logprobs as SGLang expects
                output_token_logprobs = []
                for token_id, logprob in zip(output_tokens, output_logprobs):
                    token_text = trainer.tokenizer.decode([token_id])
                    output_token_logprobs.append((logprob, token_id, token_text))

                results.append(
                    {
                        "text": output_text,
                        "meta_info": {
                            "prompt_tokens": len(prompt_tokens),
                            "completion_tokens": len(output_tokens),
                            "finish_reason": "stop",
                            "output_token_logprobs": output_token_logprobs,
                        },
                    }
                )

            return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


def run_fastapi_server():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


async def main():
    global trainer

    trainer = TinkerAtroposTrainer(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        lora_rank=int(os.getenv("LORA_RANK", "32")),
        learning_rate=float(os.getenv("LEARNING_RATE", "4e-5")),
        num_steps=50,
    )

    import threading

    server_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    server_thread.start()

    print("Waiting for FastAPI server to start...")
    await asyncio.sleep(3)

    await trainer.run()


if __name__ == "__main__":
    asyncio.run(main())
