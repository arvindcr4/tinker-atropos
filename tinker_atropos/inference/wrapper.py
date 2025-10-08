from typing import List
from tinker.types import ModelInput, SamplingParams

import asyncio
import tinker


# Wraps Tinker SamplingClient for use with Atropos environments.
class TinkerInferenceWrapper:
    def __init__(
        self,
        service_client: tinker.ServiceClient,
        base_model: str,
        initial_sampling_client: tinker.SamplingClient | None = None,
    ):
        self.service_client = service_client
        self.base_model = base_model

        # Set up initial sampling client
        if initial_sampling_client is None:
            self.current_sampling_client = service_client.create_sampling_client(
                base_model=base_model
            )
        else:
            self.current_sampling_client = initial_sampling_client

        # Get tokenizer from training client (or sampling client if available)
        # TODO: Figure out the best way to get tokenizer
        # This will probably be pulling and loading the one from HF based on basemodel params
        self.tokenizer = None

    async def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        stop: List[str] | None = None,
        **kwargs,
    ) -> List[str]:
        # Convert prompts to ModelInput and generate
        tasks = [
            self._generate_one(self.current_sampling_client, prompt, max_tokens, temperature, stop)
            for prompt in prompts
        ]
        completions = await asyncio.gather(*tasks)
        return completions

    async def _generate_one(
        self,
        client: tinker.SamplingClient,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: List[str] | None,
    ) -> str:
        # TODO: Need tokenizer to convert prompt string to tokens
        # For now, assuming we can get tokens somehow
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Need to set up tokenizer.")

        # Encode prompt to tokens
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        model_input = ModelInput.from_ints(tokens)

        # Create sampling params
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop if stop else [],
        )

        # Sample from Tinker (doub le await pattern)
        result_future = await client.sample_async(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1,
        )
        result = await result_future.result_async()

        # Decode tokens back to string
        completion_tokens = result.sequences[0].tokens
        completion_text = self.tokenizer.decode(completion_tokens)

        return completion_text

    async def update_weights(self, model_path: str) -> None:
        new_client = self.service_client.create_sampling_client(model_path=model_path)

        self.current_sampling_client = new_client

        print(f"Updated inference weights to: {model_path}")
