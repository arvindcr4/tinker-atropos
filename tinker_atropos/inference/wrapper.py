from typing import List, Dict, Tuple
from tinker.types import ModelInput, SamplingParams
from transformers import AutoTokenizer

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

        # Get tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        print(f"Loaded tokenizer for {base_model}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        stop: List[str] | None = None,
        num_samples: int = 1,
    ) -> Tuple[List[str], List[float]]:
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

        # Sample from Tinker with num_samples
        result = await self.current_sampling_client.sample_async(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=num_samples,
        )

        # Decode all sequences to strings
        completions = [self.tokenizer.decode(sequence.tokens) for sequence in result.sequences]
        logprobs = [sequence.logprobs for sequence in result.sequences]

        return completions, logprobs

    async def update_weights(self, model_path: str) -> None:
        new_client = self.service_client.create_sampling_client(model_path=model_path)

        self.current_sampling_client = new_client

        print(f"Updated inference weights to: {model_path}")

    def messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        # Try to use tokenizer's chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")

        return "\n".join(prompt_parts) + "\nassistant:"
