from typing import List
import tinker


# Wraps Tinker SamplingClient for use with Atropos environments.
class TinkerInferenceWrapper:
    def __init__(self, service_client: tinker.ServiceClient, base_model: str):
        self.service_client = service_client
        self.base_model = base_model
        self.current_sampling_client = None
        self.tokenizer = None  # TODO: figure out tokenizer

    async def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        stop: List[str] | None = None,
        **kwargs,
    ) -> List[str]:
        # TODO: implement
        raise NotImplementedError()

    async def update_weights(self, model_path: str) -> None:
        # TODO: implement
        raise NotImplementedError()
