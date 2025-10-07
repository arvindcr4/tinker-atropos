from typing import Protocol, List, Dict, Any
import tinker


class InferenceWrapper(Protocol):
    # Generate completions for a batch of prompts.
    async def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        stop: List[str] | None = None,
        **kwargs,
    ) -> List[str]:
        ...

    # Update to new model weights from training.
    async def update_weights(self, model_path: str) -> None:
        ...


# Interface for converting Atropos data to Tinker format.
class TrainingDataProcessor(Protocol):
    # Convert Atropos trajectories to Tinker Datum objects.
    def trajectories_to_data(
        self, trajectories: List[Any]
    ) -> tuple[List[tinker.types.Datum], Dict[str, Any]]:
        """
        Returns:
            (data, metadata): Training data and optional metadata
        """
        ...
