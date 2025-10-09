"""Data processing for converting Atropos to Tinker format."""
from typing import List, Dict, Any, Tuple
import math
from tinker.types import Datum
import numpy as np
import torch


def trajectories_to_data(
    trajectory_groups: List[Dict[str, Any]]
) -> Tuple[List[Datum], Dict[str, Any]]:
    data = []
    metadata = {
        "num_groups": len(trajectory_groups),
        "num_trajectories": 0,
        "mean_reward": 0.0,
    }

    all_rewards = []

    for group in trajectory_groups:
        # Each group has: prompts, responses, scores
        prompts = group.get("prompts", [])
        responses = group.get("responses", [])
        scores = group.get("scores", [])

        if not prompts or not responses or not scores:
            continue

        # Compute advantages (center rewards within group)
        advantages = compute_advantages(scores)
        all_rewards.extend(scores)

        # Convert each trajectory to Datum
        for prompt, response, advantage in zip(prompts, responses, advantages):
            datum = trajectory_to_datum(prompt, response, advantage)
            if datum is not None:
                data.append(datum)

        metadata["num_trajectories"] += len(prompts)

    if all_rewards:
        metadata["mean_reward"] = np.mean(all_rewards)

    return data, metadata


def compute_advantages(scores: List[float]) -> List[float]:
    scores_arr = np.array(scores)
    mean_score = scores_arr.mean()
    advantages = scores_arr - mean_score
    return advantages.tolist()


def pad_data_to_good_offset(
    data: Dict[str, Any], batch_size: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Pad data to good GPU-friendly offsets and prepare batches.

    This implements the same logic as grpo.py's pad_data_to_good_offset.

    Args:
        data: Dictionary with 'batch' key containing list of items
        batch_size: Size of each batch

    Returns:
        Tuple of (token_batches, label_batches, advantage_batches)
    """
    max_token_len = max([max([len(x) for x in item["tokens"]]) for item in data["batch"]])
    # Usually 64 is a good choice to ensure non-weird scaling behavior on GPUs
    # So we pad to the nearest multiple of 64
    good_multiple = 64
    if (max_token_len - 1) % (good_multiple) != 0:
        max_token_len = math.ceil((max_token_len - 1) / (good_multiple)) * good_multiple
        token_setup_len = max_token_len + 1  # Add 1 so we can make it causal at the proper length
    else:
        token_setup_len = max_token_len
        max_token_len = max_token_len - 1  # Since it's causal we need to remove the last bit...

    # Pad all tokens to max_token_len and add to lists
    input_ids = list()
    labels = list()
    advantages = list()
    lengths = list()

    for item in data["batch"]:
        scores = item["scores"]
        scores = np.array(scores)
        # Check if we have more than 1 score...
        if len(scores) > 1:
            scores = scores - scores.mean()
            scores = scores / max(scores.std(), 1e-8)
        item["scores"] = scores
        if item["overrides"] is not None:
            for i in range(len(item["overrides"])):
                if item["overrides"][i].get("set_advantage_to_zero", False):
                    item["scores"][i] = 0

        for i in range(len(item["tokens"])):
            lengths.append(
                math.ceil((len(item["tokens"][i]) - 1) / (good_multiple)) * good_multiple
            )
            label_item = np.concatenate(
                [
                    np.array(item["masks"][i]),
                    np.full(
                        max(0, token_setup_len - len(item["tokens"][i])),
                        -100,
                        dtype=np.int32,
                    ),
                ]
            )
            item["tokens"][i] = np.concatenate(
                [
                    np.array(item["tokens"][i]),
                    np.zeros(max(0, token_setup_len - len(item["tokens"][i])), dtype=np.int32),
                ]
            )
            input_ids.append(item["tokens"][i][:-1])
            labels.append(label_item[1:])
            advantages.append(item["scores"][i])

    # Combine all lists into tensors
    token_batches = []
    label_batches = []
    advantage_batches = []
    for i in range(len(input_ids) // batch_size):
        token_batches.append(
            torch.tensor(np.stack(input_ids[i * batch_size : (i + 1) * batch_size], axis=0))
        )
        label_batches.append(
            torch.tensor(np.stack(labels[i * batch_size : (i + 1) * batch_size], axis=0))
        )
        advantage_batches.append(
            torch.tensor(np.stack(advantages[i * batch_size : (i + 1) * batch_size], axis=0)).view(
                -1, 1
            )
        )
    return token_batches, label_batches, advantage_batches


def trajectory_to_datum(
    prompt: str,
    response: str,
    advantage: float,
) -> Datum | None:
    """Convert a single trajectory to a Tinker Datum.

    Note: This is a placeholder. Need to figure out:
    1. How to get tokens and logprobs from Atropos trajectories
    2. Proper format for loss_fn_inputs
    """
    # TODO: Implement proper conversion
    # For now, this is a placeholder that will fail

    # Need to:
    # 1. Get target_tokens from response
    # 2. Get logprobs from sampling (stored in trajectory)
    # 3. Build ModelInput from prompt + response tokens

    raise NotImplementedError(
        "trajectory_to_datum needs implementation. "
        "Need to figure out how Atropos stores tokens and logprobs in trajectories."
    )
