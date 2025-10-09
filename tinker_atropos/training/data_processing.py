"""Data processing for converting Atropos to Tinker format."""
from typing import List, Dict, Any, Tuple
import math
from tinker.types import Datum, ModelInput
import numpy as np
import torch


def compute_advantages(scores: List[float]) -> List[float]:
    scores_arr = np.array(scores)
    mean_score = scores_arr.mean()
    advantages = scores_arr - mean_score
    return advantages.tolist()


def pad_data_to_good_offset(
    data: Dict[str, Any], batch_size: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
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


def convert_batch_to_tinker_data(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
) -> List[Datum]:
    batch_size = tokens.shape[0]
    data = []

    for i in range(batch_size):
        # Get token sequence for this item (as list of ints)
        token_ids = tokens[i].tolist()

        # Create ModelInput from tokens
        model_input = ModelInput.from_ints(token_ids)

        # Get labels and advantage for this item
        label_ids = labels[i]  # Keep as tensor for now
        advantage_value = advantages[i].item()  # Get scalar value

        # Create loss_fn_inputs dictionary
        # The labels tensor tells us which tokens to compute loss on
        # Advantage is used for weighting the loss
        loss_fn_inputs = {
            "target_tokens": label_ids,  # Tinker will convert tensor automatically
            "advantage": torch.tensor([advantage_value]),  # Convert to tensor
        }

        # Create Datum
        datum = Datum(
            model_input=model_input,
            loss_fn_inputs=loss_fn_inputs,
        )
        data.append(datum)

    return data


def trajectory_to_datum(
    tokens: List[int],
    mask: List[int],
    advantage: float,
) -> Datum:
    # Create ModelInput from tokens
    model_input = ModelInput.from_ints(tokens)

    # Convert mask to labels format (-100 for ignored tokens, token_id for trained tokens)
    # In the mask: 1 means we compute loss on this token, 0 means we ignore it
    labels = []
    for i, (token, mask_val) in enumerate(zip(tokens, mask)):
        if mask_val == 1:
            labels.append(token)
        else:
            labels.append(-100)  # -100 is the ignore index

    # Create loss_fn_inputs
    loss_fn_inputs = {
        "target_tokens": torch.tensor(labels, dtype=torch.long),
        "advantage": torch.tensor([advantage]),  # Convert to tensor
    }

    # Create and return Datum
    return Datum(
        model_input=model_input,
        loss_fn_inputs=loss_fn_inputs,
    )
