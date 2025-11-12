"""
Simple metrics tracker that outputs TXT matching the Tinker baseline format.

Output format:
step,time_seconds,avg_reward,avg_loss,train_tokens,sampling_tokens,avg_logprob_diff
"""

import time
from pathlib import Path


class SimpleMetricsTracker:
    """
    Lightweight metrics tracker that outputs TXT in exact format for comparison.

    Columns:
    - step: Training step number
    - time_seconds: Wall-clock time for the step
    - avg_reward: Mean reward across batch
    - avg_loss: Training loss
    - train_tokens: Number of tokens used in training (forward-backward)
    - sampling_tokens: Number of tokens generated during rollouts
    - avg_logprob_diff: Mean difference between reference and training logprobs
    """

    def __init__(self, output_file: str = "tinker_atropos_metrics.txt"):
        self.output_file = Path(output_file)
        self.txt_file = None

        # Current step tracking
        self.step_start_time = None
        self.current_step_data = {}

        # Initialize TXT file with headers
        self._init_txt()

    def _init_txt(self):
        """Initialize TXT file with headers."""
        self.txt_file = open(self.output_file, "w")

        # Write header
        header = (
            "step,time_seconds,avg_reward,avg_loss,train_tokens,sampling_tokens,avg_logprob_diff"
        )
        self.txt_file.write(header + "\n")
        self.txt_file.flush()

        print(f"Initialized metrics TXT: {self.output_file}")

    def start_step(self, step: int):
        """Start tracking a new step."""
        self.step_start_time = time.time()
        self.current_step_data = {"step": step}

    def end_step(
        self,
        avg_reward: float,
        avg_loss: float,
        train_tokens: int,
        sampling_tokens: int,
        avg_logprob_diff: float,
    ):
        """
        Complete the step and write to TXT.

        Args:
            avg_reward: Mean reward across batch
            avg_loss: Training loss for this step
            train_tokens: Number of tokens in forward-backward pass
            sampling_tokens: Number of tokens generated during rollouts
            avg_logprob_diff: Mean difference between reference and training logprobs
        """
        if self.step_start_time is None:
            print("Warning: end_step called without start_step")
            return

        time_seconds = time.time() - self.step_start_time

        # Write row to TXT
        row = f"{self.current_step_data['step']},{time_seconds:.4f},{avg_reward:.4f},{avg_loss:.4f},{train_tokens},{sampling_tokens},{avg_logprob_diff:.4f}"

        self.txt_file.write(row + "\n")
        self.txt_file.flush()

        # Print summary
        step = self.current_step_data["step"]
        print(f"\nStep {step} Metrics:")
        print(f"  Time: {time_seconds:.2f}s")
        print(f"  Reward: {avg_reward:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Train tokens: {train_tokens:,}")
        print(f"  Sampling tokens: {sampling_tokens:,}")
        print(f"  Logprob diff: {avg_logprob_diff:.4f}")

    def close(self):
        """Close the TXT file."""
        if self.txt_file:
            self.txt_file.close()
            print(f"\nMetrics saved to {self.output_file}")


def count_train_tokens_from_data(data) -> int:
    """
    Count training tokens from Tinker Datum objects.

    This counts the FULL sequences (prompt + completion) used in forward-backward.

    Args:
        data: List of tinker.Datum objects

    Returns:
        Total number of tokens used in forward-backward pass (prompt + completion)
    """
    train_tokens = 0
    for datum in data:
        # Count from target_tokens length + 1 (for input_tokens which is tokens[:-1])
        # This gives us the full sequence length
        target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch()
        train_tokens += len(target_tokens) + 1
    return train_tokens


def count_sampling_tokens_from_batch(batch: dict) -> int:
    """
    Count sampling tokens from Atropos batch.

    This counts ONLY the generated completions (not including prompts).
    Prompt tokens have logprob = 1.0, so we count tokens where logprob != 1.0.

    Args:
        batch: Raw batch data from Atropos API (before conversion to Datums)

    Returns:
        Total number of tokens generated during rollouts (completions only)
    """
    sampling_tokens = 0

    if "batch" in batch and batch["batch"] is not None:
        for item in batch["batch"]:
            if "tokens" in item and "inference_logprobs" in item and isinstance(item, dict):
                # Count completion tokens only (where logprob != 1.0)
                for token_seq, logprob_seq in zip(item["tokens"], item["inference_logprobs"]):
                    # Count tokens where logprob is not 1.0 (1.0 = prompt token)
                    for logprob in logprob_seq:
                        if logprob != 1.0:
                            sampling_tokens += 1

    return sampling_tokens


def calculate_avg_logprob_diff(
    reference_logprobs: list,
    training_logprobs: list,
) -> float:
    """
    Calculate average difference between reference and training logprobs.

    Args:
        reference_logprobs: List of reference logprobs (from original policy)
        training_logprobs: List of training logprobs (from updated policy)

    Returns:
        Mean difference (reference - training)
    """
    if not reference_logprobs or not training_logprobs:
        return 0.0

    if len(reference_logprobs) != len(training_logprobs):
        print(
            f"Warning: Logprob length mismatch ({len(reference_logprobs)} vs {len(training_logprobs)})"
        )

    diffs = []
    for ref_lp, train_lp in zip(reference_logprobs, training_logprobs):
        diffs.append(ref_lp - train_lp)

    return sum(diffs) / len(diffs) if diffs else 0.0
