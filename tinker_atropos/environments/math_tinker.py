"""
MATH Competition environment for Tinker-Atropos.
Uses the Hendrycks MATH dataset (competition-level math problems).
Same scoring approach as GSM8K — parse \boxed{} answers and verify symbolically.
"""

import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from tinker_atropos.config import TinkerAtroposConfig


def _get_config_path():
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.environ.get("TINKER_CONFIG_PATH", "configs/default.yaml")


CONFIG_PATH = _get_config_path()

question_suffix = " Provide your answer inside \\boxed{}."

convo_prefix = [
    {
        "role": "user",
        "content": "What is the sum of all positive integers n such that n^2 divides 12!?" + question_suffix,
    },
    {
        "role": "assistant",
        "content": "We need n^2 | 12!. First, 12! = 2^10 * 3^5 * 5^2 * 7 * 11. For n^2 | 12!, each prime in n can appear at most floor(exponent/2) times. So n divides 2^5 * 3^2 * 5 = 1440. The sum of divisors of 1440 = (1+2+4+8+16+32)(1+3+9)(1+5) = 63 * 13 * 6 = 4914. \\boxed{4914}",
    },
]


class MATHRow(TypedDict):
    problem: str
    solution: str


class MATHEnv(BaseEnv):
    """
    Atropos environment for MATH competition problems.
    """

    name = "math_comp"

    def __init__(self, config, server_configs, slurm=True, testing=False):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        config = TinkerAtroposConfig.from_yaml(CONFIG_PATH) if CONFIG_PATH else TinkerAtroposConfig()

        env_config = BaseEnvConfig(
            tokenizer_name=config.base_model,
            group_size=config.group_size,
            use_wandb=config.use_wandb,
            rollout_server_url=config.atropos_api_url,
            total_steps=config.num_steps,
            batch_size=config.batch_size,
            steps_per_eval=config.steps_per_eval,
            max_token_length=config.max_token_env_length,
            max_num_workers=config.max_num_workers,
            max_batches_offpolicy=config.max_batches_offpolicy,
            wandb_name=f"{config.wandb_run_name}-env",
            ensure_scores_are_not_same=config.ensure_scores_are_not_same,
        )
        server_configs = [
            APIServerConfig(
                model_name=config.base_model,
                base_url=config.inference_api_url + "/v1",
                api_key="x",
                server_type="sglang",
                num_requests_for_eval=config.num_requests_for_eval,
            ),
        ]
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics=None):
        if wandb_metrics is None:
            wandb_metrics = {}
        try:
            wandb_metrics["train/percent_correct"] = sum(self.percent_correct_buffer) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            pass
        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% endif %}"
                "{% if loop.last and message['role'] != 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
                "{% endfor %}"
            )

        # Load MATH dataset (all subjects combined)
        from datasets import concatenate_datasets
        subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
        train_splits = []
        test_splits = []
        for subject in subjects:
            train_splits.append(load_dataset("EleutherAI/hendrycks_math", subject, split="train"))
            test_splits.append(load_dataset("EleutherAI/hendrycks_math", subject, split="test"))
        self.train = concatenate_datasets(train_splits).shuffle(seed=42)
        test_data = concatenate_datasets(test_splits).shuffle(seed=42)
        self.test = list()
        for item in test_data:
            self.test.append({"problem": item["problem"], "solution": item["solution"]})
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    def _extract_boxed_answer(self, solution: str) -> str:
        """Extract the final \\boxed{} answer from a MATH solution."""
        # Find the last \boxed{} in the solution
        idx = solution.rfind("\\boxed{")
        if idx == -1:
            return solution.strip()
        # Extract content inside \boxed{}
        depth = 0
        start = idx + len("\\boxed{")
        for i in range(start, len(solution)):
            if solution[i] == "{":
                depth += 1
            elif solution[i] == "}":
                if depth == 0:
                    return solution[start:i]
                depth -= 1
        return solution[start:].strip()

    async def rollout_and_score_eval(self, problem: str, solution: str) -> dict:
        gold_answer = self._extract_boxed_answer(solution)
        completion = await self.server.chat_completion(
            messages=[*convo_prefix, {"role": "user", "content": problem + question_suffix}],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        response_content = completion.choices[0].message.content
        gold_parsed = parse("\\boxed{" + gold_answer + "}", extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(
            response_content.split("</think>")[-1],
            extraction_config=[LatexExtractionConfig(normalization_config=NormalizationConfig(nits=False, malformed_operators=False, basic_latex=True, boxed="all", units=True), boxed_match_priority=0, try_extract_without_anchor=False)],
            extraction_mode="first_match",
        )
        score = 1 if verify(answer_parsed, gold_parsed) else 0
        return {"score": score, "sample": {"problem": problem, "gold_answer": gold_answer, "score": int(score), "correct": bool(score)}}

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()
        eval_tasks = [self.rollout_and_score_eval(item["problem"], item["solution"]) for item in self.test]
        results = await tqdm_asyncio.gather(*eval_tasks)
        scores = [r["score"] for r in results]
        samples = [r["sample"] for r in results]
        percent_correct = sum(scores) / len(scores)
        end_time = time.time()
        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        await self.evaluate_log(metrics={"eval/percent_correct": percent_correct}, samples=samples, start_time=start_time, end_time=end_time, generation_parameters={"temperature": 0.0, "max_tokens": self.config.max_token_length})

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list[Item]]:
        user_message = {"role": "user", "content": item["problem"] + question_suffix}
        gold_answer = "\\boxed{" + self._extract_boxed_answer(item["solution"]) + "}"
        messages = [*convo_prefix, user_message]

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completion = await managed.chat_completion(messages=messages, n=self.config.group_size, max_tokens=self.config.max_token_length, temperature=1.0, stop=[self.tokenizer.eos_token_id])
            state = managed.get_state()
            nodes = state["nodes"]

        to_score = list()
        for choice, node in zip(chat_completion.choices, nodes):
            completion_messages = (*convo_prefix, user_message, {"role": "assistant", "content": choice.message.content})
            to_score.append({"messages": completion_messages, "gold_answer": gold_answer, "finish_reason": choice.finish_reason, "tokens": node.tokens, "masked_tokens": node.masked_tokens, "logprobs": node.logprobs})
        to_postprocess = await self.score(to_score)
        return to_postprocess, []

    async def score(self, rollout_group_data) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["inference_logprobs"] = list()

        gold_parsed = parse(rollout_group_data[0]["gold_answer"], extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])

        if len(gold_parsed) != 0:
            random.shuffle(rollout_group_data)
            for item in rollout_group_data:
                answer_parsed = parse(
                    item["messages"][-1]["content"].split("</think>")[-1],
                    extraction_config=[LatexExtractionConfig(normalization_config=NormalizationConfig(nits=False, malformed_operators=False, basic_latex=True, boxed="all", units=True), boxed_match_priority=0, try_extract_without_anchor=False)],
                    extraction_mode="first_match",
                )
                reward = verify(answer_parsed, gold_parsed)
                tokens = item["tokens"]
                masked_tokens = item["masked_tokens"]
                logprobs = item["logprobs"]
                if len([1 for i in masked_tokens if i != -100]) < 10:
                    continue
                scores["tokens"].append(tokens)
                scores["masks"].append(masked_tokens)
                scores["inference_logprobs"].append(logprobs)
                scores["scores"].append(1.0 if reward else 0.0)
                if len(scores["tokens"]) >= self.config.group_size:
                    break
            for score in scores["scores"]:
                self.percent_correct_buffer.append(max(score, 0))
            return scores
        else:
            return None

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    MATHEnv.cli()
