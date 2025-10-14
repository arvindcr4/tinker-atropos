import random
import time
from typing import Dict, List, Optional, Tuple, TypedDict, Union
import aiohttp

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

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

system_prompt += """You are allocated a maximum of 2048 tokens, please strive to use less.

You will then provide your answer like this: \\boxed{your answer here}
It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.
So please end your answer with \\boxed{your answer here}"""


class GSM8kRow(TypedDict):
    question: str
    answer: str


class GSM8kEnv(BaseEnv):
    name = "gsm8k"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=128,
            steps_per_eval=100,
            max_token_length=2048,
            max_num_workers=256,
            max_batches_offpolicy=256,
            wandb_name="gsm8k-tinker-test",
        )
        server_configs = [
            APIServerConfig(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                base_url="http://localhost:8001/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Try to calculate percent_correct, pass if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.train = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
        test_data = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)
        self.test = list()
        for item in test_data:
            self.test.append(
                {
                    "question": item["question"],
                    "gold_answer": item["answer"].split("#")[-1].strip().replace(",", ""),
                }
            )
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def _generate_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ) -> tuple[list, list, list, list]:
        url = "http://localhost:8001/v1/chat/completions"

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop if stop else [],
                    "n": n,
                },
            ) as response:
                result = await response.json()

        output_tokens_list = []
        output_logprobs_list = []
        finish_reasons_list = []

        for choice in result["choices"]:
            # Get full tokens
            full_tokens = choice["tokens"]
            logprobs = choice["logprobs"]

            prefix_len = len(prompt_tokens)
            output_tokens = full_tokens[prefix_len:]

            output_tokens_list.append(output_tokens)
            output_logprobs_list.append(logprobs)
            finish_reasons_list.append(choice["finish_reason"])

        return prompt_tokens, output_tokens_list, output_logprobs_list, finish_reasons_list

    async def rollout_and_score_eval(self, question: str, answer: str) -> dict:
        """Rollout and score evaluation with detailed sample data collection."""
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )

        response_content = completion.choices[0].message.content

        # Parse gold answer
        gold_parsed = parse(
            "\\boxed{" + answer + "}",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        # Parse model answer
        answer_parsed = parse(
            response_content.split("</think>")[-1],
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )

        score = 1 if verify(answer_parsed, gold_parsed) else 0

        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response_content},
            ],
            "question": question,
            "gold_answer": answer,
            "gold_parsed": str(gold_parsed) if gold_parsed else None,
            "model_parsed": str(answer_parsed) if answer_parsed else None,
            "score": int(score),
            "correct": bool(score),
            "finish_reason": completion.choices[0].finish_reason,
            "response_after_think": (
                response_content.split("</think>")[-1]
                if "</think>" in response_content
                else response_content
            ),
        }

        return {"score": score, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()

        eval_tasks = []
        for item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(item["question"], item["gold_answer"]))
        results = await tqdm_asyncio.gather(*eval_tasks)

        # Extract scores and samples
        scores = [result["score"] for result in results]
        samples = [result["sample"] for result in results]

        percent_correct = sum(scores) / len(scores)

        end_time = time.time()

        # Add to existing metrics for wandb
        self.eval_metrics.append(("eval/percent_correct", percent_correct))

        # Log evaluation results
        eval_metrics = {
            "eval/percent_correct": percent_correct,
        }

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": self.config.max_token_length,
            },
        )

    async def collect_trajectories(self, item: GSM8kRow) -> Tuple[ScoredDataGroup, list[Item]]:
        user_message = {"role": "user", "content": item["question"]}
        gold_answer = "\\boxed{" + item["answer"].split("#")[-1].strip().replace(",", "") + "}"

        messages = [{"role": "system", "content": system_prompt}, user_message]
        (
            prompt_tokens,
            output_tokens_list,
            output_logprobs_list,
            finish_reasons_list,
        ) = await self._generate_with_logprobs(
            messages=messages,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )

        to_score = list()
        to_backlog = list()
        for i, (output_tokens, logprobs, finish_reason) in enumerate(
            zip(output_tokens_list, output_logprobs_list, finish_reasons_list)
        ):
            full_tokens = prompt_tokens + output_tokens

            output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)

            completion_messages = (
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": output_text},
            )
            to_score.append(
                {
                    "messages": completion_messages,
                    "gold_answer": gold_answer,
                    "finish_reason": finish_reason,
                    "tokens": full_tokens,
                    "logprobs": logprobs,
                    "prompt_tokens": prompt_tokens,
                }
            )
        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["ref_logprobs"] = list()
        gold_parsed = parse(
            rollout_group_data[0]["gold_answer"],
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            random.shuffle(rollout_group_data)
            for item in rollout_group_data:
                # print(item[0][-1]["content"])
                answer_parsed = parse(
                    item["messages"][-1]["content"].split("</think>")[-1],
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = verify(answer_parsed, gold_parsed)

                tokens = item["tokens"]
                prompt_tokens = item["prompt_tokens"]

                # Create masks
                prefix_len = len(prompt_tokens)
                masks = [-100] * prefix_len + tokens[prefix_len:]

                # Handle finish_reason == "length" case
                if item["finish_reason"] == "length":
                    if tokens[-1] == self.tokenizer.eos_token_id:
                        # truncate the last token
                        tokens = tokens[:-1]
                        masks = masks[:-1]

                # remove obviously bad examples
                if len([1 for i in masks if i != -100]) < 10:
                    continue
                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["ref_logprobs"].append(item["logprobs"])
                scores["scores"].append(1.0 if reward else -1.0)
                if len(scores["tokens"]) >= self.config.group_size:
                    break
            for score in scores["scores"]:
                self.percent_correct_buffer.append(max(score, 0))
            return scores
        else:
            # If the gold solution is not parseable, we return None
            return None

    async def get_next_item(self) -> GSM8kRow:
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    GSM8kEnv.cli()
