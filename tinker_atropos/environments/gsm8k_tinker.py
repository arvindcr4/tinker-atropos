import re
from typing import Tuple, Optional
from atroposlib.envs.base import BaseEnv, BaseEnvConfig
from atroposlib.envs.server_handling.server_manager import ServerBaseline
from atroposlib.envs.server_handling.openai_server import APIServerConfig
from atroposlib.rollout_api_types import ScoredDataGroup
from datasets import load_dataset

import textwrap


class GSM8KTinkerEnv(BaseEnv):
    env_config_cls = BaseEnvConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None
        self.train_data = None
        self.test_data = None

    async def setup(self):
        print("Loading GSM8K dataset...")
        dataset = load_dataset("openai/gsm8k", "main", split="train")

        split = dataset.train_test_split(test_size=0.02, seed=42)
        self.train_data = list(split["train"])
        self.test_data = list(split["test"])

        print(
            f"Loaded {len(self.train_data)} training examples, {len(self.test_data)} test examples"
        )

    async def get_next_item(self):
        if self.train_data is None:
            await self.setup()

        idx = self.curr_step % len(self.train_data)
        item = self.train_data[idx]

        return {
            "question": item["question"],
            "answer": item["answer"],
            "idx": idx,
        }

    async def collect_trajectories(self, item):
        question = item["question"]
        ground_truth = self._extract_answer(item["answer"])

        prompt = self._build_prompt(question)

        sampling_params = {
            "max_tokens": self.config.max_token_length,
            "temperature": 0.7,
            "stop": ["\n\n"],
        }

        completions = await self.server_manager.generate(
            prompts=[prompt] * self.config.group_size,
            sampling_params=sampling_params,
        )

        scores = []
        for completion in completions:
            predicted = self._extract_boxed_answer(completion)
            correct = self._verify_answer(predicted, ground_truth)

            score = 1.0 if correct else 0.0
            scores.append(score)

        scored_group = ScoredDataGroup(
            prompts=[prompt] * self.config.group_size,
            responses=completions,
            scores=scores,
        )

        return scored_group, []

    def _build_prompt(self, question: str) -> str:
        return_string = textwrap.dedent(
            f"""You are a deep thinking AI. You should enclose your thoughts and reasoning inside <think> </think> tags, then provide your final answer.

                        You will provide your answer like this: \\boxed{{your answer here}}

                        Question: {question}

                        Answer:"""
        )

        return return_string

    def _extract_answer(self, answer_text: str) -> Optional[str]:
        match = re.search(r"#### ([\d,]+)", answer_text)
        if match:
            return match.group(1).replace(",", "")
        return None

    def _extract_boxed_answer(self, completion: str) -> Optional[str]:
        match = re.search(r"\\boxed\{([^}]+)\}", completion)
        if match:
            answer = match.group(1).strip()

            answer = re.sub(r"[^\d.]", "", answer)
            return answer
        return None

    def _verify_answer(self, predicted: Optional[str], ground_truth: Optional[str]) -> bool:
        if predicted is None or ground_truth is None:
            return False

        try:
            pred_num = float(predicted)
            gt_num = float(ground_truth)
            return abs(pred_num - gt_num) < 1e-6
        except (ValueError, TypeError):
            return False

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, ServerBaseline]:
        """Initialize configuration for Tinker inference service."""
        env_config = BaseEnvConfig(
            group_size=4,
            max_token_length=2048,
            steps_per_eval=50,
            total_steps=1000,
        )

        # Points to our Tinker FastAPI service
        server_config = APIServerConfig(
            base_url="http://localhost:8001",
            api_key="x",
            model_name="meta-llama/Llama-3.1-8B",
            timeout=120,
            num_max_requests_at_once=32,
        )

        return env_config, ServerBaseline(), [server_config]
