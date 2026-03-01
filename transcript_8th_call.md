# 8th Guidance Call — ELI12 Transcript
## Tinker RL Project, Group 6 | 1 March 2026

---

## Slide 1: Title

"Hi everyone, we're Group 6, working on the Tinker RL Project. This is our 8th guidance call update. We're using reinforcement learning to make language models better at math — and today we have some really exciting results to share."

---

## Slide 2: Addressing Feedback

"Last time, you asked us two things: 'What's the current state-of-the-art?' and 'Can you try bigger models?' We've done both.

For SOTA — we did a full survey of recent papers: DeepSeek-R1, DeepScaleR, SimpleRL-Zoo, TinyZero, and DeepCoder. These are the best results in the field right now for training small models with RL.

For bigger models — we went from our original 1B model all the way up to 30 billion parameters. We ran 4 complete experiments on the GSM8K math benchmark, and we have 2 more running right now on a harder math benchmark called MATH. We also tested two different model families — Qwen and Llama — to see if our method works across different architectures."

---

## Slide 3: SOTA Comparison

"Here's where we stand compared to the best published results.

The top methods — like DeepSeek-R1 — use 'distillation' from a massive 671 billion parameter teacher model. That costs over $4,500 in compute just for training.

Our approach is different. We use GRPO — Group Relative Policy Optimization — directly on base models, with no teacher and no distillation. And we run everything on the Tinker cloud API, so we don't even need GPUs on our machines. It's much more accessible.

The key insight: SOTA gets better raw scores because they start from a distilled checkpoint. But our contribution isn't about beating SOTA — it's about showing that pure GRPO, applied systematically across model sizes and families, can achieve strong results without any of that expensive setup."

---

## Slide 4: GSM8K Scaling Results

"This is the main results slide. We ran GSM8K — grade school math problems — on 5 different models.

Let me walk through the highlights:
- Llama-3.2-1B from our earlier work: reached about 63% accuracy
- Llama-3.2-3B (base model, no instruction tuning): stayed around 1-2%. This tells us that a 3B base model doesn't have enough capacity to learn math through GRPO alone.
- Llama-3.1-8B-Instruct: started at 79% because it's already instruction-tuned, and quickly reached 100%.
- Qwen3-8B (base): started at just 7% and climbed all the way to 100% in 50 training steps. This is the most impressive result — pure GRPO turned a base model into a strong math solver.
- Qwen3-30B-A3B MoE: started at 17% and also reached 100%, but faster — it hit 50% by step 14, while the 8B took until step 25.

The key finding: bigger models learn faster under GRPO, and you need at least ~8B parameters for pure GRPO to work on GSM8K without instruction tuning."

---

## Slide 5: Reward Trajectories

"This slide shows how the training reward changed over 50 steps for three of our models.

Qwen3-8B shows a clear 'phase transition' — it stays flat around 4-7% for the first 11 steps, then gradually climbs, and suddenly jumps to 75% around step 25. After that, it stays consistently at 90-100%. This kind of sudden improvement is similar to the 'aha moments' that TinyZero reported in their paper.

Qwen3-30B MoE starts higher at 17% and converges faster, reaching 100% by step 28.

Llama-3.1-8B starts at 79% because it was already instruction-tuned, so GRPO is more like fine-tuning an already-capable model.

And Llama-3.2-3B (base) stays flat near 1% — this is an important negative result. It tells us there's a minimum model size needed for GRPO to work."

---

## Slide 6: Beyond GSM8K

"We're not just doing GSM8K. We have 4 benchmarks planned:

1. MATH Competition — this is running RIGHT NOW on both Qwen3-8B and Llama-8B. These are much harder problems: algebra, geometry, number theory at competition level.

2. LogP Steering — a different kind of RL where the model teaches itself by using a different system prompt. This tests if GRPO can work beyond math.

3. Code Generation — using HumanEval and MBPP. The reward is binary: does the code pass the test cases? This tests generalization to a completely different domain.

4. Instruction Following — IFEval benchmark. Can the model follow constraints like 'use exactly 100 words' or 'write in bullet points'?

Our full experiment matrix is 5 models times 4 benchmarks = 20 experiments. We have 6 done, 3 running, and 11 planned."

---

## Slide 7: Technical Architecture

"A quick overview of how this all works.

There are three components:
1. The Environment runs locally — it picks a math problem, sends it to the model, and checks if the answer is correct using symbolic math verification.
2. The Atropos API coordinates between the environment and the trainer — it batches data and manages the training loop.
3. The Tinker Trainer runs on cloud GPUs — it does the actual LoRA weight updates using GRPO's importance sampling loss.

Key advantages:
- GRPO needs no critic or reward model — saves memory
- LoRA trains only 0.1% of parameters — very efficient
- Tinker cloud means we don't need our own GPUs
- Math verification gives us perfect binary rewards — no noise

All experiments use the same hyperparameters: LoRA rank 32, learning rate 3-5e-5, batch size 128, group size 16, 50 steps. This consistency is what makes our comparison fair."

---

## Slide 8: Our Contribution

"So what are we actually contributing to the field?

We identified four gaps in the literature:
1. Most GRPO papers only test on Qwen models
2. Nobody has done a systematic scaling study across model families
3. Existing work needs expensive GPU clusters
4. Most focus on a single benchmark

Our contribution fills all four: a systematic GRPO scaling study across Llama and Qwen families, from 1B to 30B parameters, on multiple benchmarks, all running on cloud infrastructure that anyone can reproduce.

The five dimensions we compare: model size, model family, benchmarks, infrastructure, and reproducibility."

---

## Slide 9: Timeline

"Here's where we are and where we're going.

Today we have 6 complete experiments and 3 running. By March 5th, we want to finish MATH and start code generation experiments. By March 7th for the 9th guidance call, we should have the full multi-benchmark results.

Then March 14-21 for ablation studies and conference paper outline. The interim report is due March 28th, and the final report April 11th."

---

## Slide 10: Discussion

"We have four questions for you:

1. Evaluation — our training reward hits 100%, but we need proper held-out test evaluation. How should we structure this?

2. Conference angle — is 'systematic GRPO scaling across model families on cloud infrastructure' a strong enough angle? Any venue suggestions?

3. Bigger models — Tinker supports up to 235B parameters. Should we try a 70B+ model for the final report?

4. Depth vs breadth — should we do more benchmarks, or go deeper on math with ablations and longer training?

Thank you! Questions?"
