# Experiment Notebooks — Presentation Transcript

## How to Present

Open each HTML file in a browser tab. Walk through them in this order:
1. `experiment_overview.html` — Start here (the big picture)
2. `gsm8k_qwen_8b.html` — Your main experiment
3. `gsm8k_qwen_30b_moe.html` — Scaling up with MoE
4. `gsm8k_llama_8b.html` — Cross-family comparison
5. `gsm8k_llama_3b.html` — Negative result (important!)
6. `math_qwen_8b.html` — Harder benchmark
7. `math_llama_8b.html` — Harder benchmark + instruct model

---

## 1. Overview Notebook (`experiment_overview.html`)

### Opening (30 seconds)
> "This is the overview of all our RL experiments. We ran 6 experiments across 2 model families — Qwen and Llama — and 2 benchmarks — GSM8K grade school math and MATH competition math."

### Results Table (1 minute)
> "Here's the summary table. The key numbers to notice:
> - Qwen3-8B went from 7% to 100% on GSM8K — that's a base model learning math from scratch through RL
> - Llama-3.2-3B stayed at 2% — this is our negative result, the model is too small
> - Llama-8B-Instruct started at 79% because it's already instruction-tuned, then reached 100%
> - On MATH competition problems, even the best model only reaches about 50-85% — much harder"

### GSM8K Comparison Chart (1 minute)
> "This chart shows all 4 GSM8K experiments overlaid. You can clearly see three tiers:
> 1. The flat line at the bottom — that's Llama-3B, it couldn't learn at all
> 2. The steep curve in the middle — that's Qwen-8B going from near-zero to 100%
> 3. The high line — Llama-8B-Instruct starting high and perfecting itself
> 
> The MoE model (30B total, 3B active) is interesting — it reaches 99% but with more variance in the curve, probably because of the sparse routing."

### MATH Comparison Chart (30 seconds)
> "On competition math, you see a completely different picture. Rewards are much lower, training is more volatile. The instruct model has a clear advantage here too."

### Key Insights Section (1 minute)
> "Five takeaways:
> 1. There's a model size threshold — 3B can't learn, 8B can. This is a scaling law insight.
> 2. Instruct models have a massive head start — 79% vs 7% at step 0.
> 3. MoE is cost-effective — only 3B active params but near-perfect accuracy.
> 4. GSM8K vs MATH shows diminishing returns — success on easy benchmarks doesn't mean success on hard ones.
> 5. LoRA with only 0.1% of parameters modified can achieve full task mastery."

---

## 2. GSM8K Qwen3-8B (`gsm8k_qwen_8b.html`)

### Config (30 seconds)
> "Here's the exact configuration. Group size 16 means we generate 16 completions per question, LoRA rank 32, learning rate 4e-5. All training happens on Tinker's cloud GPUs — we don't need local GPU access."

### Reward Trajectory (1 minute)
> "This is the star of the show. Watch the reward curve:
> - Steps 0-20: hovering around 5-15%, the model is mostly guessing wrong
> - Steps 20-25: sudden jump to 75% — the model has figured out the pattern
> - Steps 25-30: reaches 100% — full mastery
> 
> The smoothed curve on the right makes the trend clearer. This is classic RL emergence — there's a phase transition where the model suddenly 'gets it'."

### Loss Plot (30 seconds)
> "The loss is volatile because we're using importance sampling. When the model is already correct, loss is zero. When it makes mistakes, loss spikes as the gradient pushes hard. This is expected behavior for GRPO."

### Step-by-Step Data (15 seconds)
> "The raw data table is here for reference — you can see each step's exact reward and loss values."

---

## 3. GSM8K Qwen3-30B MoE (`gsm8k_qwen_30b_moe.html`)

### Key talking point (45 seconds)
> "This is a Mixture of Experts model — 30 billion total parameters but only 3 billion active at any time. It's like having 10 specialist sub-networks that the model routes to.
> 
> It reaches 99.2% — almost as good as the dense 8B model. But the training curve is more volatile, with rewards bouncing between 25% and 100% in the middle stages. This is likely because the sparse routing adds instability during RL.
> 
> The practical upside: at inference time, it only uses 3B active parameters, so it's faster and cheaper than a full 30B dense model."

---

## 4. GSM8K Llama-8B-Instruct (`gsm8k_llama_8b.html`)

### Key talking point (45 seconds)
> "This is the cross-family comparison — Llama instead of Qwen. But there's a twist: this is an instruct model, meaning Meta already fine-tuned it to follow instructions.
> 
> Look at step 0: 79% accuracy right out of the box! That's the power of instruction tuning. Our GRPO training then pushes it to 100% by step 35.
> 
> The takeaway: RL is complementary to instruction tuning. Even when a model is already good, GRPO can push it to perfection on specific tasks."

---

## 5. GSM8K Llama-3.2-3B — Negative Result (`gsm8k_llama_3b.html`)

### Key talking point (1 minute)
> "This is our most important negative result. Llama-3.2-3B is a 3 billion parameter base model, and it completely failed to learn GSM8K.
> 
> Look at the reward trajectory — it never exceeds 3%. Most batches have zero reward, meaning the model couldn't solve a single problem. Without any correct examples, GRPO has nothing to learn from.
> 
> Compare this with Qwen3-8B (base) which went from 7% to 100%. The jump from 3B to 8B is critical — there appears to be a minimum model capacity needed for RL to bootstrap math reasoning.
> 
> This is valuable because it tells us where NOT to invest compute. If you have a 3B base model, supervised fine-tuning or distillation might be better than RL."

---

## 6. MATH Qwen3-8B (`math_qwen_8b.html`)

### Key talking point (45 seconds)
> "MATH competition problems are much harder than GSM8K — they come from AMC, AIME, and other competitions. 
> 
> After 25 steps, Qwen3-8B only reaches about 14% accuracy. Compare that to 100% on GSM8K at the same point. The model can handle grade school arithmetic but struggles with competition-level proofs and abstract reasoning.
> 
> This experiment is still running — we'll see if more training helps, but the early signal suggests this needs either a larger model or a different approach (maybe curriculum learning, starting with easier subjects)."

---

## 7. MATH Llama-8B-Instruct (`math_llama_8b.html`)

### Key talking point (45 seconds)
> "Again the instruct model dramatically outperforms the base model — rewards reaching 50-87% on competition math vs only 14% for Qwen-8B base.
> 
> But notice the high variance — rewards swing between 25% and 93% from step to step. This is because competition math problems vary enormously in difficulty. An algebra problem might be easy while a geometry proof is nearly impossible for the model.
> 
> Still running — but already showing that instruction tuning is even MORE important on harder benchmarks."

---

## Closing Summary (1 minute)

> "To summarize what we've demonstrated:
> 1. GRPO works — we can train language models to solve math through pure RL, no supervised data needed
> 2. Model size matters — 3B fails, 8B succeeds, there's a threshold
> 3. Instruction tuning helps enormously — 79% vs 7% starting point
> 4. LoRA makes this practical — only 0.1% of parameters, cloud training via Tinker API
> 5. Harder benchmarks need more work — GSM8K is solved, MATH is still challenging
> 
> Next steps: code generation benchmarks, IFEval for instruction following, and possibly curriculum learning for MATH."
