#!/usr/bin/env python3
"""Generate experiment notebooks and convert to HTML for sharing."""

import json
import os
import re
import subprocess
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")


def parse_log(log_path):
    """Extract step, reward, loss data from trainer log."""
    steps, rewards, losses = [], [], []
    step_num = 0
    current_loss = None
    current_reward = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Loss:") and "complete" not in line:
                try:
                    current_loss = float(line.split("Loss:")[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Reward/mean:"):
                try:
                    current_reward = float(line.split("Reward/mean:")[1].strip())
                except ValueError:
                    pass
            elif "complete - Loss:" in line:
                if current_reward is not None:
                    steps.append(step_num)
                    rewards.append(current_reward)
                    losses.append(current_loss if current_loss is not None else 0.0)
                    step_num += 1
                    current_loss = None
                    current_reward = None

    return steps, rewards, losses


def read_config(config_path):
    """Read YAML config as text."""
    with open(config_path, "r") as f:
        return f.read()


# ── Experiment definitions ──────────────────────────────────────────────

experiments = {
    "gsm8k_qwen_8b": {
        "title": "GSM8K - Qwen3-8B",
        "config": "gsm8k_qwen_8b.yaml",
        "log_dir": "gsm8k_qwen_8b",
        "model": "Qwen/Qwen3-8B",
        "params": "8.2B",
        "benchmark": "GSM8K (Grade School Math)",
        "status": "Complete",
        "tinker_uri": "tinker://380ee7fe-a0fc-5224-b755-49256a020831:train:0/sampler_weights/step_50",
        "description": """
This experiment trains Qwen3-8B on GSM8K grade school math problems using GRPO
(Group Relative Policy Optimization) with LoRA rank 32 fine-tuning via the Tinker cloud API.

**Key findings:**
- Started at ~7% accuracy (near random for a base model on math)
- Reached 100% reward by step 30, maintained through step 50
- Demonstrates that even a base model can learn structured math reasoning through RL
- The LoRA adapter only modifies ~0.1% of parameters but achieves full task mastery
""",
    },
    "gsm8k_qwen_30b_moe": {
        "title": "GSM8K - Qwen3-30B-A3B (MoE)",
        "config": "gsm8k_qwen_30b_moe.yaml",
        "log_dir": "gsm8k_qwen_30b_moe",
        "model": "Qwen/Qwen3-30B-A3B",
        "params": "30B total / 3B active (MoE)",
        "benchmark": "GSM8K (Grade School Math)",
        "status": "Complete",
        "tinker_uri": "tinker://f48983ea-.../sampler_weights/step_50",
        "description": """
This experiment trains Qwen3-30B-A3B, a Mixture-of-Experts model with 30B total parameters
but only 3B active per token, on GSM8K.

**Key findings:**
- Started higher (~17%) due to larger model capacity
- Reached 99.2% by step 49 — nearly perfect but slightly below the dense 8B model
- MoE architecture is cost-effective: only 3B active params during inference
- More volatile training curve than the dense 8B, likely due to sparse routing
""",
    },
    "gsm8k_llama_8b": {
        "title": "GSM8K - Llama 3.1-8B-Instruct",
        "config": "gsm8k_llama_8b.yaml",
        "log_dir": "gsm8k_llama8b",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "params": "8B",
        "benchmark": "GSM8K (Grade School Math)",
        "status": "Complete",
        "tinker_uri": "tinker://9f53109d-.../sampler_weights/step_50",
        "description": """
This experiment trains Llama 3.1-8B-Instruct on GSM8K — a cross-family comparison with Qwen3-8B.

**Key findings:**
- Started at ~79% accuracy — much higher baseline because this is an **instruct** model (already fine-tuned for following instructions)
- Reached 100% by step 35, sustained through step 50
- Instruct models have a massive head start over base models for structured tasks
- GRPO still adds value: pushed from 79% → 100% consistently
""",
    },
    "gsm8k_llama_3b": {
        "title": "GSM8K - Llama 3.2-3B (Base)",
        "config": "gsm8k_llama_3b.yaml",
        "log_dir": "gsm8k_llama3b",
        "model": "meta-llama/Llama-3.2-3B",
        "params": "3B",
        "benchmark": "GSM8K (Grade School Math)",
        "status": "Complete (Negative Result)",
        "tinker_uri": "tinker://03d1371d-.../sampler_weights/step_50",
        "description": """
This experiment trains Llama 3.2-3B (base, not instruct) on GSM8K — testing the lower end of the scaling ladder.

**Key findings (NEGATIVE RESULT):**
- Started near 0% and never exceeded ~3% accuracy across 50 steps
- This is an important scaling insight: 3B base parameters may be **below the threshold** for learning math reasoning via GRPO alone
- Most batches had zero reward (model couldn't solve any problems), causing zero-advantage skipping
- The model likely lacks sufficient pre-trained math knowledge to bootstrap RL learning
- Contrast with Qwen3-8B (base) which went 7% → 100% — the jump from 3B to 8B is critical
""",
    },
    "math_qwen_8b": {
        "title": "MATH Competition - Qwen3-8B",
        "config": "math_qwen_8b.yaml",
        "log_dir": "math_qwen8b",
        "model": "Qwen/Qwen3-8B",
        "params": "8.2B",
        "benchmark": "MATH (Competition-Level Mathematics)",
        "status": "In Progress",
        "tinker_uri": "N/A (running)",
        "description": """
This experiment trains Qwen3-8B on the MATH competition dataset (EleutherAI/hendrycks_math),
which includes problems from AMC, AIME, and other math competitions — significantly harder than GSM8K.

**Key observations so far:**
- Much harder than GSM8K — rewards are ~3-14% range after 25 steps
- The model struggles with competition-level problems requiring deeper reasoning
- Longer token lengths (1024 vs 512) needed for complex proofs and derivations
- This validates that GSM8K success doesn't automatically transfer to harder math
""",
    },
    "math_llama_8b": {
        "title": "MATH Competition - Llama 3.1-8B-Instruct",
        "config": "math_llama_8b.yaml",
        "log_dir": "math_llama8b",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "params": "8B",
        "benchmark": "MATH (Competition-Level Mathematics)",
        "status": "In Progress",
        "tinker_uri": "N/A (running)",
        "description": """
This experiment trains Llama 3.1-8B-Instruct on competition-level math.

**Key observations so far:**
- Much stronger than Qwen-8B on MATH — rewards reaching 50-87% range
- The instruct fine-tuning gives a massive advantage on complex reasoning
- Volatile training curve — competition math problems have high variance in difficulty
- Confirms the instruct vs base model gap is even larger on harder benchmarks
""",
    },
}


def create_experiment_notebook(exp_key, exp_info):
    """Create a Jupyter notebook for a single experiment."""
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    # Title
    nb.cells.append(new_markdown_cell(f"""# {exp_info['title']}

**Tinker RL Project — PES University MTech Capstone (Group 6)**

| Field | Value |
|-------|-------|
| **Model** | `{exp_info['model']}` |
| **Parameters** | {exp_info['params']} |
| **Benchmark** | {exp_info['benchmark']} |
| **Method** | GRPO (Group Relative Policy Optimization) + LoRA rank 32 |
| **Training API** | Tinker (Thinking Machines) — cloud GPU training |
| **Environment** | Atropos (NousResearch) — RL rollout framework |
| **Status** | {exp_info['status']} |
| **Tinker URI** | `{exp_info['tinker_uri']}` |
"""))

    # Description
    nb.cells.append(new_markdown_cell(f"""## Experiment Description
{exp_info['description']}
"""))

    # Config
    config_path = os.path.join(CONFIGS_DIR, exp_info["config"])
    config_text = read_config(config_path)
    nb.cells.append(new_markdown_cell("## Training Configuration"))
    nb.cells.append(new_code_cell(f"""# Training configuration ({exp_info['config']})
config_yaml = \"\"\"
{config_text}\"\"\"
print(config_yaml)"""))

    # Data extraction + plotting
    log_path = os.path.join(LOGS_DIR, exp_info["log_dir"], "trainer.log")
    steps, rewards, losses = parse_log(log_path)

    nb.cells.append(new_markdown_cell("## Training Results"))

    # Embed the data directly
    nb.cells.append(new_code_cell(f"""import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['figure.figsize'] = (12, 5)

steps = {steps}
rewards = {rewards}
losses = {losses}

print(f"Total steps completed: {{len(steps)}}")
print(f"Initial reward: {{rewards[0]:.4f}}" if rewards else "No data")
print(f"Final reward: {{rewards[-1]:.4f}}" if rewards else "No data")
print(f"Peak reward: {{max(rewards):.4f}}" if rewards else "No data")
print(f"Mean reward (last 10): {{sum(rewards[-10:])/len(rewards[-10:]):.4f}}" if len(rewards) >= 10 else "")
"""))

    # Reward plot
    nb.cells.append(new_code_cell(f"""# Reward trajectory
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Reward curve
ax1.plot(steps, rewards, color='#2ecc71', linewidth=2, alpha=0.8)
ax1.fill_between(steps, rewards, alpha=0.15, color='#2ecc71')
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('Mean Reward', fontsize=12)
ax1.set_title('{exp_info["title"]}\\nReward Trajectory', fontsize=14, fontweight='bold')
ax1.set_ylim(-0.05, 1.1)
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='Perfect score')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Moving average
window = min(5, len(rewards))
if window > 1:
    moving_avg = [sum(rewards[max(0,i-window+1):i+1])/len(rewards[max(0,i-window+1):i+1]) for i in range(len(rewards))]
    ax2.plot(steps, rewards, color='#2ecc71', alpha=0.3, linewidth=1, label='Raw')
    ax2.plot(steps, moving_avg, color='#e74c3c', linewidth=2, label=f'{{window}}-step moving avg')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Mean Reward', fontsize=12)
    ax2.set_title('Smoothed Reward Curve', fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

plt.tight_layout()
plt.show()
"""))

    # Loss plot
    nb.cells.append(new_code_cell(f"""# Loss trajectory
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(steps, losses, color='#3498db', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Importance Sampling Loss', fontsize=12)
ax.set_title('{exp_info["title"]}\\nTraining Loss', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
"""))

    # Data table
    nb.cells.append(new_markdown_cell("## Step-by-Step Data"))
    table_code = f"""import pandas as pd
df = pd.DataFrame({{
    'Step': {steps},
    'Reward': {rewards},
    'Loss': {losses},
}})
df['Reward_pct'] = (df['Reward'] * 100).round(2).astype(str) + '%'
print(df[['Step', 'Reward_pct', 'Loss']].to_string(index=False))
"""
    nb.cells.append(new_code_cell(table_code))

    # Architecture diagram
    nb.cells.append(new_markdown_cell("""## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRPO Training Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  Atropos     │───▶│  Environment │───▶│  Tinker Trainer   │  │
│  │  (run-api)   │    │  (GSM8K/MATH)│    │  (GRPO + LoRA)   │  │
│  │  Coordinator │◀───│  Scoring     │◀───│  Cloud GPUs      │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
│        │                    │                      │            │
│        │              Rollouts with           LoRA weights      │
│        │              binary rewards         updated via        │
│        │              (0 or 1)            importance sampling    │
│        │                    │                      │            │
│        └────────────────────┴──────────────────────┘            │
│                      Data flow loop                             │
└─────────────────────────────────────────────────────────────────┘
```
"""))

    return nb


def create_overview_notebook():
    """Create an overview notebook comparing all experiments."""
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    nb.cells.append(new_markdown_cell("""# Tinker RL Experiments — Overview & Comparison

**PES University MTech Capstone — Group 6**
**8th Guidance Call Results**

## Project Summary

This project explores **reinforcement learning for language models** using:
- **GRPO** (Group Relative Policy Optimization) — RL without a value/critic model
- **Tinker API** (Thinking Machines) — cloud-based LoRA training on GPUs
- **Atropos** (NousResearch) — environment framework for generating rollouts

We ran experiments across **2 model families** (Qwen, Llama), **4 model sizes** (3B to 30B),
and **2 benchmarks** (GSM8K grade school math, MATH competition math).
"""))

    # Collect all data
    all_data = {}
    for key, info in experiments.items():
        log_path = os.path.join(LOGS_DIR, info["log_dir"], "trainer.log")
        if os.path.exists(log_path):
            steps, rewards, losses = parse_log(log_path)
            all_data[key] = {
                "steps": steps,
                "rewards": rewards,
                "losses": losses,
                "info": info,
            }

    nb.cells.append(new_markdown_cell("## Results Summary"))

    # Summary table
    summary_rows = []
    for key, data in all_data.items():
        info = data["info"]
        r = data["rewards"]
        initial = f"{r[0]*100:.1f}%" if r else "N/A"
        final = f"{r[-1]*100:.1f}%" if r else "N/A"
        peak = f"{max(r)*100:.1f}%" if r else "N/A"
        steps_done = len(r)
        summary_rows.append(
            f"| {info['title']} | {info['params']} | {info['benchmark'].split('(')[0].strip()} | {initial} | {final} | {peak} | {steps_done} | {info['status']} |"
        )

    summary_table = """| Experiment | Parameters | Benchmark | Initial Reward | Final Reward | Peak Reward | Steps | Status |
|------------|-----------|-----------|---------------|-------------|-------------|-------|--------|
""" + "\n".join(summary_rows)

    nb.cells.append(new_markdown_cell(summary_table))

    # Comparison plots - embed data
    gsm8k_data_code = ""
    math_data_code = ""
    for key, data in all_data.items():
        info = data["info"]
        var_name = key.replace("-", "_")
        if "gsm8k" in key:
            gsm8k_data_code += f"""
data_{var_name} = {{
    'steps': {data['steps']},
    'rewards': {data['rewards']},
    'label': '{info["title"]}',
}}
gsm8k_experiments.append(data_{var_name})
"""
        else:
            math_data_code += f"""
data_{var_name} = {{
    'steps': {data['steps']},
    'rewards': {data['rewards']},
    'label': '{info["title"]}',
}}
math_experiments.append(data_{var_name})
"""

    nb.cells.append(new_markdown_cell("## GSM8K Comparison — All Models"))
    nb.cells.append(new_code_cell(f"""import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120

gsm8k_experiments = []
{gsm8k_data_code}

colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
fig, ax = plt.subplots(figsize=(14, 6))

for i, exp in enumerate(gsm8k_experiments):
    ax.plot(exp['steps'], exp['rewards'], color=colors[i % len(colors)],
            linewidth=2, alpha=0.8, label=exp['label'])

ax.set_xlabel('Training Step', fontsize=13)
ax.set_ylabel('Mean Reward', fontsize=13)
ax.set_title('GSM8K — Model Scaling Comparison\\n(GRPO + LoRA rank 32, 50 steps)', fontsize=15, fontweight='bold')
ax.set_ylim(-0.05, 1.1)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='lower right')
plt.tight_layout()
plt.show()
"""))

    if math_data_code.strip():
        nb.cells.append(new_markdown_cell("## MATH Competition Comparison"))
        nb.cells.append(new_code_cell(f"""math_experiments = []
{math_data_code}

colors = ['#9b59b6', '#e67e22']
fig, ax = plt.subplots(figsize=(14, 6))

for i, exp in enumerate(math_experiments):
    ax.plot(exp['steps'], exp['rewards'], color=colors[i % len(colors)],
            linewidth=2, alpha=0.8, label=exp['label'])

ax.set_xlabel('Training Step', fontsize=13)
ax.set_ylabel('Mean Reward', fontsize=13)
ax.set_title('MATH Competition — Model Comparison\\n(GRPO + LoRA rank 32)', fontsize=15, fontweight='bold')
ax.set_ylim(-0.05, 1.1)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
"""))

    # Key insights
    nb.cells.append(new_markdown_cell("""## Key Insights

### 1. Model Size Threshold
- **3B base model (Llama-3.2-3B)**: Failed to learn — stayed at ~2% for 50 steps
- **8B base model (Qwen3-8B)**: 7% → 100% — crossed the threshold for RL learning
- There appears to be a **critical model size** below which GRPO cannot bootstrap reasoning

### 2. Base vs Instruct Models
- **Qwen3-8B (base)**: 7% → 100% over 50 steps (slow initial ramp)
- **Llama-3.1-8B (instruct)**: 79% → 100% in 35 steps (massive head start)
- Instruct fine-tuning provides strong prior knowledge for structured tasks

### 3. MoE Efficiency
- **Qwen3-30B-A3B (MoE)**: Only 3B active parameters, yet reaches 99.2%
- More cost-effective than a dense 30B model at inference time
- Slightly more volatile training than dense models

### 4. GSM8K vs MATH Difficulty
- GSM8K (grade school): Most models reach 90-100% quickly
- MATH (competition): Much harder — Qwen-8B only reaches ~14% after 25 steps
- Llama-8B-Instruct does better on MATH (~55-87%) due to instruct pretraining

### 5. LoRA Efficiency
- All experiments use LoRA rank 32 (~0.1% of model parameters modified)
- Despite tiny parameter budget, achieves full task mastery on GSM8K
- Cloud training via Tinker API makes this accessible without local GPUs
"""))

    # Method section
    nb.cells.append(new_markdown_cell("""## Method: GRPO (Group Relative Policy Optimization)

GRPO trains language models through reinforcement learning **without a value/critic model**:

1. **Group Sampling**: For each question, generate N completions (group_size=16)
2. **Binary Reward**: Each completion is scored 0 (wrong) or 1 (correct) via symbolic math verification
3. **Relative Advantage**: Within each group, compute advantages relative to the group mean
4. **Policy Update**: Update model using importance-sampled policy gradient with LoRA adapters

```
Advantage_i = Reward_i - mean(Rewards_group)

Loss = -sum(IS_ratio * Advantage * log_prob)
     where IS_ratio = P_new(token) / P_old(token)
```

This eliminates the need for a separate value network, reducing memory and compute costs.
"""))

    return nb


def execute_and_export(nb, notebook_path, html_path):
    """Execute notebook and export to HTML."""
    # Save notebook
    with open(notebook_path, "w") as f:
        nbformat.write(nb, f)

    # Execute notebook
    print(f"Executing {notebook_path}...")
    result = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=120",
            notebook_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Warning: execution issue: {result.stderr[:200]}")

    # Convert to HTML
    print(f"Converting to HTML: {html_path}...")
    result = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--no-input",
            f"--output={os.path.basename(html_path)}",
            f"--output-dir={os.path.dirname(html_path)}",
            notebook_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Try with input visible
        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                f"--output={os.path.basename(html_path)}",
                f"--output-dir={os.path.dirname(html_path)}",
                notebook_path,
            ],
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        print(f"  ✓ HTML saved to {html_path}")
    else:
        print(f"  ✗ HTML conversion failed: {result.stderr[:200]}")


if __name__ == "__main__":
    output_dir = os.path.join(PROJECT_ROOT, "notebooks", "html")
    os.makedirs(output_dir, exist_ok=True)

    # Generate individual experiment notebooks
    for key, info in experiments.items():
        log_path = os.path.join(LOGS_DIR, info["log_dir"], "trainer.log")
        if not os.path.exists(log_path):
            print(f"Skipping {key}: no log file found")
            continue

        print(f"\n{'='*60}")
        print(f"Generating: {info['title']}")
        print(f"{'='*60}")

        nb = create_experiment_notebook(key, info)
        nb_path = os.path.join(NOTEBOOKS_DIR, f"{key}.ipynb")
        html_path = os.path.join(output_dir, f"{key}.html")
        execute_and_export(nb, nb_path, html_path)

    # Generate overview notebook
    print(f"\n{'='*60}")
    print("Generating: Overview & Comparison")
    print(f"{'='*60}")

    nb = create_overview_notebook()
    nb_path = os.path.join(NOTEBOOKS_DIR, "experiment_overview.ipynb")
    html_path = os.path.join(output_dir, "experiment_overview.html")
    execute_and_export(nb, nb_path, html_path)

    print(f"\n{'='*60}")
    print("All notebooks generated!")
    print(f"Notebooks: {NOTEBOOKS_DIR}/")
    print(f"HTML files: {output_dir}/")
    print(f"{'='*60}")
