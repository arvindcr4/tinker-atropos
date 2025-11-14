# tinker-atropos

An integration layer connecting Atropos with the Thinking Machines Tinker API (https://thinkingmachines.ai/tinker/). This package enables seamless model training with Atropos environments from your local machine, abstracting away compute management and infrastructure concerns.

## Installation

```bash
pip install -e .
```

## Quickstart

First, obtain a Tinker API key from https://tinker-console.thinkingmachines.ai/keys.

Run the following commands in separate terminal windows to start a training run:

```bash
# Terminal 1: Start Atropos API
run-api

# Terminal 2: Start training
export TINKER_API_KEY="<your-key>"
python launch_training.py --config configs/default.yaml --num-steps 10

# Terminal 3: Start environment
python tinker_atropos/environments/gsm8k_tinker.py serve --config configs/default.yaml
```

This runs a 10-step training example with Llama-3.1-8B-Instruct on the GSM8k environment. To use a different configuration file for the environment, modify the `CONFIG_PATH` variable at the top of `gsm8k_tinker.py`.

## Integration with Atropos Environments

Atropos environments that utilize the following inference pattern are compatible with the Tinker trainer:

```python
async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
    chat_completion = await managed.chat_completion(
        messages=messages,
        n=self.config.group_size,
        max_tokens=self.config.max_token_length,
        temperature=1.0,
    )

    state = managed.get_state()
    nodes = state["nodes"]
```

### Implementation Guide for Existing Atropos Environments

1. Load the `TinkerAtroposConfig` within your environment's initialization (inside `config.init`), specifying the path to your desired configuration file:

```python
config = TinkerAtroposConfig.from_yaml("configs/default.yaml")
```

2. Configure the `BaseEnvConfig` (or your custom configuration class) to utilize the loaded values:

```python
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
    ensure_scores_not_the_same=config.ensure_scores_not_the_same,
)
```

3. Ensure your environment uses the `self.server.managed_server` pattern for inference requests as demonstrated above. No additional modifications are required for Tinker integration.


## Downloading Weights

The trainer outputs a Tinker path at the end of training:

```
tinker://<training_run_id>/sampler_weights/final
```

where `<training_run_id>` is the Training Run ID from https://tinker-console.thinkingmachines.ai.

To download the weights, set the `TINKER_PATH` in `tinker_atropos/utils/download_weights.py` and run:

```bash
python tinker_atropos/utils/download_weights.py
```

Weights will be saved to the specified location.

## Configuration

Both the trainer and environment support YAML configuration files to ensure parameter consistency across your training pipeline. The provided environment file (`gsm8k_tinker.py`) references a configuration path that should match the trainer's configuration. Update the `CONFIG_PATH` variable in the environment file when switching between configurations.

### Usage

```bash
# Use default configuration
python launch_training.py

# Use a specific config file
python launch_training.py --config configs/quick_test.yaml

# Override parameters via CLI
python launch_training.py --config configs/default.yaml --num-steps 100 --no-wandb
```

### Available Configs

- `default.yaml` - Standard configuration for typical training runs
- `quick_test.yaml` - Minimal configuration for testing and debugging

### Configuration Structure

Configs follow the Atropos format with a `tinker` section for Tinker-specific settings:

```yaml
env:
  # Standard Atropos environment config
  group_size: 16
  batch_size: 128
  tokenizer_name: "meta-llama/Llama-3.1-8B-Instruct"
  # ... more settings

openai:
  # Standard Atropos server config
  - model_name: "meta-llama/Llama-3.1-8B-Instruct"
    base_url: "http://localhost:8001/v1"
    # ... more settings

tinker:
  # Tinker-specific training config
  lora_rank: 32
  learning_rate: 0.00004
  # ... more settings
```

See `configs/default.yaml` for the complete configuration options.

### Adapting Existing Atropos Configs

To use an existing Atropos environment config with Tinker:

1. Add a `tinker` section with training parameters:
   ```yaml
   tinker:
     lora_rank: 32
     learning_rate: 0.00004
     max_token_trainer_length: 2048
     checkpoint_dir: "./checkpoints/"
     save_checkpoint_interval: 0
     wandb_project: "my-project"
     wandb_group: null
     wandb_run_name: "my-run"
   ```

2. Ensure your environment uses `managed_server` with stop sequences:
   ```python
   async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
       chat_completion = await managed.chat_completion(
           messages=messages,
           n=self.config.group_size,
           max_tokens=self.config.max_token_length,
           temperature=1.0,
           stop=[self.tokenizer.eos_token_id],  # Add stop sequences
       )
   ```

3. That's it! The config is ready to use with both Atropos and Tinker.

### Programmatic Usage

```python
from tinker_atropos.config import TinkerAtroposConfig
from tinker_atropos.trainer import TinkerAtroposTrainer

# Load from YAML
config = TinkerAtroposConfig.from_yaml("configs/default.yaml")

# Access nested config values
print(f"Model: {config.env.tokenizer_name}")
print(f"LoRA rank: {config.tinker.lora_rank}")
print(f"Group size: {config.env.group_size}")

# Or use convenience properties
print(f"Model: {config.base_model}")  # -> config.env.tokenizer_name
print(f"LoRA rank: {config.lora_rank}")  # -> config.tinker.lora_rank
print(f"Learning rate: {config.learning_rate}")  # -> config.tinker.learning_rate

# Use defaults (creates default config)
config = TinkerAtroposConfig()

# Initialize and run trainer
trainer = TinkerAtroposTrainer(config=config)
await trainer.run()
```

## Testing

```bash
python -m pytest tinker_atropos/tests/ -v
```

## Cost

The Tinker Rate Card and available models are listed here: https://tinker-console.thinkingmachines.ai/rate-card

## Documentation

- Atropos: https://github.com/NousResearch/atropos
- Tinker: https://tinker-docs.thinkingmachines.ai
