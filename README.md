# tinker-atropos

Atropos integration with Thinking Machines Tinker API (https://thinkingmachines.ai/tinker/). Train models with your Atropos environments from your CPU, without having to worry about underlying compute management or implementation.

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
python launch_training.py --config configs/quick_test.yaml

# Terminal 3: Start environment
python tinker_atropos/environments/gsm8k_tinker.py serve
```

This runs a 10-step training example with Llama-3.1-1B on the GSM8k environment.

## Integration with Atropos Environments

Atropos environments using the following pattern for inference can integrate directly with the Tinker trainer:

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

### How To Implement With An Existing Atropos Environment

1. The `TinkerAtroposConfig` should be loaded in the environment (inside `config.init`), pointing to your desired `config.yaml`, like so:

```
config = TinkerAtroposConfig.from_yaml("configs/default.yaml")
```

2. The `BaseEnvConfig`, or whatever is overridden for configuration there, should be set up to use these values, like so:

```
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

3. As long as your environment uses the `self.server.managed_server` pattern for inference requests like above, this should *just work*.


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

The trainer supports YAML configuration files for environment management.

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

### Configuration Options

- **Tinker / Model Parameters**: `base_model`, `lora_rank`, `learning_rate`
- **Training**: `num_steps`, `batch_size`, `group_size`, `max_token_env_length`, `max_token_trainer_length`
- **Wandb**: `use_wandb`, `wandb_project`, `wandb_group`, `wandb_run_name`
- **APIs**: `atropos_api_url`, `inference_api_url`

See `configs/` for complete parameter lists.

### Programmatic Usage

```python
from tinker_atropos.config import TinkerAtroposConfig
from tinker_atropos.trainer import TinkerAtroposTrainer

# Load from YAML
config = TinkerAtroposConfig.from_yaml("configs/default.yaml")

# Or use defaults
config = TinkerAtroposConfig()

# Initialize trainer
trainer = TinkerAtroposTrainer(config=config)
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
