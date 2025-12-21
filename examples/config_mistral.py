"""Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information.

Usage:
```
python config_tiny_mistral.py
```
"""

import os
from dataclasses import dataclass
from typing import Optional

from nanotron.config import (
    CheckpointsArgs,
    Config,
    DataArgs,
    GeneralArgs,
    LoggingArgs,
    LRSchedulerArgs,
    MistralConfig,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    PretrainDatasetsArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

MODEL_CONFIG = MistralConfig(
    # Config for Mistral 7B
    attn_pdrop=0.0,
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=4096,
    initializer_range=0.02,
    intermediate_size=14336,
    max_position_embeddings=32768,
    num_attention_heads=32,
    num_hidden_layers=32,
    num_key_value_heads=8,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_theta=10000.0,
    sliding_window_size=4096,
    tie_word_embeddings=False,
    use_cache=True,
    vocab_size=32000,
)

num_params = human_format(get_num_params(MODEL_CONFIG)).replace(".", "p")

print(f"Model has {num_params} parameters")

PARALLELISM = ParallelismArgs(
    dp=2,
    pp=2,
    tp=2,
    pp_engine="1f1b",
    tp_mode="REDUCE_SCATTER",
    tp_linear_async_communication=True,
    recompute_granularity="selective",
)

CONFIG = Config(
    general=GeneralArgs(project="mistralai", run="Mistral-7B-v0.1", seed=42, step=0),
    checkpoints=None,
    parallelism=PARALLELISM,
    model=ModelArgs(init_method=RandomInit(std=0.025), model_config=MODEL_CONFIG),
    tokenizer=TokenizerArgs("mistralai/Mistral-7B-v0.1"),
    optimizer=None,
    logging=None,
    tokens=None,
    data=None,
    profiler=None,
    lighteval=None,
)

if __name__ == "__main__":
    file_path = os.path.abspath(__file__)

    file_path = file_path.replace(".py", ".yaml")
    # Save config as YAML file
    config.save_as_yaml(file_path)

    # You can now train a model with this config using `/run_train.py`
