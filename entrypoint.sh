#!/bin/bash
set -e

# Login to Hugging Face
if [ -n "$HF_API_KEY" ]; then
    echo "Logging in to Hugging Face..."
    huggingface-cli login --token ${HF_API_KEY}
else
    echo "HF_API_KEY not set, exiting"
    exit 1
fi

# Run the main command
echo "Running main command: accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/llama-31-8b/sft/config_qlora.yaml --load_in_4bit=true"
accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/llama-31-8b/sft/config_qlora.yaml --load_in_4bit=true

# Allow passing additional commands at runtime
exec "$@"