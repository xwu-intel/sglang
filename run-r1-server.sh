
export SGLANG_HPU_SKIP_WARMUP=1
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"

# MODEL_PATH="/data/models/deepseek/models--deepseek-ai--DeepSeek-R1/snapshots/8a58a132790c9935686eb97f042afa8013451c9f"
MODEL_PATH="/data/wgs/deepseek-r1-fp8"

# python -m sglang.launch_server --model-path $MODEL_PATH --dtype bfloat16 --port 8000 --tensor-parallel-size 8
# python3 -m sglang.launch_server --model $MODEL_PATH --enable-dp-attention --dp-size 2 --tp 8 --trust-remote-code
# python3 -m sglang.launch_server --device hpu --model $MODEL_PATH --enable-dp-attention --dp-size 2 --tp 8 --trust-remote-code
python3 -m sglang.launch_server --device hpu --model $MODEL_PATH --tp 8 --trust-remote-code --mem-fraction-static 0.95
