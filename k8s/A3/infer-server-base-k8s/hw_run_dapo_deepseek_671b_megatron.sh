#! ------------------------------------------------------------
TRAIN_FILE="/mnt/hpfs_test/data/data/dapo-math-17k.parquet"
MODEL_PATH="/mnt/hpfs_test/weights/dsv3-base-fp8-zy-bf16"

n_resp_per_prompt=4

gen_tp=8
gen_pp=4

gen_dp=8


vllm serve ${MODEL_PATH} --port 8001 \
    --gpu-memory-utilization 0.95  \
    --max-model-len 16384 \
    --served-model-name DeepSeek-V3-Base \
    --tensor-parallel-size ${gen_tp} \
    --pipeline-parallel-size ${gen_pp} \
    --host 0.0.0.0 \
    --trust-remote-code \
    --tokenizer ${MODEL_PATH}