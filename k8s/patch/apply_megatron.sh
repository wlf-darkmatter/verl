set -x

cwd=$(dirname $(realpath $0))
echo "dealing $cwd"
cd $cwd

\cp ./megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/multi_token_prediction.py /opt/Megatron-LM/megatron/core/transformer/multi_token_prediction.py
\cp ./megatron.patch/0.12.1/Megatron-LM/megatron/training/tokenizer/tokenizer.py /opt/Megatron-LM/megatron/training/tokenizer/tokenizer.py
echo -e "/033[32mApplied megatron MTP done./033[0m"

set +x
