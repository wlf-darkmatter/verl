
set -x # 开启调试模式
cwd=$(dirname $(realpath $0))
echo "dealing $cwd"
cd $cwd

#! #################  【回退代码修复】  #####################


#! 规避模型加载时 权重读取错误的问题
cd $cwd
bash /opt/verl/k8s/patch/apply_vllm-ascend.sh

#! 回退旧的mindspeed
cd /opt/MindSpeed
git reset --hard ab90bbce894603b502a09025bcdf306f16b7a89f
# cd $cwd
# \cp ../../patch/mindspeed.patch/2.2.0_core_r0.12.1/MindSpeed/mindspeed/te/pytorch/module/grouped_linear.py /opt/MindSpeed/mindspeed/te/pytorch/module/grouped_linear.py
# git apply $cwd/../../fallback/resolve_mindspeed_v0.12.0_weight_keyerror/layernorm_column_parallel_linear.py.diff

#! megatron修改
cd $cwd
\cp ../../patch/megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/dot_product_attention.py /opt/Megatron-LM/megatron/core/transformer/dot_product_attention.py
\cp ../../patch/megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/multi_token_prediction.py /opt/Megatron-LM/megatron/core/transformer/multi_token_prediction.py


set +x # 关闭调试模式
