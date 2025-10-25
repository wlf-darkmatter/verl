megatron_version=$(cat /opt/Megatron-LM/megatron/core/package_info.py | grep 'MINOR =' | awk '{print $3}')

set -x # 开启调试模式


if [[ ${megatron_version} == 12 ]];then

    rm -f /opt/Megatron-LM/megatron/core/transformer/dot_product_attention.py
    cp -f /home/code/verl/k8s/patch/megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/dot_product_attention.py /opt/Megatron-LM/megatron/core/transformer/dot_product_attention.py

    rm -f /opt/Megatron-LM/megatron/core/transformer/multi_token_prediction.py
    cp -f /home/code/verl/k8s/patch/megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/multi_token_prediction.py /opt/Megatron-LM/megatron/core/transformer/multi_token_prediction.py

    rm -f /opt/Megatron-LM/megatron/core/transformer/module.py
    cp -f /home/code/verl/k8s/patch/megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/module_strict_false.py /opt/Megatron-LM/megatron/core/transformer/module.py
    echo -e "\033[32mApplied Megatron-core ${megatron_version}!\033[0m"

fi

set +x # 关闭调试模式