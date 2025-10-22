megatron_version=$(cat /opt/Megatron-LM/megatron/core/package_info.py | grep 'MINOR =' | awk '{print $3}')



if [[ ${megatron_version} == 12 ]];then

    rm -f /opt/Megatron-LM/megatron/core/transformer/dot_product_attention.py
    cp -f /home/code/verl/k8s/patch/megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/dot_product_attention.py /opt/Megatron-LM/megatron/core/transformer/dot_product_attention.py

    rm -f /opt/Megatron-LM/megatron/core/transformer/multi_token_prediction.py
    cp -f /home/code/verl/k8s/patch/megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/multi_token_prediction.py /opt/Megatron-LM/megatron/core/transformer/multi_token_prediction.py

    echo -e "\033[32mApplied Megatron-core ${megatron_version}!\033[0m"

fi

