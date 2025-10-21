
if [[ ${USE_CP_PATCH} == 1 ]];then
    #! 需要使用新的mindspeed
    rm -rf /opt/MindSpeed
    cp -r /home/code/MindSpeed /opt/MindSpeed
    cd /opt/MindSpeed
    git reset --hard 2.2.0_core_r0.12.1
    git apply /home/code/verl/k8s/patch/mindspeed.patch/2.2.0_core_r0.12.1-cp/reset_attention_mask_adaptor.diff
    git apply /home/code/verl/k8s/patch/mindspeed.patch/2.2.0_core_r0.12.1-cp/reset_attention_mask_feature.diff
    git apply /home/code/verl/k8s/patch/mindspeed.patch/2.2.0_core_r0.12.1-cp/dot_product_attention.diff
    git apply /home/code/verl/k8s/patch/mindspeed.patch/2.2.0_core_r0.12.1-cp/grouped_linear.fallback.diff
    echo -e "\033[32mApplied MindSpeed CP\033[0m"
fi

