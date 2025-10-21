if [[ ${USE_CP_PATCH} == 1 ]];then
    #! 需要使用新的mindspeed
    cd /opt/verl
    git apply /home/code/verl/k8s/patch/verl.patch/verl0928-cp/mcore_util.diff
    git apply /home/code/verl/k8s/patch/verl.patch/verl0928-cp/model_forward.diff
    git apply /home/code/verl/k8s/patch/verl.patch/verl0928-cp/patch_v012.diff
    echo -e "\033[32mApplied Verl CP\033[0m"
fi

