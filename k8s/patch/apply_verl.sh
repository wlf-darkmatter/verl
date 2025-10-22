set -x # 开启调试模式
if [[ ${USE_CP_PATCH} == 1 ]];then
    #! 需要使用新的mindspeed
    cd /opt/verl
    \cp -f /home/code/verl/k8s/patch/verl.patch/verl0928-cp/verl/models/mcore/model_forward.py /opt/verl/verl/models/mcore/model_forward.py
    \cp -f /home/code/verl/k8s/patch/verl.patch/verl0928-cp/verl/models/mcore/patch_v012.py /opt/verl/verl/models/mcore/patch_v012.py
    \cp -f /home/code/verl/k8s/patch/verl.patch/verl0928-cp/verl/models/mcore/util.py /opt/verl/verl/models/mcore/util.py

    echo -e "/033[32mApplied Verl CP/033[0m"
fi

set +x # 关闭调试模式