
set -x # 开启调试模式
cwd=$(dirname $(realpath $0))
echo "dealing $cwd"
cd $cwd

#! 需要使用新的mindspeed
cd /opt/MindSpeed
if [[ $(git branch -a | grep -c 2.2.0_core_r0.12.1) -eq 0 ]];then
    echo -e "\033[1;33m内置的 /opt/MindSpeed版本太落后，需要手动复制一个 \033[0m"
    cd $cwd
    if [[ -d ../../tmp/MindSpeed ]];then
        rm -rf /opt/MindSpeed
        cp -r ../../tmp/MindSpeed /opt/MindSpeed
    else
        echo -e "\033[1;31mMindSpeed not found, please clone it from gitcode. 然后记得放在 <大verl>/tmp/MindSpeed 里面 \033[0m"
        exit 1
    fi
fi

cd /opt/MindSpeed
git reset --hard origin/2.2.0_core_r0.12.1

cd $cwd

# cd /opt/MindSpeed
# git apply $cwd/mindspeed.patch/2.2.0_core_r0.12.1/grouped_linear.diff

\cp ./mindspeed.patch/2.2.0_core_r0.12.1/MindSpeed/mindspeed/te/pytorch/module/grouped_linear.py /opt/MindSpeed/mindspeed/te/pytorch/module/grouped_linear.py
echo -e "\033[1;32mApplied MindSpeed grouped_linear\033[0m"

\cp $cwd/mindspeed.patch/2.2.0_core_r0.12.1/MindSpeed/mindspeed/core/transformer/flash_attention/flash_attention/adaptor.py /opt/MindSpeed/mindspeed/core/transformer/flash_attention/flash_attention/adaptor.py
echo -e "\033[1;32mApplied MindSpeed Attention Scale\033[0m"

if [[ ${USE_CP_PATCH} == 1 ]];then
    set -e
    cd /opt/MindSpeed
    git apply $cwd/mindspeed.patch/2.2.0_core_r0.12.1-cp/reset_attention_mask_adaptor.diff
    git apply $cwd/mindspeed.patch/2.2.0_core_r0.12.1-cp/reset_attention_mask_feature.diff
    git apply $cwd/mindspeed.patch/2.2.0_core_r0.12.1-cp/dot_product_attention.diff
    echo -e "\033[1;33mApplied MindSpeed CP \033[0m"
    set +e


    cd /opt/verl
    git apply $cwd/verl.patch/patch_v012.diff
    git apply $cwd/verl.patch/model_forward.diff
    git apply $cwd/verl.patch/mcore_util.diff

    echo -e "\033[1;33mApplied Verl CP \033[0m"

fi


set +x # 关闭调试模式
