
set -x # 开启调试模式

#! 需要使用新的mindspeed
rm -rf /opt/MindSpeed
cp -r /home/code/verl-gpu/tmp/MindSpeed /opt/MindSpeed
cd /opt/MindSpeed
git reset --hard origin/2.2.0_core_r0.12.1
\cp /home/code/verl-gpu/k8s/patch/mindspeed.patch/2.2.0_core_r0.12.1/MindSpeed/mindspeed/te/pytorch/module/grouped_linear.py /opt/MindSpeed/mindspeed/te/pytorch/module/grouped_linear.py

echo -e "\033[32mApplied MindSpeed CP\033[0m"

\cp /home/code/verl-gpu/k8s/patch/mindspeed.patch/2.2.0_core_r0.12.1/MindSpeed/mindspeed/core/transformer/flash_attention/flash_attention/adaptor.py /opt/MindSpeed/mindspeed/core/transformer/flash_attention/flash_attention/adaptor.py

echo -e "\033[32mApplied MindSpeed Attention Scale\033[0m"

set +x # 关闭调试模式
