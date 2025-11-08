
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
        # exit 1
    fi
fi

cd /opt/MindSpeed
git reset --hard origin/2.2.0_core_r0.12.1

cd $cwd
\cp ./mindspeed.patch/2.2.0_core_r0.12.1/MindSpeed/mindspeed/te/pytorch/module/grouped_linear.py /opt/MindSpeed/mindspeed/te/pytorch/module/grouped_linear.py

echo -e "\033[1;32mApplied MindSpeed CP\033[0m"

\cp ./mindspeed.patch/2.2.0_core_r0.12.1/MindSpeed/mindspeed/core/transformer/flash_attention/flash_attention/adaptor.py /opt/MindSpeed/mindspeed/core/transformer/flash_attention/flash_attention/adaptor.py

echo -e "\033[1;32mApplied MindSpeed Attention Scale\033[0m"

set +x # 关闭调试模式
