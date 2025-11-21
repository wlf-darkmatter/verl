
set -x # 开启调试模式
cwd=$(dirname $(realpath $0))
echo "dealing $cwd"
cd $cwd

vllm_version=$(cat /opt/vllm/vllm/_version.py | grep 'version =' | awk '{print $5}')
vllm_version=${vllm_version//\'}
echo -e "\033[1;33m vllm_version: ${vllm_version}\033[0m"

if [[ ${vllm_version} == '0.9.1' ]];then
  # rm -f /opt/vllm/vllm/model_executor/models/deepseek_v2.py
  # cp -f /home/code/verl/k8s/patch/0827/deepseek_v2.py /opt/vllm/vllm/model_executor/models/deepseek_v2.py
  echo -e "\033[1;33mApplied VLLM-ASCEND ${vllm_version}! 这个版本的patch不在当前代码仓中！\033[0m"
fi
if [[ ${vllm_version} == '0.10.0' ]];then
  rm -f /opt/vllm-ascend/vllm_ascend/models/deepseek_v2.py
  cp -f ./vllm.patch/0.10.0/vllm-ascend/vllm_ascend/models/deepseek_v2.py /opt/vllm-ascend/vllm_ascend/models/deepseek_v2.py
  cp -f ./vllm.patch/0.10.0/vllm-ascend/vllm_ascend/ascend_config.py /opt/vllm-ascend/vllm_ascend/ascend_config.py #! 为了使能mla的chunck prefill
  echo -e "\033[32mApplied VLLM-ASCEND ${vllm_version}!\033[0m"
fi
if [[ ${vllm_version} == '0.11.0' ]];then

  # rm -f /opt/vllm-ascend/vllm_ascend/models/deepseek_v2.py
  # cp -f ./vllm.patch/0.10.0/vllm-ascend/vllm_ascend/models/deepseek_v2.py /opt/vllm-ascend/vllm_ascend/models/deepseek_v2.py
  echo -e "\033[1;33mApplied VLLM-ASCEND ${vllm_version}! 这个版本暂时没有patch\033[0m"
fi
set +x # 关闭调试模式
