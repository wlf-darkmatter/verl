
set -x # 开启调试模式
vllm_version=$(cat /opt/vllm/vllm/_version.py | grep 'version =' | awk '{print $5}')
vllm_version=${vllm_version//\'}

cd $(dirname $0)
if [[ ${vllm_version} == '0.9.1' ]];then
  rm -f /opt/vllm/vllm/model_executor/models/deepseek_v2.py
  cp -f /home/code/verl/k8s/patch/0827/deepseek_v2.py /opt/vllm/vllm/model_executor/models/deepseek_v2.py
  echo -e "\033[32mApplied VLLM-ASCEND ${vllm_version}!\033[0m"
fi
if [[ ${vllm_version} == '0.10.0' ]];then
  rm -f /opt/vllm-ascend/vllm_ascend/models/deepseek_v2.py
  cp -f ./vllm.patch/0.10.0/vllm-ascend/vllm_ascend/models/deepseek_v2.py /opt/vllm-ascend/vllm_ascend/models/deepseek_v2.py
  echo -e "\033[32mApplied VLLM-ASCEND ${vllm_version}!\033[0m"
fi

set +x # 关闭调试模式
