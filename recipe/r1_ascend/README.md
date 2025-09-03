# DeepSeekV3 RLHF on NPU
本sample主要是DeepseekV3模型在NPU上进行RL后训练的适配点介绍，基于[veRL开源框架](https://github.com/volcengine/verl)，通过一系列补丁对开源代码进行面向NPU的适配改造。

---

# 1. 环境准备
veRL上的NPU环境准备，可参考[ascend_quick_start.rst]()

> 其中部分包的版本需要注意替换：
1. [vLLM：0.9.1]()
2. [vLLM_Ascend：0.9.1dev]() commit-id:
3. [MindSpeed: master](https://gitee.com/ascend/MindSpeed) commit-id: 7ff81
4. [veRL: master](https://github.com/volcengine/verl) commit-id:55e3c

# 2 准备训练数据集与模型
数据集放入 ./data, 数据集准备参考: [veRL官方文档](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
模型放入 ./DeepSeek-V3-hf, 模型下载地址：[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
权重切分可使用 verl/scripts/converter_hf_to_mcore.py

# 3 执行RL后训练
```shell
# verl目录下启动DeepSeekV3的RL后训练
bash ./recipe/r1_ascend/ray_start_grpo.sh
```

# 4 训练结果