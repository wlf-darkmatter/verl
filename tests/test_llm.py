from vllm import LLM, SamplingParams
# from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from transformers import AutoModelForCausalLM

MODEL_PATH="/mnt/hpfs_test/weights/Qwen3-30B-A3B-Instruct"
MCORE_MODEL_PATH="/mnt/hpfs_test/weights/Qwen3-30B-A3B-Instruct-Mcore"
tp=2
enforce_eager=True
gpu_memory_utilization=0.7
# max_model_len=2048
# max_num_seqs=2048
load_format="dummy"
# mode=LLM(
#             model=MODEL_PATH,
#             enable_sleep_mode=True,
#             tensor_parallel_size=tp,
#             distributed_executor_backend="external_launcher",
#             dtype="bfloat16",
#             enforce_eager=enforce_eager,
#             gpu_memory_utilization=gpu_memory_utilization,
#             disable_custom_all_reduce=True,
#             skip_tokenizer_init=False,
#             max_model_len=max_model_len,
#             max_num_seqs=max_num_seqs,
#             load_format=load_format,  #! 如果是减层
#             disable_log_stats=False,
#             max_num_batched_tokens=max_num_batched_tokens,
#             enable_chunked_prefill=False,
#             enable_prefix_caching=True,  # todo
#             trust_remote_code=True,
#             seed=0,
#             enable_expert_parallel=enable_expert_parallel,
#         )
# patch_vllm_moe_model_weight_loader(model)

HF_model = AutoModelForCausalLM.from_pretrained("/mnt/hpfs_test/weights/Qwen3-30B-A3B-Instruct")

# model.load_weights(HF_MODEL)


from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
# Create an LLM.
llm = LLM(model=MODEL_PATH,
          load_format=load_format,
          tensor_parallel_size=tp
          )
llm.load_weights(HF_MODEL)
# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
