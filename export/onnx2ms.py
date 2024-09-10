import os
import ctypes
import subprocess
import argparse
import math
from transformers.models.qwen2 import Qwen2Config

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
onnx_model_dir = os.path.join(output_dir, "onnx")
if not os.path.exists(onnx_model_dir):
    os.mkdir(onnx_model_dir)
model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dtype" ,
    type=str,
    help="support float16/float32/int8..., if use CPU, only support fp32",
    choices=["float16", "float", "int8", "uint8", "int32", "int64", "default"],
    default="default",
)
parser.add_argument(
    '--hf_model_dir',
    type=str,
    help="model and tokenizer path, only support huggingface model",
    default=os.path.join(project_dir, "download", "Qwen2-1.5B-Instruct")
)
parser.add_argument(
    "--onnx_model_path",
    help="output onnx path",
    type=str,
    default=os.path.join(onnx_model_dir, "qwen2_1.5b_chat.onnx")
)
parser.add_argument(
    "--ms_model_path",
    help=".ms/.mindir model path",
    type=str,
    default= os.path.join(model_dir, "qwen2_1.5b_chat")
)
parser.add_argument(
    "--save_type",
    help="The type of saved model.",
    choices=["mindir", "mindir_lite"],
    type=str,
    default="mindir_lite"
)

parser.add_argument(
    "--ms_optimize",
    help="optimize in MindSpore with gpu/cpu/npu",
    choices=["none", "general", "gpu_oriented", "ascend_oriented"],
    type=str,
    default="general",
)
parser.add_argument(
    "--optimize_transformer",
    help="whether enable Fast-Transformer fusion",
    choices=["true", "false"],
    type=str,
    default="false"
)
parser.add_argument(
    "--max_batch",
    help="max batch",
    type=int,
    default=1,
)
parser.add_argument(
    "--max_prefill_length",
    help="max prefill length in first inference. "
        "Attention max_prefill_length + max_output_length <= kv_cache_length. "
        "the number must by 2^xx, like 1, 2, 4, 8, 16, 32, 64, 128, 256... "
        "Note! The higher this number, the longer it will take to compile.",
    type=int,
    default=1,
)
parser.add_argument(
    "--kv_cache_length",
    help="kv-cache length",
    type=int,
    default=1024,
)


args = parser.parse_args()
max_batch = args.max_batch
model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
num_hidden_layers = model_config.num_hidden_layers
num_key_value_heads = model_config.num_key_value_heads
hidden_size = model_config.hidden_size
num_attention_heads = model_config.num_attention_heads
per_head_dim = hidden_size // num_attention_heads
kv_cache_length = args.kv_cache_length
max_prefill_log2 = int(math.log2(args.max_prefill_length))
max_prefill_length = 2 ** max_prefill_log2 
prefill_length_range = list(range(0, max_prefill_log2 + 1))
prefill_length_range = [2 ** idx for idx in prefill_length_range]
assert (max_prefill_length < kv_cache_length), \
    print("max_input_length max be smaller than kv_cache_length, because max_input_length + max_output_length <= kv_cache")
input_ids_length_range = prefill_length_range
attention_length_range = [
    length + kv_cache_length
    for length in prefill_length_range
]
position_length_range = prefill_length_range
input_ids_shape = [
    f"1~{max_batch}" if max_batch > 1 else "1",
    "1",
    "1",
    "-1" if max_prefill_length > 1 else "1",
]
attention_mask_shape = [
    f"1~{max_batch}" if max_batch > 1 else "1",
    "1",
    "1",
    "-1" if max_prefill_length > 1 else str(1 + kv_cache_length)
]
position_ids_shape = [
    f"1~{max_batch}" if max_batch > 1 else "1",
    "1",
    "1",
    "-1" if max_prefill_length > 1 else "1"
]
dynamic_dims = []
for dynamic_dim in zip(
    input_ids_length_range, attention_length_range, position_length_range
):
    dynamic_dim = [str(dim) for dim in dynamic_dim]
    dynamic_dims.append(",".join(dynamic_dim))
past_key_values_shape = [
    f"1~{max_batch}" if max_batch > 1 else "1",
    num_hidden_layers * 2 * num_key_value_heads,
    kv_cache_length,
    per_head_dim
]
past_key_values_shape = [str(x) for x in past_key_values_shape]
command_lines = [
    "converter_lite",
    "--fmk=ONNX",
    # "--device=Ascend", # 支持在Ascend上运行，以后再加上
    # "--fp16=on"  # 支持fp16运行，先关上，等CPU验证完再打开
    '--modelFile="{}"'.format(args.onnx_model_path),
    '--outputFile="{}"'.format(args.ms_model_path),
    # "--saveType={}".format(args.save_type.upper()),
    "--inputDataFormat=NCHW",  # 华为手机上面只支持NCHW，同时mindspore_lite也只支持NCHW,相关链接：https://developer.huawei.com/consumer/cn/doc/harmonyos-faqs-V5/hiaifoundation-faqs-3-V5
    "--inputDataType={}".format(args.dtype.upper()),
    "--optimize={}".format(args.ms_optimize),
    "--optimizeTransformer={}".format(args.optimize_transformer),
    # "--precision_mode=must_keep_origin_dtype",
    '--inputShape="input_ids:{};attention_mask:{};position_ids:{};past_key_values:{}"'.format(
        ",".join(input_ids_shape),
        ",".join(attention_mask_shape),
        ",".join(position_ids_shape),
        ",".join(past_key_values_shape)
    ),
]
if max_prefill_length > 1:
    command_lines.append(
        "--dynamic_dims \"{}\"".format(";".join(dynamic_dims))
    )
print("============ run command ==============")
print(" ".join(command_lines))
print("=======================================")
subprocess.run(
    " ".join(command_lines),
    shell=True,
    check=True,
)