import os
import shutil
import argparse
import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero.")
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for <class 'numpy.float32'> type is zero.")
from transformers.models.qwen2 import Qwen2Config
from mindspore_lite.converter import Converter, FmkType
import mindspore_lite as mslite

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
    "--input_data_type" ,
    type=str,
    help="Data type of input tensors, default is same with the type defined in model. FLOAT16 | FLOAT | INT8 | UINT8 | INT32 | INT64 | DEFAULT",
    choices=["float16", "float", "int8", "uint8", "int32", "int64", "default"],
    default="default",
)

parser.add_argument(
   "--fp16",
   type=bool,
   choices=[True, False],
   help="Serialize const tensor in Float16 data type, only effective for const tensor in Float32 data type.",
   default=False
)
parser.add_argument(
    '--hf_model_dir',
    type=str,
    help="model and tokenizer path, only support huggingface model",
    default=os.path.join(project_dir, "download", "Qwen2-0.5B-Instruct")
)
parser.add_argument(
    "--onnx_model_path",
    help="output onnx path",
    type=str,
    default=os.path.join(onnx_model_dir, "qwen2_0.5b_chat.onnx")
)
parser.add_argument(
    "--ms_model_path",
    help=".ms/.mindir model path",
    type=str,
    default= os.path.join(model_dir, "qwen2_0.5b_chat")
)

parser.add_argument(
    "--ms_optimize",
    help="optimize in MindSpore with gpu/cpu/npu",
    choices=["none", "general", "gpu_oriented", "ascend_oriented"],
    type=str,
    default="general",
)

parser.add_argument(
    "--max_batch",
    help="max batch",
    type=int,
    default=1,
)
parser.add_argument(
    "--kv_cache_length",
    help="kv-cache length",
    type=int,
    default=2048,
)

parser.add_argument(
    "--config_file",
    help="config file",
    type=str,
    default=None,
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

converter = Converter()
converter.weight_fp16 = args.fp16
converter.input_shape = {
    "input_ids": [args.max_batch, 1, 1, 1],
    "attention_mask": [args.max_batch, 1, 1, 1 + kv_cache_length],
    "position_ids": [args.max_batch, 1, 1, 1],
    "past_key_values": [
        args.max_batch,
        num_hidden_layers * 2 * num_key_value_heads,
        kv_cache_length,
        per_head_dim
    ]
}
converter.input_format = mslite.Format.NCHW
converter.input_data_type = mslite.DataType.FLOAT32
converter.output_data_type = mslite.DataType.FLOAT32
converter.save_type = mslite.ModelType.MINDIR_LITE
converter.optimize = args.ms_optimize
print("====== convert =====")
print(converter)
converter.convert(
    fmk_type=FmkType.ONNX,
    model_file=args.onnx_model_path,
    output_file=args.ms_model_path
)
# 将msw文件放到项目根目录, 否则msw不识别
for file in os.listdir(model_dir):
    if file.endswith("msw"):
        old_path = os.path.join(model_dir, file)
        new_path = os.path.join(project_dir, file)
        shutil.move(old_path, new_path) 