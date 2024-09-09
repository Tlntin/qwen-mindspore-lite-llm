### 说明
- 本项目基于之前的[qwen-ascend-llm](https://github.com/Tlntin/qwen-ascend-llm)项目魔改而来，基本和之前相差不大，只是模型推理从CANN变成了mindspore-lite。同时为了后续在手机部署，需要将python换成c++；数据格式需要从ND换成NCHW(华为手机上面的mindspore-lite只支持NCHW格式)。

### 准备工作
1. 下载本项目
  ```bash
  git clone https://github.com/Tlntin/qwen-mindspore-lite-llm.git
  ```

2. 下载qwen1.5/qwen2的模型，选择chat模型或者instruct模型，将其放到download文件夹，仅支持huggingface下载的模型，网络不好的可以用镜像站：https://hf-mirror.com/Qwen


### 详细运行步骤
1. 安装python依赖（用于后续模型结构转onnx，以及onnx验证）。
  ```bash
  cd qwen-mindspore-lite-llm
  conda create -n mindspore_lite python==3.10 
  pip install -r ./requirements.txt
  ```
2. 导出onnx，默认kv-cache长度为1024，可以根据自己的内存、显存来设置更大参数。
  - 对于NPU设备
    ```bash
    python3 export/export_onnx.py \
      --device_str="npu" \
      --dtype="float16" \
      --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
      --onnx_model_path="./output/onnx/qwen2_1.5b_chat.onnx" \
      --kv_cache_length=1024
    ```
  - 对于CPU设备
    ```bash
    python3 export/export_onnx.py \
      --device_str="cpu" \
      --dtype="float32" \
      --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
      --onnx_model_path="./output/onnx/qwen2_1.5b_chat.onnx" \
      --kv_cache_length=1024
    ```

3. 验证onnx，需要分别运行pytorch和onnx，观察两边输出误差，若误差较小(最大误差低于3位小数，平均误差低于5位小数)，则说明onnx导出是ok的。
  - 先用cpu跑pytorch
    ```bash
    python3 export/test_pytorch_run.py \
      --device_str="cpu" \
      --dtype="float32" \
      --hf_model_dir="./download/Qwen2-1.5B-Instruct"
    ```
  - 再用cpu跑onnx
    ```bash
    python3 export/test_onnx_run.py \
      --dtype="float32" \
      --onnx_model_path="./output/onnx/qwen2_1.5b_chat.onnx"
    ```

4. 改变onnx结构，目前导出的Trilu算子和Cast算子有些问题，atc命令无法识别，需要改一下结构。
  ```bash
  python3 export/change_node.py \
    --input_model_path="./output/onnx/qwen2_1.5b_chat.onnx" \
    --output_model_path="./output/onnx2/qwen2_1.5b_chat.onnx"
  ```
