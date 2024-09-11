### 说明
- 本项目基于之前的[qwen-ascend-llm](https://github.com/Tlntin/qwen-ascend-llm)项目魔改而来，基本和之前相差不大，只是模型推理从CANN变成了mindspore-lite。数据格式需要从ND换成NCHW(华为手机上面的mindspore-lite只支持NCHW格式)。

### 准备工作
1. 下载本项目
  ```bash
  git clone https://github.com/Tlntin/qwen-mindspore-lite-llm.git
  ```

2. 下载qwen1.5/qwen2的模型，选择chat模型或者instruct模型，将其放到download文件夹，仅支持huggingface下载的模型，网络不好的可以用镜像站：https://hf-mirror.com/Qwen

3. 需要已经配置好mindspore-lite的环境，可以参考下面这个环境配置。
  ```bash
  # mindspore-lite
  export LITE_HOME=/usr/local/mindspore-lite
  export PATH=$PATH:${LITE_HOME}/tools/converter/converter:${LITE_HOME}/tools/benchmark
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LITE_HOME}/tools/converter/lib:${LITE_HOME}/runtime/lib:${LITE_HOME}/runtime/third_party/dnnl
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LITE_HOME}/runtime/third_party/glog:${LITE_HOME}/runtime/third_party/libjpeg-turbo/lib:${LITE_HOME}/runtime/third_party/securec
  ```
4. 并且按照了mindspore-lite python包。[下载页面](https://www.mindspore.cn/lite/docs/zh-CN/r2.3.1/use/downloads.html)，下面是一个参考安装指令。
  ```bash
  pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.1/MindSpore/lite/release/l
inux/aarch64/cloud_fusion/python310/mindspore_lite-2.3.1-cp310-cp310-linux_aarch64.whl
  ```


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
      --hf_model_dir="./download/Qwen2-0.5B-Instruct" \
      --onnx_model_path="./output/onnx/qwen2_0.5b_chat.onnx" \
      --kv_cache_length=1024
    ```
  - 对于CPU设备(CPU导出的东西仅做代码结构验证，推荐还是用NPU导出更好)
    ```bash
    python3 export/export_onnx.py \
      --device_str="cpu" \
      --dtype="float32" \
      --hf_model_dir="./download/Qwen2-0.5B-Instruct" \
      --onnx_model_path="./output/onnx/qwen2_0.5b_chat.onnx" \
      --kv_cache_length=1024
    ```

3. 验证onnx，需要分别运行pytorch和onnx，观察两边输出误差，若误差较小(最大误差低于3位小数，平均误差低于5位小数)，则说明onnx导出是ok的。
  - 先用cpu跑pytorch
    ```bash
    python3 export/test_pytorch_run.py \
      --device_str="cpu" \
      --dtype="float32" \
      --hf_model_dir="./download/Qwen2-0.5B-Instruct"
    ```
  - 再用cpu跑onnx
    ```bash
    python3 export/test_onnx_run.py \
      --dtype="float32" \
      --hf_model_dir="./download/Qwen2-0.5B-Instruct" \
      --onnx_model_path="./output/onnx/qwen2_0.5b_chat.onnx"
    ```
  - 也可以用npu开发板上面的cpu跑onnx（一般测试结构，统一用上面的cpu会好一些）
    ```bash
    python3 export/test_onnx_run.py \
      --dtype="float16" \
      --hf_model_dir="./download/Qwen2-0.5B-Instruct" \
      --onnx_model_path="./output/onnx/qwen2_0.5b_chat.onnx"
    ```
4. 测试使用onnx对话，用于验证onnx整体效果，若无明显乱码则说明正常。（注意：由于是CPU运行，所以速度较慢，请耐心等待）
  ```bash
  python3 ./cli_chat.py \
    --session_type=onnx \
    --dtype="float32" \
    --hf_model_dir="./download/Qwen2-0.5B-Instruct" \
    --onnx_model_path="./output/onnx/qwen2_0.5b_chat.onnx"
  ```

5. （可选？）改变onnx结构，目前导出的Trilu算子和Cast算子有些问题，atc命令无法识别，需要改一下结构。
  ```bash
  python3 export/change_node.py \
    --input_model_path="./output/onnx/qwen2_0.5b_chat.onnx" \
    --output_model_path="./output/onnx2/qwen2_0.5b_chat.onnx"
  ```

6. 将onnx转成MindSpore-Lite的文件（推荐在NPU开发板上面转，得到的ms模型更小。）
  ```bash
  python3 export/onnx2ms.py \
    --hf_model_dir="${PWD}/download/Qwen2-0.5B-Instruct" \
    --onnx_model_path="${PWD}/output/onnx/qwen2_0.5b_chat.onnx" \
    --ms_model_path="${PWD}/output/model/qwen2_0.5b_chat" \
    --save_type="mindir_lite" \
    --ms_optimize="general" \
    --kv_cache_length=1024
  ```


7. 测试使用mindspore-lite生成的模型文件对话（由于Mindspore主要服务端侧CPU、NPU，对于NPU开发板，默认启用CPU，所以还是比较慢）。
  ```bash
  python3 ./cli_chat.py \
    --session_type="ms_lite" \
    --dtype="float32" \
    --hf_model_dir="./download/Qwen2-0.5B-Instruct" \
    --ms_model_path="./output/model/qwen2_0.5b_chat.ms"
  ```


