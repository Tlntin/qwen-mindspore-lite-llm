### 说明
- 本项目基于之前的[qwen-ascend-llm](https://github.com/Tlntin/qwen-ascend-llm)项目魔改而来，基本和之前相差不大，只是模型推理从CANN变成了mindspore-lite。数据格式需要从ND换成NCHW(华为手机上面的mindspore-lite只支持NCHW格式)。
- 在做鸿蒙Next开发时，发现mindspore-lite版本号要求必须为2.1.0。
- 根据[mindspore-lite 官方文档](https://www.mindspore.cn/lite/docs/zh-CN/r2.1/use/downloads.html#2-1-0), mindspore_lite提供的编译好的的python版本仅为python3.7，所以用2.1.0的mindspore-lite C库做转换，2.1.1的mindspore-lite python库做验证。

### 准备工作
1. 下载本项目
  ```bash
  git clone https://github.com/Tlntin/qwen-mindspore-lite-llm.git
  ```

2. 下载qwen1.5/qwen2的模型，选择chat模型或者instruct模型，将其放到download文件夹，仅支持huggingface下载的模型，网络不好的可以用镜像站：https://hf-mirror.com/Qwen
3. 配置MIndSpore-Lite环境
  - 对于x86-64 Linux设备
  ```bash
  wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.1.0/MindSpore/lite/release/linux/x86_64/cloud_fusion/python37/mindspore-lite-2.1.0-linux-x64.tar.gz
  tar -zxvf mindspore-lite-2.1.0-linux-x64.tar.gz
  sudo mv mindspore-lite-2.1.0-linux-x64 /usr/local/
  sudo ln -s /usr/local/mindspore-lite-2.1.0-linux-x64 /usr/local/mindspore-lite
  ```
  - 对于AArch64 Linux设备
  ```bash
  wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.1.0/MindSpore/lite/release/linux/aarch64/cloud_fusion/python37/mindspore-lite-2.1.0-linux-aarch64.tar.gz
  tar -zxvf mindspore-lite-2.1.0-linux-aarch64.tar.gz
  sudo mv mindspore-lite-2.1.0-linux-aarch64 /usr/local/
  sudo ln -s /usr/local/mindspore-lite-2.1.0-linux-aarch64 /usr/local/mindspore-lite
  ```
  - 配置环境，将下面的内容写到`~/.bashrc`或者`~/.zshrc`中
  ```bash
  export LITE_HOME=/usr/local/mindspore-lite
  export PATH=$PATH:${LITE_HOME}/tools/converter/converter:${LITE_HOME}/tools/benchmark
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LITE_HOME}/tools/converter/lib:${LITE_HOME}/runtime/lib:${LITE_HOME}/runtime/third_party/dnnl
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LITE_HOME}/runtime/third_party/glog:${LITE_HOME}/runtime/third_party/libjpeg-turbo/lib:${LITE_HOME}/runtime/third_party/securec
  ```
4. 从[官方文档](https://www.mindspore.cn/lite/docs/zh-CN/r2.3.1/use/downloads.html#2-1-0)来看，mindspore_lite 2.1.1对应的python开发包最高3.9，所以可以构建一个python3.9虚拟环境。
  ```bash
  conda create -n ms_lite python=3.9
  conda activate ms_lite
  ```
5. 安装mindspore-lite 2.1.1
  - 对于x86-64 Linux 设备
  ```bash
  pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.1.1/MindSpore/lite/release/linux/x86_64/cloud_fusion/python39/mindspore_lite-2.1.1-cp39-cp39-linux_x86_64.whl
  ```

  - 对于aarch64 linux设备
  ```bash
  pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.1.1/MindSpore/lite/release/linux/aarch64/cloud_fusion/python39/mindspore_lite-2.1.1-cp39-cp39-linux_aarch64.whl
  ```
6. 安装其它python依赖（用于后续模型结构转onnx，以及onnx验证）。
  ```bash
  cd qwen-mindspore-lite-llm
  pip install -r ./requirements.txt
  ```
7. (可选)安装torch-npu，用于导出fp16的模型（如果你的CPU支持fp16或者你有昇腾NPU设备）
  ```bash
  wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc2-pytorch2.1.0/torch_npu-2.1.0.post6-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl 
  pip install ./torch_npu-2.1.0.post6-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
  ```

### 详细运行步骤
1. 导出onnx，默认kv-cache长度为1024，可以根据自己的内存、显存来设置更大参数。
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

2. 验证onnx，需要分别运行pytorch和onnx，观察两边输出误差，若误差较小(最大误差低于3位小数，平均误差低于5位小数)，则说明onnx导出是ok的。
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
3. 测试使用onnx对话，用于验证onnx整体效果，若无明显乱码则说明正常。（注意：由于是CPU运行，所以速度较慢，请耐心等待）
  ```bash
  python3 ./cli_chat.py \
    --session_type=onnx \
    --dtype="float32" \
    --hf_model_dir="./download/Qwen2-0.5B-Instruct" \
    --onnx_model_path="./output/onnx/qwen2_0.5b_chat.onnx"
  ```


4. 将onnx转成MindSpore-Lite的文件（推荐在NPU开发板上面转，得到的ms模型更小。）
  ```bash
  python3 export/onnx2ms.py \
    --hf_model_dir="${PWD}/download/Qwen2-0.5B-Instruct" \
    --onnx_model_path="${PWD}/output/onnx/qwen2_0.5b_chat.onnx" \
    --ms_model_path="${PWD}/output/model/qwen2_0.5b_chat" \
    --save_type="mindir_lite" \
    --ms_optimize="general" \
    --kv_cache_length=1024
  ```


5. 测试使用mindspore-lite生成的模型文件对话（由于Mindspore主要服务端侧CPU、NPU，对于NPU开发板，默认启用CPU，所以还是比较慢）。
  ```bash
  python3 ./cli_chat.py \
    --session_type="ms_lite" \
    --dtype="float32" \
    --hf_model_dir="./download/Qwen2-0.5B-Instruct" \
    --ms_model_path="./output/model/qwen2_0.5b_chat.ms"
  ```


