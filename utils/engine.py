import numpy as np
import mindspore_lite as mslite
import os
import sys
from typing import List

now_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(now_dir)
sys.path.append(parent_dir)


from config import InferenceConfig


class MindSporeLiteModel:
    def __init__(self, config: InferenceConfig):
        self.model = None
        self.build_model(config.ms_model_path, cpu_support_fp16=config.cpu_support_fp16)

    def build_model(self, ms_model_path: str, cpu_support_fp16=False, support_ascend=True):
        """
        编译模型
        Args:
            ms_model_path (str): _description_
        """
        # 先创建上下文
        print("=== create context ===")
        context = mslite.Context()
        if support_ascend:
            context.target = ["cpu", "ascend"]
            context.ascend.precision_mode = "enforce_origin"
        else:
            context.target = ["cpu"]
        if cpu_support_fp16:    
            print("cpu support float16")
            context.cpu.precision_mode = "preferred_fp16"
        print("=== create model ===")
        self.model = mslite.Model()
        self.model.build_from_file(
            ms_model_path,
            mslite.ModelType.MINDIR_LITE,
            context
        )
        print("=== create model ok ===")

    def inference(self, input_data_list: List[np.ndarray], seq_length=1, is_dynamic=False) -> List[np.ndarray]:
        """
        执行推理，同步方式
        Args:
            input_data_list (_type_): _description_
            seq_length: 推理长度

        Returns:
            List[np.ndarray]: _description_
        """
        inputs = self.model.get_inputs()
        # new_shape_list = []
        for i in range(len(inputs)):
            # print(i, " ==> shape ", input_data_list[i].shape)
            # temp_shape = input_data_list[i].shape
            # if (i < len(inputs) - 1):
            #     temp_shape = list(temp_shape)
            #     temp_shape[3] = temp_shape[3] + 1
            # new_shape_list.append(temp_shape)
            inputs[i].set_data_from_numpy(input_data_list[i])
        # self.model.resize(inputs, new_shape_list)
        outputs: List[mslite.Tensor] = self.model.predict(inputs)
        output_data_list = []
        for output in outputs:
            output_data = output.get_data_to_numpy()
            output_data_list.append(output_data)
        return output_data_list


if __name__ == "__main__":
    # 测试一下模型编译
    import os
    
    conf = InferenceConfig(
        hf_model_dir=os.path.join(
            parent_dir,
            "download",
            "Qwen2-0.5B-Instruct"
        ),
        ms_model_path=os.path.join(
            parent_dir,
            "output",
            "model",
            "qwen2_0.5b_chat.ms"
        ),
        onnx_model_path="",
    )
    engine = MindSporeLiteModel(config=conf)