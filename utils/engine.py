import numpy as np
import mindspore_lite as mslite
from config import InferenceConfig

class MindSporeLiteModel:
    def __init__(self, config: InferenceConfig):
        self.model = None
        self.build_model(config.ms_model_path)

    def build_model(self, ms_model_path: str):
        """
        编译模型
        Args:
            ms_model_path (str): _description_
        """
        # 先创建上下文
        print("=== create context ===")
        context = mslite.Context()
        context.target = ["cpu", "ascend"]
        context.cpu.precision_mode = "preferred_fp16"
        context.ascend.precision_mode = "preferred_optimal"
        print("=== create model ===")
        self.model = mslite.Model()
        self.model.build_from_file(
            ms_model_path,
            mslite.ModelType.MINDIR_LITE, context
        )


if __name__ == "__main__":
    # 测试一下模型编译
    import os
    now_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(now_dir)
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