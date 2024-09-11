import os
from transformers.models.qwen2 import Qwen2Config, Qwen2Tokenizer


class InferenceConfig:
    def __init__(
        self,
        hf_model_dir: str,
        ms_model_path: str,
        onnx_model_path: str,
        session_type: str = "ms_lite", # 支持ms_lite和onnx两种，ms_lite即MindSpore-Lite
        device_id: int = 0,
        sampling_method: str = "top_p", # 支持 greedy, top_p, top_k
        sampling_value: float = 0.8,
        temperature: float = 0.7,
        max_batch: int = 1,
        max_input_length: int = 512, # 输入长度的最大数值
        max_output_length: int = 1024, # 输出长度的最大值
        max_prefill_length: int = 1, # prefile阶段，单次最大推理长度
        kvcache_method: str = "fixsize", # kv_cache类型，支持basic,fixsize,streamllm,H2O四种，具体可以去kvcache.py查看
        kv_cache_length: int = 1024, # kvcache的最大长度
        cache_format: str = 'huggingface-tensor', # kv_cache的格式
        dtype:str="float16",
    ):
        self.tokenizer_dir = hf_model_dir
        self.session_type = session_type
        if self.session_type == "ms_lite":
            assert os.path.exists(ms_model_path), print(ms_model_path, "not exists")
        elif self.session_type == "onnx":
            assert os.path.exists(onnx_model_path), print(onnx_model_path, "not exists")
        self.ms_model_path = ms_model_path
        self.onnx_model_path = onnx_model_path
        self.device_id = device_id
        self.sampling_method = sampling_method
        self.sampling_value = sampling_value
        self.temperature = temperature
        self.max_batch = max_batch
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.kvcache_method = kvcache_method
        self.kv_cache_length = kv_cache_length  # max_cache_size
        self.cache_format = cache_format
        self.dtype = dtype
        self.model_config = Qwen2Config.from_pretrained(hf_model_dir)
        self.num_hidden_layers = self.model_config.num_hidden_layers # n_layer
        self.num_key_value_heads = self.model_config.num_key_value_heads # head_num
        self.hidden_size = self.model_config.hidden_size # hidden_dim
        self.num_attention_heads = self.model_config.num_attention_heads
        self.per_head_dim = self.hidden_size // self.num_attention_heads # head_dim
        # 下面的kv-cache shape要改成NCHW格式的
        self.past_key_value_shape = (
            self.max_batch,
            self.num_hidden_layers * 2 * self.num_key_value_heads,
            self.kv_cache_length,
            self.per_head_dim
        )
        self.max_prefill_length = max_prefill_length
        self.vocab_size = self.model_config.vocab_size
