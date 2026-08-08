"""
=============================================================================
  config/common.py —— Model / LoRA / Data 共用配置
=============================================================================
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    use_chat_template: bool = True


@dataclass
class LoraConfig:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = (
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class DataConfig:
    data_format: str = "conversations"
    medical_ratio: float = 0.8
    general_ratio: float = 0.2
    system_prompt: str = (
        "你是一个专业的医疗健康助手，具备丰富的医学知识。"
        "请用专业但易懂的语言回答用户的健康问题。"
        "如果用户描述的症状可能很严重，请建议就医。"
    )
