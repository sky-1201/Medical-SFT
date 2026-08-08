from .common import ModelConfig, LoraConfig, DataConfig
from .sft import SFTConfig
from .dpo import DPOConfig

# 向后兼容旧名称
TrainingConfig = SFTConfig

