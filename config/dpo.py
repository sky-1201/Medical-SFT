"""
=============================================================================
  config/dpo.py —— DPO 训练配置
=============================================================================
"""
from dataclasses import dataclass


@dataclass
class DPOConfig:
    # 输出
    output_dir: str = "./output/dpo"
    sft_lora_path: str = "./output/sft"              # 从 SFT 训练好的 LoRA 出发

    # 数据
    dpo_data_path: str = "data/medical_reward_5000.jsonl"

    # DPO 核心
    dpo_beta: float = 0.1                            # KL 惩罚：越小越激进
    dpo_loss_type: str = "sigmoid"

    # 训练（比 SFT 小，因为偏好对数据少）
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2              # chosen+rejected 双份
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    max_prompt_length: int = 512

    learning_rate: float = 5e-5                       # DPO 比 SFT 低
    warmup_ratio: float = 0.1

    # 显存
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True

    # 分布式
    deepspeed_config: str = "configs/deepspeed_zero2.json"

    # 日志
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
