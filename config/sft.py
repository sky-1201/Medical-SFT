"""
=============================================================================
  config/sft.py —— SFT 训练配置
=============================================================================
"""
from dataclasses import dataclass


@dataclass
class SFTConfig:
    # 输出
    output_dir: str = "./output/sft"

    # 数据
    train_data_path: str = "data/processed/train.jsonl"
    eval_data_path: str = "data/processed/eval.jsonl"

    # 训练
    num_train_epochs: int = 2                        # 10万条数据跑3轮效果更好
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1            # effective batch = 16×1×4卡 = 64
    max_seq_length: int = 1024                       # A800 80G 完全够，不截断长回答

    learning_rate: float = 3e-4                       # batch 32→64，lr 按比例上调
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # 显存
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True

    # 分布式
    deepspeed_config: str = "configs/deepspeed_zero2.json"

    # 日志
    logging_steps: int = 200
    save_steps: int = 1500
    eval_steps: int = 1500
    save_total_limit: int = 3
