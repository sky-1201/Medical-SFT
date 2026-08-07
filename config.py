"""
=============================================================================
  config.py —— 所有训练参数集中管理
=============================================================================
  作用：替代 LLaMA-Factory 的 YAML 配置文件，让你精确控制每一个参数

  TODO-Day1: 理解每个参数的含义
  TODO-Day3: 配置 LoRA 相关参数 (lora_r, lora_alpha 等)
  TODO-Day5: 根据训练效果调整参数
=============================================================================
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """模型相关配置"""

    # [已补齐] Day1: 医疗微调使用 Qwen2.5-1.5B-Instruct
    #   1.5B 模型显存占用 ~3GB，单张8GB显卡即可训练（LoRA后更省）
    #   如果显存充裕（>16GB），可以换成 Qwen/Qwen2.5-7B-Instruct
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # [已补齐] Day1: torch_dtype 和 device_map
    #   torch_dtype="auto" → HuggingFace 自动选最优精度（bf16 > fp16 > fp32）
    #   device_map="auto"  → accelerate 自动把模型层分配到 GPU/CPU
    torch_dtype: str = "auto"
    device_map: str = "auto"

    # [已补齐] Day1: 使用 tokenizer 内置的 chat_template（Qwen 用的是 ChatML 格式）
    #   Qwen 的对话格式: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n
    use_chat_template: bool = True


@dataclass
class LoraConfig:
    """
    LoRA (Low-Rank Adaptation) 配置

    [已补齐] Day3: 核心概念 —— LoRA 在原始权重矩阵 W 旁边加一个低秩矩阵 ΔW = B·A
        - 原始 W: (d × d) 全量参数，比如 4096 × 4096 = 16M 个参数
        - A: (d × r), B: (r × d)，参数总量 = 2·d·r
        - r=8 时: 2 × 4096 × 8 ≈ 65K，只有原来的 0.4%
        - 训练时冻结 W，只更新 A 和 B → 显存暴降，训练加速

    [已补齐] Day3: 面试必考题 —— 为什么 target_modules 通常选 Q 和 V？
        因为 Attention 中 Q 决定"查什么"、V 决定"取什么"，
        两者对语义理解影响最大。K 和 O 的影响相对小。
        实测：Q+V 即可达到 Q+K+V+O 的 95% 效果，参数省一半。
    """

    # [已补齐] 1.5B 模型用 rank=8，7B 模型用 rank=16
    lora_r: int = 8

    # [已补齐] alpha = 2 × r，这是业界经验值
    lora_alpha: int = 16

    # [已补齐] dropout 防止 LoRA 层过拟合，0.05 是常用值
    lora_dropout: float = 0.05

    # [已补齐] Qwen2 的 target_modules：
    #   查看方法: python -c "from transformers import AutoModel; m=AutoModel.from_pretrained('Qwen/Qwen2.5-1.5B'); [print(n) for n,_ in m.named_modules() if 'Linear' in str(type(_))]"
    target_modules: tuple = (
        "q_proj", "v_proj", "k_proj", "o_proj",     # 注意力四件套
        "gate_proj", "up_proj", "down_proj",          # FFN 三层
    )


@dataclass
class TrainingConfig:
    """训练相关配置"""

    # [已补齐] Day2: 数据路径 —— 指向预处理后的文件
    train_data_path: str = "data/processed/train.jsonl"
    eval_data_path: str = "data/processed/eval.jsonl"

    # [已补齐] 最大序列长度 —— 1000条数据以短问答为主，512 足够且省显存
    #   长对话场景（多轮问诊）可以开到 1024 或 2048
    max_seq_length: int = 512                       # [GPU] 先保持512，后面数据多了再调大

    # [已补齐] Day4: batch_size 和 gradient_accumulation 的关系
    #   有效 batch = per_device_batch × gradient_accumulation × GPU数
    #   2 × 4 × 1 = 8 条/step，适合 1000 条数据
    per_device_train_batch_size: int = 4            # [GPU] 显存充裕，开大加速训练
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # [已补齐] 1000 条数据，epoch=5 保证模型充分学习（数据量小可以多跑几轮）
    num_train_epochs: int = 5

    # [已补齐] Day4: 学习率 —— LoRA 微调通常用 1e-4 ~ 5e-4
    #   比全量微调（1e-5）大一个数量级，因为只更新少量参数
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # [已补齐] Day5: 显存优化
    gradient_checkpointing: bool = True    # 用时间换显存
    fp16: bool = False
    bf16: bool = True                      # [GPU] 4090 支持 bf16，加速+省显存

    # 日志与保存
    logging_steps: int = 10
    save_steps: int = 200                             # 必须被 eval_steps 整除，否则报错
    eval_steps: int = 200                  # [已补齐] 1000条数据，200步评估一次更合理
    save_total_limit: int = 3

    # 输出
    output_dir: str = "./output"


@dataclass
class DataConfig:
    """
    数据构造配置

    [已补齐] Day2: 医疗微调使用 conversations 格式
        你的数据格式:
        {
            "conversations": [
                {"from": "human", "value": "头疼怎么办？"},
                {"from": "gpt", "value": "头疼的原因有..."}
            ]
        }

        系统内部会转成 HuggingFace messages 格式（见 dataset.py）
    """

    # [已补齐] 数据格式：你的是 conversations（human/gpt），dataset.py 会自动检测
    data_format: str = "conversations"

    # [已补齐] 医疗助手的 system prompt
    system_prompt: str = (
        "你是一个专业的医疗健康助手，具备丰富的医学知识。"
        "请用专业但易懂的语言回答用户的健康问题。"
        "如果用户描述的症状可能很严重，请建议就医。"
    )


# ============================================
# [已补齐] Day1: 运行这个文件确认配置加载成功
#   python config.py
# ============================================
if __name__ == "__main__":
    model_cfg = ModelConfig()
    lora_cfg = LoraConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()

    print("[OK] 配置加载成功！")
    print(f"   模型: {model_cfg.model_name_or_path}")
    print(f"   LoRA rank: {lora_cfg.lora_r}, alpha: {lora_cfg.lora_alpha}")
    print(f"   训练轮数: {train_cfg.num_train_epochs}")
    print(f"   最大序列长度: {train_cfg.max_seq_length}")
    print(f"   有效 batch size: {train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps}")
    print(f"   输出目录: {train_cfg.output_dir}")
    print(f"   数据格式: {data_cfg.data_format}")
    print(f"   System Prompt: {data_cfg.system_prompt[:50]}...")
