"""
=============================================================================
  train.py —— 模型微调主脚本
=============================================================================

  这是整个项目的核心。在这里你会串联起所有学到的知识。

  流程图:
    config.py (参数)
      ↓
    AutoModelForCausalLM.from_pretrained()  ← 加载预训练模型
      ↓
    get_peft_model(model, lora_config)      ← 注入 LoRA
      ↓
    SFTDataset(data_path, tokenizer)        ← 加载数据
      ↓
    Trainer / 手写训练循环                   ← 训练！
      ↓
    model.save_pretrained(output_dir)       ← 保存 LoRA 权重
    tokenizer.save_pretrained(output_dir)   ← 保存 tokenizer

  学习策略:
    Day1-3: 用 HuggingFace Trainer（封装好的训练器，类似 LLaMA-Factory）
    Day4-5: 手写训练循环（真正理解 backward/optimizer/scheduler）

=============================================================================
"""
import os
import sys
import math
import torch
import logging
from pathlib import Path

# 把项目根目录加入 path，方便导入自己的模块
sys.path.insert(0, str(Path(__file__).parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from config import ModelConfig, LoraConfig as LoraCfg, TrainingConfig, DataConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================
# Step 1: 加载 Tokenizer
# ============================================
def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """
    [已补齐] Day1: 加载 tokenizer 并修补缺失的特殊 token

    关键概念:
    - pad_token: 批处理时填充短句用的 token
    - eos_token: 句子结束符
    - bos_token: 句子开始符
    - chat_template: 对话格式模板（Qwen 用的是 ChatML 格式）

    很多开源模型（如 Qwen2）的 tokenizer 没有 pad_token 和 bos_token，
    训练前必须补上，否则会报错。
    """
    logger.info(f"加载 Tokenizer: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,       # Qwen 需要这个
        use_fast=False,               # Qwen 的 fast tokenizer 有时有 bug
    )

    # [已补齐] 检查并修补特殊 token（借鉴 MedicalGPT supervised_finetuning.py:377-393）
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "</s>"
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"  补 eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"  补 bos_token_id: {tokenizer.bos_token_id} (复用 eos_token_id)")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        logger.info(f"  补 pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    logger.info(f"  Tokenizer 加载完成")
    logger.info(f"    vocab_size: {tokenizer.vocab_size}")
    logger.info(f"    pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    logger.info(f"    eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    logger.info(f"    bos_token_id: {tokenizer.bos_token_id}")
    logger.info(f"    chat_template: {'已配置' if tokenizer.chat_template else '未配置!'}")

    return tokenizer


# ============================================
# Step 2: 加载模型 + 注入 LoRA
# ============================================
def load_model_with_lora(
    model_name_or_path: str,
    lora_cfg: LoraCfg,
    torch_dtype: str = "auto",
    device_map: str = "auto",
) -> PeftModel:
    """
    [已补齐] Day1 + Day3: 加载预训练模型并注入 LoRA

    关键概念:
    - from_pretrained: 从 HuggingFace Hub 或本地加载模型权重
    - torch_dtype: 权重的数据类型
      - float32: 最精确但最慢最吃显存
      - float16: 快但数值范围小，容易溢出
      - bfloat16: 快且范围大，精度稍低但够用（推荐，需 GPU 支持）
      - auto: HuggingFace 自动选最优
    - device_map="auto": accelerate 库自动把模型分配到可用的 GPU/CPU

    LoRA 注入后的变化:
    - 原始模型参数冻结（requires_grad=False）
    - 只在 target_modules 上加可训练的 LoRA 层（A 和 B 矩阵）
    - 可训练参数通常只有原始模型的 0.5%-2%
    """
    logger.info(f"加载模型: {model_name_or_path}")

    # [已补齐] Day1: 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,          # [已补齐] 减少 CPU 内存占用
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  模型参数量: {total_params / 1e9:.2f}B")

    # [已补齐] Day3: 注入 LoRA
    logger.info("注入 LoRA...")
    logger.info(f"  rank={lora_cfg.lora_r}, alpha={lora_cfg.lora_alpha}, "
                f"dropout={lora_cfg.lora_dropout}")
    logger.info(f"  target_modules: {lora_cfg.target_modules}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.lora_r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
        bias="none",                     # 不训练 bias
    )

    model = get_peft_model(model, peft_config)

    # [已补齐] 打印 LoRA 注入后的参数统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  可训练参数: {trainable/1e6:.2f}M / {total/1e9:.2f}B "
                f"({trainable/total*100:.2f}%)")

    # [已补齐] 把 LoRA 可训练参数转为 float32（保证训练精度）
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)
    logger.info("  LoRA 参数已转为 float32")

    return model


# ============================================
# Step 3: 加载数据
# ============================================
def load_data(tokenizer, train_cfg: TrainingConfig, data_cfg: DataConfig):
    """
    [已补齐] Day2: 加载训练和验证数据（从 data/dataset.py 导入 SFTDataset）
    """
    from data.dataset import SFTDataset

    logger.info("加载数据...")

    train_dataset = SFTDataset(
        data_path=train_cfg.train_data_path,
        tokenizer=tokenizer,
        max_seq_length=train_cfg.max_seq_length,
        system_prompt=data_cfg.system_prompt,
    )

    eval_dataset = SFTDataset(
        data_path=train_cfg.eval_data_path,
        tokenizer=tokenizer,
        max_seq_length=train_cfg.max_seq_length,
        system_prompt=data_cfg.system_prompt,
    )

    logger.info(f"  训练集: {len(train_dataset)} 条")
    logger.info(f"  验证集: {len(eval_dataset)} 条")

    return train_dataset, eval_dataset


# ============================================
# Step 4: 训练！(使用 HuggingFace Trainer)
# ============================================
def train(model, tokenizer, train_dataset, eval_dataset, train_cfg: TrainingConfig):
    """
    [已补齐] Day4: 配置 Trainer 并开始训练

    HuggingFace Trainer 内部等价于:

        for epoch in range(num_epochs):
            for batch in dataloader:
                outputs = model(**batch)                   # forward
                loss = outputs.loss / accumulation_steps   # 梯度累积
                loss.backward()                            # backward
                if step % accumulation_steps == 0:
                    optimizer.step()                       # 更新参数
                    scheduler.step()                       # 更新学习率
                    optimizer.zero_grad()                  # 清空梯度

    详见 train_manual.py 的手写版本。
    """
    logger.info("配置训练参数...")

    training_args = TrainingArguments(
        # 输出
        output_dir=train_cfg.output_dir,

        # 训练
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,

        # 精度
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        # [已补齐] CPU训练：关闭gradient_checkpointing（只在GPU上有意义）
        gradient_checkpointing=False,

        # 日志与保存
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        eval_steps=train_cfg.eval_steps,
        save_total_limit=train_cfg.save_total_limit,

        # 评估
        eval_strategy="steps",                   # 按步数评估
        load_best_model_at_end=True,             # 训练完自动加载最佳模型
        metric_for_best_model="eval_loss",       # 用验证集 loss 选最佳

        # 其他
        report_to="tensorboard",                 # [GPU] 开 tensorboard 看 loss 曲线
        remove_unused_columns=False,             # 保留 labels 列
        dataloader_num_workers=2,                # [GPU] 恢复多进程加载
        seed=42,
    )

    # [已补齐] DataCollatorForSeq2Seq: labels 的 pad 位置自动填 -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,                 # ★ 关键参数
        pad_to_multiple_of=8,                    # padding 到 8 的倍数（硬件对齐）
    )

    # [已补齐] 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # [已补齐] 计算总步数并打印训练配置
    total_steps_approx = (
        len(train_dataset)
        // (train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps)
        * train_cfg.num_train_epochs
    )
    logger.info(f"开始训练...")
    logger.info(f"  数据量: {len(train_dataset)} 条训练 / {len(eval_dataset)} 条验证")
    logger.info(f"  Epoch: {train_cfg.num_train_epochs}")
    logger.info(f"  总步数(约): {total_steps_approx}")
    logger.info(f"  有效 batch size: {train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps}")
    logger.info(f"  学习率: {train_cfg.learning_rate}")

    # 训练
    trainer.train()

    # 训练完检查 loss 趋势:
    #   train_loss ↓ eval_loss ↓ → 正常，可以继续训
    #   train_loss ↓ eval_loss ↑ → 过拟合！加 lora_dropout 或减少 epoch
    #   train_loss → eval_loss → → 学习率太小或数据有问题

    return trainer


# ============================================
# Step 5: 保存模型
# ============================================
def save_model(model, tokenizer, output_dir: str):
    """
    [已补齐] Day3: 保存微调后的模型

    注意：保存的是 LoRA adapter（~30MB），不是完整模型（~3GB）。
    推理时需要先加载 base model，再用 PeftModel.from_pretrained() 挂载 LoRA。

    如果想合并成完整模型：
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir)
    """
    logger.info(f"保存模型到: {output_dir}")

    # [已补齐] 推理前把 padding_side 切回 left
    tokenizer.padding_side = "left"
    tokenizer.init_kwargs["padding_side"] = "left"

    # [已补齐] 重新启用 KV Cache（训练时因 gradient_checkpointing 关闭了）
    model.config.use_cache = True

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("[OK] 模型保存完成")
    logger.info(f"  LoRA 权重: {output_dir}/adapter_model.safetensors")
    logger.info(f"  LoRA 配置: {output_dir}/adapter_config.json")
    logger.info(f"  Tokenizer:  {output_dir}/tokenizer_config.json")

    # [已补齐] 估算文件大小
    adapter_path = Path(output_dir) / "adapter_model.safetensors"
    if adapter_path.exists():
        size_mb = adapter_path.stat().st_size / (1024 * 1024)
        logger.info(f"  adapter 大小: {size_mb:.1f} MB")


# ============================================
# 主函数
# ============================================
def main():
    """
    [已补齐] 完整的训练主流程

    Day1: 只跑 Step1（加载模型），确认环境和显存
    Day2: 加上数据加载
    Day3: 加上 LoRA 注入
    Day4: 开始训练！
    """
    print("=" * 60)
    print("医疗模型 SFT 微调")
    print("=" * 60)

    # [已补齐] 加载配置
    model_cfg = ModelConfig()
    lora_cfg = LoraCfg()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()

    # 检查数据文件是否存在
    if not Path(train_cfg.train_data_path).exists():
        print(f"\n[ERROR] 训练数据文件不存在: {train_cfg.train_data_path}")
        print("  请先运行: python data/preprocess.py")
        return

    # Step 1: Tokenizer
    print("\n" + "=" * 40)
    print("Step 1/5: 加载 Tokenizer")
    print("=" * 40)
    tokenizer = load_tokenizer(model_cfg.model_name_or_path)

    # Step 2: 模型 + LoRA
    print("\n" + "=" * 40)
    print("Step 2/5: 加载模型 + 注入 LoRA")
    print("=" * 40)

    model = load_model_with_lora(
        model_cfg.model_name_or_path,
        lora_cfg,
        torch_dtype=model_cfg.torch_dtype,
        device_map=model_cfg.device_map,
    )

    # Step 3: 数据
    print("\n" + "=" * 40)
    print("Step 3/5: 加载数据")
    print("=" * 40)
    train_dataset, eval_dataset = load_data(tokenizer, train_cfg, data_cfg)

    # Step 4: 训练
    print("\n" + "=" * 40)
    print("Step 4/5: 开始训练")
    print("=" * 40)
    trainer = train(model, tokenizer, train_dataset, eval_dataset, train_cfg)

    # Step 5: 保存
    print("\n" + "=" * 40)
    print("Step 5/5: 保存模型")
    print("=" * 40)
    save_model(model, tokenizer, train_cfg.output_dir)

    # [已补齐] 评估
    eval_metrics = trainer.evaluate()
    try:
        perplexity = math.exp(eval_metrics["eval_loss"])
        print(f"\n  验证集 Loss: {eval_metrics['eval_loss']:.4f}")
        print(f"  Perplexity:   {perplexity:.2f}")
    except OverflowError:
        print(f"\n  验证集 Loss: {eval_metrics['eval_loss']:.4f}")

    print(f"\n[OK] 训练完成！")
    print(f"   模型保存在: {train_cfg.output_dir}")
    print(f"   下一步: python inference.py 体验对话")
    print(f"   评估: python evaluate.py")


if __name__ == "__main__":
    main()
