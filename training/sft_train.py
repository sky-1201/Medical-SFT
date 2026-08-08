"""
=============================================================================
  training/sft_train.py —— SFT 监督微调训练
=============================================================================
"""
import os
import math
import logging
from pathlib import Path

from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

from config.common import ModelConfig, LoraConfig as LoraCfg, DataConfig
from config.sft import SFTConfig
from training.common import load_tokenizer, load_model_with_lora

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_data(tokenizer, train_cfg: SFTConfig, data_cfg: DataConfig):
    """加载训练和验证数据"""
    from data.dataset import SFTDataset

    logger.info("加载数据...")
    train_dataset = SFTDataset(
        data_path=train_cfg.train_data_path, tokenizer=tokenizer,
        max_seq_length=train_cfg.max_seq_length, system_prompt=data_cfg.system_prompt,
    )
    eval_dataset = SFTDataset(
        data_path=train_cfg.eval_data_path, tokenizer=tokenizer,
        max_seq_length=train_cfg.max_seq_length, system_prompt=data_cfg.system_prompt,
    )
    logger.info(f"  训练集: {len(train_dataset)} 条, 验证集: {len(eval_dataset)} 条")
    return train_dataset, eval_dataset


def train(model, tokenizer, train_dataset, eval_dataset, train_cfg: SFTConfig):
    """配置 Trainer 并开始训练"""
    logger.info("配置训练参数...")

    training_args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        fp16=train_cfg.fp16, bf16=train_cfg.bf16,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps, eval_steps=train_cfg.eval_steps,
        save_total_limit=train_cfg.save_total_limit,
        eval_strategy="steps", load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        deepspeed=train_cfg.deepspeed_config if Path(train_cfg.deepspeed_config).exists() else None,
        report_to=[], remove_unused_columns=False, dataloader_num_workers=4, seed=42,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator,
    )

    total_steps = (len(train_dataset) //
                   (train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps)
                   * train_cfg.num_train_epochs)
    logger.info(f"  数据: {len(train_dataset)} train / {len(eval_dataset)} eval")
    logger.info(f"  总步数≈{total_steps}, 有效batch={train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps}")

    trainer.train()
    return trainer


def save_model(model, tokenizer, output_dir: str):
    """保存 LoRA 权重"""
    logger.info(f"保存模型到: {output_dir}")
    tokenizer.padding_side = "left"
    tokenizer.init_kwargs["padding_side"] = "left"
    model.config.use_cache = True
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    adapter = Path(output_dir) / "adapter_model.safetensors"
    logger.info(f"  adapter: {adapter.stat().st_size / 1024 / 1024:.1f} MB" if adapter.exists() else "  [WARN] adapter 未找到")


def run():
    """SFT 训练主流程"""
    print("=" * 60)
    print("SFT 监督微调")
    print("=" * 60)

    model_cfg = ModelConfig()
    lora_cfg = LoraCfg()
    train_cfg = SFTConfig()
    data_cfg = DataConfig()

    if not Path(train_cfg.train_data_path).exists():
        print(f"\n[ERROR] 数据不存在: {train_cfg.train_data_path}")
        print("  请先: python data/preprocess.py")
        return

    # Step 1-2: 加载 tokenizer + 模型
    tokenizer = load_tokenizer(model_cfg.model_name_or_path)
    # DeepSpeed 分布式时不能设 device_map，DeepSpeed 自己管显卡
    device_map = "auto" if int(os.getenv("WORLD_SIZE", "1")) <= 1 else None
    model = load_model_with_lora(model_cfg.model_name_or_path, lora_cfg,
                                 torch_dtype=model_cfg.torch_dtype, device_map=device_map)

    # Step 3: 数据
    train_dataset, eval_dataset = load_data(tokenizer, train_cfg, data_cfg)

    # Step 4: 训练
    trainer = train(model, tokenizer, train_dataset, eval_dataset, train_cfg)

    # Step 5: 保存
    save_model(model, tokenizer, train_cfg.output_dir)

    # 评估
    eval_metrics = trainer.evaluate()
    try:
        ppl = math.exp(eval_metrics["eval_loss"])
        print(f"\n  eval_loss={eval_metrics['eval_loss']:.4f}, PPL={ppl:.2f}")
    except OverflowError:
        print(f"\n  eval_loss={eval_metrics['eval_loss']:.4f}")

    print(f"\n[OK] SFT 完成 → {train_cfg.output_dir}")
