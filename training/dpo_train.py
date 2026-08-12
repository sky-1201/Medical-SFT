"""
=============================================================================
  training/dpo_train.py —— DPO 偏好对齐训练
=============================================================================
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from config.common import ModelConfig, LoraConfig as LoraCfg, DataConfig
from config.dpo import DPOConfig
from training.common import load_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_dpo_data(data_path: str, max_samples: int = None) -> List[Dict]:
    """加载 DPO 偏好对数据"""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                convs = item.get("conversations", [])
                rejected = item.get("rejected", "")
                if len(convs) >= 2 and rejected:
                    data.append(item)
            except json.JSONDecodeError:
                continue
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
    logger.info(f"加载 DPO 数据: {len(data)} 条偏好对")
    return data


def format_dpo_sample(item: Dict, tokenizer, system_prompt: str,
                       max_prompt_length: int = 512) -> Optional[Dict[str, torch.Tensor]]:
    """把一条偏好对转成 DPOTrainer 需要的 prompt/chosen/rejected 格式"""
    convs = item.get("conversations", [])
    rejected = item.get("rejected", "")
    if len(convs) < 2:
        return None

    user_msgs = [c for c in convs if c.get("from") in ["human", "user"]]
    assistant_msgs = [c for c in convs if c.get("from") in ["gpt", "assistant"]]
    if not user_msgs or not assistant_msgs:
        return None

    user_content = user_msgs[-1].get("value", "")
    chosen_content = assistant_msgs[-1].get("value", "")
    if not user_content or not chosen_content or not rejected:
        return None

    prompt_msgs = []
    if system_prompt:
        prompt_msgs.append({"role": "system", "content": system_prompt})
    prompt_msgs.append({"role": "user", "content": user_content})
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

    return {
        "prompt": prompt_text,
        "chosen": chosen_content + tokenizer.eos_token,
        "rejected": rejected + tokenizer.eos_token,
    }


def prepare_dpo_dataset(data: List[Dict], tokenizer, system_prompt: str,
                         max_prompt_length: int = 512) -> List[Dict]:
    """全部偏好对 → DPOTrainer 格式"""
    formatted = [format_dpo_sample(it, tokenizer, system_prompt, max_prompt_length) for it in data]
    formatted = [f for f in formatted if f is not None]
    logger.info(f"有效 DPO 样本: {len(formatted)} / {len(data)}")
    return formatted


def run():
    """DPO 训练主流程"""
    print("=" * 60)
    print("DPO 偏好对齐训练")
    print("=" * 60)

    model_cfg = ModelConfig()
    lora_cfg = LoraCfg()
    dpo_cfg = DPOConfig()
    data_cfg = DataConfig()

    # Step 1: Tokenizer
    print("\n[Step 1/5] Tokenizer")
    tokenizer = load_tokenizer(model_cfg.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Step 2: 加载 SFT 模型
    print("\n[Step 2/5] 加载 SFT 模型")
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    sft_path = dpo_cfg.sft_lora_path
    if not Path(sft_path).exists():
        print(f"[ERROR] SFT 模型不存在: {sft_path}")
        print("  请先: python train.py --stage sft")
        return

    import os as _os
    _device_map = "auto" if int(_os.getenv("WORLD_SIZE", "1")) <= 1 else None
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path, torch_dtype="auto", device_map=_device_map, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, sft_path, is_trainable=True)
    logger.info("SFT LoRA 已加载")

    # Step 3: 加载 DPO 数据
    print("\n[Step 3/5] 加载 DPO 数据")
    if not Path(dpo_cfg.dpo_data_path).exists():
        print(f"[ERROR] DPO 数据不存在: {dpo_cfg.dpo_data_path}")
        print("  python scripts/download_medical_data.py --subset reward --num 5000")
        return

    raw_data = load_dpo_data(dpo_cfg.dpo_data_path)
    dpo_dataset = prepare_dpo_dataset(raw_data, tokenizer, data_cfg.system_prompt, dpo_cfg.max_prompt_length)
    if len(dpo_dataset) < 10:
        print(f"[ERROR] DPO 数据太少 ({len(dpo_dataset)} 条)")
        return

    # Step 4: DPO 训练
    print("\n[Step 4/5] DPO 训练")
    from trl import DPOTrainer, DPOConfig as TRLDPOConfig

    dpo_training_args = TRLDPOConfig(
        output_dir=dpo_cfg.output_dir,
        num_train_epochs=dpo_cfg.num_train_epochs,
        per_device_train_batch_size=dpo_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=dpo_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        learning_rate=dpo_cfg.learning_rate, warmup_ratio=dpo_cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=dpo_cfg.bf16, fp16=dpo_cfg.fp16,
        gradient_checkpointing=dpo_cfg.gradient_checkpointing,
        logging_steps=dpo_cfg.logging_steps,
        save_steps=dpo_cfg.save_steps, eval_steps=dpo_cfg.eval_steps,
        save_total_limit=dpo_cfg.save_total_limit,
        max_length=dpo_cfg.max_seq_length,
        beta=dpo_cfg.dpo_beta, loss_type=dpo_cfg.dpo_loss_type,
        deepspeed=dpo_cfg.deepspeed_config if Path(dpo_cfg.deepspeed_config).exists() else None,
        remove_unused_columns=False, report_to=[], seed=42,
    )

    # ref_model: 冻结的 SFT 模型作为参考
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path, torch_dtype="auto", device_map=_device_map, trust_remote_code=True)
    ref_model = PeftModel.from_pretrained(ref_model, sft_path)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    dpo_trainer = DPOTrainer(
        model=model, ref_model=ref_model, args=dpo_training_args,
        train_dataset=dpo_dataset, processing_class=tokenizer,
    )
    logger.info(f"  DPO 样本: {len(dpo_dataset)}, beta={dpo_cfg.dpo_beta}")
    dpo_trainer.train()

    # Step 5: 保存
    print("\n[Step 5/5] 保存 DPO 模型")
    tokenizer.padding_side = "left"
    tokenizer.init_kwargs["padding_side"] = "left"
    model.save_pretrained(dpo_cfg.output_dir)
    tokenizer.save_pretrained(dpo_cfg.output_dir)

    adapter = Path(dpo_cfg.output_dir) / "adapter_model.safetensors"
    if adapter.exists():
        logger.info(f"  adapter: {adapter.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\n[OK] DPO 完成 → {dpo_cfg.output_dir}")
