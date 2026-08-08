"""
=============================================================================
  training/common.py —— SFT / DPO 共用的工具函数
=============================================================================
"""
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

logger = logging.getLogger(__name__)


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """加载 tokenizer 并修补缺失的特殊 token"""
    logger.info(f"加载 Tokenizer: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False,
    )

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

    logger.info(f"  vocab_size={tokenizer.vocab_size}, pad={tokenizer.pad_token_id}, "
                f"eos={tokenizer.eos_token_id}")
    return tokenizer


def load_model_with_lora(model_name_or_path: str, lora_cfg,
                         torch_dtype: str = "auto", device_map: str = "auto") -> PeftModel:
    """加载预训练模型 + 注入 LoRA"""
    logger.info(f"加载模型: {model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, device_map=device_map,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    logger.info(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.lora_r, lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules), bias="none",
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  可训练参数: {trainable/1e6:.2f}M / {total/1e9:.2f}B ({trainable/total*100:.2f}%)")

    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    return model
