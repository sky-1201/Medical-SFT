"""
=============================================================================
  evaluate.py —— 模型效果评估
=============================================================================

  [已补齐] Day6: "怎么证明你的微调有效？"

  评估维度：
  1. Perplexity (PPL): 模型对文本的"惊讶程度"，越低越好
  2. 生成样本对比：微调前后的模型回答同一个医疗问题，直观对比效果
  3. 查看训练 loss 曲线: tensorboard --logdir ./output

  借鉴：MedicalGPT supervised_finetuning.py:980-996 + demo/inference.py:72-122
=============================================================================
"""
import os
import sys
import math
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config.common import ModelConfig as ModelCfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================
# Step 1: 加载微调后的模型
# ============================================
def load_finetuned_model(
    base_model_name: str,
    lora_weights_path: str,
    device_map: str = "auto",
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    [已补齐] Day6: 加载微调后的模型（base model + LoRA weights）

    注意加载顺序：
    1. 先加载 tokenizer（需要 padding_side="left" 用于生成）
    2. 再加载 base model
    3. 最后用 PeftModel.from_pretrained 挂 LoRA
    """
    logger.info(f"加载基座模型: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="left",             # [已补齐] 评估用左 padding
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    logger.info(f"加载 LoRA 权重: {lora_weights_path}")
    model = PeftModel.from_pretrained(
        base_model, lora_weights_path,
        torch_dtype="auto", device_map="auto",
    )
    model.eval()

    return model, tokenizer


# ============================================
# Step 2: 计算 Perplexity
# ============================================
def compute_perplexity(
    model,
    tokenizer,
    eval_data_path: str,
    max_seq_length: int = 512,
    max_samples: int = 200,
    system_prompt: str = "",
) -> float:
    """
    [已补齐] Day6: 计算 Perplexity (困惑度)

    PPL = exp(cross_entropy_loss)

    直观理解:
      PPL=10  → 模型在每个 token 上平均在 10 个选项中犹豫 → 熟悉
      PPL=100 → 模型平均在 100 个选项中犹豫 → 不熟悉

    微调后 PPL 应该显著低于微调前。

    借鉴 MedicalGPT 的结构，但用我们自己 dataset.py 的 SFTDataset 做数据加载。
    """
    from data.dataset import SFTDataset
    from torch.utils.data import DataLoader

    logger.info("计算 Perplexity...")

    if not Path(eval_data_path).exists():
        logger.error(f"评估数据文件不存在: {eval_data_path}")
        return float("inf")

    dataset = SFTDataset(
        data_path=eval_data_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        system_prompt=system_prompt,
    )

    # [已补齐] 单条评估（batch_size=1 避免 padding 影响 PPL 计算）
    dataloader = DataLoader(dataset, batch_size=1)

    total_loss = 0.0
    total_tokens = 0
    processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if processed >= max_samples:
                break

            batch = {k: v.to(model.device) for k, v in batch.items()}

            # [已补齐] 前向传播获取 loss
            try:
                outputs = model(**batch)
                loss = outputs.loss.item()

                # [已补齐] 乘以实际的 token 数（排除 padding）
                num_tokens = batch["attention_mask"].sum().item()
                total_loss += loss * num_tokens
                total_tokens += num_tokens
            except Exception as e:
                logger.warning(f"  第 {processed} 条计算失败: {e}")
                continue

            processed += 1

    if total_tokens == 0:
        logger.error("没有成功计算任何 token 的 loss")
        return float("inf")

    avg_loss = total_loss / total_tokens
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")

    logger.info(f"  Eval Loss: {avg_loss:.4f}")
    logger.info(f"  Perplexity: {ppl:.2f}")
    logger.info(f"  评估样本数: {processed}, 总 tokens: {int(total_tokens)}")

    return ppl


# ============================================
# Step 3: 生成回答
# ============================================
@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    messages: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """
    [已补齐] Day6: 用模型生成回答

    关键参数:
    - temperature=0.1: 评估用低温度（确定性高，便于对比）
    - temperature=0.7: 对话用高温度（多样性高）
    - top_p=0.9: nucleus sampling，从累积概率 90% 的 token 中采样
    - repetition_penalty=1.1: >1.0 惩罚重复 token，防止复读
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=(temperature > 0.0),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # [已补齐] 只取生成部分（去掉 prompt）
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================
# Step 4: 微调前 vs 微调后对比
# ============================================
def side_by_side_compare(
    base_model,
    finetuned_model,
    tokenizer,
    test_questions: List[str],
    system_prompt: str,
    output_path: str = "evaluation_report.md",
) -> List[Dict]:
    """
    [已补齐] Day6: 并排对比 —— 微调前后的模型回答同一个问题

    这是面试时最直观展示微调效果的方式。
    选 5-10 个有代表性的医疗问题，对比两边回答的质量。
    """
    results = []

    for i, question in enumerate(test_questions):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        logger.info(f"  [{i+1}/{len(test_questions)}] {question[:50]}...")

        # [已补齐] 微调前的回答
        try:
            base_answer = generate_response(base_model, tokenizer, messages)
        except Exception as e:
            base_answer = f"[生成失败: {e}]"

        # [已补齐] 微调后的回答
        try:
            finetuned_answer = generate_response(finetuned_model, tokenizer, messages)
        except Exception as e:
            finetuned_answer = f"[生成失败: {e}]"

        results.append({
            "question": question,
            "base_model_answer": base_answer,
            "finetuned_model_answer": finetuned_answer,
        })

    # [已补齐] 生成 Markdown 报告
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 医疗模型微调效果评估报告\n\n")
        f.write(f"**基座模型:** Qwen2.5-1.5B-Instruct  \n")
        f.write(f"**测试问题数:** {len(results)}  \n\n")
        f.write("---\n\n")

        for i, r in enumerate(results):
            f.write(f"### Q{i+1}: {r['question']}\n\n")
            f.write(f"**微调前（基座模型）:**\n\n> {r['base_model_answer']}\n\n")
            f.write(f"**微调后（医疗 LoRA）:**\n\n> {r['finetuned_model_answer']}\n\n")
            f.write("---\n\n")

    logger.info(f"评估报告已保存: {output_path}")
    return results


# ============================================
# [已补齐] 医疗测试问题（专业问答 + 多轮对话）
# ============================================
MEDICAL_TEST_QUESTIONS = [
    "我最近频繁头痛，可能是哪些原因导致的？需要做什么检查？",
    "高血压患者日常饮食要注意什么？",
    "什么是2型糖尿病？和1型糖尿病有什么区别？",
    "小孩发烧到39度，应该怎么处理？什么时候需要去医院？",
    "颈椎病有哪些症状？如何缓解？",
    "感冒和流感有什么区别？怎么判断自己得的是哪种？",
    "胃疼怎么办？有哪些常见原因？",
    "轻度抑郁症有哪些早期信号？应该寻求什么帮助？",
]

# [升级] 通用对话测试（验证灾难性遗忘是否解决）
GENERAL_TEST_QUESTIONS = [
    "你好，今天天气真好",
    "请简单介绍一下中国",
    "我刚刚跟你说了什么？",
    "请用50字总结人工智能",
    "讲一个简短的笑话",
]

# [升级] 多轮对话测试（验证上下文记忆能力）
MULTITURN_TEST = [
    ["扭伤后应该怎么处理？", "抽筋呢？", "我第一个问题问的是什么？"],
    ["感冒了吃什么药？", "这个药有副作用吗？", "回到第一个问题，感冒了还能做什么？"],
]


def main():
    print("=" * 60)
    print("医疗模型评估")
    print("=" * 60)

    # [已补齐] 配置
    base_model_name = ModelCfg().model_name_or_path
    # [升级] 支持 SFT 或 DPO 模型路径
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_weights", default="./output/sft", help="LoRA 路径")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--skip-base", action="store_true", help="跳过基座模型对比（省显存）")
    args_eval, _ = parser.parse_known_args()

    lora_weights_path = args_eval.lora_weights
    if args_eval.base_model:
        base_model_name = args_eval.base_model
    eval_data_path = "data/processed/eval.jsonl"
    system_prompt = (
        "你是一个专业的医疗健康助手，具备丰富的医学知识。"
        "请用专业但易懂的语言回答用户的健康问题。"
    )

    # [已补齐] 检查 LoRA 权重是否存在
    if not Path(lora_weights_path).exists() or not Path(f"{lora_weights_path}/adapter_model.safetensors").exists():
        logger.error(f"LoRA 权重不存在: {lora_weights_path}")
        logger.error("请先运行 python train.py 完成训练")
        return

    # 加载微调后的模型
    model, tokenizer = load_finetuned_model(base_model_name, lora_weights_path)

    # [已补齐] 加载 base model（未微调）用于对比
    logger.info("加载基座模型（未微调）用于对比...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    # [已补齐] 计算 Perplexity
    print("\n" + "=" * 40)
    print("Perplexity 评估")
    print("=" * 40)

    ppl_finetuned = compute_perplexity(
        model, tokenizer, eval_data_path,
        system_prompt=system_prompt,
    )

    ppl_base = compute_perplexity(
        base_model, tokenizer, eval_data_path,
        system_prompt=system_prompt,
    )

    if ppl_base != float("inf") and ppl_finetuned != float("inf"):
        improvement = (ppl_base - ppl_finetuned) / ppl_base * 100
        print(f"\n  Base model PPL:     {ppl_base:.2f}")
        print(f"  微调后 PPL:         {ppl_finetuned:.2f}")
        print(f"  提升:               {improvement:.1f}%")

    # [已补齐] 生成对比
    print("\n" + "=" * 40)
    print("微调前后生成对比")
    print("=" * 40)

    results = side_by_side_compare(
        base_model, model, tokenizer,
        MEDICAL_TEST_QUESTIONS, system_prompt,
        "evaluation_report.md",
    )

    # [已补齐] 打印摘要
    print(f"\n[OK] 评估完成！")
    print(f"  Perplexity（微调后）: {ppl_finetuned:.2f}")
    print(f"  对比报告: evaluation_report.md")

    # [已补齐] 快速预览
    print(f"\n{'='*60}")
    print(f"快速预览（前 3 个医疗问题）")
    print(f"{'='*60}")
    for i, r in enumerate(results[:3]):
        print(f"\nQ{i+1}: {r['question'][:60]}...")
        print(f"  微调前: {r['base_model_answer'][:120]}...")
        print(f"  微调后: {r['finetuned_model_answer'][:120]}...")

    # [升级] 通用对话测试（检查灾难性遗忘）
    print(f"\n{'='*60}")
    print(f"通用对话能力测试")
    print(f"{'='*60}")
    for q in GENERAL_TEST_QUESTIONS:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        ans = generate_response(model, tokenizer, messages)
        print(f"  Q: {q}")
        print(f"  A: {ans[:200]}\n")

    # [升级] 多轮对话测试
    print(f"{'='*60}")
    print(f"多轮对话测试")
    print(f"{'='*60}")
    for turn_set in MULTITURN_TEST:
        print(f"  对话: {' -> '.join(turn_set)}")
        messages = [{"role": "system", "content": system_prompt}]
        for j, turn in enumerate(turn_set):
            messages.append({"role": "user", "content": turn})
            ans = generate_response(model, tokenizer, messages, temperature=0.3)
            messages.append({"role": "assistant", "content": ans})
            print(f"    Q{j+1}: {turn}")
            print(f"    A{j+1}: {ans[:200]}\n")
        print(f"   {'-' * 56}\n")


if __name__ == "__main__":
    main()
