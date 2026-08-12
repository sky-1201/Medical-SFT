"""
=============================================================================
  evaluate.py —— 模型效果评估（PPL + 综合报告）
=============================================================================

  用法:
    python evaluate.py --lora_weights ./output/sft
    python evaluate.py --lora_weights ./output/dpo
=============================================================================
"""
import os
import sys
import math
import random
import argparse
import torch
import logging
from pathlib import Path
from datetime import datetime
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
# 测试集
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

GENERAL_TEST_QUESTIONS = [
    "你好，今天天气真好",
    "请简单介绍一下中国",
    "我刚刚跟你说了什么？",
    "请用50字总结人工智能",
    "讲一个简短的笑话",
]

MULTITURN_TEST = [
    ["扭伤后应该怎么处理？", "抽筋呢？", "我第一个问题问的是什么？"],
    ["感冒了吃什么药？", "这个药有副作用吗？", "回到第一个问题，感冒了还能做什么？"],
]


# ============================================
# Step 1: 加载模型
# ============================================
def load_finetuned_model(
    base_model_name: str, lora_weights_path: str, device_map: str = "auto",
) -> Tuple[PeftModel, AutoTokenizer]:
    logger.info(f"加载基座模型: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype="auto", device_map=device_map,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    logger.info(f"加载 LoRA 权重: {lora_weights_path}")
    model = PeftModel.from_pretrained(base_model, lora_weights_path, torch_dtype="auto", device_map="auto")
    model.eval()
    return model, tokenizer


# ============================================
# Step 2: Perplexity（随机抽样）
# ============================================
def compute_perplexity(
    model, tokenizer, eval_data_path: str, max_seq_length: int = 512,
    max_samples: int = 500, system_prompt: str = "",
) -> float:
    from data.dataset import SFTDataset
    from torch.utils.data import DataLoader, Subset

    logger.info("计算 Perplexity...")
    if not Path(eval_data_path).exists():
        logger.error(f"数据不存在: {eval_data_path}")
        return float("inf")

    dataset = SFTDataset(eval_data_path, tokenizer, max_seq_length=max_seq_length, system_prompt=system_prompt)

    if len(dataset) > max_samples:
        random.seed(42)  # 固定种子，确保每次评估同一批数据
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = Subset(dataset, indices)
        logger.info(f"  从 {len(SFTDataset(eval_data_path, tokenizer).raw_data)} 条中固定抽取 {max_samples} 条（seed=42）")

    dataloader = DataLoader(dataset, batch_size=1)

    total_loss, total_tokens, processed = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            try:
                outputs = model(**batch)
                num_tokens = batch["attention_mask"].sum().item()
                total_loss += outputs.loss.item() * num_tokens
                total_tokens += num_tokens
            except Exception as e:
                logger.warning(f"  第 {processed} 条失败: {e}")
                continue
            processed += 1

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    logger.info(f"  Eval Loss: {avg_loss:.4f}, PPL: {ppl:.2f} ({processed} 样本)")
    return ppl


# ============================================
# Step 3: 生成回答
# ============================================
@torch.inference_mode()
def generate_response(
    model, tokenizer, messages: List[Dict], max_new_tokens: int = 512,
    temperature: float = 0.1, top_p: float = 0.9, repetition_penalty: float = 1.1,
) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, repetition_penalty=repetition_penalty,
        do_sample=(temperature > 0.0),
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ============================================
# Step 4: 生成完整评估报告
# ============================================
def generate_full_report(
    output_path: str,
    ppl_base: float, ppl_finetuned: float,
    medical_results: List[Dict],
    general_results: List[Tuple[str, str]],
    multiturn_results: List[Tuple[List[str], List[str]]],
    model_name: str = "Qwen2.5-7B-Instruct",
):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 医疗模型微调效果评估报告\n\n")
        f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**基座模型:** {model_name}  \n")
        f.write(f"**微调方法:** LoRA (rank=16) + SFT  \n\n")
        f.write("---\n\n")

        # 一、PPL
        f.write("## 一、Perplexity（困惑度）\n\n")
        improvement = (ppl_base - ppl_finetuned) / ppl_base * 100 if ppl_base != float("inf") else 0
        f.write("*PPL 越低表示模型对医疗文本越熟悉。评估时从验证集中随机抽取 500 条。*\n\n")
        f.write("| 指标 | 基座模型 | 微调后 | 提升 |\n")
        f.write("|------|------|------|------|\n")
        f.write(f"| PPL | {ppl_base:.2f} | {ppl_finetuned:.2f} | **{improvement:.1f}%** |\n")
        f.write("\n---\n\n")

        # 二、医疗专业问答对比
        f.write("## 二、医疗专业问答对比\n\n")
        for i, r in enumerate(medical_results):
            f.write(f"### Q{i+1}: {r['question']}\n\n")
            f.write(f"**基座模型:**\n\n> {r['base_model_answer'][:800]}\n\n")
            f.write(f"**微调后:**\n\n> {r['finetuned_model_answer'][:800]}\n\n")
            f.write("---\n\n")

        # 三、通用对话测试
        f.write("## 三、通用对话能力测试\n\n")
        f.write("*验证微调后是否保留了通用对话能力（灾难性遗忘检查）*\n\n")
        for q, a in general_results:
            f.write(f"**Q:** {q}\n\n> {a}\n\n---\n\n")

        # 四、多轮对话测试
        f.write("## 四、多轮对话测试\n\n")
        f.write("*验证微调后是否保留上下文记忆能力*\n\n")
        for turn_set, answers in multiturn_results:
            f.write(f"**对话:** {' → '.join(turn_set)}\n\n")
            for j, (turn, ans) in enumerate(zip(turn_set, answers)):
                f.write(f"**Q{j+1}:** {turn}\n\n> {ans}\n\n")
            f.write("---\n\n")

    logger.info(f"[OK] 评估报告: {output_path}")


# ============================================
# 主流程
# ============================================
def main():
    parser = argparse.ArgumentParser(description="医疗模型评估")
    parser.add_argument("--lora_weights", default="./output/sft", help="LoRA 路径")
    parser.add_argument("--base_model", default=None, help="基座模型（默认用 config）")
    parser.add_argument("--report", default="evaluation_report.md", help="报告输出路径")
    parser.add_argument("--skip-base", action="store_true", help="跳过基座模型对比")
    args = parser.parse_args()

    base_model_name = args.base_model or ModelCfg().model_name_or_path
    lora_path = args.lora_weights
    eval_data_path = "data/processed/eval.jsonl"
    system_prompt = (
        "你是一个专业的医疗健康助手，具备丰富的医学知识。"
        "请用专业但易懂的语言回答用户的健康问题。"
    )

    if not Path(lora_path).exists() or not Path(f"{lora_path}/adapter_model.safetensors").exists():
        logger.error(f"LoRA 不存在: {lora_path}")
        return

    print("=" * 60)
    print("医疗模型评估")
    print("=" * 60)

    # 加载微调模型
    model, tokenizer = load_finetuned_model(base_model_name, lora_path)

    # 加载基座模型
    base_model = None
    if not args.skip_base:
        logger.info("加载基座模型（未微调）用于对比...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype="auto", device_map="auto",
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        base_model.eval()

    # ========================================
    # Perplexity
    # ========================================
    print("\n" + "=" * 40)
    print("Perplexity 评估（验证集随机抽取 500 条）")
    print("=" * 40)

    ppl_finetuned = compute_perplexity(model, tokenizer, eval_data_path, system_prompt=system_prompt)
    ppl_base = compute_perplexity(base_model, tokenizer, eval_data_path, system_prompt=system_prompt) if base_model else float("inf")

    if ppl_base != float("inf") and ppl_finetuned != float("inf"):
        improvement = (ppl_base - ppl_finetuned) / ppl_base * 100
        print(f"\n  基座 PPL: {ppl_base:.2f}  →  微调 PPL: {ppl_finetuned:.2f}  →  提升 {improvement:.1f}%")

    # ========================================
    # 医疗问答对比
    # ========================================
    print("\n" + "=" * 40)
    print("医疗问答对比")
    print("=" * 40)

    medical_results = []
    for i, question in enumerate(MEDICAL_TEST_QUESTIONS):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        logger.info(f"  [{i+1}/{len(MEDICAL_TEST_QUESTIONS)}] {question[:50]}...")

        base_ans = ""
        if base_model:
            try:
                base_ans = generate_response(base_model, tokenizer, messages)
            except Exception as e:
                base_ans = f"[生成失败: {e}]"
        try:
            fine_ans = generate_response(model, tokenizer, messages)
        except Exception as e:
            fine_ans = f"[生成失败: {e}]"
        medical_results.append({
            "question": question,
            "base_model_answer": base_ans,
            "finetuned_model_answer": fine_ans,
        })

    # ========================================
    # 通用对话测试
    # ========================================
    print("\n" + "=" * 40)
    print("通用对话测试")
    print("=" * 40)
    general_results = []
    for q in GENERAL_TEST_QUESTIONS:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
        ans = generate_response(model, tokenizer, messages)
        general_results.append((q, ans))
        print(f"  Q: {q}")
        print(f"  A: {ans[:200]}\n")

    # ========================================
    # 多轮对话测试
    # ========================================
    print("=" * 40)
    print("多轮对话测试")
    print("=" * 40)
    multiturn_results = []
    for turn_set in MULTITURN_TEST:
        print(f"  对话: {' → '.join(turn_set)}")
        messages = [{"role": "system", "content": system_prompt}]
        answers = []
        for j, turn in enumerate(turn_set):
            messages.append({"role": "user", "content": turn})
            ans = generate_response(model, tokenizer, messages, temperature=0.3)
            messages.append({"role": "assistant", "content": ans})
            answers.append(ans)
            print(f"    Q{j+1}: {turn} → A: {ans[:200]}\n")
        multiturn_results.append((turn_set, answers))
        print(f"   {'─' * 56}\n")

    # ========================================
    # 生成报告
    # ========================================
    print("=" * 40)
    print("生成评估报告")
    generate_full_report(
        args.report, ppl_base, ppl_finetuned,
        medical_results, general_results, multiturn_results,
        model_name=base_model_name,
    )

    print(f"\n[OK] 评估完成")
    print(f"  PPL: {ppl_base:.2f} → {ppl_finetuned:.2f} ({improvement:.1f}%)")
    print(f"  报告: {args.report}")


if __name__ == "__main__":
    main()
