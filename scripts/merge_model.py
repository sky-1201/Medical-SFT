"""
=============================================================================
  merge_model.py —— 合并 LoRA 到 base model，用于部署
=============================================================================

  用法:
    python scripts/merge_model.py                          # 默认合并 DPO
    python scripts/merge_model.py --lora ./output/sft      # 合并 SFT

  输出: ./merged-model（完整的 14GB 模型，推理不需要 peft）
=============================================================================
"""
import argparse
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 到 base model")
    parser.add_argument("--lora", default="./output/dpo", help="LoRA 权重路径")
    parser.add_argument("--base_model", default=None,
                        help="base model 路径（默认用 config）")
    parser.add_argument("--output", default="./merged-model", help="输出路径")
    args = parser.parse_args()

    # base model 路径
    if args.base_model:
        base_model_path = args.base_model
    else:
        from config.common import ModelConfig
        base_model_path = ModelConfig().model_name_or_path

    lora_path = Path(args.lora)
    if not (lora_path / "adapter_model.safetensors").exists():
        print(f"[ERROR] LoRA 不存在: {lora_path}")
        return

    print("=" * 60)
    print("合并 LoRA 模型")
    print("=" * 60)
    print(f"  base model: {base_model_path}")
    print(f"  LoRA:       {lora_path}")
    print(f"  输出:       {args.output}")

    # 1. 加载 base model
    print("\n[1/3] 加载 base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # 2. 加载 LoRA 并合并
    print(f"[2/3] 加载 LoRA 并合并...")
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    merged = model.merge_and_unload()
    print("  合并完成")

    # 3. 保存
    print(f"[3/3] 保存到 {args.output}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    merged.save_pretrained(args.output, max_shard_size="5GB")
    tokenizer.save_pretrained(args.output)

    # 输出文件大小
    total_size = sum(f.stat().st_size for f in Path(args.output).glob("*.safetensors"))
    print(f"\n[OK] 合并完成: {args.output}")
    print(f"  模型大小: {total_size / 1024**3:.1f} GB")
    print(f"  下一步: vllm serve {args.output} --port 8000")


if __name__ == "__main__":
    main()
