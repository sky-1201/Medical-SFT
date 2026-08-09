"""
=============================================================================
  download_medical_data.py —— 下载 shibing624/medical 中文医疗数据集
=============================================================================

  用法:
    python scripts/download_medical_data.py                     # 默认: finetune 子集 3 万条
    python scripts/download_medical_data.py --subset zh         # zh 子集（195万条综合数据）
    python scripts/download_medical_data.py --subset finetune   # finetune 子集（指令微调）
    python scripts/download_medical_data.py --subset pretrain   # pretrain 子集（预训练）
    python scripts/download_medical_data.py --subset reward     # reward 子集（奖励模型）
    python scripts/download_medical_data.py --num 50000         # 下载 5 万条
    python scripts/download_medical_data.py --num all           # 下载全部
    python scripts/download_medical_data.py --subset finetune --num 50000


  四个子集说明:
    finetune    → SFT 微调用，instruction/input/output 格式，最适合你
    zh          → 综合子集（195万条），覆盖广泛但质量参差不齐
    pretrain    → 预训练用，纯文本，不需要你关注
    reward      → 奖励模型用，chosen/rejected 偏好对，做 DPO 时用

  来源: https://huggingface.co/datasets/shibing624/medical
=============================================================================
"""
import argparse
import json
import os
import sys
from pathlib import Path

# 各子集 → 输出字段 → 你项目 conversations 格式的映射
SUBSET_MAPPING = {
    "finetune": {
        "instruction": "human",
        "input": "human",         # input 可为空，拼到 instruction 后面
        "output": "gpt",
    },
    "zh": {
        "instruction": "human",
        "output": "gpt",
    },
    "pretrain": {
        "text": "gpt",  # 预训练纯文本，直接作为 assistant 内容
    },
    "reward": {
        "question": "human",
        "response_chosen": "gpt",  # 只取优质回答做 SFT
    },
}


def convert_finetune(example):
    """finetune 子集：instruction + input + output"""
    instruction = example["instruction"]
    user_input = example.get("input", "")
    if user_input:
        user_value = f"{instruction}\n{user_input}" if instruction else user_input
    else:
        user_value = instruction
    return {
        "conversations": [
            {"from": "human", "value": user_value},
            {"from": "gpt", "value": example["output"]},
        ]
    }


def convert_zh(example):
    """zh 子集：instruction + output"""
    return {
        "conversations": [
            {"from": "human", "value": example["instruction"]},
            {"from": "gpt", "value": example["output"]},
        ]
    }


def convert_reward(example):
    """reward 子集：chosen 放 conversations，rejected 单独存，供 DPO 使用"""
    return {
        "conversations": [
            {"from": "human", "value": example["question"]},
            {"from": "gpt", "value": example["response_chosen"]},
        ],
        "rejected": example["response_rejected"],
    }


def convert_pretrain(example):
    """pretrain 子集：纯文本"""
    return {
        "conversations": [
            {"from": "gpt", "value": example["text"]},
        ]
    }


CONVERTERS = {
    "finetune": convert_finetune,
    "zh": convert_zh,
    "pretrain": convert_pretrain,
    "reward": convert_reward,
}


def main():
    parser = argparse.ArgumentParser(description="下载中文医疗SFT数据集")
    parser.add_argument("--subset", type=str, default="finetune",
                        choices=["zh", "finetune", "pretrain", "reward"],
                        help="数据集子集: zh(综合) / finetune(指令微调,推荐) / pretrain(预训练) / reward(奖励模型)")
    parser.add_argument("--num", type=str, default="30000",
                        help="下载条数，默认30000。传 'all' 下载全部")
    parser.add_argument("--shuffle", action="store_true",
                        help="随机采样而不是取前N条（195万数据建议开启）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（默认: data/medical_{subset}_{num}.jsonl）")
    parser.add_argument("--no-mirror", action="store_true",
                        help="不使用 HF 镜像")
    args = parser.parse_args()

    # 国内镜像加速
    if not args.no_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[INFO] 使用 HF 国内镜像: https://hf-mirror.com")

    print("[1/3] 导入依赖...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 请先安装: pip install datasets")
        sys.exit(1)

    subset = args.subset
    converter = CONVERTERS[subset]

    print(f"[2/3] 下载 shibing624/medical ({subset}) 数据集...")
    ds = load_dataset("shibing624/medical", subset, split="train", trust_remote_code=True)
    total = len(ds)
    print(f"  总数据量: {total} 条")

    if args.num == "all":
        num_samples = total
    else:
        num_samples = min(int(args.num), total)

    print(f"  选取 {num_samples} 条")

    # 自动生成输出路径
    output_path = args.output or f"data/medical_{subset}_{num_samples}.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[3/3] 格式转换 → conversations (human/gpt) + 保存...")
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed).select(range(num_samples))
        print(f"  随机采样 (seed={args.seed})")
    else:
        ds = ds.select(range(num_samples))
        print(f"  顺序取前 {num_samples} 条")
    ds = ds.map(converter, remove_columns=ds.column_names)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] 保存完成: {output_path}")
    print(f"  子集: {subset}")
    print(f"  条数: {count}")
    print(f"  大小: {file_size:.1f} MB")
    print(f"  下一步: python data/preprocess.py")


if __name__ == "__main__":
    main()
