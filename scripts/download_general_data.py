"""
=============================================================================
  download_general_data.py —— 下载中文通用对话数据集
=============================================================================

  用法:
    python scripts/download_general_data.py                # 默认 Belle，2 万条
    python scripts/download_general_data.py --num 30000    # 3 万条
    python scripts/download_general_data.py --source belle  # 指定 Belle
    python scripts/download_general_data.py --source alpaca # 换 alpaca-zh

  数据集:
    Belle:  100 万条中文对话，覆盖聊天/写作/代码/翻译
    alpaca: 5 万条中文指令，质量高但量少

  来源:
    Belle:  https://huggingface.co/datasets/BelleGroup/train_1M_CN
    alpaca: https://huggingface.co/datasets/shibing624/alpaca-zh
=============================================================================
"""
import argparse
import json
import os
import sys
from pathlib import Path


DATASETS = {
    "belle": {
        "name": "BelleGroup/train_1M_CN",
        "subset": None,
        "desc": "BELLE 100万条中文对话",
    },
    "alpaca": {
        "name": "shibing624/alpaca-zh",
        "subset": None,
        "desc": "Alpaca 中文翻译版 5万条",
    },
}


def convert_belle(example):
    """Belle: instruction + output → conversations"""
    instruction = example.get("instruction", "")
    output = example.get("output", "")

    # 有些 Belle 数据有 input 字段
    user_input = example.get("input", "")
    if user_input:
        user_value = f"{instruction}\n{user_input}" if instruction else user_input
    else:
        user_value = instruction

    if not user_value or not output:
        return None

    return {
        "conversations": [
            {"from": "human", "value": user_value},
            {"from": "gpt", "value": output},
        ]
    }


def convert_alpaca(example):
    """Alpaca: instruction + input + output → conversations"""
    instruction = example.get("instruction", "")
    user_input = example.get("input", "")
    output = example.get("output", "")

    if user_input:
        user_value = f"{instruction}\n{user_input}" if instruction else user_input
    else:
        user_value = instruction

    if not user_value or not output:
        return None

    return {
        "conversations": [
            {"from": "human", "value": user_value},
            {"from": "gpt", "value": output},
        ]
    }


CONVERTERS = {
    "belle": convert_belle,
    "alpaca": convert_alpaca,
}


def main():
    parser = argparse.ArgumentParser(description="下载中文通用对话数据集")
    parser.add_argument("--source", type=str, default="belle", choices=["belle", "alpaca"])
    parser.add_argument("--num", type=int, default=20000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    if not args.no_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[INFO] 使用 HF 国内镜像")

    print("[1/3] 导入依赖...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 请先安装: pip install datasets")
        sys.exit(1)

    ds_info = DATASETS[args.source]
    converter = CONVERTERS[args.source]

    print(f"[2/3] 下载 {ds_info['desc']}...")

    if ds_info["subset"]:
        ds = load_dataset(ds_info["name"], ds_info["subset"], split="train", trust_remote_code=True)
    else:
        ds = load_dataset(ds_info["name"], split="train", trust_remote_code=True)

    total = len(ds)
    num = min(args.num, total)
    print(f"  总数据量: {total}, 选取 {num} 条")

    output_path = args.output or f"data/general_{args.source}_{num}.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[3/3] 格式转换 + 保存...")
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(num):
            example = ds[i]
            item = converter(example)
            if item is None:
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] 保存完成: {output_path}")
    print(f"  条数: {count}")
    print(f"  大小: {file_size:.1f} MB")


if __name__ == "__main__":
    main()
