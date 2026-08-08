"""
=============================================================================
  mix_data.py —— 混合多份数据 + shuffle
=============================================================================

  用法:
    python scripts/mix_data.py data/medical_80k.jsonl data/general_20k.jsonl
    python scripts/mix_data.py medical.jsonl general.jsonl --output data/mixed.jsonl
    python scripts/mix_data.py medical.jsonl general.jsonl --split  # 同时划分 train/eval

=============================================================================
"""
import argparse
import json
import random
from pathlib import Path


def load_jsonl(file_path: str) -> list:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"  加载 {Path(file_path).name}: {len(data)} 条")
    return data


def save_jsonl(data: list, file_path: str):
    """保存为 JSONL"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="混合多份训练数据")
    parser.add_argument("files", nargs="+", help="要混合的 JSONL 文件")
    parser.add_argument("--output", type=str, default="data/mixed_train.jsonl",
                        help="输出文件（默认 data/mixed_train.jsonl）")
    parser.add_argument("--split", action="store_true",
                        help="同时划分 train/eval（5% 验证集）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载所有数据
    all_data = []
    for f in args.files:
        if not Path(f).exists():
            print(f"[WARN] 文件不存在，跳过: {f}")
            continue
        all_data.extend(load_jsonl(f))

    print(f"\n合计: {len(all_data)} 条")

    # Shuffle
    random.shuffle(all_data)
    print(f"随机打乱完成（seed={args.seed}）")

    if args.split:
        split_idx = int(len(all_data) * (1 - args.eval_ratio))
        train_data = all_data[:split_idx]
        eval_data = all_data[split_idx:]

        train_path = args.output.rsplit(".", 1)[0] + ".jsonl" if not args.output.endswith(".jsonl") else args.output
        eval_path = train_path.rsplit("/", 1)[0] + "/eval.jsonl" if "/" in train_path else "eval.jsonl"

        save_jsonl(train_data, train_path)
        save_jsonl(eval_data, eval_path)
        print(f"\n[OK] 混合完成:")
        print(f"  训练集: {train_path} ({len(train_data)} 条)")
        print(f"  验证集: {eval_path} ({len(eval_data)} 条)")
    else:
        save_jsonl(all_data, args.output)
        print(f"\n[OK] 混合完成: {args.output} ({len(all_data)} 条)")


if __name__ == "__main__":
    main()
