"""
=============================================================================
  preprocess.py —— 数据预处理与质量检查
=============================================================================

  [已补齐] Day2+Day6: 完整的数据预处理流水线

  你在实习中做的"清洗数据"和"泛化训练语句"，本质上就是这里的逻辑。
  区别是：这里你写代码实现，而不是手动在 Excel 里挑。

  常见的数据问题：
    1. 重复数据 —— 模型会过拟合到重复样本
    2. 截断数据 —— output 被 max_length 裁掉一半
    3. 格式混乱 —— 奇怪的符号、换行、编码问题
    4. 质量低 —— answer 太短或包含"无法回答"等拒绝模板
    5. 分布不均 —— 某个类型的 QA 太多，导致模型偏科

=============================================================================
"""
import json
import hashlib
import re
from typing import List, Dict, Set, Tuple
from pathlib import Path


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件，自动跳过空行和解析失败的行"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] 第 {line_num} 行 JSON 解析失败: {e}")
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """保存为 JSONL 文件（每行一个 JSON）"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def deduplicate(data: List[Dict], key_field: str = "conversations") -> List[Dict]:
    """
    [已补齐] Day2: MD5 去重

    跟你在 RAG 项目里做的 MD5 去重原理一样，但是去的是训练数据。
    重复的训练数据会让模型在这一条上过拟合。

    原理：对每条数据的 conversations 字段做 MD5 哈希，相同哈希 = 重复数据。
    """
    seen: Set[str] = set()
    unique_data: List[Dict] = []

    for item in data:
        content = json.dumps(
            item.get(key_field, item), ensure_ascii=False, sort_keys=True
        )
        h = hashlib.md5(content.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_data.append(item)

    removed = len(data) - len(unique_data)
    if removed > 0:
        print(f"  去重: 移除 {removed} 条重复数据 ({removed/len(data)*100:.1f}%)")
    else:
        print(f"  去重: 无重复数据")

    return unique_data


# [已补齐] 拒绝回答的模板 —— 这些回答在训练中没有价值
REJECT_PATTERNS = [
    r"你的提问过于简单",
    r"无法给你满意的答复",
    r"抱歉.*无法",
    r"作为AI.*无法",
    r"请提供更.*信息",
    r"建议.*就医",         # 这个保留——医疗场景下建议就医是合理回答
]


def check_quality(data: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """
    [已补齐] Day6: 数据质量检查

    检查项：
    1. 数据格式是否正确（必须有 conversations 字段）
    2. 每条数据至少有一轮 human+gpt 对话
    3. gpt 回答不能太短（< 5 个字 = 训练价值低）
    4. gpt 回答不能包含拒绝模板
    5. human 提问不能太短（< 2 个字 = 无法构成有效问题）
    """
    stats = {
        "total": len(data),
        "filtered": 0,
        "bad_format": 0,
        "answer_too_short": 0,
        "question_too_short": 0,
        "reject_answer": 0,
    }

    clean_data = []
    for item in data:
        # 检查1：格式正确
        if "conversations" not in item or not isinstance(item["conversations"], list):
            stats["bad_format"] += 1
            stats["filtered"] += 1
            continue

        convs = item["conversations"]
        if len(convs) < 2:
            stats["bad_format"] += 1
            stats["filtered"] += 1
            continue

        # 检查2：最后一条 gpt 回答的长度
        gpt_values = [c["value"] for c in convs if c.get("from") == "gpt"]
        if gpt_values:
            last_answer = gpt_values[-1]
            if len(last_answer) < 5:       # [已补齐] 不到5个字，训练价值低
                stats["answer_too_short"] += 1
                stats["filtered"] += 1
                continue

            # 检查3：是否包含拒绝模板
            is_reject = False
            for pattern in REJECT_PATTERNS[:2]:  # 只检查前两个（最明确的拒绝模板）
                if re.search(pattern, last_answer):
                    is_reject = True
                    break
            if is_reject:
                stats["reject_answer"] += 1
                stats["filtered"] += 1
                continue

        # 检查4：human 提问的长度
        human_values = [c["value"] for c in convs if c.get("from") == "human"]
        if human_values:
            last_question = human_values[-1]
            if len(last_question) < 2:     # [已补齐] 不到2个字，太短
                stats["question_too_short"] += 1
                stats["filtered"] += 1
                continue

        clean_data.append(item)

    return clean_data, stats


def split_train_eval(
    data: List[Dict],
    eval_ratio: float = 0.05,
    shuffle_seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    划分训练集和验证集

    为什么要设 shuffle_seed？
      固定随机种子 → 每次划分结果一致 → 实验结果可复现
    """
    import random
    random.seed(shuffle_seed)

    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - eval_ratio))
    train_data = shuffled[:split_idx]
    eval_data = shuffled[split_idx:]

    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(eval_data)} 条")

    return train_data, eval_data


def print_sample(data: List[Dict], n: int = 3):
    """打印几条样本供人工检查"""
    print(f"\n{'='*60}")
    print(f"数据样本 (共 {len(data)} 条，展示前 {n} 条)")
    print(f"{'='*60}")
    for i, item in enumerate(data[:n]):
        convs = item.get("conversations", [])
        for c in convs:
            role = "用户" if c.get("from") == "human" else "助手"
            value = c.get("value", "")
            print(f"  [{role}] {value[:80]}{'...' if len(value) > 80 else ''}")
        print(f"  {'─' * 56}")


# ============================================
# 主流程
# ============================================
def main():
    """
    [已补齐] Day2: 完整的数据预处理流水线

    处理你的 medical_sft_1K_format.jsonl：
      加载 → 去重 → 质量检查 → 划分 → 保存
    """
    print("=" * 60)
    print("医疗数据预处理流水线")
    print("=" * 60)

    # [已补齐] 输入文件：支持 3 万条高质量医疗数据
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # 默认：优先用 3 万条数据，回退到 1000 条
        candidates = [
            "data/medical_finetune_30000.jsonl",
            "data/medical_30k.jsonl",
            "data/medical_sft_1K_format.jsonl",
            "data/raw_data.jsonl",
        ]
        input_file = None
        for f in candidates:
            if Path(f).exists():
                input_file = f
                break
        if input_file is None:
            print("[ERROR] 没有找到数据文件")
            return
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)

    if not Path(input_file).exists():
        print(f"[ERROR] 原始数据文件 {input_file} 不存在")
        print("  请确认文件路径是否正确")
        return

    # Step 1: 加载
    print("\n[1/4] 加载数据...")
    data = load_jsonl(input_file)
    print(f"  加载 {len(data)} 条数据")
    print_sample(data)

    # Step 2: 去重
    print("\n[2/4] 去重...")
    data = deduplicate(data, key_field="conversations")

    # Step 3: 质量检查
    print("\n[3/4] 质量检查...")
    data, stats = check_quality(data)
    print(f"  过滤掉 {stats['filtered']} 条数据:")
    print(f"    格式错误: {stats['bad_format']}")
    print(f"    回答太短 (<5字): {stats['answer_too_short']}")
    print(f"    提问太短 (<2字): {stats['question_too_short']}")
    print(f"    拒绝回答: {stats['reject_answer']}")
    print(f"  剩余: {len(data)} 条")

    if len(data) < 10:
        print("[ERROR] 数据太少，训练效果会很差。请检查数据质量。")
        return

    # Step 4: 划分训练/验证集
    print("\n[4/4] 划分数据集...")
    train_data, eval_data = split_train_eval(data, eval_ratio=0.05)

    # 保存
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    save_jsonl(train_data, str(train_path))
    save_jsonl(eval_data, str(eval_path))

    print(f"\n[OK] 数据预处理完成！")
    print(f"   训练集: {train_path} ({len(train_data)} 条)")
    print(f"   验证集: {eval_path} ({len(eval_data)} 条)")
    print(f"   下一步: python train.py")


if __name__ == "__main__":
    main()
