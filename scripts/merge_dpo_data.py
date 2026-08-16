"""
合并两份高质量 DPO 数据 → 统一 {prompt, chosen, rejected} 格式

来源:
  1. data/master/dpo_answer.jsonl (1399 条, prompt/chosen/rejected)
  2. data/master/clinical_dpo_final_dataset.json (997 条, question/chosen/rejected)

输出: data/medical_dpo_merged.jsonl
"""
import json
from pathlib import Path


def main():
    data_dir = Path("data/master")
    output_path = Path("data/medical_dpo_merged.jsonl")

    all_data = []

    # 1. dpo_answer.jsonl
    f1 = data_dir / "dpo_answer.jsonl"
    count1 = 0
    if f1.exists():
        with open(f1, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt = item.get("prompt", "").strip()
                chosen = item.get("chosen", "").strip()
                rejected = item.get("rejected", "").strip()
                if prompt and chosen and rejected:
                    all_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                    count1 += 1
    print(f"dpo_answer.jsonl: {count1} 条")

    # 2. clinical_dpo_final_dataset.json
    f2 = data_dir / "clinical_dpo_final_dataset.json"
    count2 = 0
    if f2.exists():
        with open(f2, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            prompt = item.get("question", "").strip()
            chosen = item.get("chosen", "").strip()
            rejected = item.get("rejected", "").strip()
            if prompt and chosen and rejected:
                all_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                count2 += 1
    print(f"clinical_dpo_final_dataset.json: {count2} 条")

    # 去重（按 prompt）
    seen = set()
    deduped = []
    for item in all_data:
        if item["prompt"] not in seen:
            seen.add(item["prompt"])
            deduped.append(item)

    print(f"\n合并后: {len(all_data)} 条, 去重后: {len(deduped)} 条")

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        for item in deduped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] 保存到 {output_path} ({size:.1f} MB)")
    print(f"  下一步: 修改 config/dpo.py 的 dpo_data_path 为 data/medical_dpo_merged.jsonl")


if __name__ == "__main__":
    main()
