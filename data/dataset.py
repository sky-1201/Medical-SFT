"""
=============================================================================
  dataset.py —— 自定义 Dataset，把原始文本变成模型吃进去的 Tensor
=============================================================================

  [已补齐] Day2 (核心学习任务):
  这是整个项目最核心的文件。理解它 = 理解了 SFT 微调的本质。

  torch.utils.data.Dataset —— 告诉 PyTorch "我的数据长什么样"
  torch.utils.data.DataLoader —— 把 Dataset 变成一批一批的 batch

  你需要继承 Dataset 并实现三个方法：
    1. __init__()  —— 加载数据
    2. __len__()   —— 返回数据总量
    3. __getitem__() —— 取第 i 条数据，返回 tokenized 后的 tensor

  流程图:
    原始 JSONL
      → __init__() 读到内存
      → __getitem__(i) 取第i条
        → 格式检测（conversations 或 messages）
        → apply_chat_template() 拼接成对话格式文本
        → tokenizer() 转成 input_ids + attention_mask + labels
        → 返回 PyTorch Tensor

  ==========================================================================
  ★ Labels 构造原理（整个微调最核心的概念）★

  训练目标是"给定上文，预测下一个 token"。模型不应该学习"记住用户的提问"，
  只应该学习"如何生成好的回答"。

  因此：
    input_ids: [sys, ..., user, 头疼, 怎么办, assistant, 头疼, 的, 原因, eos]
    labels:    [-100,-100,-100,-100,-100,-100,-100,    头疼,  的, 原因, eos]
                ↑ prompt 部分全是 -100（不计算 loss）   ↑ response 部分保留（要学）

  CrossEntropyLoss(ignore_index=-100) 遇到 -100 自动跳过 → 只在回答部分优化模型。

  借鉴：MedicalGPT supervised_finetuning.py 的 preprocess_function（第541-567行）
=============================================================================
"""
import json
import copy
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer


# [已补齐] 兼容的角色名映射 —— 同时支持 conversations(human/gpt) 和 messages(user/assistant)
ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "observation": "user",
    "function_call": "assistant",
}


def detect_format(item: Dict) -> str:
    """
    [已补齐] 自动检测数据格式

    返回 "conversations" 或 "messages" 或 "unknown"
    """
    if "conversations" in item:
        return "conversations"
    if "messages" in item:
        return "messages"
    return "unknown"


def parse_to_messages(item: Dict) -> List[Dict[str, str]]:
    """
    [已补齐] 把各种格式统一转成 HuggingFace 标准的 messages 格式:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    支持两种输入格式：
      1. conversations: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
      2. messages:      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    fmt = detect_format(item)
    raw = item.get("conversations") or item.get("messages") or []

    messages = []
    for msg in raw:
        # 获取角色
        role = msg.get("from") or msg.get("role") or ""
        role = ROLE_MAPPING.get(role, role)

        # 获取内容
        content = msg.get("value") or msg.get("content") or ""

        if role and content:
            messages.append({"role": role, "content": content})

    return messages


def extract_system_prompt(item: Dict, default_system: Optional[str] = None) -> Optional[str]:
    """
    [已补齐] 提取 system prompt

    优先顺序：数据自带 system_prompt > conversations 里 role=system > default_system
    """
    # 方式1: 数据自带 system_prompt 字段
    if item.get("system_prompt"):
        return item["system_prompt"]

    # 方式2: conversations 里有 role=system
    convs = item.get("conversations") or []
    for c in convs:
        role = c.get("from") or c.get("role") or ""
        if role == "system":
            return c.get("value") or c.get("content") or ""

    # 方式3: 用默认值
    return default_system


class SFTDataset(Dataset):
    """
    SFT (Supervised Fine-Tuning) 数据集

    职责：把对话数据转成训练用的 Tensor

    [已补齐] Day2: 完整实现了 __getitem__ 的 tokenization 和 labels 构造
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        system_prompt: Optional[str] = None,
    ):
        """
        [已补齐] Day2: 加载并验证数据

        Args:
            data_path: JSONL 文件路径，每行一个对话
            tokenizer: HuggingFace tokenizer（必须已设置 pad_token）
            max_seq_length: 最大序列长度，超过的截断
            system_prompt: 默认系统提示词（数据自带时优先用数据自带的）
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.default_system = system_prompt

        # [已补齐] 加载 JSONL 数据
        self.raw_data: List[Dict] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        self.raw_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[WARN] Dataset: 第 {line_num} 行 JSON 解析失败: {e}")

        if not self.raw_data:
            raise ValueError(f"[ERROR] 数据文件 {data_path} 为空或无法解析")

        # [已补齐] 预检查：确认数据格式可识别
        first_item = self.raw_data[0]
        fmt = detect_format(first_item)
        if fmt == "unknown":
            raise ValueError(
                f"[ERROR] 无法识别数据格式。数据必须包含 'conversations' 或 'messages' 字段。\n"
                f"  第一条数据: {json.dumps(first_item, ensure_ascii=False)[:200]}..."
            )
        print(f"  检测到数据格式: {fmt}")
        print(f"  加载 {len(self.raw_data)} 条数据")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        [已补齐] Day2: 最核心的方法

        分四步：
          1. 解析原始数据 → messages 列表
          2. 分步 tokenize（prompt 和 response 分开处理）
          3. 构造 labels（prompt=-100, response=原始token_id）
          4. 截断并返回 Tensor

        Returns:
            {
                "input_ids": torch.LongTensor,
                "attention_mask": torch.LongTensor,
                "labels": torch.LongTensor,
            }

        ================================================================
        为什么不直接 tokenize 完整对话然后用正则找 assistant 位置？

        因为不同模型的 chat_template 格式不同：
          Qwen:   <|im_start|>assistant\n...<|im_end|>
          Llama3: <|start_header_id|>assistant<|end_header_id|>\n\n...<|eot_id|>
          ChatGLM: <|assistant|>...<|user|>

        正则方案在不同模型间不稳定。分步 tokenize 更可靠——
        先 tokenize prompt（不带 response），再 tokenize response，然后拼接。
        ================================================================
        """
        # ============================================
        # Step 1: 解析原始数据
        # ============================================
        item = self.raw_data[idx]
        messages = parse_to_messages(item)
        system = extract_system_prompt(item, self.default_system)

        if len(messages) < 2:
            # 最少需要一轮对话（user + assistant），不够的话返回一个占位
            print(f"[WARN] 第 {idx} 条数据对话轮数不足 ({len(messages)} 条消息)，跳过")
            return self.__getitem__((idx + 1) % len(self.raw_data))

        # ============================================
        # Step 2: 分步 tokenize
        # ============================================

        # 构造完整 messages（带 system prompt）
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # 取最后一个 assistant 消息作为 response
        # 其余所有消息（包含最后一个 assistant 之前的全部内容）= prompt
        last_assistant_idx = None
        for i in range(len(full_messages) - 1, -1, -1):
            if full_messages[i]["role"] == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            print(f"[WARN] 第 {idx} 条数据没有 assistant 消息，跳过")
            return self.__getitem__((idx + 1) % len(self.raw_data))

        # [已补齐] 关键：分别 tokenize prompt 和 response

        # Prompt 部分：从开头到最后一个 assistant 之前
        # 用 add_generation_prompt=True 让 tokenizer 在末尾自动加 assistant 标记
        prompt_messages = full_messages[:last_assistant_idx]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,    # [已补齐] 重要！自动加 "<|im_start|>assistant\n"
        )

        # Response 部分：最后一个 assistant 的内容 + eos
        response_content = full_messages[last_assistant_idx]["content"]
        response_text = response_content + self.tokenizer.eos_token

        # [已补齐] Tokenize prompt
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False,     # chat_template 已经加了特殊 token
            return_tensors=None,          # 返回 Python list，不是 tensor
        )
        prompt_ids = prompt_enc["input_ids"]

        # [已补齐] Tokenize response
        response_enc = self.tokenizer(
            response_text,
            add_special_tokens=False,
            return_tensors=None,
        )
        response_ids = response_enc["input_ids"]

        # ============================================
        # Step 3: 构造 labels
        # ============================================

        # 拼接 input_ids
        input_ids = prompt_ids + response_ids

        # [已补齐] ★ 核心：labels 构造 ★
        # prompt 部分全部标记为 IGNORE_INDEX(-100)
        # response 部分保留原始 token_id
        IGNORE_INDEX = -100
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids

        # ============================================
        # Step 4: 长度截断和校验
        # ============================================
        if len(input_ids) > self.max_seq_length:
            # [已补齐] 超长：截断，优先保留 response 部分
            overflow = len(input_ids) - self.max_seq_length
            # 从 prompt 部分截断（保留 response 的完整性）
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        # 长度校验
        assert len(input_ids) == len(labels), (
            f"input_ids 和 labels 长度不一致: {len(input_ids)} vs {len(labels)}"
        )
        assert len(input_ids) <= self.max_seq_length, (
            f"序列长度 {len(input_ids)} 超过 max_seq_length {self.max_seq_length}"
        )

        # [已补齐] attention_mask: 全部是1（因为没有 padding，padding 由 DataCollator 处理）
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ============================================
# [已补齐] DataCollator —— 你已经不需要自己写了
#
#   HuggingFace 的 DataCollatorForSeq2Seq 已经做了所有事：
#     - padding 到 batch 内最大长度
#     - labels 的 padding 位置填 -100
#     - attention_mask 自动生成
#
#   所以 train.py 里直接用它。但你保留这个类来理解原理。
# ============================================
class ManualDataCollator:
    """
    [已补齐] 手动实现的 DataCollator，用于理解原理

    HuggingFace Trainer 会自动使用 DataCollatorForSeq2Seq，
    所以这个类在训练中不会被调用。放在这里的目的是让你理解
    padding 和 labels 对齐的底层逻辑。
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        把一个 batch 里不等长的数据 padding 到相同长度
        """
        # 找到最大长度
        max_len = max(item["input_ids"].size(0) for item in batch)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)

            # input_ids: 尾部填充 pad_token_id
            padded_input_ids.append(
                torch.cat([
                    item["input_ids"],
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                ])
            )

            # attention_mask: 真实 token=1, pad=0
            padded_attention_mask.append(
                torch.cat([
                    item["attention_mask"],
                    torch.zeros(pad_len, dtype=torch.long),
                ])
            )

            # labels: 尾部填充 -100（CrossEntropyLoss 自动忽略）
            padded_labels.append(
                torch.cat([
                    item["labels"],
                    torch.full((pad_len,), -100, dtype=torch.long),
                ])
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }


# ============================================
# [已补齐] Day2: 测试 Dataset 是否正确
#   运行: python data/dataset.py
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("Dataset 模块测试")
    print("=" * 60)

    from transformers import AutoTokenizer

    # 加载 tokenizer（先用小模型测试）
    print("\n加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

    # [已补齐] 用你的真实医疗数据测试
    print("\n加载医疗数据集...")
    import os
    data_path = "data/medical_sft_1K_format.jsonl"
    if not os.path.exists(data_path):
        print(f"[ERROR] 数据文件不存在: {data_path}")
        print("  请先确认数据文件路径")
        exit(1)

    dataset = SFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=512,
        system_prompt="你是一个专业的医疗健康助手。",
    )

    print(f"\n数据集大小: {len(dataset)} 条")

    # 取第一条测试
    print("\n" + "=" * 60)
    print("取第 0 条数据测试")
    print("=" * 60)
    sample = dataset[0]
    print(f"  input_ids:      shape={sample['input_ids'].shape}, dtype={sample['input_ids'].dtype}")
    print(f"  attention_mask: shape={sample['attention_mask'].shape}, dtype={sample['attention_mask'].dtype}")
    print(f"  labels:         shape={sample['labels'].shape}, dtype={sample['labels'].dtype}")

    # [已补齐] 关键验证：检查 labels 中 prompt 部分是否为 -100
    labels_list = sample["labels"].tolist()
    ignore_count = labels_list.count(-100)
    learn_count = len(labels_list) - ignore_count
    print(f"\n  总 token 数: {len(labels_list)}")
    print(f"  被 mask 的 token 数 (-100): {ignore_count} ({ignore_count/len(labels_list)*100:.1f}%)")
    print(f"  参与训练的 token 数: {learn_count} ({learn_count/len(labels_list)*100:.1f}%)")

    # [已补齐] 解码查看 prompt 和 response 的内容
    print(f"\n  --- 解码 input_ids（模型读到的完整文本）---")
    decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
    print(decoded_input[:400])

    print(f"\n  --- 解码 labels（只有非 -100 部分 = 模型要学的）---")
    learn_labels = [l if l != -100 else tokenizer.pad_token_id for l in labels_list]
    decoded_labels = tokenizer.decode(learn_labels, skip_special_tokens=False)
    print(decoded_labels[:400])

    # [已补齐] 验证 labels 构造正确性
    print(f"\n  --- 验证 ---")
    # 找到 labels 中第一个非 -100 的位置
    first_learn_idx = next((i for i, l in enumerate(labels_list) if l != -100), -1)
    print(f"  第一个参与训练的 token 在第 {first_learn_idx} 位")
    if first_learn_idx > 0:
        # 前面的 token 应该都是 prompt 部分
        prompt_decoded = tokenizer.decode(sample["input_ids"][:first_learn_idx].tolist(), skip_special_tokens=False)
        print(f"  之前的内容（prompt，被 mask）:\n    {prompt_decoded[-200:]}")
        # 从第一个学习位置开始的内容
        response_start = tokenizer.decode(sample["input_ids"][first_learn_idx:first_learn_idx+1].tolist(), skip_special_tokens=False)
        print(f"  第一个学习到的 token:\n    {response_start!r}")

    print(f"\n[OK] Dataset 模块测试通过！")
    print(f"  如果 labels 中 prompt 部分是 -100 且 response 部分是原始 token_id，")
    print(f"  说明 labels 构造正确。下一步: python train.py")
