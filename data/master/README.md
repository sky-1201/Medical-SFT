---
annotations_creators:
- expert-generated
language:
- zh
license:
- MIT
multilinguality:
- monolingual
pretty_name: Medical Evidence DPO Dataset
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- sequence-modeling
- text-generation
task_ids:
- preference-alignment
---

# Medical Evidence DPO Dataset

## 简介

Medical Evidence DPO Dataset 是一个面向医学领域的中文偏好对齐数据集，专为 Direct Preference Optimization (DPO) 训练设计。该数据集包含医学问答三元组（问题、高质量回答、低质量回答），可用于微调语言模型以提升其医学领域的回答质量。

## 数据集组成

| 数据集名称 | 分割 | 样本数 | 说明 |
|-----------|------|--------|------|
| medical_evidence_dpo | train | ~1400 | DPO 训练数据 |

## 数据格式

每条数据包含三个字段：

- **prompt** (string): 医学问题或查询
- **chosen** (string): 高质量回答（作为偏好目标）
- **rejected** (string): 低质量回答（作为负样本）

### 数据示例

```json
{
  "prompt": "在阿尔茨海默病与溃疡性结肠炎患者中，PPARG 和 NOS2 作为共同基因，是否通过调控巨噬细胞和小胶质细胞极化参与疾病的发生发展？",
  "chosen": "从目前的人类与动物实验证据来看，PPARG 和 NOS2 很有可能作为共同炎症枢纽基因，通过调控巨噬细胞/小胶质细胞的极化状态参与阿尔茨海默病和溃疡性结肠炎的发生发展...",
  "rejected": "这是一个非常具体且专业的问题，涉及到两种疾病的共同机制。根据现有的生物医学研究，我们可以进行一个基于科学逻辑的推理和分析..."
}
```

## 主题覆盖

数据集涵盖多个医学主题领域，包括：

- **临床指南与治疗管理**：肝硬化、川崎病、器官移植等疾病的循证治疗
- **疾病机制与病理生理**：肿瘤耐药机制、神经退行性疾病机制
- **药物疗法与药理学**：免疫抑制剂、化疗药物作用机制
- **医学诊断与检验**：生物标志物解读、影像学分析
- **医学研究方法**：临床研究设计、meta分析解读

## 用途

本数据集主要用于：

1. **DPO (Direct Preference Optimization)**：训练语言模型偏好对齐
2. **RLHF 训练**：作为偏好反馈数据用于人类反馈强化学习
3. **医学问答模型微调**：提升模型在医学领域的回答质量
4. **医学知识评估**：评估模型在医学领域的知识水平和推理能力

## 使用方法

### 使用 Hugging Face Datasets 加载

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("path/to/medical_evidence_DPO.py", name="medical_evidence_dpo")

# 查看数据
print(dataset["train"][0])
```

### DPO 训练示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer
import torch

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 准备数据
def format_dpo_sample(sample):
    return {
        "prompt": sample["prompt"],
        "chosen": sample["chosen"],
        "rejected": sample["rejected"]
    }

train_dataset = dataset["train"].map(format_dpo_sample)

# DPO 训练
dpo_trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    beta=0.1,
)

dpo_trainer.train()
```

## 数据来源

本数据集基于医学领域专家撰写的循证医学问答内容构建，每个问题都经过专业筛选，回答内容参考最新临床指南和医学文献。

## 许可证

本数据集采用MIT许可证。

## 引用

```bibtex
@misc{medical_evidence_dpo,
  title        = {Medical Evidence DPO Dataset},
  author       = {Medical Evidence Team},
  year         = {2026},
  url          = {https://modelscope.cn/datasets/modelzhang/medical_evidence_DPO}
}
```

## 注意事项
1. 模型输出的医学建议仅供参考，不能替代专业医疗诊断
2. 使用前请确保了解数据的内容和局限性

# DPO的Agent的训练数据集
filter_dpo_dataset.jsonl
