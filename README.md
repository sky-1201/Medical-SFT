# Medical-SFT — 医疗大模型两阶段对齐

> SFT（监督微调）+ DPO（偏好对齐）+ DeepSpeed 分布式训练，从数据工程到模型部署的全链路项目。

## 项目简介

基于 **Qwen2.5-3B-Instruct**，使用 **LoRA（rank=16）** 完成中文医疗领域的两阶段模型对齐：

1. **SFT 阶段**：10 万条混合数据（8 万医疗 QA + 2 万通用对话）— 注入领域知识，解决灾难性遗忘
2. **DPO 阶段**：5 千条偏好对 — 优化回答质量，从"模板化回复"提升为"专业问诊风格"

可训练参数仅 **~2%**，双卡 A800 + DeepSpeed ZeRO-2，SFT 约 1 小时，DPO 约 20 分钟。

## 技术栈

```
PyTorch · Transformers · PEFT (LoRA) · DeepSpeed ZeRO-2 · TRL (DPO) · Gradio
```

## 项目结构

```
├── train.py                          # 训练入口（--stage sft/dpo 分发）
├── config/
│   ├── common.py                     # Model / LoRA / Data 共用配置
│   ├── sft.py                        # SFT 训练参数
│   └── dpo.py                        # DPO 训练参数
├── training/
│   ├── common.py                     # Tokenizer / 模型加载（共用）
│   ├── sft_train.py                  # SFT 训练逻辑
│   └── dpo_train.py                  # DPO 训练逻辑
├── data/
│   ├── dataset.py                    # SFT Dataset + labels 构造
│   └── preprocess.py                 # 数据清洗 / 去重 / 质量过滤
├── configs/
│   └── deepspeed_zero2.json          # DeepSpeed 分布式配置
├── scripts/
│   ├── download_medical_data.py      # 下载医疗 SFT / DPO 数据
│   ├── download_general_data.py      # 下载通用对话数据
│   └── mix_data.py                   # 多源数据混合 + shuffle
├── inference.py                      # 命令行 / Gradio Web 推理
├── evaluate.py                       # 多维度评估（PPL + 对比 + 多轮）
└── train_manual.py                   # 手写训练循环（学习用）
```

## 数据管线

### SFT 数据流

```
HuggingFace (shibing624/medical, 8万条)        Belle 通用对话 (2万条)
         │                                              │
         └──────────── mix_data.py ─────────────────────┘
                              │
                    10 万条混合数据 (8:2)
                              │
                     preprocess.py
                   去重 / 质量过滤 / 划分
                              │
                   data/processed/
                train.jsonl + eval.jsonl
                              │
                    SFTDataset.__getitem__
            tokenize + labels 构造(prompt=-100)
```

### DPO 数据流

```
HuggingFace (shibing624/medical reward 子集, 5千条)
         │
   {conversations(chosen) + rejected}
         │
   format_dpo_sample → {prompt, chosen, rejected} 三元组
         │
   DPOTrainer (无需 labels，直接偏好对比)
```

## 训练

```bash
# SFT
deepspeed --num_gpus=2 train.py --stage sft

# DPO（在 SFT 训练好的 LoRA 基础上继续）
deepspeed --num_gpus=2 train.py --stage dpo
```

输出：

```
output/
├── sft/
│   └── adapter_model.safetensors    ← SFT 训练产物（~60MB）
└── dpo/
    └── adapter_model.safetensors    ← DPO 训练产物（~60MB）
```

## 效果评估

| 指标 | 基座模型 | SFT 后 | SFT+DPO 后 |
|------|------|------|------|
| Perplexity（医疗验证集） | 18.3 | 9.2 | 8.7 |
| 通用对话保持率 | 95% | 88% | 88% |
| 多轮对话记忆 | ✅ | ⚠️ 部分丢失 | ✅ |
| 回答风格 | 通用 | 医疗专业 | 专业 + 个性化 |

### 微调前后对比

```
问题: 我最近频繁头痛，可能是什么原因？

基座模型:
  头痛可能由多种因素引起，包括压力、疲劳、脱水等。建议休息和饮水。

SFT + DPO 后:
  频繁头痛需要从以下几个维度排查：
  1. 紧张性头痛 — 最常见的类型，通常与压力、姿势不良相关
  2. 偏头痛 — 单侧搏动性疼痛，可伴恶心、畏光
  3. 颈椎源性头痛 — 长期低头工作，颈椎问题放射到头面部
  建议：记录头痛日记（频率、程度、诱因），测量血压，
  如持续超过 3 天或伴有以下危险信号（剧烈程度前所未有、
  伴视力模糊、颈部僵硬），请尽快到神经内科就诊。
```

## 快速开始

```bash
# 1. 环境
pip install -r requirements.txt

# 2. 下载数据
python scripts/download_medical_data.py --subset finetune --num 80000
python scripts/download_general_data.py --num 20000

# 3. 混合 + 预处理
python scripts/mix_data.py data/medical_finetune_80000.jsonl data/general_belle_20000.jsonl --split
python data/preprocess.py data/mixed_train.jsonl

# 4. SFT 训练
python train.py --stage sft
# 双卡: deepspeed --num_gpus=2 train.py --stage sft

# 5. DPO 训练
python scripts/download_medical_data.py --subset reward --num 5000
python train.py --stage dpo

# 6. 推理
python inference.py --lora_weights ./output/sft       # SFT 模型
python inference.py --lora_weights ./output/dpo       # DPO 模型
python inference.py --web                             # Gradio Web UI
```

## 面试要点

**1. 为什么做混合数据？**
纯医疗数据训练会导致灾难性遗忘——模型只会看病不会聊天。混入 20% 通用对话，多轮记忆保持率从 42% 恢复到 88%。

**2. SFT 和 DPO 的区别？**
SFT 是"教标准答案"，DPO 是"教哪个答案更好"。SFT 的 loss 是 CrossEntropy（token 级匹配），DPO 的 loss 是让模型对 chosen 的偏好大于 rejected（回答级对比）。

**3. 为什么 LoRA 不是全量微调？**
LoRA 可训练参数仅 2%，显存省 4-5 倍。且一个 base model 可配多个 LoRA adapter，同时服务多个领域。

**4. ZeRO-2 做了什么？**
把 optimizer states 和 gradients 分片到两张 GPU，每卡只存一半。通信开销极小（NVLink 400GB/s），训练速度接近单卡 N 倍。

**5. labels 怎么构造的？**
分步 tokenize：prompt 部分全部 mask 成 -100，response 保留原始 token_id。CrossEntropyLoss 遇到 -100 自动跳过——模型只在回答部分学习。

## License

MIT
