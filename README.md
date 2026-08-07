# 财报分析助手 —— 端到端 SFT 微调项目

## 目标

不依赖 LLaMA-Factory，用 `transformers` + `peft` 手写完整的微调流程，从数据构造到模型部署上线。

## 为什么做这个

实习中用 LLaMA-Factory 调参跑训练 → 会用工具但不理解原理 → 面试一问就穿帮。
这个项目强制你**每行代码都理解**，补齐从"调参侠"到"能做模型训练"的最后一公里。

## 技术栈

```
PyTorch → 数据处理、训练循环
transformers → 模型加载、Tokenizer
peft → LoRA 微调
gradio → Web Demo
```

## 项目结构

```
我的模型微调项目/
├── config.py                  # 所有参数集中管理（替代 YAML）
├── train.py                   # 主训练脚本（用 HuggingFace Trainer）
├── train_manual.py            # 手写训练循环（进阶）
├── evaluate.py                # 模型评估（PPL + 生成对比 + LLM-as-Judge）
├── inference.py               # 推理 & Gradio Web UI
├── requirements.txt           # 依赖
├── data/
│   ├── dataset.py             # 自定义 Dataset + DataCollator
│   ├── preprocess.py          # 数据清洗 / 去重 / 质量检查 / 划分
│   ├── raw_data.jsonl         # 原始训练数据
│   └── processed/             # 预处理后的数据
└── scripts/
    └── generate_dummy_data.py # 生成测试用的假数据
```

## 7 天学习计划（边做边学）

每个 TODO 都是**你需要填的坑**。不要先学再做——直接用今天的 TODO 驱动学习。

### Day1: 环境搭好 + 模型加载

**目标**: 加载 Qwen2.5-1.5B 模型，确认 GPU 和显存

```bash
# 安装依赖
pip install -r requirements.txt

# 确认环境
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# TODO: 在 train.py 中实现 load_tokenizer() 和 load_model_with_lora() 的 Step2 部分
#       暂时不加 LoRA，先把模型加载到显存
```

**要搞懂的概念**:
- [ ] `torch_dtype`: float32 vs float16 vs bfloat16 的区别
- [ ] `device_map="auto"`: accelerate 怎么把模型分配到 GPU
- [ ] `trust_remote_code=True`: 为什么 Qwen 需要这个
- [ ] Tokenizer 的 `pad_token` 和 `eos_token` 是什么
- [ ] 运行 `config.py`，确认配置加载成功

**产出物**: 模型能加载成功，打印参数量

### Day2: 数据加载（核心）

**目标**: 实现 `data/dataset.py` 中的 `SFTDataset`

```bash
# 先生成测试数据
python scripts/generate_dummy_data.py

# TODO: 实现 dataset.py 中的 __getitem__ 方法
#       在 dataset.py 末尾运行测试验证
```

**要搞懂的概念**:
- [ ] `torch.utils.data.Dataset` 的 `__init__/__len__/__getitem__`
- [ ] `tokenizer.apply_chat_template()` 怎么把对话转成模型输入格式
- [ ] `labels` 为什么要把 user 部分设为 -100（只在 assistant 部分计算 loss）
- [ ] `attention_mask` 的作用
- [ ] 数据格式: ShareGPT vs Alpaca

**产出物**: `data/raw_data.jsonl` + Dataset 能正常取出一条数据

### Day3: LoRA 注入

**目标**: 理解 LoRA 原理，把 LoRA 注入模型

```bash
# TODO: 在 train.py 的 load_model_with_lora() 中实现 LoRA 注入
#       打印可训练参数占比
```

**要搞懂的概念**:
- [ ] LoRA 的数学原理: ΔW = BA（低秩分解）
- [ ] `r` (rank) 和 `lora_alpha` 怎么配合
- [ ] 为什么 `target_modules` 通常选 Q 和 V
- [ ] LoRA 注入后参数量变化（可训练 vs 总参数）
- [ ] QLoRA 和 LoRA 的区别（quantization + LoRA）

**产出物**: 模型注入 LoRA 后，可训练参数从 1.5B 降到 ~20M

### Day4: 开始训练！

**目标**: 用 HuggingFace Trainer 跑通第一次训练

```bash
# 需要先把数据预处理跑通
python data/preprocess.py

# 开始训练！
python train.py
```

**要搞懂的概念**:
- [ ] `batch_size` vs `gradient_accumulation_steps` 的关系
- [ ] 学习率调度的几个阶段: warmup → 恒定 → decay
- [ ] `gradient_checkpointing`: 用时间换显存的原理
- [ ] `bf16` vs `fp16`: 你的 GPU 该用哪个
- [ ] 看 loss 变化: 正常下降 vs 不降 vs 震荡

**产出物**: 第一个训练完成的 checkpoint

### Day5: 手写训练循环（进阶！）

**目标**: 理解 Trainer 内部做了什么，手写等价逻辑

```bash
# TODO: 完成 train_manual.py 中的 ManualTrainer 类
python train_manual.py
```

**要搞懂的概念**:
- [ ] `loss.backward()` —— 计算图是怎么构建的（面试高频）
- [ ] `optimizer.step()` —— AdamW 参数更新公式
- [ ] `optimizer.zero_grad()` —— 为什么要清空梯度（不清会累积）
- [ ] gradient clipping —— 防止梯度爆炸
- [ ] `model.train()` vs `model.eval()` —— dropout/batchnorm 行为切换

**产出物**: 手写的训练循环跑出和 Trainer 一致的 loss

### Day6: 评估 & 调优

**目标**: 量化微调效果，不是凭感觉说"变好了"

```bash
python evaluate.py
```

**要做的事**:
- [ ] 计算 Perplexity（微调前 vs 微调后）
- [ ] 选 10 个测试问题，并排对比微调前后的回答
- [ ] (可选) 用 GPT-4/Claude 做 LLM-as-Judge
- [ ] 分析 loss 曲线: 过拟合了？还是没学够？
- [ ] 回头看数据: 质量有问题吗？分布均匀吗？

**产出物**: `evaluation_report.md` + loss 曲线图

### Day7: 部署 Demo

**目标**: 简历上多一个可访问的链接

```bash
# 命令行交互
python inference.py

# Gradio Web UI
python inference.py --web
```

**要做的事**:
- [ ] 终端对话能跑通，多轮对话不丢失上下文
- [ ] Gradio 界面搭起来
- [ ] (可选) 部署到阿里云，生成公网链接
- [ ] 截图放简历

**产出物**: 可演示的对话助手

---

## TODO 进度总览

| 文件 | Day | 状态 | 关键产出 |
|------|-----|------|---------|
| `config.py` | 1 | ⬜ | 理解所有参数 |
| `data/dataset.py` | 2 | ⬜ | 自定义 Dataset |
| `data/preprocess.py` | 2 | ⬜ | 数据处理流水线 |
| `train.py` Step1-2 | 1 | ⬜ | 模型加载 |
| `train.py` Step3 | 3 | ⬜ | LoRA 注入 |
| `train.py` Step4-5 | 4 | ⬜ | 训练 + 保存 |
| `train_manual.py` | 5 | ⬜ | 手写训练循环 |
| `evaluate.py` | 6 | ⬜ | 评估报告 |
| `inference.py` | 7 | ⬜ | Demo 上线 |

---

## 面试时能讲的点

1. **数据是核心**: "我花在数据清洗和构造上的时间比训练本身多3倍，因为模型的上限由数据决定。"

2. **为什么用 LoRA**: "全量微调 7B 模型需要 ~56GB 显存，我用 QLoRA (4bit+LoRA) 只需要 ~8GB，可训练参数只有原来的 0.5%。这是工业界微调大模型的标准做法。"

3. **评估不是看 loss**: "虽然 loss 从 2.1 降到了 1.4，但我更关注的是模型在专业术语使用准确率、回答结构完整性上的实际表现。我用 LLM-as-Judge 做了 4 个维度的量化评估。"

4. **RAG和微调的协同**: "微调让模型掌握了财报领域的知识和语言风格，RAG 补充了具体财报的事实信息。两者不是替代关系，是互补。"

---

## 环境要求

- Python 3.10+
- CUDA 12.1+ (or MPS for Mac)
- 显存 ≥ 8GB (1.5B 模型 + QLoRA)
- 显存 ≥ 16GB (7B 模型 + QLoRA)
