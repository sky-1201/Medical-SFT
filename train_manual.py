"""
=============================================================================
  train_manual.py —— 手写训练循环（进阶）
=============================================================================

  TODO-Day5: 这是从"会用工具"到"理解原理"的关键一步

  HuggingFace Trainer 内部做的就是你在这里看到的：

    for batch in dataloader:
        outputs = model(**batch)              # 前向传播
        loss = outputs.loss                   # 计算损失
        loss.backward()                       # 反向传播 → 计算梯度
        optimizer.step()                      # 更新参数
        scheduler.step()                      # 调整学习率
        optimizer.zero_grad()                 # 清空梯度（否则会累积）

  你要理解以下概念：
  1. loss.backward() —— 自动求导链是什么？
  2. optimizer.step() —— 参数是怎么更新的？
  3. gradient_accumulation —— 为什么要分多步才做一次 step？
  4. warmup —— 为什么学习率要从小到大？
  5. gradient_clipping —— 防止梯度爆炸

  TODO: 完成这个脚本，替换 train.py 中的 Trainer

=============================================================================
"""
import os
import sys
import math
import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from config import ModelConfig, LoraConfig as LoraCfg, TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManualTrainer:
    """
    TODO-Day5: 手写训练器

    这个类做和 HuggingFace Trainer 一样的事，但每一行都是你写的。
    """

    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config

        # TODO-Day5: 创建 optimizer
        #   为什么用 AdamW 而不是 Adam？
        #   AdamW 把 weight_decay 从 梯度更新 中解耦出来，效果更好
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,  # L2 正则化系数
        )

        # TODO-Day5: 计算总步数
        num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
        self.total_steps = num_update_steps_per_epoch * config.num_train_epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)

        # TODO-Day5: 创建 scheduler
        #   warmup: 前 warmup_steps 步，学习率从 0 线性增长到 learning_rate
        #   decay:  之后按 cosine 曲线衰减到 0
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        # 日志
        self.writer = SummaryWriter(log_dir=config.output_dir)
        self.global_step = 0
        self.best_eval_loss = float("inf")

        logger.info(f"ManualTrainer 初始化完成")
        logger.info(f"  每轮更新步数: {num_update_steps_per_epoch}")
        logger.info(f"  总步数: {self.total_steps}")
        logger.info(f"  Warmup 步数: {self.warmup_steps}")

    def train(self):
        """
        TODO-Day5: 主训练循环

        你需要理解的循环结构:

        for epoch in range(num_epochs):
            for batch in dataloader:
                # 1. 前向
                outputs = model(**batch)
                loss = outputs.loss

                # 2. 缩放 loss（梯度累积需要）
                loss = loss / gradient_accumulation_steps

                # 3. 反向
                loss.backward()

                # 4. 每 accumulation_steps 步更新一次参数
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪（防止梯度爆炸）
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()         # 更新参数
                    scheduler.step()         # 更新学习率
                    optimizer.zero_grad()    # 清空梯度
        """
        logger.info("=" * 50)
        logger.info("开始训练")
        logger.info("=" * 50)

        for epoch in range(self.config.num_train_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_train_epochs}")

            # ============================================
            # TODO-Day5: 训练阶段
            # ============================================
            self.model.train()
            total_loss = 0.0

            progress_bar = tqdm(self.train_dataloader, desc="Training")

            for step, batch in enumerate(progress_bar):
                # TODO: 把 batch 移到 GPU
                # batch = {k: v.to(device) for k, v in batch.items()}

                # TODO: 前向传播
                # outputs = self.model(**batch)
                # loss = outputs.loss / self.config.gradient_accumulation_steps

                # TODO: 反向传播
                # loss.backward()

                # TODO: 每 accumulation_steps 步更新一次
                # if (step + 1) % self.config.gradient_accumulation_steps == 0:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                #     self.optimizer.step()
                #     self.scheduler.step()
                #     self.optimizer.zero_grad()
                #     self.global_step += 1

                # TODO: 记录 loss
                # total_loss += loss.item()
                # progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                pass  # 占位

            # ============================================
            # TODO-Day5: 评估阶段（每个 epoch 结束后）
            # ============================================
            eval_loss = self.evaluate()
            logger.info(f"  Epoch {epoch+1} 完成 | Train Loss: {total_loss/len(self.train_dataloader):.4f} | Eval Loss: {eval_loss:.4f}")

            # TODO: 如果 eval_loss 下降，保存 checkpoint
            # if eval_loss < self.best_eval_loss:
            #     self.best_eval_loss = eval_loss
            #     self.model.save_pretrained(...)

        self.writer.close()
        logger.info("训练完成!")

    def evaluate(self) -> float:
        """
        TODO-Day5: 在验证集上计算 loss

        注意: 评估时用 torch.no_grad() 关闭梯度计算，节省显存
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                # TODO: 同训练的前向，但不做 backward
                # batch = {k: v.to(device) for k, v in batch.items()}
                # outputs = self.model(**batch)
                # total_loss += outputs.loss.item()
                pass

        return total_loss / len(self.eval_dataloader)


# ============================================
# TODO-Day5: 画图 —— 训练过程的 Loss 曲线
# ============================================
def plot_loss_curve(train_losses, eval_losses, output_dir: str):
    """
    用 matplotlib 画出训练/验证 loss 曲线

    观察:
    - 两条线都在下降 → 正常，可以继续训练
    - 训练 loss 下降但验证 loss 上升 → 过拟合！停！
    - 两条线都不动 → 学习率太小或数据有问题
    """
    # TODO-Day6: 实现
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label="Train Loss", color="#2563EB")
        ax.plot(eval_losses, label="Eval Loss", color="#EF4444")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Evaluation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(f"{output_dir}/loss_curve.png", dpi=150, bbox_inches="tight")
        logger.info(f"Loss 曲线已保存到 {output_dir}/loss_curve.png")
    except ImportError:
        logger.warning("matplotlib 未安装，跳过画图")
    except Exception as e:
        logger.warning(f"画图失败: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("手写训练循环 (进阶)")
    print("=" * 60)
    print()
    print("TODO-Day5: 完成 ManualTrainer 类")
    print("完成后: python train_manual.py 启动训练")
    print()
    print("这比 Trainer 多写 100 行代码，但你会真正理解训练过程。")
    print("面试时你能画出 forward → backward → optimize 的完整流程。")
