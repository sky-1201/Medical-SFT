"""
=============================================================================
  train.py —— 训练入口（分发到 training/sft_train 或 training/dpo_train）
=============================================================================

  用法:
    python train.py --stage sft     # SFT 监督微调
    python train.py --stage dpo     # DPO 偏好对齐

  双卡 DeepSpeed:
    deepspeed --num_gpus=2 train.py --stage sft
    deepspeed --num_gpus=2 train.py --stage dpo
=============================================================================
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医疗模型训练")
    parser.add_argument("--stage", type=str, default="sft", choices=["sft", "dpo"])
    parser.add_argument("--local_rank", type=int, default=0, help="DeepSpeed 自动传入")
    args, _ = parser.parse_known_args()

    if args.stage == "sft":
        from training.sft_train import run
        run()
    elif args.stage == "dpo":
        from training.dpo_train import run
        run()
