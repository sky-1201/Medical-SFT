"""
=============================================================================
  download_model.py —— 从 ModelScope 下载模型（国内加速）
=============================================================================

  用法:
    python scripts/download_model.py                              # 默认: Qwen2.5-3B-Instruct
    python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/download_model.py --model qwen3-4b             # 模糊匹配

  ModelScope 模型列表:
    https://modelscope.cn/models?page=1
=============================================================================
"""
import argparse
import os
import sys
from pathlib import Path

# ModelScope ↔ HuggingFace 模型名映射
MODEL_MAP = {
    "Qwen/Qwen2.5-3B-Instruct": "qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen/Qwen2.5-0.5B-Instruct",
}


def main():
    parser = argparse.ArgumentParser(description="从 ModelScope 下载模型")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace 模型名")
    parser.add_argument("--cache", type=str, default="/root/autodl-tmp/models",
                        help="下载到哪个目录")
    args = parser.parse_args()

    hf_name = args.model
    ms_name = MODEL_MAP.get(hf_name, hf_name)

    print(f"HuggingFace: {hf_name}")
    print(f"ModelScope:  {ms_name}")
    print(f"缓存目录:    {args.cache}")

    # 安装 modelscope
    os.system(f"{sys.executable} -m pip install modelscope -q")

    from modelscope import snapshot_download

    local_path = snapshot_download(ms_name, cache_dir=args.cache)
    print(f"\n[OK] 下载完成: {local_path}")
    print(f"\n在 config/common.py 中将 model_name_or_path 改为:")
    print(f'  model_name_or_path: str = "{local_path}"')


if __name__ == "__main__":
    main()
