"""
=============================================================================
  inference.py —— 模型推理 / 交互式对话
=============================================================================

  [已补齐] Day7: 把你的微调模型用起来！

  用法:
    python inference.py --base_model /path/to/model --lora_weights ./output/sft
    python inference.py --web                # Gradio Web UI

  服务器上（模型已下载到本地）:
    python inference.py \
        --base_model /root/autodl-tmp/models/models/qwen--Qwen2.5-7B-Instruct/snapshots/master \
        --lora_weights ./output/sft

  借鉴：MedicalGPT demo/inference.py + demo/gradio_demo.py
=============================================================================
"""
import os
import sys
import argparse
from threading import Thread
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel


# ============================================
# 模型加载
# ============================================
def load_model(base_model_name: str, lora_weights_path: str):
    """
    [已补齐] 加载微调后的模型（base model + LoRA adapter）

    加载顺序：
    1. AutoTokenizer：加载 tokenizer
    2. AutoModelForCausalLM：加载 base model（3GB）
    3. PeftModel.from_pretrained：挂载 LoRA adapter（~30MB）
    """
    print("加载模型...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="left",             # [已补齐] 推理用左 padding
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  加载 base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"  加载 LoRA adapter: {lora_weights_path}")
    if Path(lora_weights_path).exists():
        model = PeftModel.from_pretrained(
            base_model, lora_weights_path,
            torch_dtype="auto", device_map="auto",
        )
        print(f"    [OK] LoRA 权重已挂载")
    else:
        print(f"    [WARN] LoRA 路径不存在，使用未微调的 base model")
        model = base_model

    model.eval()  # 关闭 dropout
    print(f"[OK] 模型加载完成")
    return model, tokenizer


# ============================================
# [已补齐] 流式生成回答
# ============================================
@torch.inference_mode()   # 等价于 torch.no_grad() + 额外优化
def generate_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stop_str: str = "<|im_end|>",
):
    """
    [已补齐] Day7: 流式生成回答

    原理（借鉴 MedicalGPT demo/inference.py:26-68）：
    1. TextIteratorStreamer 是一个队列
    2. model.generate() 在后台线程中运行，生成一个 token 就往队列里放一个
    3. 主线程从队列里取 token 并实时打印 → 实现"逐字输出"效果

    Thread 是必需的——model.generate() 是同步阻塞调用，
    不分线程的话主线程会卡在 generate() 里，用户看不到流式输出。
    """
    # [已补齐] 创建流式输出器
    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=60.0,
        skip_prompt=True,             # 不输出 prompt 部分
        skip_special_tokens=True,     # 不输出 <|im_end|> 等特殊 token
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=(temperature > 0.0),    # temperature=0 → 贪心解码
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # [已补齐] 在独立线程中运行 generate()
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # [已补齐] 主线程逐 token 收集
    generated_text = ""
    for new_text in streamer:
        pos = new_text.find(stop_str)   # 检测停止词
        if pos != -1:
            new_text = new_text[:pos]
            generated_text += new_text
            yield generated_text
            break
        generated_text += new_text
        yield generated_text


# ============================================
# [已补齐] 交互式对话
# ============================================
def chat(model, tokenizer, system_prompt: str):
    """
    [已补齐] Day7: 交互式对话循环

    实现要点（借鉴 MedicalGPT demo/inference.py:199-259）：
    1. 维护对话历史（多轮对话）
    2. 每次把完整 history 送入模型（模型本身是无状态的）
    3. 控制生成长度和随机性
    4. 处理特殊输入（/clear 清空历史, /exit 退出）
    """
    history = []  # [[user_msg, assistant_msg], ...]

    print(f"\n{'='*60}")
    print("医疗助手 (输入 /clear 清空历史, /exit 退出)")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("再见!")
            break

        if user_input == "/clear":
            history = []
            print("[OK] 对话历史已清空\n")
            continue

        # [已补齐] 构造完整 messages：system + 历史对话 + 当前问题
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({"role": "user", "content": user_input})

        # [已补齐] 用 chat_template 拼成模型认识的文本
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,   # 末尾加 assistant 标记
        )

        # [已补齐] 流式输出
        print("助手: ", end="", flush=True)
        response = ""
        for partial in generate_stream(model, tokenizer, prompt):
            # 只打印新增的部分
            new_part = partial[len(response):]
            if new_part:
                print(new_part, end="", flush=True)
            response = partial
        print()

        # [已补齐] 保存到历史
        history.append([user_input, response.strip()])


# ============================================
# [已补齐] Gradio Web UI
# ============================================
def launch_web_ui(model, tokenizer, system_prompt: str):
    """
    [已补齐] Day7: 用 Gradio 搭建 Web 界面

    借鉴 MedicalGPT demo/gradio_demo.py:84-125

    简历上有 demo 链接非常加分——面试官点开就能用。
    部署方式: python inference.py --web --port 7860
    """
    try:
        import gradio as gr
    except ImportError:
        print("[ERROR] 请先安装 gradio: pip install gradio")
        return

    def respond(message, history):
        """
        [已补齐] Gradio 回调函数

        history: list of [user_msg, assistant_msg]（Gradio 自动维护的对话历史）
        """
        # 构造 messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})

        messages.append({"role": "user", "content": message})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # [已补齐] 流式输出
        response = ""
        for partial in generate_stream(model, tokenizer, prompt):
            response = partial
            yield response

    demo = gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="请输入您的健康问题...", lines=3),
        title="医疗健康助手 (Medical SFT)",
        description=(
            "基于 Qwen2.5-1.5B  + LoRA 微调的医疗问答模型。\n\n"
            "⚠️ **免责声明：** 本模型仅供学习和研究用途，所有回答仅供参考，"
            "不能替代专业医生的诊断和建议。如有健康问题，请及时就医。"
        ),
        examples=[
            "我最近总是头疼，可能是什么原因？",
            "什么是高血压？如何预防？",
            "感冒和流感有什么区别？",
            "小孩发烧到38度怎么办？",
        ],
        theme="soft",
    )

    print(f"\n启动 Gradio Web UI (端口: 7860)...")
    demo.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医疗模型推理")
    parser.add_argument("--web", action="store_true", help="启动 Gradio Web 界面")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="基座模型名称或路径")
    parser.add_argument("--lora_weights", default="./output/sft",
                        help="LoRA adapter 路径")
    parser.add_argument("--port", type=int, default=7860,
                        help="Gradio 端口")
    args = parser.parse_args()

    SYSTEM_PROMPT = (
        "你是一个专业的医疗健康助手，具备丰富的医学知识。"
        "请用专业但易懂的语言回答用户的健康问题。"
        "如果用户描述的症状可能很严重，请建议就医。"
    )

    model, tokenizer = load_model(args.base_model, args.lora_weights)

    if args.web:
        launch_web_ui(model, tokenizer, SYSTEM_PROMPT)
    else:
        chat(model, tokenizer, SYSTEM_PROMPT)
