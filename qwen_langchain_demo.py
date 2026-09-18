"""使用 LangChain 调用通义千问的最小示例。

运行：
    python qwen_langchain_demo.py
    python qwen_langchain_demo.py "请介绍一下 LangChain"
    python qwen_langchain_demo.py --model qwen-turbo "写一首短诗"
"""

from __future__ import annotations

import argparse
import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


def create_model(model_name: str = DEFAULT_MODEL) -> ChatOpenAI:
    """创建一个通过 OpenAI 兼容接口访问千问的 LangChain 模型。"""
    if not QWEN_API_KEY:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量。")
    return ChatOpenAI(
        model=model_name,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        temperature=0.7,
        max_retries=2,
    )


def ask_qwen(question: str, model_name: str = DEFAULT_MODEL) -> str:
    """向千问发送一轮对话并返回文本回答。"""
    model = create_model(model_name)
    messages = [
        SystemMessage(content="你是一个乐于助人的中文助手，请简洁、准确地回答问题。"),
        HumanMessage(content=question),
    ]
    response = model.invoke(messages)
    return str(response.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangChain + 通义千问示例")
    parser.add_argument(
        "question",
        nargs="?",
        default="请用一句话解释什么是 LangChain。",
        help="要发送给千问的问题",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="千问模型名称，例如 qwen-plus、qwen-turbo、qwen-max",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        answer = ask_qwen(args.question, args.model)
    except Exception as exc:
        print(f"调用千问失败：{exc}", file=sys.stderr)
        print(
            "请检查网络、API Key、模型名称，以及是否已安装 langchain-openai。",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(answer)


if __name__ == "__main__":
    main()
