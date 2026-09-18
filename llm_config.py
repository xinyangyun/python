"""统一的大模型配置与初始化。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI 兼容接口配置。

    默认值与当前项目的模型服务配置一致；生产环境建议通过环境变量覆盖，
    不要把 API Key 直接提交到代码仓库。
    """

    api_key: str = os.getenv(
        "DEEPSEEK_API_KEY",
        os.getenv("OPENAI_API_KEY", ""),
    )
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_retries: int = 2

    def __post_init__(self) -> None:
        """将配置写入 LangChain/OpenAI 默认读取的环境变量。"""
        if not self.api_key:
            raise ValueError(
                "请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量。"
            )
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_API_BASE"] = self.base_url
        os.environ["OPENAI_BASE_URL"] = self.base_url

    @property
    def request_base_url(self) -> str:
        """返回用于 HTTP 请求的地址，不改变原始环境变量配置。"""
        if self.base_url.startswith(("http://", "https://")):
            return self.base_url
        return f"https://{self.base_url}"

    def create_chat_model(self, **kwargs: object) -> ChatOpenAI:
        """根据当前配置创建一个 ChatOpenAI 实例。"""
        options: dict[str, object] = {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.request_base_url,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
        }
        options.update(kwargs)
        return ChatOpenAI(**options)


def create_llm(**kwargs: object) -> ChatOpenAI:
    """便捷方法：使用默认配置初始化大模型。"""
    return LLMConfig().create_chat_model(**kwargs)


if __name__ == "__main__":
    llm = create_llm()
    print(f"已初始化模型：{llm.model_name}")
