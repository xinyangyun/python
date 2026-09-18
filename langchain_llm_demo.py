"""使用统一配置调用大模型的示例。"""

from langchain_core.messages import HumanMessage, SystemMessage

from llm_config import create_llm


model = create_llm()

def main() -> None:

    messages = [
        SystemMessage(content="你是一个乐于助人的助手。"),
        HumanMessage(content="请用一句话解释什么是 FastAPI。"),
    ]

    response = model.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    main()
