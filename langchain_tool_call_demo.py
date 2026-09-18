"""示例：让大模型自己判断是否调用工具。"""

import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# 设置 DeepSeek API Base 地址；API Key 从环境变量读取
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# 创建 DeepSeek 模型
model = ChatOpenAI(model="deepseek-chat", temperature=0.7)


@tool
def get_product_price(product_name: str) -> float:
    """查询商品价格，参数为商品名称。"""
    prices = {
        "apple": 5.5,
        "banana": 3.2,
        "milk": 12.0,
    }
    return prices.get(product_name, 12.0)


@tool
def get_product_stock(product_name: str) -> int:
    """查询商品库存，参数为商品名称。"""
    stocks = {
        "apple": 100,
        "banana": 50,
        "milk": 0,
    }
    return stocks.get(product_name, 0)


def run_tool_call_loop(question: str) -> str:
    llm_with_tools = model.bind_tools(
        [get_product_price, get_product_stock]
    )

    messages = [
        SystemMessage(content="你是一个乐于助人的助手。"),
        HumanMessage(content=question),
    ]

    tool_map = {
        "get_product_price": get_product_price,
        "get_product_stock": get_product_stock,
    }

    for _ in range(5):
        response = llm_with_tools.invoke(messages)

        print(response.tool_calls)
        if not response.tool_calls:
            return response.content

        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            result = tool_map[tool_name].invoke(tool_args)

            print(result)
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    return "工具调用次数过多，已停止。"


def main() -> None:
    answer = run_tool_call_loop("苹果的价格是多少？")
    print(answer)


if __name__ == "__main__":
    main()
