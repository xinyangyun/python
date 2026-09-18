"""示例：让大模型自主选择并调用本地 skills。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from intent_recognizer import recognize_intent
from llm_config import create_llm
from product import list_products


@tool
def query_product_price(product_name: str) -> str:
    """查询商品价格，参数为商品名称。"""
    products = {
        "苹果": "13 元/斤",
        "香蕉": "3.2 元/斤",
        "牛奶": "12 元/盒",
    }
    return products.get(product_name, f"未找到商品：{product_name}")


@tool
def query_product_stock(product_name: str) -> str:
    """查询商品库存，参数为商品名称。"""
    stocks = {
        "苹果": "库存 100",
        "香蕉": "库存 50",
        "牛奶": "库存 0，暂时缺货",
    }
    return stocks.get(product_name, f"未找到商品：{product_name}")


@tool
def classify_intent(user_input: str) -> dict[str, object]:
    """识别用户请求属于知识查询、数据报表还是任务执行。"""
    result = recognize_intent(user_input)
    return {
        "intent": result.intent.value,
        "confidence": result.confidence,
        "matched_keywords": list(result.matched_keywords),
    }


@tool
def list_available_products() -> list[dict[str, object]]:
    """列出当前商品列表。"""
    return [item.model_dump() for item in list_products()]


def run_skill_call(user_input: str) -> str:
    """运行一轮或多轮 skill 调用，直到模型返回最终答案。"""
    skills = [
        query_product_price,
        query_product_stock,
        classify_intent,
        list_available_products,
    ]
    skill_map = {skill.name: skill for skill in skills}
    model = create_llm(temperature=0).bind_tools(skills)
    messages = [
        SystemMessage(
            content=(
                "你是一个业务助手。需要商品信息或意图分类时调用对应 skill，"
                "拿到结果后用简洁中文回答；不需要 skill 时直接回答。"
            )
        ),
        HumanMessage(content=user_input),
    ]

    for _ in range(5):
        response = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return str(response.content)

        for tool_call in response.tool_calls:
            skill = skill_map.get(tool_call["name"])
            if skill is None:
                raise ValueError(f"模型请求了未注册的 skill：{tool_call['name']}")
            result = skill.invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    return "skill 调用次数超过上限，已停止。"


def main() -> None:
    examples = (
        "苹果现在多少钱？",
        "牛奶还有库存吗？",
        "帮我判断：查询本月销售报表属于什么类型的请求？",
    )
    for example in examples:
        print(f"用户：{example}")
        print(f"助手：{run_skill_call(example)}\n")


if __name__ == "__main__":
    main()
