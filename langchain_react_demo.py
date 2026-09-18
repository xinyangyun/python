"""示例：ReAct 模式，让大模型按 Thought/Action/Observation 自主调用工具。"""

import json
import os
import re

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# 设置 DeepSeek API Base 地址；API Key 从环境变量读取
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# 创建 DeepSeek 模型
model = ChatOpenAI(model="deepseek-chat", temperature=0)


@tool
def get_product_price(product_name: str) -> float:
    """查询商品价格，参数为商品名称。"""
    prices = {
        "苹果": 13,
        "香蕉": 3.2,
        "牛奶": 12.0,
    }
    return prices.get(product_name, 0.0)


@tool
def get_product_stock(product_name: str) -> int:
    """查询商品库存，参数为商品名称。"""
    stocks = {
        "苹果": 100,
        "香蕉": 50,
        "牛奶": 0,
    }
    return stocks.get(product_name, 0)


tools = [get_product_price, get_product_stock]
tool_map = {item.name: item for item in tools}


REACT_PROMPT = PromptTemplate.from_template(
    """你是一个乐于助人的助手。请使用工具回答用户问题。

你可以使用以下工具：
{tools}

工具名列表：{tool_names}

请严格按以下格式回答：
Question: 用户问题
Thought: 你的思考过程
Action: 要使用的工具名
Action Input: 工具参数
每轮只输出到 Action Input 为止，不要输出 Observation 或 Final Answer。
Observation: 工具返回结果
...（可以重复 Thought/Action/Action Input/Observation）
Thought: 我现在知道了最终答案
Final Answer: 最终回答

开始！

Question: {input}
Thought: {agent_scratchpad}"""
)


def parse_action(text: str):
    print("parse_action:", text)
    action_match = re.search(r"Action:\s*([A-Za-z_]+)", text)
    input_match = re.search(r"Action Input:\s*([^\n]+)", text)

    if action_match and input_match:
        action_name = action_match.group(1).strip()
        raw_input = input_match.group(1).strip()

        try:
            args = json.loads(raw_input)
            if not isinstance(args, dict):
                args = {"product_name": raw_input}
        except json.JSONDecodeError:
            args = {"product_name": raw_input.strip('"')}

        return "action", action_name, args, None

    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        return "final", final_match.group(1).strip(), None, None

    return "error", text, None, None


def run_react(question: str) -> str:
    scratchpad = ""

    for _ in range(5):
        prompt_text = REACT_PROMPT.format(
            tools="\n".join(item.name + ": " + item.description for item in tools),
            tool_names=", ".join(item.name for item in tools),
            input=question,
            agent_scratchpad=scratchpad,
        )

        print("提示词模板：", prompt_text)
        response = model.invoke([HumanMessage(content=prompt_text)])
        print("大模型返回：", response)
        text = response.content

        print("response:", text)
        print("---")

        kind, action_name, args, _ = parse_action(text)
        print("kind:, action_name:,  args:", kind, action_name, args)

        if kind == "final":
            return action_name

        if kind == "error" or action_name not in tool_map:
            scratchpad += text + "\nObservation: 格式错误，请重新输出 Action 和 Action Input。\n"
            continue

        result = tool_map[action_name].invoke(args)
        print("Observation2:", result)
        print("---")

        step_text = text.split("Observation:", 1)[0].strip()
        scratchpad += f"{step_text}\nObservation: {result}\n"
        print("scratchpad:", scratchpad)

    return "超过最大步数，已停止。"


def main() -> None:
    answer = run_react("苹果的价格是多少？")
    print("最终回答1:", answer)


if __name__ == "__main__":
    main()
