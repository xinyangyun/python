"""使用大模型识别设备运维子流程意图。"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from llm_config import create_llm


def create_model() -> ChatOpenAI:
    """创建用于设备运维意图识别的低温度聊天模型。"""
    return create_llm(temperature=0)


def _parse_json(content: str) -> dict[str, object]:
    """解析模型返回的 JSON，兼容代码块和前后附加文字。"""
    cleaned = content.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型未返回合法 JSON")
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("模型返回结果必须是 JSON 对象")
    return data


def recognize_maintenance_intent_with_llm(
    user_input: str,
    model: ChatOpenAI | None = None,
) -> dict[str, str]:
    """调用大模型识别设备运维子流程并返回毕昇节点结果。

    当模型置信度低于 0.8、意图不合法或返回内容格式错误时，
    返回 {"result": "unknown_intent"}。
    """
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input 必须是非空字符串")

    intent_descriptions = {
        "root_cause_analysis": "分析设备故障、异常、停机的根本原因，定位故障来源",
        "repair_guidance": "提供设备维修、检修、部件更换的处理方法和操作步骤",
        "repair_work_order": "创建、提交、派发、查询或处理设备维修工单",
        "equipment_operation_report": "查询或生成设备运维、故障、维修、停机等统计报表",
    }
    valid_intents = tuple(intent_descriptions)
    descriptions = "\n".join(
        f"- {intent}: {description}"
        for intent, description in intent_descriptions.items()
    )
    system_prompt = f"""你是设备运维系统的意图分类器。
请根据用户输入，从以下四个意图中选择且只能选择一个：
{descriptions}

只返回一个合法 JSON 对象，不要输出 Markdown、解释文字或代码块，格式如下：
{{
  "intent": "root_cause_analysis | repair_guidance | repair_work_order | equipment_operation_report",
  "confidence": 0.0,
  "reason": "一句话说明判断依据"
}}

confidence 必须是 0 到 1 之间的数字。无法确定时选择最接近的类别并降低置信度。"""


    print("system_prompt :", system_prompt)

    response = (model or create_model()).invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input.strip()),
        ]
    )

    try:
        data = _parse_json(str(response.content))
        intent = str(data["intent"])
        confidence = float(data["confidence"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {"result": "unknown_intent"}

    if intent not in valid_intents or not 0 <= confidence <= 1:
        return {"result": "unknown_intent"}
    if confidence < 0.8:
        return {"result": "unknown_intent"}

    return {"result": intent}


def main() -> None:
    """调用真实模型测试四类设备运维意图。"""
    examples = (
        # "分析一下空压机频繁停机的根本原因",
        # "给我一份更换水泵轴承的维修步骤",
        # "为二号生产线的故障创建维修工单",
        # "生成本月设备故障率和停机时长的运维报表",
        "金铲铲之战是什么",
    )
    for example in examples:
        try:
            result = recognize_maintenance_intent_with_llm(example)
        except Exception as exc:
            print(f"识别失败：{exc}")
            continue
        print(example, "->", result)


if __name__ == "__main__":
    main()
