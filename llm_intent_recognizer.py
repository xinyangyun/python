"""使用大模型进行用户意图识别的最小示例。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from intent_recognizer import Intent
from llm_config import create_llm


@dataclass(frozen=True)
class LLMIntentResult:
    """大模型返回的结构化意图结果。"""

    intent: Intent
    confidence: float
    reason: str
    matched_keywords: tuple[str, ...]


_INTENT_DESCRIPTIONS = {
    Intent.KNOWLEDGE_RETRIEVAL: "查询制度、文档、手册、说明书等非结构化知识",
    Intent.DATA_REPORT_QUERY: "查询业务数据、统计指标、报表或趋势",
    Intent.TASK_EXECUTION: "执行工作流、发送通知、导出或其他操作任务",
}

_SYSTEM_PROMPT = """你是一个用户意图分类器。
请从以下三类意图中选择且只能选择一类：
{intent_descriptions}

只返回一个合法 JSON 对象，不要输出 Markdown、解释文字或代码块，格式如下：
{{
  "intent": "knowledge_retrieval | data_report_query | task_execution",
  "confidence": 0.0,
  "reason": "一句话说明判断依据",
  "matched_keywords": ["命中的关键词"]
}}

confidence 必须是 0 到 1 之间的数字；无法确定时选择最接近的类别并降低置信度。"""


def create_model() -> ChatOpenAI:
    """创建用于意图识别的低温度聊天模型。"""
    return create_llm(temperature=0)


def _parse_json(content: str) -> dict[str, object]:
    """解析模型返回的 JSON，兼容偶尔出现的代码块包裹。"""
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
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


def recognize_intent_with_llm(
    user_input: str,
    model: ChatOpenAI | None = None,
) -> LLMIntentResult:
    """调用大模型识别用户意图并校验返回结果。"""
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input 必须是非空字符串")

    descriptions = "\n".join(
        f"- {intent.value}: {description}"
        for intent, description in _INTENT_DESCRIPTIONS.items()
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT.format(intent_descriptions=descriptions)),
        HumanMessage(content=user_input.strip()),
    ]
    response = (model or create_model()).invoke(messages)
    data = _parse_json(str(response.content))

    try:
        intent = Intent(str(data["intent"]))
        confidence = float(data["confidence"])
        reason = str(data["reason"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"模型返回字段不完整或格式错误: {data}") from exc

    if not 0 <= confidence <= 1:
        raise ValueError("confidence 必须在 0 到 1 之间")

    raw_keywords = data.get("matched_keywords", [])
    if not isinstance(raw_keywords, list):
        raise ValueError("matched_keywords 必须是数组")

    return LLMIntentResult(
        intent=intent,
        confidence=round(confidence, 2),
        reason=reason,
        matched_keywords=tuple(str(item) for item in raw_keywords),
    )


def main() -> None:
    examples = (
        "帮我查一下员工请假制度",
        "统计本月华东区的销售额并生成报表",
        "启动合同审批流程并通知负责人",
    )
    for example in examples:
        try:
            result = recognize_intent_with_llm(example)
        except Exception as exc:
            print(f"识别失败：{exc}")
            continue
        print(example, "->", result)


if __name__ == "__main__":
    main()
