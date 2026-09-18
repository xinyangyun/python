"""用户意图识别模块。

支持三类意图：
1. 知识库检索
2. 数据或报表查询
3. 工作流或其他任务执行
"""

from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    """系统支持的用户意图。"""

    # 查询制度、文档、手册等非结构化知识。
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    # 查询业务数据、统计指标或报表。
    DATA_REPORT_QUERY = "data_report_query"
    # 执行工作流、发送通知或处理其他操作型任务。
    TASK_EXECUTION = "task_execution"


@dataclass(frozen=True)
class IntentResult:
    """一次意图识别的结果。"""

    # 最终识别出的意图类型。
    intent: Intent
    # 最高意图得分占全部意图得分的比例，取值范围为 0 到 1。
    confidence: float
    # 用户输入中实际命中的、属于最终意图的关键词。
    matched_keywords: tuple[str, ...]


# 每个关键词后面的整数是权重。表达越明确，权重越高：
# 例如“数据报表”比含义较宽泛的“多少”更能确定数据查询意图。
_INTENT_KEYWORDS: dict[Intent, dict[str, int]] = {
    Intent.KNOWLEDGE_RETRIEVAL: {
        "知识库": 3,
        "知识文档": 3,
        "操作手册": 3,
        "使用手册": 3,
        "规章制度": 3,
        "制度": 2,
        "文档": 2,
        "资料": 2,
        "说明书": 2,
        "怎么使用": 2,
        "如何操作": 2,
        "是什么": 1,
    },
    Intent.DATA_REPORT_QUERY: {
        "数据报表": 4,
        "统计报表": 4,
        "数据查询": 3,
        "查询数据": 3,
        "查数据": 3,
        "数据分析": 3,
        "报表": 3,
        "数据库": 3,
        "销售额": 3,
        "销售数据": 3,
        "订单数据": 3,
        "库存数据": 3,
        "统计": 2,
        "汇总": 2,
        "同比": 2,
        "环比": 2,
        "趋势": 2,
        "指标": 2,
        "数量": 1,
        "多少": 1,
    },
    Intent.TASK_EXECUTION: {
        "工作流": 4,
        "workflow": 4,
        "审批流程": 4,
        "执行任务": 3,
        "执行": 2,
        "运行": 2,
        "启动": 2,
        "创建": 2,
        "生成": 2,
        "发送": 2,
        "通知": 2,
        "审批": 2,
        "导出": 2,
        "下载": 2,
        "提醒": 2,
    },
}


def recognize_intent(user_input: str) -> IntentResult:
    """识别用户输入所属的意图。

    Args:
        user_input: 用户输入的自然语言文本。

    Returns:
        包含意图、置信度和命中关键词的识别结果。

    Raises:
        ValueError: 输入为空或不是字符串时抛出。
    """
    # 提前拒绝无效输入，避免空文本被错误归入默认意图。
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input 必须是非空字符串")

    # 去除首尾空格并转为小写，使英文关键词匹配不区分大小写。
    normalized_text = user_input.strip().lower()

    # scores 保存每类意图的累计得分，matches 保存各类命中的关键词。
    scores: dict[Intent, int] = {}
    matches: dict[Intent, tuple[str, ...]] = {}

    # 逐类检查关键词，并将所有命中关键词的权重相加。
    for intent, weighted_keywords in _INTENT_KEYWORDS.items():
        matched = tuple(
            keyword
            for keyword in weighted_keywords
            if keyword.lower() in normalized_text
        )
        matches[intent] = matched
        scores[intent] = sum(weighted_keywords[keyword] for keyword in matched)

    # 找出三个意图中的最高分。
    highest_score = max(scores.values())

    # 没有命中任何关键词时，将请求交给通用任务处理逻辑。
    if highest_score == 0:
        return IntentResult(
            intent=Intent.TASK_EXECUTION,
            confidence=0.5,
            matched_keywords=(),
        )

    # 相同得分时，优先选择语义更明确的知识库或数据报表意图。
    priority = (
        Intent.KNOWLEDGE_RETRIEVAL,
        Intent.DATA_REPORT_QUERY,
        Intent.TASK_EXECUTION,
    )
    selected_intent = next(
        intent for intent in priority if scores[intent] == highest_score
    )

    # 用最高分在总分中的占比表示置信度。
    # 若用户同时表达多个意图，该比例会下降，提示上层系统存在歧义。
    total_score = sum(scores.values())
    confidence = round(highest_score / total_score, 2)

    return IntentResult(
        intent=selected_intent,
        confidence=confidence,
        matched_keywords=matches[selected_intent],
    )


if __name__ == "__main__":
    # 直接运行此文件时，分别演示三类意图的识别结果。
    examples = (
        "帮我从知识库查一下员工请假制度",
        "查询本月的销售数据并展示报表",
        "启动合同审批工作流并通知负责人",
    )
    for example in examples:
        print(example, "->", recognize_intent(example))
