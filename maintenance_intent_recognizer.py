"""设备运维子流程意图识别模块。

支持四类子流程：
1. 根因分析
2. 维修指导
3. 维修工单
4. 设备运维报表
"""

def recognize_maintenance_intent(user_input: str) -> dict[str, str]:
    """识别用户输入应进入的设备运维子流程。

    Args:
        user_input: 用户输入的自然语言文本。

    Returns:
        包含毕昇平台子流程节点名称的字典，键名为 result。

    Raises:
        ValueError: 输入为空或不是字符串时抛出。
    """
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input 必须是非空字符串")

    # 毕昇代码节点无法读取函数外部声明，因此配置全部定义在函数内部。
    intent_keywords: dict[str, dict[str, int]] = {
        "root_cause_analysis": {
            "根因分析": 5,
            "原因分析": 4,
            "故障诊断": 4,
            "故障原因": 4,
            "根本原因": 4,
            "为什么故障": 3,
            "为什么停机": 3,
            "异常原因": 3,
            "问题定位": 3,
            "故障定位": 3,
            "排查原因": 3,
            "诊断": 2,
            "原因": 1,
            "异常": 1,
        },
        "repair_guidance": {
            "维修指导": 5,
            "维修方案": 4,
            "维修步骤": 4,
            "检修指导": 4,
            "检修方案": 4,
            "处理方法": 3,
            "解决办法": 3,
            "怎么维修": 3,
            "如何维修": 3,
            "怎么修": 3,
            "如何处理": 3,
            "操作步骤": 3,
            "更换部件": 2,
            "维修手册": 2,
            "检修": 1,
            "维修": 1,
        },
        "repair_work_order": {
            "维修工单": 5,
            "检修工单": 5,
            "报修工单": 5,
            "创建工单": 4,
            "新建工单": 4,
            "提交工单": 4,
            "派发工单": 4,
            "工单状态": 3,
            "工单进度": 3,
            "工单记录": 3,
            "工单处理": 3,
            "报修": 3,
            "派单": 2,
            "接单": 2,
            "完工": 2,
            "工单": 2,
        },
        "equipment_operation_report": {
            "设备运维报表": 5,
            "运维报表": 5,
            "设备报表": 4,
            "故障报表": 4,
            "维修报表": 4,
            "设备统计": 4,
            "运维统计": 4,
            "故障统计": 3,
            "维修统计": 3,
            "停机统计": 3,
            "设备台账": 3,
            "故障率": 3,
            "完好率": 3,
            "维修时长": 3,
            "停机时长": 3,
            "趋势分析": 2,
            "同比": 2,
            "环比": 2,
            "报表": 2,
            "统计": 1,
        },
    }

    normalized_text = user_input.strip().lower()
    scores: dict[str, int] = {}

    for intent, weighted_keywords in intent_keywords.items():
        matched = tuple(
            keyword
            for keyword in weighted_keywords
            if keyword.lower() in normalized_text
        )
        scores[intent] = sum(weighted_keywords[keyword] for keyword in matched)

    highest_score = max(scores.values())

    # 未命中关键词时无法判断所属子流程。
    if highest_score == 0:
        return {"result": "unknown_intent"}

    # 最高类别得分占所有类别总得分的比例作为置信度。
    confidence = highest_score / sum(scores.values())
    if confidence < 0.8:
        return {"result": "unknown_intent"}

    # 同分时优先选择更明确、对动作要求更强的工单和报表子流程。
    priority = (
        "repair_work_order",
        "equipment_operation_report",
        "root_cause_analysis",
        "repair_guidance",
    )
    selected_intent = next(
        intent for intent in priority if scores[intent] == highest_score
    )

    return {"result": selected_intent}


if __name__ == "__main__":
    examples = (
        "分析一下空压机频繁停机的根本原因",
        "给我一份更换水泵轴承的维修步骤",
        "为二号生产线的故障创建维修工单",
        "生成本月设备故障率和停机时长的运维报表",
        "金铲铲之战是什么",
    )
    for example in examples:
        print(example, "->", recognize_maintenance_intent(example))
