"""示例：让大模型自主选择 Skill，再加载对应 SKILL.md 执行。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from llm_config import create_llm


ROOT = Path(__file__).parent
SKILL_INDEX_PATH = ROOT / "skills" / "index.json"


@dataclass(frozen=True)
class SkillMeta:
    """用于路由阶段的 Skill 元数据，不包含详细提示词。"""

    name: str
    description: str
    path: Path


def load_skill_index(index_path: Path = SKILL_INDEX_PATH) -> tuple[SkillMeta, ...]:
    """从统一的 index.json 文件读取 Skill 元数据。"""
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Skill 目录必须是 JSON 数组")

    skills: list[SkillMeta] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Skill 目录中的每一项必须是 JSON 对象")
        name = item.get("name")
        description = item.get("description")
        relative_path = item.get("path")
        if not all(isinstance(value, str) and value for value in (name, description, relative_path)):
            raise ValueError("Skill 元数据必须包含非空的 name、description 和 path")
        path = index_path.parent / relative_path
        skills.append(SkillMeta(name, description, path))
    return tuple(skills)


# 路由阶段只使用目录文件中的 name 和 description；path 仅在选中后使用。
SKILLS = load_skill_index()
SKILL_INDEX = {skill.name: skill for skill in SKILLS}


def parse_json(content: str) -> dict[str, object]:
    """解析模型 JSON，并兼容 Markdown 代码块。"""
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE
    )
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"模型未返回合法 JSON：{content}")
        result = json.loads(cleaned[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("模型返回结果必须是 JSON 对象")
    return result


def choose_skill(user_input: str) -> str:
    """让模型从技能目录中选择最匹配的 Skill。"""
    catalog = "\n".join(
        f"- {skill.name}: {skill.description}" for skill in SKILLS
    )
    prompt = f"""你是 Skill 路由器。
根据用户问题，从下面的 Skill 中选择一个最匹配的技能：
{catalog}

只返回 JSON，不要输出其他内容：
{{"skill": "技能名称", "reason": "选择原因"}}

用户问题：{user_input}"""
    response = create_llm(temperature=0).invoke([HumanMessage(content=prompt)])
    result = parse_json(str(response.content))
    skill_name = str(result.get("skill", ""))
    if skill_name not in SKILL_INDEX:
        raise ValueError(f"模型选择了未注册的 Skill：{skill_name}")
    return skill_name


def load_skill_content(skill_name: str) -> str:
    """模型选中 Skill 后，按名称读取对应的详细文档。"""
    skill = SKILL_INDEX.get(skill_name)
    if skill is None:
        raise ValueError(f"未找到 Skill 元数据：{skill_name}")
    if not skill.path.exists():
        raise FileNotFoundError(f"Skill 文档不存在：{skill.path}")
    return skill.path.read_text(encoding="utf-8")


def run_skill(user_input: str) -> dict[str, object]:
    """先选 Skill，再加载对应文档并执行。"""
    skill_name = choose_skill(user_input)
    skill_document = load_skill_content(skill_name)
    model = create_llm(temperature=0)
    messages = [
        SystemMessage(
            content=(
                f"你选择了 `{skill_name}` Skill。请严格按照下面的 Skill 文档处理用户问题，"
                "只返回文档规定格式的 JSON。\n\n"
                f"{skill_document}"
            )
        ),
        HumanMessage(content=user_input),
    ]
    response = model.invoke(messages)
    result = parse_json(str(response.content))
    result["selected_skill"] = skill_name
    return result


def main() -> None:
    examples = (
        "牛奶还有多少库存？",
        "请查询订单 1001 的物流状态",
        "用户反馈耳机左耳没有声音，应该怎么处理？",
    )
    for example in examples:
        print(f"用户：{example}")
        try:
            result = run_skill(example)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"处理失败：{exc}")
        print()


if __name__ == "__main__":
    main()
