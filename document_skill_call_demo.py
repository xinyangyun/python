"""示例：读取 SKILL.md 文档，并让大模型按文档规则执行任务。"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from llm_config import create_llm


SKILL_PATH = Path(__file__).parent / "skills" / "customer_service" / "SKILL.md"


def load_skill(skill_path: Path = SKILL_PATH) -> str:
    """读取文档型 skill 的完整内容。"""
    if not skill_path.exists():
        raise FileNotFoundError(f"找不到 skill 文档：{skill_path}")
    return skill_path.read_text(encoding="utf-8")


def run_skill(user_input: str) -> dict[str, object]:
    """将 skill 文档作为系统指令，调用大模型处理用户输入。"""
    skill_document = load_skill()
    model = create_llm(temperature=0)
    messages = [
        SystemMessage(
            content=(
                "请严格按照下面的 SKILL 文档执行任务。"
                "SKILL 文档中的规则优先于一般回答习惯。\n\n"
                f"{skill_document}"
            )
        ),
        HumanMessage(content=user_input),
    ]
    response = model.invoke(messages)
    content = str(response.content).strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型未按 SKILL 要求返回 JSON：{content}") from exc
    if not isinstance(result, dict):
        raise ValueError("SKILL 返回结果必须是 JSON 对象")
    return result


def main() -> None:
    feedback = "用户说收到的耳机左耳没有声音，明天要出差，希望尽快处理。"
    print(json.dumps(run_skill(feedback), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
