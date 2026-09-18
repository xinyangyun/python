# Demo 说明

本目录中的示例主要用于演示 Web API、文本处理、意图识别，以及 LangChain 调用大模型的常见方式。

## 示例总览

| 文件 | 主要用途 | 设计方式 |
| --- | --- | --- |
| `fastapi_demo.py` | 提供商品增删查示例 API | 使用 FastAPI 定义路由和响应模型；业务逻辑委托给 `product.py`；批量查询通过异步并发执行。 |
| `product.py` | 提供商品领域的基础业务逻辑 | 使用 Pydantic 定义商品模型；用内存字典模拟数据存储；封装新增、列表查询、单个查询和批量查询。 |
| `time_extractor.py` | 从中文文本中识别并标准化时间表达 | 通过正则匹配绝对日期、相对日期、星期、时间偏移和时刻；统一转换为 `datetime`；去除重叠结果。 |
| `intent_recognizer.py` | 判断用户请求属于哪类意图 | 预置三类意图及关键词权重；按命中关键词累计得分；返回意图、置信度和命中词。 |
| `llm_intent_recognizer.py` | 使用大模型识别用户意图 | 复用三类意图定义；通过系统提示词约束模型输出 JSON；代码负责解析、校验意图和置信度。 |
| `llm_config.py` | 统一配置并初始化大模型 | 用 `LLMConfig` 集中管理 API Key、接口地址、模型名和生成参数，并提供 `create_llm()` 快捷初始化方法。 |
| `llm_skill_call_demo.py` | 演示大模型自主调用本地 skills | 使用 `@tool` 注册价格、库存、商品列表和意图识别技能；模型通过 `bind_tools` 选择技能，程序执行后用 `ToolMessage` 回传结果。 |
| `document_skill_call_demo.py` | 演示调用文档型 Skill | 从 `skills/customer_service/SKILL.md` 读取角色、流程和输出规范，将文档注入系统提示词，再校验模型返回的 JSON。 |
| `skill_router_demo.py` | 演示多 Skill 自动路由 | 从统一的 `skills/index.json` 读取所有 Skill 的 `name` 和 `description`；模型选中名称后，再按名称加载对应的 `SKILL.md` 执行。 |
| `qwen_langchain_demo.py` | 用 LangChain 调用通义千问 | 使用 OpenAI 兼容接口创建 `ChatOpenAI`；组合系统消息和用户消息；支持命令行传入问题和模型名。 |
| `langchain_llm_demo.py` | 演示最简单的一轮大模型调用 | 初始化聊天模型；构造 `SystemMessage` 与 `HumanMessage`；调用 `invoke` 获取文本回答。 |
| `langchain_tool_call_demo.py` | 让模型自主决定是否调用工具 | 使用 `@tool` 声明价格和库存查询工具；模型输出工具调用请求；程序执行工具并通过 `ToolMessage` 回传，循环直到得到最终回答。 |
| `langchain_react_demo.py` | 演示 ReAct 思路下的工具调用 | 通过提示词约束模型输出 `Thought/Action/Action Input`；正则解析动作；执行工具后把结果写入 scratchpad，再继续推理。 |
| `test.py` | 将 JSON 字符串格式化为易读文本 | 先用 `json.loads` 校验并解析输入，再用 `json.dumps` 按缩进输出；解析失败时返回错误信息。 |

## 运行方式

### FastAPI 示例

```bash
python fastapi_demo.py
```

启动后可访问 `http://127.0.0.1:8000/docs` 查看接口文档。商品数据保存在进程内存中，服务重启后会清空。

### 时间抽取与意图识别

```bash
python time_extractor.py
python intent_recognizer.py
python llm_intent_recognizer.py
```

三个文件都内置了示例输入，直接运行即可查看识别结果。大模型示例统一通过 `llm_config.py` 初始化；可在 `LLMConfig` 中配置 API Key、接口地址、模型名和生成参数，生产环境建议改为从环境变量读取密钥。

### LangChain 示例

```bash
python langchain_llm_demo.py
python qwen_langchain_demo.py "请介绍一下 LangChain"
python langchain_tool_call_demo.py
python langchain_react_demo.py
python llm_skill_call_demo.py
python document_skill_call_demo.py
```

运行前需要安装相应依赖，并准备可用的模型接口配置。当前部分示例将 API Key 直接写在代码中，仅适合本地演示，实际使用应改为环境变量并及时轮换密钥。

## 模块关系

`fastapi_demo.py` 是接口层，`product.py` 是商品业务层；两者共同组成一个最小的分层 Web API 示例。其余文件相互独立，分别展示文本规则处理、分类逻辑和大模型集成方式。
