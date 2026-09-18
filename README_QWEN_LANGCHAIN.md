# LangChain 调用千问 Demo

## 安装依赖

```bash
pip install -U langchain-core langchain-openai
```

## 运行

```bash
python qwen_langchain_demo.py
python qwen_langchain_demo.py "请介绍一下 LangChain"
python qwen_langchain_demo.py --model qwen-turbo "写一首关于秋天的短诗"
```

脚本通过千问的 OpenAI 兼容接口调用模型，默认模型是 `qwen-plus`。API Key 已按你的要求直接写在 `qwen_langchain_demo.py` 中；如果将代码提交到远程仓库，建议及时改成环境变量并轮换密钥。
