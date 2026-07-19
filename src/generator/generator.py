"""生成模块：拼接 Prompt + 调用 DeepSeek LLM + 返回回答 + 引用来源"""
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载 .env（未传 key 时兜底）
load_dotenv()

DEFAULT_SYSTEM_PROMPT = """你是企业知识库助手。根据提供的参考资料回答用户问题。
- 如果参考资料中有答案，请引用资料内容回答。
- 关键：引用时必须带完整的来源标注，格式为【参考X·来源: 文件名】，不可省略文件名。
- 如果参考资料中没有相关信息，请明确说"根据现有资料无法回答"，不要编造。
- 回答简洁，不超过 200 字。"""


class Generator:
    """LLM 生成器：拼接 Prompt → 调 DeepSeek → 结构化返回"""

    def __init__(
        self,
        api_key: str = None, 
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        system_prompt: str = None,
    ):
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url,
        )
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def generate(
        self,
        question: str,
        contexts: list[str],
        sources: list[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """
        检索结果 → 拼接 Prompt → LLM 生成。

        参数:
          question: 用户问题
          contexts: 检索到的文档块列表
          sources: 每个文档块的来源标注（文件名/页码），长度与 contexts 一致

        返回: 带引用来源的完整回答
        """
        # 拼接参考资料
        refs = []
        for i, ctx in enumerate(contexts):
            src = sources[i] if sources and i < len(sources) else "未知来源"
            refs.append(f"【参考{i+1}·来源: {src}】\n{ctx}")

        context_block = "\n\n".join(refs)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"参考资料：\n{context_block}\n\n问题：{question}"},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def generate_stream(
        self,
        question: str,
        contexts: list[str],
        sources: list[str] = None,
    ):
        """流式生成（SSE，前端逐字显示用）"""
        refs = []
        for i, ctx in enumerate(contexts):
            src = sources[i] if sources and i < len(sources) else "未知来源"
            refs.append(f"【参考{i+1}·来源: {src}】\n{ctx}")

        context_block = "\n\n".join(refs)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"参考资料：\n{context_block}\n\n问题：{question}"},
            ],
            temperature=0.0,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content