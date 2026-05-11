from service.retrieval_engine import RetrievalEngine
from model.llm import LLM
from utils.logger import logger
from config import *


class RAGChain:
    """
    完整 RAG 问答过程
    整合：混合检索 → 上下文格式化 → 提示词构建 → 大模型生成
    """

    def __init__(self):
        self.retrieval_engine = None
        self.llm = None
        
        self._init_components()

    def _init_components(self):
        """初始化所有组件"""
        logger.info("正在初始化 RAG 系统组件...")
        
        self.retrieval_engine = RetrievalEngine()
        self.retrieval_engine.load_index()
        
        self.llm = LLM()
        
        logger.info("RAG 系统所有组件初始化完成")

    def _format_context(self, context_list: list[str]) -> str:
        """格式化上下文"""
        if not context_list:
            return "无相关参考内容"
        
        formatted_context = ""
        for idx, content in enumerate(context_list, 1):
            formatted_context += f"【参考内容{idx}】\n{content}\n\n"
        
        return formatted_context

    def _build_prompt(self, question: str, context: str) -> str:
        """构建提示词"""
        prompt = f"""
        你是一个专业的文档问答助手，请严格遵守以下规则：
        1. 只能使用下方【匹配参考内容】中的信息回答问题，绝对不能编造任何内容
        2. 如果参考内容中没有相关信息，直接回复：「未在文档中找到该问题的相关内容」
        3. 回答要简洁明了、逻辑清晰，分点说明，不要重复啰嗦
        4. 不要输出任何与问题无关的内容，不要解释你的回答过程

        {context}

        用户问题：{question}
        回答：
        """
        return prompt.strip()

    def query(self, question: str) -> str:
        """执行RAG 问答"""
        if not question.strip():
            return "请输入你想要咨询的问题~"
        
        logger.info(f"用户问题: {question}")
        
        context_list = self.retrieval_engine.hybrid_search(question)
        formatted_context = self._format_context(context_list)
        logger.info(f"检索到 {len(context_list)} 条相关上下文")
        
        prompt = self._build_prompt(question, formatted_context)
        answer = self.llm.generate(prompt)
        
        return answer
