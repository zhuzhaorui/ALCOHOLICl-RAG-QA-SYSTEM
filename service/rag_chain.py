from typing import List
from langchain.schema import Document
from service.retrieval_engine import RetrievalEngine
from model.llm import LLM
from utils.logger import logger
from utils.prompt_loader import PromptLoader
from config import *
import tiktoken


class RAGChain:
    """
    完整RAG问答流水线
    整合：混合检索 → 上下文压缩 → Prompt构建 → 大模型生成
    """

    def __init__(self, prompt_template: str = None):
        """
        初始化RAG系统
        :param prompt_template: 使用的Prompt模板名称，None使用默认模板
        """
        self.retrieval_engine = None
        self.llm = None
        self.prompt_loader = PromptLoader()
        self.prompt_template = prompt_template
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # 用于token计数
        
        self._init_components()

    def _init_components(self):
        """初始化所有组件"""
        logger.info("正在初始化RAG系统组件...")
        self.retrieval_engine = RetrievalEngine()
        self.retrieval_engine.load_index()
        self.llm = LLM()
        logger.info("RAG系统所有组件初始化完成")

    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.tokenizer.encode(text))

    def compress_context(
        self,
        docs: List[Document],
        query: str,
        max_total_tokens: int = MAX_CONTEXT_TOKENS,
        max_doc_length: int = MAX_DOC_LENGTH,
        compressed_length: int = COMPRESSED_DOC_LENGTH
    ) -> List[Document]:
        """
        基于问题感知的智能上下文压缩
        :param docs: 检索到的原始文档列表
        :param query: 用户问题
        :param max_total_tokens: 总上下文最大token数
        :param max_doc_length: 单个文档最大长度阈值
        :param compressed_length: 压缩后单条信息最大长度
        :return: 压缩后的文档列表
        """
        if not ENABLE_CONTEXT_COMPRESSION or not docs:
            return docs

        # 先计算原始上下文的总token数
        total_tokens = sum(self._count_tokens(doc.page_content) for doc in docs)
        logger.info(f"原始上下文总token数: {total_tokens}，阈值: {max_total_tokens}")

        # 如果总token数未超过阈值，直接返回原始文档
        if total_tokens <= max_total_tokens:
            logger.info("总token数未超过阈值，跳过压缩")
            return docs

        logger.info(f"总token数超过阈值，启动上下文压缩")
        compressed_docs = []

        for idx, doc in enumerate(docs, 1):
            original_text = doc.page_content
            source = doc.metadata.get("source", "未知文档")
            
            # 如果文档本身很短，直接保留
            if len(original_text) <= max_doc_length:
                compressed_docs.append(doc)
                continue

            logger.info(f"正在压缩文档 {idx}（来源：{source}），原始长度: {len(original_text)} 字符")
            
            try:
                # 构建压缩提示词
                compression_prompt = f"""
请从以下文档中提炼与问题「{query}」最相关的关键信息。
要求：
1. 只保留与问题直接相关的内容，删除所有无关信息
2. 保持原文的表述方式和专业术语
3. 控制在{compressed_length}字以内
4. 不要添加任何个人观点或解释

文档内容：
{original_text[:3000]}  # 防止超长文档导致LLM调用失败

关键信息：
"""

                # 调用LLM进行压缩
                compressed_text = self.llm.generate(
                    user_prompt=compression_prompt,
                    system_prompt="你是一个专业的文档摘要助手，只提取与问题相关的关键信息。"
                ).strip()

                # 验证压缩结果
                if compressed_text and len(compressed_text) > 10:
                    # 创建新的Document对象，保留原有的元数据
                    compressed_doc = Document(
                        page_content=compressed_text,
                        metadata=doc.metadata
                    )
                    compressed_docs.append(compressed_doc)
                    logger.info(f"文档 {idx} 压缩完成，压缩后长度: {len(compressed_text)} 字符")
                else:
                    # 压缩结果为空，回退到智能截断
                    logger.warning(f"文档 {idx} 压缩结果为空，回退到截断模式")
                    truncated_text = original_text[:FALLBACK_TRUNCATE_LENGTH] + "..."
                    compressed_doc = Document(
                        page_content=truncated_text,
                        metadata=doc.metadata
                    )
                    compressed_docs.append(compressed_doc)

            except Exception as e:
                # LLM调用失败，回退到智能截断
                logger.error(f"文档 {idx} 压缩失败: {str(e)}，回退到截断模式")
                truncated_text = original_text[:FALLBACK_TRUNCATE_LENGTH] + "..."
                compressed_doc = Document(
                    page_content=truncated_text,
                    metadata=doc.metadata
                )
                compressed_docs.append(compressed_doc)

        # 计算压缩后的总token数
        compressed_total_tokens = sum(self._count_tokens(doc.page_content) for doc in compressed_docs)
        logger.info(f"上下文压缩完成，压缩后总token数: {compressed_total_tokens}，压缩率: {round((1 - compressed_total_tokens/total_tokens)*100, 1)}%")

        return compressed_docs

    @staticmethod
    def _format_context(retrieved_docs: List[Document]) -> str:
        """
        格式化检索到的文档为上下文字符串
        :param retrieved_docs: 检索到的Document对象列表
        :return: 格式化后的上下文字符串
        """
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "未知文档")
            context_parts.append(
                f"参考文档{idx}（来自：{source}）：\n{doc.page_content}"
            )
        return "\n\n".join(context_parts)

    def build_rag_prompt(
        self,
        query: str,
        retrieved_docs: List[Document],
        template_name: str = None
    ) -> str:
        """
        构建完整的RAG用户Prompt
        :param query: 用户问题
        :param retrieved_docs: 检索到的文档列表
        :param template_name: 使用的Prompt模板
        :return: 格式化后的用户Prompt
        """
        # 步骤1：格式化上下文
        context = self._format_context(retrieved_docs)
        
        # 步骤2：使用PromptLoader构建用户提示
        return self.prompt_loader.build_user_prompt(
            query=query,
            context=context,
            template_name=template_name or self.prompt_template
        )

    def query(
        self,
        question: str,
        template_name: str = None,
        enable_compression: bool = ENABLE_CONTEXT_COMPRESSION
    ) -> str:
        """
        执行完整RAG问答
        :param question: 用户问题
        :param template_name: 临时使用的Prompt模板（覆盖初始化时的模板）
        :param enable_compression: 是否启用上下文压缩
        :return: 最终回答
        """
        if not question.strip():
            return "请输入你想要咨询的问题~"
        
        logger.info(f"用户问题: {question}")
        
        # 1. 混合检索（返回Document对象列表）
        retrieved_docs = self.retrieval_engine.hybrid_search(
            query=question,
            return_documents=True
        )
        logger.info(f"检索到 {len(retrieved_docs)} 条相关上下文")
        
        # 2. 上下文压缩（新增步骤）
        if enable_compression:
            retrieved_docs = self.compress_context(
                docs=retrieved_docs,
                query=question
            )
        
        # 3. 构建RAG Prompt
        user_prompt = self.build_rag_prompt(
            query=question,
            retrieved_docs=retrieved_docs,
            template_name=template_name
        )
        
        # 4. 获取系统提示
        system_prompt = self.prompt_loader.get_system_prompt(
            template_name=template_name or self.prompt_template
        )
        
        # 5. 大模型生成回答
        answer = self.llm.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        return answer
