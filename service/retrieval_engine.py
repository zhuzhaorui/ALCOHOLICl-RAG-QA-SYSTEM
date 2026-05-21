import os
import pickle
import jieba
from typing import List, Optional
from rank_bm25 import BM25Okapi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from utils.logger import logger
from config import *


class RetrievalEngine:
    """
    Chroma 向量检索 + BM25 关键词检索 + 双重去重
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        device: str = DEVICE,
        chroma_persist_dir: str = CHROMA_PERSIST_DIR,
        bm25_chunks_path: str = BM25_CHUNKS_PATH
    ):
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.chroma_persist_dir = chroma_persist_dir
        self.bm25_chunks_path = bm25_chunks_path
        
        self.embedding = None
        self.vector_db = None
        self.bm25 = None
        self.texts = None
        
        self._init_embedding()

    def _init_embedding(self):
        """初始化词嵌入模型"""
        try:
            logger.info(f"正在加载词嵌入模型: {self.embedding_model_name}")
            self.embedding = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("词嵌入模型加载成功")
        except Exception as e:
            logger.error(f"词嵌入模型加载失败: {str(e)}", exc_info=True)
            raise

    def build_index(
        self,
        chunks: Optional[List[Document]] = None,
        chunks_path: Optional[str] = None,
        force_rebuild: bool = False
    ):
        """构建检索索引（向量索引+BM25索引）"""
        if not force_rebuild and self._index_exists():
            self.load_index()
            return

        if chunks is None:
            logger.info(f"从文件加载分块数据: {chunks_path}")
            chunks = DocumentProcessor.load_chunks(chunks_path)
        
        self.texts = [doc.page_content for doc in chunks]
        logger.info(f"共加载 {len(self.texts)} 个文档分块")

        # 构建 Chroma 向量索引
        logger.info("开始构建 Chroma 向量索引")
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory=self.chroma_persist_dir
        )
        self.vector_db.persist()
        logger.info("Chroma 向量索引构建完成")

        # 构建 BM25 关键词索引
        logger.info("开始构建 BM25 关键词索引")
        tokenized_texts = [jieba.lcut(text) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # 保存 BM25 所需的文本数据
        os.makedirs(os.path.dirname(self.bm25_chunks_path), exist_ok=True)
        with open(self.bm25_chunks_path, "wb") as f:
            pickle.dump(self.texts, f)
        logger.info("BM25 关键词索引构建完成")

    def load_index(self):
        """加载已构建的索引"""

        logger.info("正在加载 Chroma 向量索引")
        self.vector_db = Chroma(
            persist_directory=self.chroma_persist_dir,
            embedding_function=self.embedding
        )

        logger.info("正在加载 BM25 关键词索引")
        with open(self.bm25_chunks_path, "rb") as f:
            self.texts = pickle.load(f)
        tokenized_texts = [jieba.lcut(text) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        logger.info("所有索引加载成功")

    def _index_exists(self) -> bool:
        """检查索引是否存在"""
        chroma_exists = os.path.exists(self.chroma_persist_dir) and len(os.listdir(self.chroma_persist_dir)) > 0
        bm25_exists = os.path.exists(self.bm25_chunks_path)
        return chroma_exists and bm25_exists

    def vector_search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Document]:
        """向量检索（语义匹配）"""
        if self.vector_db is None:
            raise ValueError("向量索引未加载，请先运行 build_index() 或 load_index()")
        
        logger.info(f"执行向量检索: {query}")
        results = self.vector_db.similarity_search(query, k=top_k)
        logger.info(f"向量检索完成，返回 {len(results)} 个结果")
        return results

    def bm25_search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Document]:
        """BM25 关键词检索（精确匹配）"""
        
        logger.info(f"执行 BM25 检索: {query}")
        tokenized_query = jieba.lcut(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [Document(page_content=self.texts[i], metadata={"score": scores[i]}) for i in top_indices]
        
        logger.info(f"BM25 检索完成，返回 {len(results)} 个结果")
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        max_context_num: int = MAX_CONTEXT_NUM
    ) -> List[str]:
        """混合检索（向量+BM25）+ 双重去重"""
        logger.info(f"执行混合检索: {query}")
        
        vector_results = self.vector_search(query, top_k=top_k)
        bm25_results = self.bm25_search(query, top_k=top_k)
        
        # 双重去重
        seen_text = set()
        final_context = []
        
        for doc in vector_results + bm25_results:
            text = doc.page_content
            
            if not text.strip() or len(text) < 20:
                continue
            
            text_clean = text.strip().replace("\n", "")
            if text_clean in seen_text:
                continue
            
            seen_text.add(text_clean)
            final_context.append(text)
            
            if len(final_context) >= max_context_num:
                break
        
        logger.info(f"混合检索完成，返回 {len(final_context)} 个去重后的结果")
        return final_context
