from service.document_processor import DocumentProcessor
from service.retrieval_engine import RetrievalEngine
from service.rag_chain import RAGChain
from config import *
from utils.logger import logger


def build_index():
    """构建文档分块和检索索引"""
    logger.info("开始构建文档分块和检索索引...")
    
    # 处理文档
    processor = DocumentProcessor(min_paragraph_length=MIN_PARAGRAPH_LENGTH)
    chunks = processor.process_folder(
        input_dir=DOCS_INPUT_DIR,
        output_pkl_path=CHUNKS_OUTPUT_PATH
    )
    
    # 构建检索索引
    retrieval_engine = RetrievalEngine()
    retrieval_engine.build_index(
        chunks=chunks,
        force_rebuild=True
    )
    
    logger.info("索引构建完成")


def test_rag():
    """初步测试 RAG 系统"""
    rag = RAGChain()
    
    test_questions = [
        "2019-2021年，茅台酒营销工作的指导思想是什么？",
        "酒企数字化运营助力企业现代化升级主要体现在哪些方面？",
        "为什么茅台老酒市场对维护茅台金融属性极其重要？"
    ]
    
    for question in test_questions:
        print("\n" + "="*60)
        print(f"问题: {question}")
        print("-"*60)
        answer = rag.query(question)
        print(f"回答: {answer}")
        print("="*60)


if __name__ == "__main__":
    # 首次运行先执行 build_index() 构建索引
    # build_index()
    
    # 命令行测试
    test_rag()
