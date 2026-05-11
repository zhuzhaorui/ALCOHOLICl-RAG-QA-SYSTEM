import os
import re
import pickle
from typing import List, Optional
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.schema import Document
from utils.logger import logger


class DocumentProcessor:
    """
    文档清洗，中文自然段落分块
    """

    def __init__(
        self,
        min_paragraph_length: int = 20,
        support_formats: tuple = (".pdf", ".txt")
    ):
        self.min_paragraph_length = min_paragraph_length
        self.support_formats = support_formats

    def _load_single_document(self, file_path: str) -> str:
        """加载单个文档并返回完整文本"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
            pages = loader.load()
            full_text = " ".join([page.page_content for page in pages])
        elif file_ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            full_text = loader.load()[0].page_content
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
            
        return full_text

    def _clean_pdf_text(self, raw_text: str) -> str:
        """清洗 PDF 文本，修复分页断行和文字粘连问题"""
        cleaned_text = raw_text.replace("\n", " ")
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        cleaned_text = cleaned_text.replace("。 ", "。\n\n")
        return cleaned_text.strip()

    def _split_by_natural_paragraph(self, cleaned_text: str) -> List[str]:
        """按中文自然段落分块，过滤无效段落"""
        paragraphs = cleaned_text.split("\n\n")
        valid_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if para and len(para) >= self.min_paragraph_length:
                valid_paragraphs.append(para)
                
        return valid_paragraphs

    def _convert_to_documents(self, paragraphs: List[str], source: str = "") -> List[Document]:
        """转换为 Langchain 标准 Document 对象"""
        documents = []
        for para in paragraphs:
            doc = Document(
                page_content=para,
                metadata={"source": source}
            )
            documents.append(doc)
        return documents

    def process_single_file(self, file_path: str) -> List[Document]:
        """处理文档"""
        filename = os.path.basename(file_path)
        logger.info(f"开始处理文档: {filename}")
        
        try:
            raw_text = self._load_single_document(file_path)
            
            if file_path.lower().endswith(".pdf"):
                cleaned_text = self._clean_pdf_text(raw_text)
            else:
                cleaned_text = raw_text.strip()
            
            paragraphs = self._split_by_natural_paragraph(cleaned_text)
            documents = self._convert_to_documents(paragraphs, source=filename)
            
            logger.info(f"文档 {filename} 处理完成，生成 {len(documents)} 个有效分块")
            return documents
            
        except Exception as e:
            logger.error(f"处理文档 {filename} 失败: {str(e)}", exc_info=True)
            return []

    def process_folder(
        self,
        input_dir: str,
        output_pkl_path: Optional[str] = None,
        merge_results: bool = True
    ) -> List[Document]:
        """批量处理文件夹中的所有文档"""
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")
            
        all_documents = []
        logger.info(f"开始批量处理文件夹: {input_dir}")
        
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            
            if (filename.startswith(".") or 
                filename.startswith("~$") or 
                not os.path.isfile(file_path)):
                continue
                
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in self.support_formats:
                logger.warning(f"跳过不支持的文件: {filename}")
                continue
                
            documents = self.process_single_file(file_path)
            
            if merge_results:
                all_documents.extend(documents)
            else:
                if output_pkl_path:
                    file_output_path = os.path.join(
                        os.path.dirname(output_pkl_path),
                        f"{os.path.splitext(filename)[0]}_chunks.pkl"
                    )
                    self.save_chunks(documents, file_output_path)
        
        if merge_results and output_pkl_path:
            self.save_chunks(all_documents, output_pkl_path)
            logger.info(f"所有文档处理完成，共生成 {len(all_documents)} 个分块")
            
        return all_documents

    @staticmethod
    def save_chunks(chunks: List[Document], output_path: str):
        """保存分块结果到本地"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"分块结果已保存至: {output_path}")

    @staticmethod
    def load_chunks(input_path: str) -> List[Document]:
        """加载本地保存的分块结果"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"分块文件不存在: {input_path}")
            
        with open(input_path, "rb") as f:
            chunks = pickle.load(f)
        logger.info(f"成功加载分块结果，共 {len(chunks)} 个文档")
        return chunks
