# service/document_processor.py
import os
import re
import pickle
from typing import List, Optional
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import logger


class DocumentProcessor:
    """
    文档处理器（混合分块+日期前缀）
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        min_chunk_length: int = 50,
        support_formats: tuple = (".pdf", ".txt")
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.support_formats = support_formats
        
        # 初始化语义分块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                "。",
                "；",
                "，",
                " ",
                ""
            ],
            length_function=len,
            is_separator_regex=False
        )
        
        # 匹配标题
        self.level1_pattern = re.compile(r"^[一二三四五六七八九十]+､\s*")  
        self.level2_pattern = re.compile(r"^\d+､\s*") 
        self.date_pattern = re.compile(r"(\d{4})(\d{2})(\d{2})") 

    def _extract_date_from_filename(self, filename: str) -> Optional[str]:
        """从文件名提取日期"""
        match = self.date_pattern.search(filename)
        if match:
            year, month, day = match.groups()
            return f"{year}年{int(month)}月{int(day)}日"
        return None

    def _load_single_document(self, file_path: str) -> str:
        """加载单个文档并返回完整文本"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
        elif file_ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            full_text = loader.load()[0].page_content
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
            
        # 统一换行符
        full_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
        return full_text

    def _clean_text(self, raw_text: str) -> str:
        """清洗PDF文本，修复常见问题"""
        # 去除多余空格
        cleaned_text = re.sub(r" +", " ", raw_text)
        # 修复中文标点后没有空格的问题
        cleaned_text = re.sub(r"([。；，！？])", r"\1 ", cleaned_text)
        # 去除页码和页眉页脚
        cleaned_text = re.sub(r"\n\s*\d+\s*\n", "\n", cleaned_text)
        # 去除空行
        cleaned_text = re.sub(r"\n+", "\n", cleaned_text)
        return cleaned_text.strip()

    def _split_by_level1(self, text: str) -> List[tuple[str, str]]:
        """按一级标题拆分文本，返回(一级标题, 内容)列表"""
        parts = []
        lines = text.split("\n")
        current_title = ""
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 匹配一级标题
            if self.level1_pattern.match(line):
                if current_title:
                    parts.append((current_title, "\n".join(current_content)))
                current_title = line
                current_content = []
            else:
                current_content.append(line)
        
        # 添加最后一个部分
        if current_title and current_content:
            parts.append((current_title, "\n".join(current_content)))
        
        # 如果没有一级标题，整个文本作为一个部分
        if not parts:
            parts.append(("", text))
            
        return parts

    def _split_by_level2(self, level1_title: str, text: str) -> List[tuple[str, str]]:
        """按二级标题拆分一级标题下的内容，返回(完整标题, 内容)列表"""
        parts = []
        lines = text.split("\n")
        current_title = ""
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 匹配二级标题
            if self.level2_pattern.match(line):
                if current_title:
                    full_title = f"{level1_title} {current_title}" if level1_title else current_title
                    parts.append((full_title, "\n".join(current_content)))
                current_title = line
                current_content = []
            else:
                current_content.append(line)
        
        # 添加最后一个部分
        if current_title and current_content:
            full_title = f"{level1_title} {current_title}" if level1_title else current_title
            parts.append((full_title, "\n".join(current_content)))
        
        # 如果没有二级标题，整个内容作为一个部分
        if not parts:
            parts.append((level1_title, text))
            
        return parts

    def _split_by_semantic(self, title: str, text: str, date_prefix: str) -> List[Document]:
        """对过长的内容进行语义分块，保留标题和日期前缀"""
        documents = []
        # 先拼接标题和内容
        full_text = f"{title}\n{text}" if title else text
        
        # 如果长度小于分块大小，直接返回
        if len(full_text) <= self.chunk_size:
            doc_content = f"{date_prefix} {full_text}"
            return [Document(page_content=doc_content, metadata={"source": "", "date": date_prefix, "title": title})]
        
        # 语义分块
        chunks = self.text_splitter.split_text(full_text)
        
        for chunk in chunks:
            if len(chunk) < self.min_chunk_length:
                continue
            # 每个分块都添加日期前缀
            doc_content = f"{date_prefix} {chunk}"
            documents.append(Document(
                page_content=doc_content,
                metadata={"source": "", "date": date_prefix, "title": title}
            ))
        
        return documents

    def process_single_file(self, file_path: str) -> List[Document]:
        """处理单个文档的完整流水线"""
        filename = os.path.basename(file_path)
        logger.info(f"开始处理文档: {filename}")
        
        try:
            # 1. 提取日期
            date_prefix = self._extract_date_from_filename(filename)
            if not date_prefix:
                logger.warning(f"无法从文件名提取日期: {filename}，将使用默认日期")
                date_prefix = "【未知日期】"
            else:
                date_prefix = f"【{date_prefix}】"
            
            # 2. 加载和清洗文本
            raw_text = self._load_single_document(file_path)
            cleaned_text = self._clean_text(raw_text)
            
            # 3. 按一级标题拆分
            level1_parts = self._split_by_level1(cleaned_text)
            logger.info(f"按一级标题拆分为 {len(level1_parts)} 个部分")
            
            # 4. 按二级标题拆分 + 语义分块
            all_documents = []
            for level1_title, level1_content in level1_parts:
                # 按二级标题拆分
                level2_parts = self._split_by_level2(level1_title, level1_content)
                logger.info(f"一级标题「{level1_title}」拆分为 {len(level2_parts)} 个二级部分")
                
                # 对每个二级部分进行语义分块
                for level2_title, level2_content in level2_parts:
                    docs = self._split_by_semantic(level2_title, level2_content, date_prefix)
                    all_documents.extend(docs)
            
            # 5. 设置source元数据
            for doc in all_documents:
                doc.metadata["source"] = filename
            
            logger.info(f"文档 {filename} 处理完成，生成 {len(all_documents)} 个有效分块")
            return all_documents
            
        except Exception as e:
            logger.error(f"处理文档 {filename} 失败: {str(e)}", exc_info=True)
            return []

    def process_folder(self, folder_path: str = "data/doc_raw") -> List[Document]:
        """批量处理文件夹中的所有文档"""
        all_documents = []
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(self.support_formats):
                documents = self.process_single_file(file_path)
                all_documents.extend(documents)
        
        logger.info(f"文件夹处理完成，共生成 {len(all_documents)} 个分块")
        return all_documents

    def save_chunks(self, documents: List[Document], save_path: str = "data/result/doc_chunks.pkl"):
        """保存分块结果到文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(documents, f)
        logger.info(f"分块结果已保存到: {save_path}")

    def load_chunks(self, load_path: str = "data/result/doc_chunks.pkl") -> List[Document]:
        """从文件加载分块结果"""
        if not os.path.exists(load_path):
            logger.error(f"分块文件不存在: {load_path}")
            return []
        
        with open(load_path, "rb") as f:
            documents = pickle.load(f)
        logger.info(f"从 {load_path} 加载了 {len(documents)} 个分块")
        return documents
