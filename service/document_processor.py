import os
import re
import pickle
from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import logger

class DocumentProcessor:
    """
    文档处理器（最终版 · 严格按标题分层拆分）
    逻辑：一级标题 → 二级标题 → 语义分块 → 日期前缀+标题保留
    兼容：PDF / TXT
    支持：保存/加载pkl
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        min_paragraph_length: int = 50,
        support_formats: tuple = (".pdf", ".txt")
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_paragraph_length = min_paragraph_length
        self.support_formats = support_formats

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len
        )

        self.level1_pattern = re.compile(r"^[一二三四五六七八九十]+、\s*")
        self.level2_pattern = re.compile(r"^\d+、\s*")
        self.date_pattern = re.compile(r"(\d{4})(\d{2})(\d{2})")

    def _extract_date_from_filename(self, filename: str) -> Optional[str]:
        match = self.date_pattern.search(filename)
        if match:
            y, m, d = match.groups()
            return f"{y}年{int(m)}月{int(d)}日"
        return None

    def _load_single_document(self, file_path: str) -> str:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
            pages = loader.load()
            return "\n".join([page.page_content for page in pages])
        elif file_ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()[0].page_content
        return ""

    def _clean_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_by_level1(self, text: str) -> List[Tuple[str, str]]:
        parts = []
        lines = text.split("\n")
        current_title = ""
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if self.level1_pattern.match(line):
                if current_title:
                    parts.append((current_title, "\n".join(current_content)))
                current_title = line
                current_content = []
            else:
                current_content.append(line)

        if current_title and current_content:
            parts.append((current_title, "\n".join(current_content)))
        if not parts:
            parts.append(("", text))
        return parts

    def _split_by_level2(self, level1_title: str, text: str) -> List[Tuple[str, str]]:
        parts = []
        lines = text.split("\n")
        current_title = ""
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if self.level2_pattern.match(line):
                if current_title:
                    full_title = f"{level1_title} {current_title}" if level1_title else current_title
                    parts.append((full_title, "\n".join(current_content)))
                current_title = line
                current_content = []
            else:
                current_content.append(line)

        if current_title and current_content:
            full_title = f"{level1_title} {current_title}" if level1_title else current_title
            parts.append((full_title, "\n".join(current_content)))
        if not parts:
            parts.append((level1_title, text))
        return parts

    def _split_by_semantic(self, title: str, text: str, date_prefix: str, filename: str) -> List[Document]:
        documents = []
        full_text = f"{title}\n{text}" if title else text

        if len(full_text) <= self.chunk_size:
            doc_content = f"{date_prefix} {full_text}"
            return [Document(page_content=doc_content, metadata={"source": filename, "title": title})]

        chunks = self.text_splitter.split_text(full_text)
        for chunk in chunks:
            if len(chunk) < self.min_paragraph_length:
                continue
            doc_content = f"{date_prefix} {chunk}"
            documents.append(Document(
                page_content=doc_content,
                metadata={"source": filename, "title": title}
            ))
        return documents

    def _detect_category(self, text: str):
        if "白酒" in text:
            return "白酒"
        elif "啤酒" in text:
            return "啤酒"
        elif "葡萄酒" in text:
            return "葡萄酒"
        else:
            return "综合"

    def process_single_file(self, file_path: str) -> List[Document]:
        filename = os.path.basename(file_path)
        logger.info(f"开始处理文档: {filename}")

        try:
            date_str = self._extract_date_from_filename(filename)
            date_prefix = f"【{date_str}】" if date_str else "【未知日期】"

            # TXT 问答对整块处理
            if file_path.endswith(".txt"):
                raw_text = self._load_single_document(file_path)
                cleaned_text = self._clean_text(raw_text)
                doc = Document(
                    page_content=f"{date_prefix} {cleaned_text}",
                    metadata={"source": filename, "title": "问答对"}
                )
                return [doc]

            # PDF 标题分层拆分
            raw_text = self._load_single_document(file_path)
            cleaned_text = self._clean_text(raw_text)
            level1_parts = self._split_by_level1(cleaned_text)
            all_documents = []

            for level1_title, level1_content in level1_parts:
                level2_parts = self._split_by_level2(level1_title, level1_content)
                for level2_title, level2_content in level2_parts:
                    docs = self._split_by_semantic(level2_title, level2_content, date_prefix, filename)
                    all_documents.extend(docs)

            # 补充元数据
            match = self.date_pattern.search(filename)
            year = int(match.group(1)) if match else 0
            month = int(match.group(2)) if match else 0

            for doc in all_documents:
                doc.metadata["year"] = year
                doc.metadata["month"] = month
                doc.metadata["category"] = self._detect_category(doc.page_content)

            logger.info(f"文档 {filename} 处理完成，生成 {len(all_documents)} 个有效分块")
            return all_documents

        except Exception as e:
            logger.error(f"处理文档失败: {e}", exc_info=True)
            return []

    def process_folder(
        self,
        input_dir: str,
        output_pkl_path: Optional[str] = None,
        merge_results: bool = True
    ) -> List[Document]:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"文件夹不存在: {input_dir}")

        all_documents = []
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if filename.startswith((".", "~$")) or not os.path.isfile(file_path):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in self.support_formats:
                continue
            docs = self.process_single_file(file_path)
            all_documents.extend(docs)

        if merge_results and output_pkl_path:
            self.save_chunks(all_documents, output_pkl_path)
            logger.info(f"全部处理完成，共生成 {len(all_documents)} 块")
        return all_documents

    @staticmethod
    def save_chunks(chunks: List[Document], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"分块已保存: {output_path}")

    @staticmethod
    def load_chunks(input_path: str) -> List[Document]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"分块文件不存在: {input_path}")
        with open(input_path, "rb") as f:
            chunks = pickle.load(f)
        logger.info(f"加载分块完成: {len(chunks)} 块")
        return chunks
