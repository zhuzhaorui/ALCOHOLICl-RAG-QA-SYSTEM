# ALCOHOLICl-RAG-QA-SYSTEMRAG 茅台内部文档智能问答系统

基于 ChatGLM2-6B + BGE中文嵌入 + Chroma向量库 实现的酒类垂直领域私有文档RAG智能问答系统

项目简介

本项目针对酒类行业专业文档、行业研报、资料文档打造私有化离线RAG问答系统，通过检索增强生成技术解决大模型行业知识不足、内容幻觉、私有数据无法访问的问题。
支持PDF批量导入、文本清洗、智能分块、向量知识库构建、混合检索、行业专属问答全流程本地化部署，数据不联网，适合垂直领域知识库搭建与学习使用。

技术栈

• Python 3.8+

• 大语言模型：ChatGLM2-6B

• 向量嵌入模型：BAAI/bge-small-zh-v1.5

• 向量数据库：Chroma

• 文档处理：PyMuPDF、jieba、LangChain文本分割器

• 检索策略：语义向量检索 + BM25关键词混合检索

• 依赖框架：PyTorch、Transformers、LangChain

功能特性

• 批量PDF文档自动加载、文本清洗与格式规整

• 递归式智能文本分块，保留行业文档语义完整性

• 本地持久化向量知识库，一键构建与加载

• 向量检索+关键词检索融合，提升专业术语匹配精度

• 基于私有文档生成合规、低幻觉、可溯源的行业问答

• 全流程离线本地化运行，无外部API依赖

• 模块化代码设计，三步式运行流程，易修改与二次开发

1. 克隆项目
git clone https://github.com/zhuzhaorui/ALCOHOLICl-RAG-QA-SYSTEM.git

cd ALCOHOLICl-RAG-QA-SYSTEM
3. 安装依赖环境
pip install -r requirements.txt
4. 放入私有文档

将酒类行业PDF文档放入 data/doc_raw/ 文件夹下。

4. 执行文档预处理与分块
python document_processor.py  
5. 构建向量知识库
python retrieval_engine.py 
6. 启动RAG问答交互
python rag_chain.py
输入问题即可基于私有文档进行专业问答。

运行说明

• 模型文件会自动下载至本地缓存目录，无需手动配置

• 可通过修改分块大小、检索数量、提示词模板优化问答效果

• 向量库已持久化存储，首次构建完成后可直接启动问答脚本
