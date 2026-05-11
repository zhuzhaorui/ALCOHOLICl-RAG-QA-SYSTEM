import gradio as gr
from service.rag_chain import RAGChain
from utils.logger import logger
from config import *

logger.info("正在启动 RAG 问答系统...")
rag_chain = RAGChain()
logger.info("RAG 问答系统启动成功")


def chat_interface(question: str) -> str:
    """Gradio 界面调用函数"""
    return rag_chain.query(question)


with gr.Blocks(title="文档智能问答系统") as demo:
    gr.Markdown("茅台内部文档智能问答系统")

    with gr.Row():
        question_input = gr.Textbox(
            label="请输入你的问题",
            placeholder="例如：请介绍一下美国智库的主要职能",
            lines=2
        )
    
    with gr.Row():
        answer_output = gr.Textbox(
            label="智能回答",
            lines=12
        )
    
    with gr.Row():
        submit_btn = gr.Button("🚀 生成回答", variant="primary")

    question_input.submit(chat_interface, inputs=question_input, outputs=answer_output)
    submit_btn.click(chat_interface, inputs=question_input, outputs=answer_output)


if __name__ == "__main__":
    demo.launch(
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
        inbrowser=True
    )
