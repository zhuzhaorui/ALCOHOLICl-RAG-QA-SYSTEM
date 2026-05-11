import torch
from transformers import AutoTokenizer, AutoModel
from utils.logger import logger
from config import *


class LLM:
    """
    ChatGLM2-6B配置
    """

    def __init__(
        self,
        model_path: str = LLM_MODEL_PATH,
        device: str = DEVICE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        repetition_penalty: float = REPETITION_PENALTY,
        no_repeat_ngram_size: int = NO_REPEAT_NGRAM_SIZE,
        length_penalty: float = LENGTH_PENALTY
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty
        
        self.tokenizer = None
        self.model = None
        
        self._load_model()

    def _load_model(self):
        """加载 ChatGLM2-6B 模型和分词器"""
        try:
            logger.info(f"正在加载 ChatGLM2-6B 模型: {self.model_path}")
            logger.info(f"检测到运行设备: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
          
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            
            logger.info(" ChatGLM2-6B 模型加载成功！")
                 
        except Exception as e:
            logger.error(f" 模型加载失败: {str(e)}", exc_info=True)
            raise

    def generate(self, prompt: str) -> str:
        """生成回答"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("大模型未加载，请先初始化")
        
        try:
            logger.info("开始生成回答")
            
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=[],
                max_length=self.max_new_tokens + 2048,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                length_penalty=self.length_penalty,
                do_sample=True
            )
            
            logger.info("回答生成完成")
            return response.strip()
            
        except Exception as e:
            logger.error(f" 回答生成失败: {str(e)}", exc_info=True)
            return "抱歉，回答生成过程中出现错误，请稍后重试。"
