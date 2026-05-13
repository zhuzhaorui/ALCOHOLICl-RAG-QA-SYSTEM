#负责加载和解析 Prompt 配置文件
import os
import yaml
from typing import Dict, Any
from utils.logger import logger


class PromptLoader:
    """
    Prompt配置加载器
    负责从YAML配置文件中加载Prompt模板，支持多模板切换
    """

    def __init__(self, config_path: str = "config/prompts.yaml"):
        """
        初始化Prompt加载器
        :param config_path: Prompt配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.default_template = "default"

    def _load_config(self) -> Dict[str, Any]:
        """
        加载YAML配置文件
        :return: 解析后的配置字典
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Prompt配置文件不存在: {self.config_path}")

            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            logger.info(f"Prompt配置文件加载成功: {self.config_path}")
            return config

        except Exception as e:
            logger.error(f" Prompt配置文件加载失败: {str(e)}", exc_info=True)
            raise

    def get_system_prompt(self, template_name: str = None) -> str:
        """
        获取系统提示
        :param template_name: 模板名称，None使用默认模板
        :return: 系统提示字符串
        """
        if template_name is None:
            return self.config["system_prompt"]

        if template_name not in self.config["templates"]:
            logger.warning(f"模板 {template_name} 不存在，使用默认模板")
            return self.config["system_prompt"]

        return self.config["templates"][template_name]["system_prompt"]

    def get_user_prompt_template(self, template_name: str = None) -> str:
        """
        获取用户提示模板
        :param template_name: 模板名称，None使用默认模板
        :return: 用户提示模板字符串
        """
        if template_name is None:
            return self.config["user_prompt_template"]

        if template_name not in self.config["templates"]:
            logger.warning(f"模板 {template_name} 不存在，使用默认模板")
            return self.config["user_prompt_template"]

        return self.config["templates"][template_name]["user_prompt_template"]

    def build_user_prompt(
        self,
        query: str,
        context: str,
        template_name: str = None
    ) -> str:
        """
        构建完整的用户提示（变量替换）
        :param query: 用户问题
        :param context: 格式化后的上下文
        :param template_name: 模板名称
        :return: 替换变量后的用户提示
        """
        template = self.get_user_prompt_template(template_name)
        return template.format(query=query, context=context)
