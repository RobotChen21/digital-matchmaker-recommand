# -*- coding: utf-8 -*-
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.core.utils.env_utils import API_KEY, BASE_URL

def get_llm(temperature: float = 0.0, model_name: str = None) -> ChatOpenAI:
    """
    获取配置好的 LLM 实例 (工厂方法)
    
    Args:
        temperature (float): 采样温度，默认 0.0 (确定性输出)
        model_name (str, optional): 模型名称，默认从配置读取
        
    Returns:
        ChatOpenAI: 配置好的 LangChain LLM 对象
    """
    return ChatOpenAI(
        model=model_name or settings.llm.model_name,
        temperature=temperature,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
