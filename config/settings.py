# -*- coding: utf-8 -*-
import os
import yaml
from typing import Dict, Any
from pydantic import BaseModel
from pathlib import Path

class DatabaseConfig(BaseModel):
    mongo_uri: str
    db_name: str
    chroma_persist_dir: str
    chroma_collection_name: str

class LLMConfig(BaseModel):
    model_name: str
    temperature_user: float
    temperature_ai: float
    chroma_embedding_model: str # <--- 新增

class GenerationConfig(BaseModel):
    num_users: int
    min_onboarding_turns: int
    max_onboarding_turns: int
    min_chat_messages: int
    max_chat_messages: int

class RAGConfig(BaseModel):
    window_size: int
    overlap: int

class Settings(BaseModel):
    """系统配置 (YAML 驱动)"""
    database: DatabaseConfig
    llm: LLMConfig
    generation: GenerationConfig
    rag: RAGConfig

    @classmethod
    def load_from_yaml(cls, path: str = "config/config.yaml") -> "Settings":
        # 尝试从多个路径加载，以兼容不同运行环境
        possible_paths = [
            Path(path), # 默认路径
            Path(os.getcwd()) / path, # 当前工作目录
            Path(__file__).parent / path, # config/config.yaml
            Path(__file__).parent.parent / path # project_root/config/config.yaml
        ]
        
        found_path = None
        for p in possible_paths:
            if p.exists():
                found_path = p
                break

        if not found_path:
            raise FileNotFoundError(f"Config file not found in any of the expected paths: {possible_paths}")

        with open(found_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

# 单例加载
try:
    settings = Settings.load_from_yaml()
except Exception as e:
    print(f"❌ 严重错误: 无法加载配置文件。请确保 config/config.yaml 存在且格式正确。错误信息: {e}")
    settings = None
