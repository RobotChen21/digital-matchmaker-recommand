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
        # 获取项目根目录 (假设 app/core/config.py 在 app/core/ 下)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent # app/core/config.py -> app/core -> app -> root
        
        possible_paths = [
            project_root / "config" / "config.yaml", # 标准路径
            Path(os.getcwd()) / "config" / "config.yaml", # 运行时路径
            Path(path) # 绝对路径或相对于 CWD
        ]
        
        found_path = None
        for p in possible_paths:
            if p.exists():
                found_path = p
                break

        if not found_path:
            # Fallback for old structure or if running from scripts
            if (Path(os.getcwd()) / "config.yaml").exists():
                 found_path = Path(os.getcwd()) / "config.yaml"

        if not found_path:
            raise FileNotFoundError(f"Config file not found. Searched in: {[str(p) for p in possible_paths]}")

        with open(found_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # 修正: 将相对路径转换为基于项目根目录的绝对路径
        if 'database' in config_data and 'chroma_persist_dir' in config_data['database']:
             p = Path(config_data['database']['chroma_persist_dir'])
             if not p.is_absolute():
                 config_data['database']['chroma_persist_dir'] = str(project_root / p)

        return cls(**config_data)

# 单例加载
try:
    settings = Settings.load_from_yaml()
except Exception as e:
    print(f"❌ 严重错误: 无法加载配置文件。请确保 config/config.yaml 存在且格式正确。错误信息: {e}")
    settings = None
