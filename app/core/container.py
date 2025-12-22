# -*- coding: utf-8 -*-
from typing import Optional

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.utils.env_utils import API_KEY, BASE_URL
from app.db.es_manager import ESManager
from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import ChromaManager


class AppContainer:
    """
    轻量级 IOC 容器 (Singleton / Service Locator)
    负责管理全局单例对象 (DB, LLM) 的生命周期和配置。
    """
    _instance = None

    def __init__(self):
        # 延迟初始化变量
        self._mongo_manager: Optional[MongoDBManager] = None
        self._chroma_manager: Optional[ChromaManager] = None
        self._es_manager: Optional[ESManager] = None
        self._workflow = None # Workflow 单例
        self._profile_service = None # ProfileService 单例
        self._session_service = None # SessionService 单例
        self._termination_manager = None # TerminationManager 单例
        
        # LLM 缓存
        self._llms = {}

    @classmethod
    def get_instance(cls) -> "AppContainer":
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    # --- Services (Singleton) ---
    @property
    def session_service(self):
        """获取 SessionService 单例"""
        if not self._session_service:
            from app.services.session_service import SessionService
            self._session_service = SessionService()
        return self._session_service

    @property
    def profile_service(self):
        """获取 ProfileService 单例"""
        if not self._profile_service:
            from app.services.ai.agents.profile_manager import ProfileService
            # 使用 chat 模型，温度适中，适合提取和生成
            self._profile_service = ProfileService(self.get_llm("reason"))
        return self._profile_service

    @property
    def termination_manager(self):
        """获取 DialogueTerminationManager 单例"""
        if not self._termination_manager:
            from app.services.ai.tools.termination import DialogueTerminationManager
            # 使用 intent 模型 (温度0) 进行逻辑判断
            self._termination_manager = DialogueTerminationManager(self.get_llm("intent"))
        return self._termination_manager

    # --- Workflow (Singleton) ---
    @property
    def recommendation_app(self):
        """获取已编译的 Recommendation Graph App"""
        if not self._workflow:
            from app.services.ai.workflows.recommendation import RecommendationWorkflow
            # Node 内部现在会自动从 container 获取依赖，所以这里不需要传参了
            workflow = RecommendationWorkflow() 
            self._workflow = workflow.build_graph()
        return self._workflow

    # --- Database Managers (Singleton) ---

    @property
    def db(self) -> MongoDBManager:
        """获取 MongoDB Manager 单例"""
        if not self._mongo_manager:
            self._mongo_manager = MongoDBManager(
                settings.database.mongo_uri, 
                settings.database.db_name
            )
        return self._mongo_manager

    @property
    def chroma(self) -> ChromaManager:
        """获取 ChromaDB Manager 单例"""
        if not self._chroma_manager:
            self._chroma_manager = ChromaManager(
                settings.database.chroma_persist_dir,
                settings.database.chroma_collection_name
            )
        return self._chroma_manager

    @property
    def es(self):
        """获取 Elasticsearch Manager 单例"""
        if not hasattr(self, "_es_manager") or not self._es_manager:
            from app.db.es_manager import ESManager
            self._es_manager = ESManager()
        return self._es_manager

    # --- LLM Factory (Cached by Type) ---

    def get_llm(self, type: str = "chat") -> ChatOpenAI:
        """
        根据业务类型获取预配置的 LLM 实例。
        
        Types:
        - "intent": 温度 0.0 (严谨，用于分类、提取)
        - "chat": 温度 0.7 (灵活，用于对话、闲聊)
        - "reason": 温度 0.4 (平衡，用于推理、总结、推荐语)
        """
        if type in self._llms:
            return self._llms[type]

        # 配置映射
        configs = {
            "intent": 0.0,
            "chat": settings.llm.temperature_user, # 通常 0.7
            "reason": settings.llm.temperature_ai  # 通常 0.3-0.4
        }
        
        temperature = configs.get(type, 0.7) # 默认 0.7
        
        llm = ChatOpenAI(
            model=settings.llm.model_name,
            temperature=temperature,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        
        self._llms[type] = llm
        return llm

# 全局单例入口
container = AppContainer.get_instance()
