# -*- coding: utf-8 -*-
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class ChatContext(BaseModel):
    """前端需要维护并传回的上下文数据"""
    seen_candidate_ids: List[str] = []
    last_candidates: List[Dict[str, Any]] = [] # 存储上一次推荐的完整候选人信息
    last_target_person: Optional[str] = None   # 上一次聊过的人名 (指代消解)
    last_search_criteria: Optional[Dict[str, Any]] = {} # 上一次的搜索条件

class ChatRequest(BaseModel):
    message: str
    context: ChatContext = Field(default_factory=ChatContext)

class CandidateDTO(BaseModel):
    """返回给前端渲染卡片用的数据"""
    id: str
    nickname: str
    gender: str
    age: Any = 0 # 兼容 int 或 str '?'
    city: str = ""
    summary: str = ""
    evidence: str = ""

class ChatResponse(BaseModel):
    reply: str
    intent: str
    final_candidates: List[CandidateDTO] = []
    new_context: ChatContext
    debug_info: dict
