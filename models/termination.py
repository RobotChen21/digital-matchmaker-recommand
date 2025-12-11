# -*- coding: utf-8 -*-
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class TerminationReason(str, Enum):
    """对话终止原因"""
    USER_HESITANT = "user_hesitant"       # 用户表现出犹豫、敷衍
    USER_TIRED = "user_tired"             # 用户表现出疲惫
    USER_REQUEST_END = "user_request_end" # 用户明确请求结束对话
    INFO_COLLECTED = "info_collected"     # 信息收集已足够
    NATURAL_END = "natural_end"           # 对话自然结束
    MAX_TURNS = "max_turns"               # 达到最大对话轮数/消息数
    FALLBACK = "fallback"                 # 默认或未知原因

class TerminationSignal(BaseModel):
    """对话终止信号模型"""
    should_terminate: bool = Field(description="是否应该终止对话")
    reason: Optional[TerminationReason] = Field(None, description="终止对话的原因")
    confidence: float = Field(description="终止的置信度 (0.0-1.0)")
    explanation: str = Field(description="终止判断的详细解释")