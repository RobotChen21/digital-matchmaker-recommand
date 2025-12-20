# -*- coding: utf-8 -*-
from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.common.models.state import MatchmakingState 

# --- Policy Model ---
class MatchPolicy(BaseModel):
    """匹配策略：定义哪些维度参与打分及其权重"""
    education_weight: int = Field(default=0, description="学历看重程度")
    job_weight: int = Field(default=0, description="工作/收入看重程度")
    family_weight: int = Field(default=0, description="家庭背景看重程度")
    
    preferred_degree: Optional[str] = Field(None, description="期望最低学历 (如: 本科, 硕士)")
    preferred_job: Optional[str] = Field(None, description="期望工作特征 (如: 稳定, 高薪, 自由)")
    preferred_family: Optional[str] = Field(None, description="期望家庭特征 (如: 独生, 开明)")

# --- Output Models ---

class IntentOutput(BaseModel):
    intent: Literal["search_candidate", "deep_dive", "chitchat"] = Field(
        description="意图: search_candidate(找人/换一批), deep_dive(问某人详情/感兴趣), chitchat(闲聊)"
    )
    keywords: str = Field(description="语义检索词")
    target_person: Optional[str] = Field(None, description="如果意图是deep_dive，提取用户感兴趣的对象名字/代词")
    match_policy: MatchPolicy = Field(default_factory=MatchPolicy, description="匹配策略")

class FilterOutput(BaseModel):
    city: List[str] = Field(default_factory=list, description="期望城市列表 (如: ['上海', '杭州'])")
    height_min: Optional[int] = Field(None, description="最小身高(cm)")
    height_max: Optional[int] = Field(None, description="最大身高(cm)")
    age_min: Optional[int] = Field(None, description="期望最小年龄")
    age_max: Optional[int] = Field(None, description="期望最大年龄")
    bmi_min: Optional[float] = Field(None, description="期望最小BMI")
    bmi_max: Optional[float] = Field(None, description="期望最大BMI")
    explanation: str = Field(description="筛选条件解释")

class RefineOutput(BaseModel):
    relaxed_query: str = Field(description="放宽后的查询描述")
    reason: str = Field(description="理由")

class EvidenceOutput(BaseModel):
    has_evidence: bool = Field(description="是否找到证据")
    evidence_summary: str = Field(description="证据总结")