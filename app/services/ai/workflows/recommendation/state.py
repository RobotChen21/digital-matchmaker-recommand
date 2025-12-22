# -*- coding: utf-8 -*-
from typing import Literal, List, Optional
from pydantic import BaseModel, Field

# --- Policy Model ---
class IntentOutput(BaseModel):
    intent: Literal["search_candidate", "refresh_candidate", "deep_dive", "chitchat"] = Field(
        description="意图: search_candidate(新搜索/改条件), refresh_candidate(换一批/翻页), deep_dive(问详情), chitchat(闲聊)"
    )

class FilterOutput(BaseModel):
    # --- 硬性指标 (Mongo) ---
    city: List[str] = Field(default_factory=list, description="期望城市列表")
    height_min: Optional[int] = Field(None, description="最小身高(cm)")
    height_max: Optional[int] = Field(None, description="最大身高(cm)")
    age_min: Optional[int] = Field(None, description="期望最小年龄")
    age_max: Optional[int] = Field(None, description="期望最大年龄")
    bmi_min: Optional[float] = Field(None, description="期望最小BMI")
    bmi_max: Optional[float] = Field(None, description="期望最大BMI")
    
    # --- 软性/半硬性指标 (ES) ---
    keywords: str = Field(description="语义检索词。包含学历、职业、家庭、性格、爱好等所有非硬性指标，用空格分隔。")
    
    explanation: str = Field(description="筛选条件解释")

class RefineOutput(BaseModel):
    criteria: FilterOutput = Field(description="放宽后的具体筛选条件")
    relaxed_query_str: str = Field(description="放宽后的自然语言描述 (用于前端展示/更新current_input)")
    reason: str = Field(description="修正理由")

class EvidenceOutput(BaseModel):
    has_evidence: bool = Field(description="是否找到证据")
    evidence_summary: str = Field(description="证据总结")

class DeepDiveOutput(BaseModel):
    target_person: Optional[str] = Field(None, description="用户感兴趣的对象姓名/代词")
    reason: str = Field(description="选择该目标的理由")