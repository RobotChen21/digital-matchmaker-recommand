# -*- coding: utf-8 -*-
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class PersonaSeed(BaseModel):
    """虚拟用户的性格种子"""
    personality_traits: List[str] = Field(description="性格关键词列表，如：内向、细致、务实、温和")
    occupation: str = Field(description="职业大类，如：程序员、设计师、教师")
    occupation_detail: str = Field(description="职业详细描述，如：在互联网公司负责Web应用前端开发，精通Vue和React框架")
    interests: List[str] = Field(description="兴趣爱好列表，如：街头摄影、山地徒步、咖啡品鉴、科幻小说")
    relationship_history: str = Field(description="感情经历总结，如：大学时有一段两年恋情，因毕业后异地发展而和平分手，目前单身一年")
    family_background: str = Field(description="家庭背景总结，如：父母是中学教师，有个妹妹在读大学，家庭氛围开明和睦")
    values: List[str] = Field(description="核心价值观列表，如：诚信负责、终身学习、工作生活平衡、尊重多样性")
    communication_style: str = Field(description="沟通风格，如：直接、委婉、幽默、严肃")
    emotional_stability: str = Field(description="情绪稳定性，如：高、中、低")
    response_speed: str = Field(description="对话反应速度，如：快、中、慢")
    ideal_partner: str = Field(description="理想伴侣描述，如：喜欢有同理心、独立自主的女性，能理解技术工作性质，愿意分享生活趣事")

class VirtualUser(BaseModel):
    """虚拟用户基础信息"""
    nickname: str = Field(description="昵称")
    gender: str = Field(description="性别，male 或 female")
    birthday: str = Field(description="出生日期，YYYY-MM-DD")
    city: str = Field(description="所在城市")
    height: int = Field(description="身高，单位cm")
    self_intro_raw: str = Field(description="用户自我介绍原始文本")
    persona_seed: PersonaSeed = Field(description="用户性格种子")