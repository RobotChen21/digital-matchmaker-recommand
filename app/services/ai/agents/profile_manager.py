# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from app.services.ai.agents.extractors import (
    PersonalityExtractor, InterestExtractor, ValuesExtractor,
    LifestyleExtractor, LoveStyleExtractor, RiskExtractor,
    EducationExtractor, OccupationExtractor, FamilyExtractor,
    DatingPrefExtractor
)
from app.common.models.profile import UserProfile

class ProfileService:
    """
    画像服务：负责协调各个细分维度的 Extractor，
    从对话文本中提取完整的用户画像。
    """
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        # 初始化所有子 Agent
        self.agents = {
            "personality_profile": PersonalityExtractor(llm),
            "interest_profile": InterestExtractor(llm),
            "values_profile": ValuesExtractor(llm),
            "lifestyle_profile": LifestyleExtractor(llm),
            "love_style_profile": LoveStyleExtractor(llm),
            "risk_profile": RiskExtractor(llm),
            "education_profile": EducationExtractor(llm),
            "occupation_profile": OccupationExtractor(llm),
            "family_profile": FamilyExtractor(llm),
            "dating_preferences": DatingPrefExtractor(llm),
        }

    def extract_from_dialogue(self, dialogue_text: str) -> Dict[str, Any]:
        """
        输入对话文本，运行所有 Agent，返回聚合后的画像字典。
        """
        full_profile_data = {}
        
        # 并行或串行执行提取 (目前为串行，未来可优化为并行)
        for field_name, agent in self.agents.items():
            try:
                result = agent.extract(dialogue_text)
                if result:
                    # 将 Pydantic 模型转为 dict，并移除 None 值
                    full_profile_data[field_name] = result.model_dump(exclude_none=True)
                else:
                    full_profile_data[field_name] = None
            except Exception as e:
                print(f"⚠️ [ProfileService] {field_name} 提取失败: {e}")
                full_profile_data[field_name] = None
        
        return full_profile_data

    @staticmethod
    def format_dialogue_for_llm(messages: list) -> str:
        """辅助函数：将数据库的消息列表格式化为文本"""
        text = []
        for msg in messages:
            role = "AI红娘" if msg.get('role') == 'ai' else "用户"
            content = msg.get('content', '')
            text.append(f"{role}: {content}")
        return "\n".join(text)
