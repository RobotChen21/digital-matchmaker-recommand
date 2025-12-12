# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional
from datetime import datetime
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

    @staticmethod
    def generate_profile_summary(basic: Dict, profile: Dict) -> str:
        """将结构化画像转换为自然语言摘要 (用于向量化)"""
        parts = []
        
        # 辅助函数: 计算年龄
        def _get_age(bday):
            if not bday: return "未知"
            try:
                if isinstance(bday, datetime):
                    return str(datetime.now().year - bday.year)
                elif isinstance(bday, str):
                    return str(datetime.now().year - int(bday.split('-')[0]))
                return "未知"
            except:
                return "未知"

        # 基础信息
        age = _get_age(basic.get('birthday'))
        parts.append(f"我是{basic.get('nickname')}，{basic.get('gender')}性，今年{age}岁，住在{basic.get('city')}。")
        
        # 职业
        occ = profile.get('occupation_profile')
        if occ:
            parts.append(f"职业是{occ.get('job_title', '未知')}，行业是{occ.get('industry', '未知')}。")
        
        # 性格
        pers = profile.get('personality_profile')
        if pers:
            mbti = pers.get('mbti', '')
            if mbti: parts.append(f"MBTI类型是{mbti}。")
            # Big5
            big5 = pers.get('big5', {})
            if big5:
                traits = []
                if big5.get('extroversion', 0) > 0.6: traits.append("外向")
                if big5.get('openness', 0) > 0.6: traits.append("充满想象力")
                if big5.get('conscientiousness', 0) > 0.6: traits.append("认真负责")
                if big5.get('agreeableness', 0) > 0.6: traits.append("随和")
                if big5.get('neuroticism', 0) > 0.6: traits.append("敏感")
                if traits: parts.append(f"性格关键词：{'、'.join(traits)}。")

        # 兴趣
        interest = profile.get('interest_profile')
        if interest:
            tags = interest.get('tags', [])
            if tags: parts.append(f"我的兴趣爱好包括：{'、'.join(tags)}。")
            
        # 价值观/生活方式 (可选)
        life = profile.get('lifestyle_profile')
        if life:
            parts.append(f"生活习惯：{life.get('smoking', '未知')}抽烟，{life.get('drinking', '未知')}喝酒。")

        return " ".join(parts)