# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional
from datetime import datetime, date # 导入 date
from langchain_openai import ChatOpenAI
from app.services.ai.agents.extractors import (
    PersonalityExtractor, InterestExtractor, ValuesExtractor,
    LifestyleExtractor, LoveStyleExtractor,
    EducationExtractor, OccupationExtractor, FamilyExtractor,
    DatingPrefExtractor, RiskExtractor
)
from app.common.models.profile import UserProfile
from app.core.llm import get_llm

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
        
        # [NEW] 进度提示 LLM
        self.completion_llm = get_llm(temperature=0.0) # 确保稳定

    def extract_from_dialogue(self, dialogue_text: str) -> Dict[str, Any]:
        """
        输入对话文本，运行所有 Agent，返回聚合后的画像字典。
        采用多线程并行执行，大幅提升提取速度。
        """
        import concurrent.futures
        
        full_profile_data = {}
        
        # 定义单个任务函数
        def _run_agent(name, agent_instance):
            try:
                result = agent_instance.extract(dialogue_text)
                return name, result.model_dump(exclude_none=True) if result else None
            except Exception as e:
                print(f"⚠️ [ProfileService] {name} 提取失败: {e}")
                return name, None

        # 并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # 提交所有任务
            future_to_agent = {
                executor.submit(_run_agent, name, agent): name 
                for name, agent in self.agents.items()
            }
            
            # 获取结果
            for future in concurrent.futures.as_completed(future_to_agent):
                name, data = future.result()
                full_profile_data[name] = data
        
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

    #TODO 这个需要优化，用户画像的摘要是硬编码，应该使用LLM进行总结
    @staticmethod
    def generate_profile_summary(basic: Dict, profile: Dict) -> str:
        """将结构化画像转换为自然语言摘要 (用于向量化)"""
        parts = []
        
        def _get_age(bday: date): # 明确类型为 date
            if not bday or not isinstance(bday, date): return "未知"
            try:
                today = date.today()
                return str(today.year - bday.year - ((today.month, today.day) < (bday.month, bday.day)))
            except:
                return "未知"

        # 基础信息
        age = _get_age(basic.get('birthday'))
        gender_mapping = {"male": "男", "female": "女士"}
        parts.append(f"我是{basic.get('nickname')}，{gender_mapping.get(basic.get('gender'), None)}性，"
                     f"今年{age}岁，住在{basic.get('city')}，身高{basic.get('height')}，"
                     f"体重是{basic.get('weight')}，"
                     f"{basic.get('self_intro_raw')}")
        
        # 职业
        occ = profile.get('occupation_profile')
        if occ:
            parts.append(f"职业是{occ.get('job_title', '未知')}, 行业是{occ.get('industry', '未知')}。")
        
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

    def generate_profile_completion_hint(self, profile: Dict) -> str:
        """
        使用 LLM 生成当前画像的完整度提示。
        """
        from langchain_core.prompts import ChatPromptTemplate
        from app.common.models.profile import REQUIRED_PROFILE_DIMENSIONS # 导入规则
        
        prompt = ChatPromptTemplate.from_template(
            """你是一个数据分析助手。请根据以下【已提取画像JSON】和【必填字段清单】，生成一段简短的【画像完整度提示】，供红娘参考。
            
            【已提取画像】:
            {profile_json}
            
            【必填字段清单】:
            {required_dimensions}
            
            【分析目标】:
            1. **简述已收集到的核心信息** (用一句话概括，如: "已知用户是硕士，程序员，独生子")。
            2. **检查必填项缺失**。对比清单，明确指出还有哪些核心字段缺失。
               - **特殊规则**：如果识别出用户是**学生/在读**，则【工作风格】和【收入水平】不算作缺失项，请勿要求红娘追问。
            3. 如果必填项都全了，请说 "核心画像已完善"。
            
            请直接输出提示内容，不要废话，控制在 50 字以内。"""
        )
        
        chain = prompt | self.completion_llm
        try:
            # 将 profile 转为字符串
            res = chain.invoke({
                "profile_json": str(profile),
                "required_dimensions": "\n".join(REQUIRED_PROFILE_DIMENSIONS)
            })
            return res.content
        except Exception as e:
            print(f"⚠️ [Hint Gen] 生成提示失败: {e}")
            return "当前画像信息分析服务暂时不可用，请根据对话历史判断缺失信息。"
