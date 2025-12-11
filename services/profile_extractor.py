# -*- coding: utf-8 -*-
from typing import List, Dict, Type, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from models.profile import (
    PersonalityProfile, InterestProfile, ValuesProfile, LifestyleProfile,
    LoveStyleProfile, RiskProfile, EducationProfile, OccupationProfile,
    FamilyProfile, DatingPreferences
)

class BaseProfileExtractor:
    """画像提取 Agent 基类"""
    
    def __init__(self, llm: ChatOpenAI, output_model: Type[BaseModel]):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=output_model)
        self.output_model = output_model

    def _get_system_prompt(self) -> str:
        """子类必须实现此方法，提供具体的 System Prompt"""
        raise NotImplementedError

    def extract(self, conversation_text: str) -> Optional[BaseModel]:
        """执行提取逻辑"""
        prompt = ChatPromptTemplate.from_template(
            """{system_prompt}

请分析以下对话记录，提取相关用户画像信息。

【重要原则】
1. **实事求是**：只提取用户明确表达或通过行为强烈暗示的信息。
2. **宁缺毋滥**：如果信息不足或无法确定，请将对应字段留为 null/None，**绝对不要猜测或编造**。
3. **保持中立**：客观描述，不要带个人情感色彩。

【对话记录】
{conversation}

【输出格式】
请严格按照以下 JSON 格式输出 (不要任何 Markdown 标记):
{format_instructions}
"""
        )

        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "system_prompt": self._get_system_prompt(),
                "conversation": conversation_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # 简单的清洗逻辑
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
                
            return self.parser.parse(content)
            
        except Exception as e:
            print(f"❌ Extractor [{self.__class__.__name__}] failed: {e}")
            return None


class PersonalityExtractor(BaseProfileExtractor):
    """性格分析 Agent (Big5 + MBTI)"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, PersonalityProfile)
    
    def _get_system_prompt(self) -> str:
        return """你是一位资深的人格心理学家。请分析用户的大五人格 (0.0-1.0) 和 MBTI 类型。
        
        - 开放性 (Openness): 想象力、好奇心
        - 尽责性 (Conscientiousness): 条理、自律
        - 外向性 (Extroversion): 活力、社交
        - 宜人性 (Agreeableness): 利他、信任
        - 神经质 (Neuroticism): 焦虑、情绪波动
        
        MBTI: 给出 4 字母代码 (如 ENFP)。如果信息不足，请留空。"""

class InterestExtractor(BaseProfileExtractor):
    """兴趣分析 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, InterestProfile)
        
    def _get_system_prompt(self) -> str:
        return """你负责提取用户的兴趣爱好。
        
        1. **Tags**: 提取具体的兴趣标签（如：摄影、马拉松、科幻电影）。
        2. **Strength**: 评估用户对该兴趣的热情程度 (0.0-1.0)。
           - 1.0 = 狂热/专业级
           - 0.5 = 业余爱好
           - 0.1 = 稍微感兴趣"""

class ValuesExtractor(BaseProfileExtractor):
    """价值观分析 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, ValuesProfile)
        
    def _get_system_prompt(self) -> str:
        return """你负责分析用户的核心价值观权重 (0.0-1.0)。
        通过用户做选择的倾向、对未来的规划来判断：
        
        - 家庭: 是否渴望结婚生子？是否听从父母？
        - 事业: 是否是工作狂？是否有野心？
        - 爱情: 是否恋爱脑？是否追求浪漫？
        - 自由: 是否讨厌被管束？是否喜欢独处？
        - 金钱: 是否看重物质条件？"""

class LifestyleExtractor(BaseProfileExtractor):
    """生活方式分析 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, LifestyleProfile)
        
    def _get_system_prompt(self) -> str:
        return """你负责提取用户的生活习惯。
        
        - 作息: 早睡早起 / 熬夜修仙 / 不规律
        - 运动: 健身房常客 / 偶尔跑步 / 躺平
        - 社交: 社牛 / 宅 / 只跟熟人玩
        - 烟酒: 抽烟 / 喝酒 / 不沾 / 偶尔"""

class LoveStyleExtractor(BaseProfileExtractor):
    """恋爱风格分析 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, LoveStyleProfile)
        
    def _get_system_prompt(self) -> str:
        return """你是一位情感咨询师，分析用户的依恋类型和恋爱风格。
        
        - 依恋类型: 安全型 / 焦虑型 / 回避型 / 恐惧型
        - 爱的语言: 喜欢怎么表达爱？(服务/言辞/礼物/接触/时间)
        - 约会风格: 慢热 / 激情 / 务实 / 浪漫"""

class RiskExtractor(BaseProfileExtractor):
    """风险分析 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, RiskProfile)
        
    def _get_system_prompt(self) -> str:
        return """你是风控专家，负责识别潜在的交往风险。
        
        - 情绪稳定性: 打分 0.0(极不稳定)-1.0(非常稳定)。注意是否有过激言论、消极抱怨。
        - 安全风险: 打分 0.0(安全)-1.0(危险)。注意是否有暴力倾向、欺诈嫌疑、极端思想。
        - 自述问题: 用户自己承认的缺点或雷点（如：欠债、脾气不好、有遗传病）。"""

class EducationExtractor(BaseProfileExtractor):
    """教育背景提取 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, EducationProfile)
        
    def _get_system_prompt(self) -> str:
        return """提取用户的教育信息。
        - 学历: 本科/硕士/博士/专科
        - 学校类型: 985/211/海外QS100/双非
        - 学校/专业: 具体名称"""

class OccupationExtractor(BaseProfileExtractor):
    """职业背景提取 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, OccupationProfile)
        
    def _get_system_prompt(self) -> str:
        return """提取用户的工作信息。
        - 职位/行业
        - 工作风格: 996/轻松/体制内
        - 收入水平: 如果用户提到，请记录（如：年薪30w+）。"""

class FamilyExtractor(BaseProfileExtractor):
    """家庭背景提取 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, FamilyProfile)
        
    def _get_system_prompt(self) -> str:
        return """提取用户的家庭原生家庭信息。
        - 独生子女？兄弟姐妹？
        - 父母健康/职业/退休？
        - 家庭经济状况？"""

class DatingPrefExtractor(BaseProfileExtractor):
    """择偶偏好提取 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, DatingPreferences)
        
    def _get_system_prompt(self) -> str:
        return """提取用户对另一半的要求。
        - 年龄/城市要求
        - Priorities (加分项): 必须有的特质
        - Dealbreakers (雷点): 绝对不能接受的特质"""