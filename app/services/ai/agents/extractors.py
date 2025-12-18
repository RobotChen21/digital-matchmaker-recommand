# -*- coding: utf-8 -*-
from typing import List, Dict, Type, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from app.common.models.profile import (
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
           - **特殊情况**: 如果用户明确表示**“没有爱好”、“平时什么都不干”**，请提取 `tags=["无特殊爱好"]`，**不要留空**。
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
        return """你负责提取用户的生活习惯。请用**简短的短语或一句话概括**。
        
        - 作息: 例如“早睡早起”、“经常熬夜”、“不规律”。
        - 运动: 例如“健身房常客”、“偶尔跑步”、“不怎么运动”。
        - 社交: 例如“社牛”、“宅家”、“只跟熟人玩”。
        - 烟酒: 例如“抽烟”、“偶尔喝酒”、“不沾烟酒”。"""

class LoveStyleExtractor(BaseProfileExtractor):
    """恋爱风格分析 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, LoveStyleProfile)
        
    def _get_system_prompt(self) -> str:
        return """你是一位情感咨询师，分析用户的依恋类型和恋爱风格。请用短语或一句话概括。
        
        - 依恋类型: 例如“安全型”、“焦虑型”、“回避型”、“恐惧型”。
        - 爱的语言: 例如“服务的行动”、“肯定的言辞”、“礼物”、“身体接触”、“高质量时间”。
        - 约会风格: 例如“慢热”、“激情”、“务实”、“浪漫”。"""

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
        - 学历: 例如“本科”、“硕士”、“博士”、“专科”等。
        - 学校类型: 例如“985”、“211”、“海外QS100”、“双非”等。
        - 学校/专业: 具体名称，例如“清华大学计算机科学与技术”。"""

class OccupationExtractor(BaseProfileExtractor):
    """职业背景提取 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, OccupationProfile)
        
    def _get_system_prompt(self) -> str:
        return """提取用户的工作信息。
        
        【特殊规则：学生群体】
        如果用户是**学生/在读** (本科/硕士/博士等):
        - 职位(job_title): 填 "学生"、"博士在读"、"研二在读" 等。
        - 行业(industry): 填 "学术/教育"。
        - 工作风格(work_style): 描述学业状态，如 "科研压力大"、"课程轻松"、"全职实习"。
        - 收入水平(income_level): 如果未提及，填 "无收入" 或 "奖学金/生活费"，不要留空，以免被判定为缺失。

        【常规规则：职场群体】
        - 职位/行业: 例如“软件工程师/互联网”、“老师/教育”。
        - 工作风格: 例如“996”、“轻松”、“体制内”、“自由职业”。
        - 收入水平: 例如“年薪30w+”、“中等偏上”，如果用户提到，请记录。"""

class FamilyExtractor(BaseProfileExtractor):
    """家庭背景提取 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, FamilyProfile)
        
    def _get_system_prompt(self) -> str:
        return """提取用户的家庭原生家庭信息。请用短语或一句话概括。
        - 独生子女？兄弟姐妹？例如“独生子女”、“有姐弟”。
        - 父母健康/职业/退休？例如“父母健在，已退休”、“母亲是教师”。
        - 家庭经济状况？例如“小康”、“富裕”、“普通”。
        - 家庭氛围与状况？**重点提取**：是否离异、单亲、重组家庭、是否和睦、父母关系如何。"""

class DatingPrefExtractor(BaseProfileExtractor):
    """择偶偏好提取 Agent"""
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm, DatingPreferences)
        
    def _get_system_prompt(self) -> str:
        return """提取用户对另一半的要求。请用短语或一句话概括。
        - 年龄/城市要求: 例如“比我小5岁以内”、“同城”。
        - Priorities (加分项): 例如“必须有共同爱好”、“希望TA有上进心”。
           - **特殊**: 如果用户说**“没要求”、“看感觉/看眼缘”**，请填 `["看眼缘/无特殊要求"]`，不要留空。
        - Dealbreakers (雷点): 例如“绝对不能接受抽烟”、“不接受异地恋”。
           - **特殊**: 如果用户明确说**“没啥雷点”**，请填 `["无明显雷点"]`。"""