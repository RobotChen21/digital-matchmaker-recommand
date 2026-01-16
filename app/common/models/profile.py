# -*- coding: utf-8 -*-
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# 1. 兴趣画像
class InterestProfile(BaseModel):
    tags: List[str] = Field(default_factory=list, description="兴趣标签列表")
    strength: Dict[str, Optional[float]] = Field(default_factory=dict, description="各项兴趣的强度(0.0-1.0)")

# 2. 人格画像
class Big5Traits(BaseModel):
    openness: Optional[float] = Field(None, description="开放性 (0.0-1.0)")
    conscientiousness: Optional[float] = Field(None, description="尽责性 (0.0-1.0)")
    extroversion: Optional[float] = Field(None, description="外向性 (0.0-1.0)")
    agreeableness: Optional[float] = Field(None, description="宜人性 (0.0-1.0)")
    neuroticism: Optional[float] = Field(None, description="神经质/情绪不稳定性 (0.0-1.0)")

class PersonalityProfile(BaseModel):
    reasoning: str = Field(description="[CoT] 在得出结论前，先分析用户的用词、语气和行为模式")
    big5: Optional[Big5Traits] = None
    mbti: Optional[str] = Field(None, description="MBTI类型 (如 INTJ, ENFP)")

# 3. 价值观画像
class ValuesProfile(BaseModel):
    reasoning: str = Field(description="[CoT] 分析用户在做选择时的倾向，以此推导价值观权重")
    family: Optional[float] = Field(None, description="家庭观念重视程度 (0.0-1.0)")
    career: Optional[float] = Field(None, description="事业重视程度 (0.0-1.0)")
    romance: Optional[float] = Field(None, description="爱情重视程度 (0.0-1.0)")
    freedom: Optional[float] = Field(None, description="自由重视程度 (0.0-1.0)")
    money: Optional[float] = Field(None, description="金钱/物质重视程度 (0.0-1.0)")

# 4. 生活方式画像
class LifestyleProfile(BaseModel):
    sleep_schedule: Optional[str] = Field(None, description="作息习惯 (早睡早起/熬夜/不规律)")
    exercise_level: Optional[str] = Field(None, description="运动频率 (从不/偶尔/经常/狂热)")
    social_activity: Optional[str] = Field(None, description="社交活跃度 (宅/偶尔社交/社交达人)")
    smoking: Optional[str] = Field(None, description="吸烟习惯")
    drinking: Optional[str] = Field(None, description="饮酒习惯")

# 5. 恋爱风格
class LoveStyleProfile(BaseModel):
    attachment_style: Optional[str] = Field(None, description="依恋类型 (安全型/焦虑型/回避型/恐惧型)")
    love_languages: List[str] = Field(default_factory=list, description="爱的语言 (服务的行动/肯定的言辞/礼物/身体接触/高质量时间)")
    dating_style: Optional[str] = Field(None, description="约会风格")

# 6. 风险画像
class RiskProfile(BaseModel):
    reasoning: str = Field(description="[CoT] 详细列举风险判定依据 (言辞过激、暴力倾向等证据)")
    emotional_stability: Optional[float] = Field(None, description="情绪稳定性评分 (0.0-1.0, 越低越不稳定)")
    safety_risk: Optional[float] = Field(None, description="安全风险评分 (0.0-1.0, 越高风险越大)")
    self_reported_issues: Optional[str] = Field(None, description="自述的问题或雷点")

# 7. 教育画像
class EducationProfile(BaseModel):
    highest_degree: Optional[str] = Field(None, description="最高学历")
    school_type: Optional[str] = Field(None, description="学校类型 (985/211/海外/双非)")
    school_name: Optional[str] = Field(None, description="学校名称")
    major: Optional[str] = Field(None, description="专业")

# 8. 工作画像
class OccupationProfile(BaseModel):
    job_title: Optional[str] = Field(None, description="职位名称")
    industry: Optional[str] = Field(None, description="所在行业")
    work_style: Optional[str] = Field(None, description="工作风格 (稳定/高压/996/自由职业)")
    income_level: Optional[str] = Field(None, description="收入水平描述")

# 9. 家庭画像
class FamilyProfile(BaseModel):
    family_structure: Optional[str] = Field(None, description="家庭结构 (独生/多子女/单亲)")
    parents_health: Optional[str] = Field(None, description="父母健康状况")
    parents_occupation: Optional[str] = Field(None, description="父母职业")
    siblings: Optional[str] = Field(None, description="兄弟姐妹情况")
    family_economy_level: Optional[str] = Field(None, description="家庭经济状况")
    family_atmosphere: Optional[str] = Field(None, description="家庭氛围与状况 (如: 和睦、离异、重组家庭)")

# 10. 约会偏好
class DatingPreferences(BaseModel):
    preferred_age_range: Optional[str] = Field(None, description="期望年龄范围")
    preferred_city: Optional[str] = Field(None, description="期望城市")
    priorities: List[str] = Field(default_factory=list, description="择偶优先看重的点")
    dealbreakers: List[str] = Field(default_factory=list, description="绝对不能接受的雷点")

# 11. 聊天行为 (基于 chat_records)
class BehaviorProfile(BaseModel):
    avg_response_speed: Optional[float] = Field(None, description="平均回复速度评分 (0-1)")
    communication_style: Optional[str] = Field(None, description="沟通风格 (主动/被动/幽默/严肃)")
    positivity_score: Optional[float] = Field(None, description="积极情绪评分 (0-1)")
    topics_liked: List[str] = Field(default_factory=list, description="喜欢聊的话题")
    topics_avoided: List[str] = Field(default_factory=list, description="回避的话题")

# --- 汇总的大画像模型 ---
class UserProfile(BaseModel):
    user_id: Optional[str] = None # MongoDB ObjectId as string
    
    interest_profile: Optional[InterestProfile] = None
    personality_profile: Optional[PersonalityProfile] = None
    values_profile: Optional[ValuesProfile] = None
    lifestyle_profile: Optional[LifestyleProfile] = None
    love_style_profile: Optional[LoveStyleProfile] = None
    risk_profile: Optional[RiskProfile] = None
    education_profile: Optional[EducationProfile] = None
    occupation_profile: Optional[OccupationProfile] = None
    family_profile: Optional[FamilyProfile] = None
    dating_preferences: Optional[DatingPreferences] = None
    behavior_profile: Optional[BehaviorProfile] = None

# --- 常量定义 ---
REQUIRED_PROFILE_DIMENSIONS = [
    "教育背景 - 学历 (本科/硕士/博士/专科)",
    "教育背景 - 学校类型 (985/211/海外/双非)",
    "教育背景 - 学校名称/专业",
    "工作职业 - 职位/行业 (或是学生身份)",
    "工作职业 - 工作风格 (996/轻松/体制内) [学生可免]",
    "工作职业 - 收入水平 (如: 年薪30w+) [学生可免]",
    "家庭背景 - 独生子女？兄弟姐妹？",
    "家庭背景 - 父母健康/职业/退休？",
    "家庭背景 - 家庭经济状况？",
    "家庭背景 - 家庭氛围/父母婚姻状况(离异/重组)?",
    # 其他非强制维度
    "兴趣爱好 (具体的活动)",
    "核心价值观 (家庭观/事业观/金钱观)", 
    "生活方式 (烟酒/社交/运动量)",
    "恋爱风格 (依恋类型/粘人程度)", 
    "约会偏好 (理想型/雷点)" 
]
