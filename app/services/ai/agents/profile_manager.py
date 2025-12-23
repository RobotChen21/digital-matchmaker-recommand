# -*- coding: utf-8 -*-
from typing import Dict, Any
from datetime import datetime, date # 导入 date
from langchain_openai import ChatOpenAI
from app.services.ai.agents.extractors import (
    PersonalityExtractor, InterestExtractor, ValuesExtractor,
    LifestyleExtractor, LoveStyleExtractor,
    EducationExtractor, OccupationExtractor, FamilyExtractor,
    DatingPrefExtractor, RiskExtractor
)

class ProfileService:
    """
    画像服务：负责协调各个细分维度的 Extractor，
    从对话文本中提取完整的用户画像。
    """
    def __init__(self, llm: ChatOpenAI):
        self.completion_llm = llm
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

    def generate_profile_summary(self, basic: Dict, profile: Dict) -> str:
        """
        使用 LLM 将结构化画像转换为自然语言摘要 (用于向量化和详情展示)。
        生成一段第三人称的、专业且生动的个人画像。
        """
        import json
        from langchain_core.prompts import ChatPromptTemplate
        
        # 数据预处理：处理日期等非序列化对象
        basic_safe = basic.copy()
        if isinstance(basic_safe.get('birthday'), (date, datetime)):
            basic_safe['birthday'] = str(basic_safe['birthday'])
            
        profile_str = json.dumps(profile, ensure_ascii=False, default=str)
        basic_str = json.dumps(basic_safe, ensure_ascii=False, default=str)

        prompt = ChatPromptTemplate.from_template(
            """请根据以下用户的【基础信息】和【详细画像】，以**第三人称**（称呼其昵称：{nickname}）写一段专业、生动、详尽的个人画像描述。
            这段描述将由专业红娘用于向其他嘉宾介绍该用户，或进行深度匹配分析。

            【基础信息】:
            {basic_info}

            【详细画像】:
            {profile_data}

            【要求】:
            1. **称呼**: 全程使用昵称“{nickname}”或“他/她”，严禁使用第一人称“我”。
            2. **内容全面**: 自然地融入年龄、学历、职业、家庭背景、性格特点(MBTI/Big5)、
            兴趣爱好、核心价值观、生活方式、恋爱观等，详细画像中提到的你都要体现出来。
            3. **文笔生动**: 像一位资深红娘在向人推介，既要客观真实，也要展现该嘉宾的人格魅力和闪光点。
            4. **逻辑清晰**: 不要简单罗列，要通过因果、转折等逻辑将各维度信息串联成一篇丝滑的文章。
            5. **字数控制**: 350-450字左右。

            请直接输出画像描述文本。"""
        )
        
        nickname = basic.get('nickname', '该嘉宾')
        chain = prompt | self.completion_llm
        try:
            res = chain.invoke({
                "nickname": nickname,
                "basic_info": basic_str,
                "profile_data": profile_str
            })
            return res.content
        except Exception as e:
            print(f"⚠️ [Summary Gen] 生成摘要失败: {e}")
            return f"我是{basic.get('nickname', '用户')}，期待在这里遇到对的人。"

    def get_profile_summary_with_cache(self, basic: Dict, profile: Dict, db_collection) -> str:
        """
        获取画像摘要的高级封装 (带缓存 + 5分钟防抖)
        :param basic: 用户基础信息
        :param profile: 用户详细画像 (需包含 timestamps)
        :param db_collection: MongoDB集合对象，用于回写缓存 (如 db["users_profile"])
        :return: 摘要文本
        """
        summary = profile.get("user_summary")
        p_up = profile.get("updated_at")
        s_up = profile.get("summary_updated_at")
        
        need_gen = False
        
        if not summary or not s_up:
            # 场景 A: 还没生成过 -> 必须生成
            need_gen = True
        elif p_up and p_up > s_up:
            # 场景 B: 缓存已过期 (Profile 新于 Summary)
            # 检查 Profile 更新了多久 (防抖: 5分钟内不重刷)
            time_since_update = datetime.now() - p_up
            if time_since_update.total_seconds() > 300: # 300秒 = 5分钟
                need_gen = True
            else:
                need_gen = False # 暂用旧的
        
        if need_gen:
            print(f"   🧠 [ProfileService] 重新生成摘要 (User: {basic.get('nickname')})...")
            summary = self.generate_profile_summary(basic, profile)
            # 回写 DB
            try:
                if "_id" in profile:
                    query = {"_id": profile["_id"]}
                else:
                    query = {"user_id": basic.get("_id")} # Fallback
                
                db_collection.update_one(
                    query,
                    {"$set": {"user_summary": summary, "summary_updated_at": datetime.now()}}
                )
            except Exception as e:
                print(f"   ⚠️ 回写摘要缓存失败: {e}")
        else:
            if summary: 
                # print("   ⚡ 使用缓存摘要")
                pass
            
        return summary or ""

    @staticmethod
    def clean_profile_data(profile: Dict) -> Dict:
        """
        数据清洗：移除 profile 中的 summary 和系统字段，只保留纯净的结构化画像。
        用于瘦身 State，避免数据冗余。
        """
        if not profile:
            return {}
        
        # 浅拷贝，防止修改原引用影响后续逻辑
        clean_data = profile.copy()
        
        # 定义要移除的字段列表
        keys_to_remove = [
            "user_summary", 
            "summary_updated_at",
            "updated_at"
            # "_id",
            # "user_id",
        ]
        
        for k in keys_to_remove:
            clean_data.pop(k, None)
            
        return clean_data

    def generate_profile_completion_hint(self, profile: Dict) -> str:
        """
        使用 LLM 生成当前画像的完整度提示。
        """
        import json
        from langchain_core.prompts import ChatPromptTemplate
        from app.common.models.profile import REQUIRED_PROFILE_DIMENSIONS # 导入规则

        prompt = ChatPromptTemplate.from_template(
            """你是一名资深的婚恋画像分析师。请根据【已提取画像JSON】对比【必填维度清单】，生成一份详尽的【当前画像状态分析】给前台红娘。

            【必填维度清单】:
            {required_dimensions}

            【已提取画像】:
            {profile_json}

            【分析要求】:
            1. ✅ **已收集信息盘点**: 请用精炼的语言概括**所有**已获取的信息。
               - 必须覆盖以下维度（如果有值）：基本资料、教育(学校/专业/学历)、职业(职位/行业/收入)、家庭背景(父母状况/兄弟姐妹/氛围)、兴趣爱好(具体项目)、价值观(人生/金钱/事业)、生活方式 (烟酒/社交/运动量)、恋爱风格(语言/依恋)、择偶标准(优先项/雷点)。
               - 格式示例: "用户是硕士(xx大学)，职业是xx，家庭氛围xx，性格xx，喜欢xx，择偶看重xx..."
            
            2. ❌ **缺失核心项检查**: 指出哪些**必填维度**完全缺失或缺乏关键细节。
               - **注意**: 只要维度下有主要内容(如兴趣有了tags)，就不算缺失，不要过于苛刻。
               - **学生特例**: 若职业信息表明是"学生/在读/科研"，则[工作风格/收入/职业稳定性]自动视为**不缺失**，请勿列入缺失项。
            
            3. 💡 **追问建议**: 基于缺失项，简要建议红娘接下来重点询问哪个方向。**不要建议红娘问具体择偶标准，现在只是对用户画像的提取，还没到让用户择偶的时候，主要是用户自己的画像。**
            
            4. **状态结论**: 如果核心维度（教育、职业、家庭、兴趣、核心价值观、生活方式、恋爱风格、约会偏好）基本齐全，请在开头明确标注 "【核心画像已完善】"。

            请直接输出分析结果，条理清晰，语气专业客观，字数控制在 350 字以内。"""
        )
        
        chain = prompt | self.completion_llm
        try:
            # 将 profile 转为格式化的 JSON 字符串
            profile_str = json.dumps(profile, ensure_ascii=False, indent=2, default=str)
            
            res = chain.invoke({
                "profile_json": profile_str,
                "required_dimensions": "\n".join(REQUIRED_PROFILE_DIMENSIONS)
            })
            return res.content
        except Exception as e:
            print(f"⚠️ [Hint Gen] 生成提示失败: {e}")
            return "当前画像信息分析服务暂时不可用，请根据对话历史判断缺失信息。"

