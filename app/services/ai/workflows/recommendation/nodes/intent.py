# -*- coding: utf-8 -*-
from datetime import datetime, date
from bson import ObjectId
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.core.config import settings
from app.common.models.state import MatchmakingState
from app.core.env_utils import API_KEY, BASE_URL
from app.services.ai.workflows.recommendation.state import IntentOutput

class IntentNode:
    def __init__(self, db_manager):
        self.db = db_manager
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            temperature=0,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        
        self.intent_parser = PydanticOutputParser(pydantic_object=IntentOutput)
        self.intent_chain = (
            ChatPromptTemplate.from_template(
                """你是一个专业红娘助手。请分析用户输入，提取意图。
                
                输入: {user_input}
                
                【判断标准】:
                1. **search_candidate**: 用户想找人、换一批、改条件 (如 "找个180的", "换个年轻点的")。
                2. **deep_dive**: 用户对**之前推荐的某个人**感兴趣，想深入了解或**询问追求建议** (如 "林薇怎么样", "说说张三的性格", "怎么追她", "如何和她相处")。
                3. **chitchat**: 纯闲聊 (如 "你好"), 或者**通用情感咨询/个人提升问题** (如 "我该怎么提升自己", "送女生什么礼物好")。
                
                【字段提取】:
                - 如果是 `search_candidate`: 提取 `match_policy` 和 `keywords`。
                - 如果是 `deep_dive`: 提取 `target_person` (名字或 "第一个人")。**如果用户使用了代词（如"她"、"他"、"这个人"），请将 target_person 设为 "THE_LAST_ONE"**。
                
                【任务 3: 提取关键词 (keywords)】
                提取用于语义检索的关键词。
                **重要**：
                1. 请包含 **学历、职业、工作内容** 等半硬性指标 (因为它们不在Mongo索引中)。
                2. 请包含 **兴趣、性格、价值观** 等软性描述。
                3. **请排除** 城市、年龄、身高、性别 等硬性指标 (因为它们已经用于数据库筛选了)。
                
                例如："找杭州的985程序员，喜欢滑雪" -> keywords: "985 程序员 喜欢滑雪" (去掉了杭州)
                
                输出JSON: {format_instructions}"""
            ) | self.llm | self.intent_parser
        )
        
        # [NEW] 通用对话 Chain (Chat/Consultation)
        self.chitchat_llm = ChatOpenAI(
            model=settings.llm.model_name,
            temperature=0.7, 
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        self.chitchat_chain = (
            ChatPromptTemplate.from_template(
                """你是一位**资深婚恋顾问**，说话**专业、知性、温暖且有边界感**。
                
                【当前用户画像】: {user_summary}
                【用户输入】: {user_input}
                
                请直接回复用户：
                1. 如果是打招呼，礼貌回应。
                2. 如果是**情感咨询**或**自我提升**问题，请结合用户画像给出客观、建设性的建议。
                3. **严禁**使用过于亲昵或油腻的称呼（如“弟弟”、“姐姐”、“亲”），保持专业形象。
                4. 回复要言之有物，不要空洞的套话。
                5. 如果话题偏离太远，可以幽默地拉回来，提醒他你最擅长的是帮他找对象。
                请直接输出回复内容，不要带任何前缀。"""
            ) | self.chitchat_llm
        )

    def load_profile(self, state: MatchmakingState):
        """Step 0: 加载当前用户画像"""
        print(f"👤 [LoadProfile] 加载用户: {state['user_id']}")
        try:
            uid = ObjectId(state['user_id'])
            user_basic = self.db.users_basic.find_one({"_id": uid})
            if not user_basic:
                user_basic = {"gender": "unknown", "city": "unknown", "birthday": date(2000, 1, 1)}
            
            state['current_user_gender'] = user_basic.get('gender')
            state['current_user_summary'] = f"性别:{user_basic.get('gender')}, 城市:{user_basic.get('city')}, 年龄:{self._calc_age(user_basic.get('birthday'))}"
            state['search_count'] = 0 
            
        except Exception as e:
            print(f"   ❌ 加载用户失败: {e}")
            state['error_msg'] = str(e)
        return state

    def analyze_intent(self, state: MatchmakingState):
        """Step 1: 意图识别 & 策略提取 & 指代消解"""
        if state.get('error_msg'): return state

        print(f"🤔 [Intent] 分析: {state['current_input']}")
        try:
            res = self.intent_chain.invoke({
                "user_input": state['current_input'],
                "format_instructions": self.intent_parser.get_format_instructions()
            })
            state['intent'] = res.intent
            state['semantic_query'] = res.keywords
            state['match_policy'] = res.match_policy.model_dump()
            
            # 深度探索逻辑 & 指代消解
            if res.intent == "deep_dive":
                target = res.target_person
                last_target = state.get('last_target_person')
                
                if target == "THE_LAST_ONE":
                    if last_target:
                        print(f"   -> 指代消解: '她/他' -> {last_target}")
                        target = last_target
                    else:
                        print("   -> 指代消解失败: 上下文无目标，尝试默认取第一个")
                        target = None 
                
                state['target_person_name'] = target
                
                if target:
                    state['last_target_person'] = target
                    
                print(f"   -> 深度探索目标: {target}")
            
            if res.intent == "search_candidate":
                print(f"   -> 搜索策略: 学历权重={res.match_policy.education_weight}, 工作={res.match_policy.job_weight}, 家庭={res.match_policy.family_weight}")
            
        except Exception as e:
            print(f"   ❌ 意图识别失败: {e}")
            state['intent'] = "chitchat"
        return state

    def chitchat(self, state: MatchmakingState):
        """通用对话/咨询节点"""
        try:
            res = self.chitchat_chain.invoke({
                "user_summary": state.get('current_user_summary', '未知用户'),
                "user_input": state['current_input']
            })
            state['reply'] = res.content
        except Exception as e:
            print(f"   ❌ 闲聊生成失败: {e}")
            state['reply'] = "我是您的专属红娘，主要负责帮您找对象哦~ (刚才脑子短路了一下)"
        return state

    def _calc_age(self, birthday_val):
        if not birthday_val: return 0
        try:
            # 统一转为 date 对象进行计算
            if isinstance(birthday_val, datetime):
                b_date = birthday_val.date()
            elif isinstance(birthday_val, date):
                b_date = birthday_val
            else:
                return 0
                
            today = date.today()
            return today.year - b_date.year - ((today.month, today.day) < (b_date.month, b_date.day))
        except:
            return 0
