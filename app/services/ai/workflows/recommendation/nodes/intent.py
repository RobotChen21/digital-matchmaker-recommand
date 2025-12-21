# -*- coding: utf-8 -*-
from datetime import datetime, date
from bson import ObjectId
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.core.container import container
from app.common.models.state import MatchmakingState
from app.services.ai.workflows.recommendation.state import IntentOutput

class IntentNode:
    def __init__(self):
        self.db = container.db
        self.llm = container.get_llm("intent") # temperature=0
        
        self.intent_parser = PydanticOutputParser(pydantic_object=IntentOutput)
        self.intent_chain = (
            ChatPromptTemplate.from_template(
                """你是一个专业红娘助手。请结合【对话历史】和【当前候选人列表】分析用户的【最新输入】，提取意图。
                
                【当前候选人列表】(指代消解的参考选项):
                {candidate_names}
                
                【对话历史】:
                {chat_history}
                
                【最新输入】: {user_input}
                
                【判断标准】:
                1. **search_candidate**: 用户想找人、换一批、改条件 (如 "找个180的", "换个年轻点的")。
                2. **deep_dive**: 用户对**之前推荐的某个人**感兴趣，想深入了解或**询问追求建议** (如 "林薇怎么样", "说说张三的性格", "怎么追她", "如何和她相处")。
                3. **chitchat**: 纯闲聊 (如 "你好"), 或者**通用情感咨询/个人提升问题** (如 "我该怎么提升自己", "送女生什么礼物好")。
                
                【字段提取】:
                - 如果是 `search_candidate`: 提取 `match_policy` 和 `keywords`。
                - 如果是 `deep_dive`: 提取 `target_person` (具体姓名)。
                    **重要**：请优先从【当前候选人列表】中匹配。将用户使用的代词（如“他”、“她”、“这个人”）或序数词（如“第一个”）**解析为列表中的标准姓名**。
                    - 如果实在无法确定，请输出 "THE_LAST_ONE"。
                
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
        self.chitchat_llm = container.get_llm("chat") # temperature=0.7
        self.profile_service = container.profile_service # 使用单例
        self.chitchat_chain = (
            ChatPromptTemplate.from_template(
                """你是一位**资深婚恋顾问**，说话**专业、知性、温暖且有边界感**。
                
                【当前用户画像】: {user_summary}
                
                【对话历史】:
                {chat_history}
                
                【用户输入】: {user_input}
                
                请直接回复用户：
                1. 如果是打招呼，礼貌回应。
                2. 如果是**情感咨询**或**自我提升**问题，请结合用户画像给出客观、建设性的建议。
                3. **严禁**使用过于亲昵或油腻的称呼（如“弟弟”、“姐姐”、“亲”），保持专业形象。
                4. 回复要言之有物，不要空洞的套话。
                
                请直接输出回复内容，不要带任何前缀。"""
            ) | self.chitchat_llm
        )

    def _format_history(self, messages: list) -> str:
        """Helper: 将 Message 对象列表转为字符串文本"""
        if not messages: return "(无历史记录)"
        text = []
        for m in messages:
            # 兼容 Pydantic 对象或 Dict (因为 State 里可能是对象，也可能是从DB读出的Dict)
            role = getattr(m, 'role', None) or m.get('role')
            content = getattr(m, 'content', None) or m.get('content')
            if role == 'user':
                text.append(f"User: {content}")
            elif role in ['ai', 'assistant']:
                text.append(f"AI: {content}")
        return "\n".join(text)

    def load_profile(self, state: MatchmakingState):
        """Step 0: 加载当前用户全量画像 (Basic + Profile)"""
        print(f"👤 [LoadProfile] 加载用户: {state['user_id']}")
        try:
            uid = ObjectId(state['user_id'])
            
            # 1. 查 Basic
            user_basic = self.db.users_basic.find_one({"_id": uid})
            
            # 2. 查 Profile
            user_profile = self.db.profile.find_one({"_id": uid}) or {}
            
            # 3. 生成 Summary
            summary = self.profile_service.generate_profile_summary(user_basic, user_profile)
            
            # 4. 更新 State
            # state['current_user_gender'] = user_basic.get('gender')
            state['current_user_basic'] = user_basic
            state['current_user_profile'] = user_profile
            state['current_user_summary'] = summary
            state['search_count'] = 0 
            
        except Exception as e:
            print(f"   ❌ 加载用户失败: {e}")
            state['error_msg'] = str(e)
        return state

    def analyze_intent(self, state: MatchmakingState):
        """Step 1: 意图识别 & 策略提取 & 指代消解"""
        if state.get('error_msg'): return state

        print(f"🤔 [Intent] 分析: {state['current_input']}")
        
        # 格式化历史记录
        history_str = self._format_history(state.get('messages', []))
        
        # 提取候选人名单 (做成类似 "[林薇, 晓晨]" 的字符串)
        candidates = state.get('final_candidates', [])
        # 兼容 candidate 可能是 dict 或 object
        cand_names = []
        for c in candidates:
            if isinstance(c, dict):
                name = c.get('nickname') or c.get('name')
            else:
                # 假设是 Pydantic 对象
                name = getattr(c, 'nickname', None) or getattr(c, 'name', None)
            if name:
                cand_names.append(name)
        
        cand_names_str = f"[{', '.join(cand_names)}]" if cand_names else "(无推荐记录)"

        try:
            res = self.intent_chain.invoke({
                "user_input": state['current_input'],
                "chat_history": history_str,
                "candidate_names": cand_names_str,
                "format_instructions": self.intent_parser.get_format_instructions()
            })
            state['intent'] = res.intent
            state['semantic_query'] = res.keywords
            state['match_policy'] = res.match_policy.model_dump()
            
            # 深度探索逻辑 & 指代消解
            if res.intent == "deep_dive":
                target = res.target_person
                last_target = state.get('last_target_person')
                
                # 现在的 LLM 应该已经能直接给出名字了 (例如 "林薇")。
                # 只有当 LLM 返回特殊的 "THE_LAST_ONE" 时，我们才动用 Python 兜底。
                if target == "THE_LAST_ONE" and last_target:
                    print(f"   -> Python 兜底消解: 'THE_LAST_ONE' -> {last_target}")
                    target = last_target
                
                state['target_person_name'] = target
                
                # 记录最后一次提到的目标，用于后续可能的 THE_LAST_ONE 兜底
                if target and target != "THE_LAST_ONE":
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
        # 格式化历史记录
        history_str = self._format_history(state.get('messages', []))
        
        try:
            res = self.chitchat_chain.invoke({
                "user_summary": state.get('current_user_summary', '未知用户'),
                "user_input": state['current_input'],
                "chat_history": history_str
            })
            state['reply'] = res.content
        except Exception as e:
            print(f"   ❌ 闲聊生成失败: {e}")
            state['reply'] = "我是您的专属红娘，主要负责帮您找对象哦~ (刚才脑子短路了一下)"
        return state