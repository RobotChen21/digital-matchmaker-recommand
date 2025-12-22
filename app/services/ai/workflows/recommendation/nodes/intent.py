# -*- coding: utf-8 -*-
from datetime import datetime, date
from bson import ObjectId
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.core.container import container
from app.common.models.state import MatchmakingState
from app.core.utils.format_utils import format_history
from app.services.ai.workflows.recommendation.state import IntentOutput

class IntentNode:
    def __init__(self):
        self.db = container.db
        self.llm = container.get_llm("intent") # temperature=0
        
        self.intent_parser = PydanticOutputParser(pydantic_object=IntentOutput)
        self.intent_chain = (
            ChatPromptTemplate.from_template(
                """你是一个专业红娘助手。请结合【对话历史】和【当前候选人列表】分析用户的【最新输入】，提取意图。
                
                【对话历史】:
                {chat_history}
                
                【最新输入】: {user_input}
                
                【判断标准】:
                1. **search_candidate**: 用户想**发起新搜索**或**修改筛选条件**。
                   - 例如: "找个180的", "换个年轻点的", "我想找上海的", "有没有程序员"。
                2. **refresh_candidate**: 用户对当前条件没意见，仅仅想**换一批人** / **翻页**。
                   - 例如: "换一批", "再推荐几个", "还有吗", "不满意", "看点别的"。
                3. **deep_dive**: 用户对**之前推荐的某个人**感兴趣，想深入了解或**询问追求建议**。
                   - 例如: "林薇怎么样", "说说张三的性格", "怎么追她", "如何和她相处"。
                4. **chitchat**: 纯闲聊 (如 "你好"), 或者**通用情感咨询/个人提升问题**。
                
                请直接进行意图分类，不要做多余的分析。
                
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
            state['current_user_basic'] = user_basic
            state['current_user_profile'] = user_profile
            state['current_user_summary'] = summary
            state['search_count'] = 0 
            
        except Exception as e:
            print(f"   ❌ 加载用户失败: {e}")
            state['error_msg'] = str(e)
        return state

    def analyze_intent(self, state: MatchmakingState):
        """Step 1: 纯意图识别 (Router)"""
        if state.get('error_msg'): return state

        print(f"🤔 [Intent] 分析: {state['current_input']}")
        
        # 格式化历史记录
        history_str = format_history(state.get('messages', []))

        try:
            res = self.intent_chain.invoke({
                "user_input": state['current_input'],
                "chat_history": history_str,
                "format_instructions": self.intent_parser.get_format_instructions()
            })
            state['intent'] = res.intent
            
        except Exception as e:
            print(f"   ❌ 意图识别失败: {e}")
            state['intent'] = "chitchat"
        return state

    def chitchat(self, state: MatchmakingState):
        """通用对话/咨询节点"""
        # 格式化历史记录
        history_str = format_history(state.get('messages', []))
        
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
