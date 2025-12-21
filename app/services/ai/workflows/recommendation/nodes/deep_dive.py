# -*- coding: utf-8 -*-
from langchain_core.prompts import ChatPromptTemplate
from bson import ObjectId

from app.core.container import container
from app.common.models.state import MatchmakingState
from app.core.utils.cal_utils import calc_age


class DeepDiveNode:
    def __init__(self):
        self.db = container.db
        self.chroma = container.chroma
        self.llm = container.get_llm("chat") # temperature=0.7
        
        self.deep_answer_chain = (
            ChatPromptTemplate.from_template(
                """你是一位资深的心理咨询师兼红娘。用户对候选人【{name}】很感兴趣，正在询问详情。
                
                【用户问题】: {user_input}
                
                【候选人详细档案 (自我介绍)】:
                {candidate_profile_summary}
                
                【过往聊天记录精选 (Evidence)】:
                {chat_evidence}
                
                请结合画像和聊天记录，深入分析并回答用户的问题。
                语气要知心、独到，不要仅仅复述信息，要给出你的专业见解。
                """
            ) | self.llm
        )

    def deep_dive(self, state: MatchmakingState):
        """处理深度询问意图"""
        target_name = state.get('target_person_name', '')
        candidates = state.get('final_candidates', [])
        
        # 1. 锁定目标
        # 由于 IntentNode 已经处理了指代消解 (将代词/序数转为了名字)，
        # 这里我们主要负责根据名字从 candidates 列表里捞出完整的对象。
        target_candidate = None
        
        print(f"   [Debug] DeepDive Target Name: '{target_name}'")

        if target_name:
            # 策略 A: 名字匹配 (优先匹配当前推荐列表)
            for c in candidates:
                # 兼容 nickname 或 name 字段
                c_name = c.get('nickname')
                if c_name == target_name:
                    target_candidate = c
                    break
            
            # 策略 B: 如果推荐列表里没有，尝试去 state['last_target_person'] 找
            # (暂时略过，因为 IntentNode 应该保证了名字的一致性)

        # 兜底: 依然找不到，生成反问
        if not target_candidate:
            print("   ⚠️ 未找到目标用户，触发反问")
            state['reply'] = f"抱歉，我不确定您指的是哪位。请告诉我具体的名字，或者您可以说“第一个”、“第二个”。"
            return state
            
        print(f"🕵️ [DeepDive] 深入分析: {target_candidate.get('nickname')}")
        
        # 2. 准备数据
        uid = ObjectId(target_candidate['id'])
        profile_doc = self.db.db["users_profile"].find_one({"user_id": uid}) or {}
        basic_doc = self.db.users_basic.find_one({'_id':uid}) or {}
        
        # 调用 ProfileService 生成全量画像摘要 (比手动拼字段更全、更自然)
        # 注意：这里需要 container.profile_service 单例，但我看 __init__ 里没引
        # 临时引入一下，或者建议在 __init__ 里加上
        from app.core.container import container
        candidate_profile_summary = container.profile_service.generate_profile_summary(basic_doc, profile_doc)

        # 3. 检索聊天记录 (作为佐证)
        query = state['current_input']
        docs = self.chroma.retrieve_related_context(
            query, 
            user_id=target_candidate['id'], 
            k=3, 
            filter={"dialogue_type": {"$in": ["onboarding", "social"]}}
        )
        chat_evidence = "\n".join([d.page_content for d in docs]) if docs else "暂无相关聊天记录"
        
        # 4. 生成回复
        try:
            res = self.deep_answer_chain.invoke({
                "name": target_candidate['nickname'],
                "user_input": state['current_input'],
                "candidate_profile_summary": candidate_profile_summary,
                "chat_evidence": chat_evidence
            })
            state['reply'] = res.content
        except Exception as e:
            print(f"   ❌ Deep dive failed: {e}")
            state['reply'] = "哎呀，分析这位嘉宾时出了点小差错，请稍后再试。"
            
        return state