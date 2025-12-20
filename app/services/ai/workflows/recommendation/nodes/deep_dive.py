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
                
                【候选人档案】:
                - 基础信息: {basic_info}
                - 性格/MBTI: {personality}
                - 价值观: {values}
                - 恋爱观: {love_style}
                
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
        target_candidate = None
        
        print(f"   [Debug] Target: '{target_name}', Candidates: {[c['nickname'] for c in candidates]}")

        # 策略 A: 精确匹配 (优先)
        for c in candidates:
            if target_name == c['nickname']:
                target_candidate = c
                break
        
        # 策略 B: 包含匹配 (次优)
        if not target_candidate:
            for c in candidates:
                if target_name in c['nickname'] or c['nickname'] in target_name:
                    target_candidate = c
                    break
                    
        # 策略 C: 序号匹配 (如 "第二个")
        if not target_candidate:
            cn_nums = {"一": 0, "二": 1, "三": 2}
            for cn, idx in cn_nums.items():
                if f"第{cn}" in target_name and idx < len(candidates):
                    target_candidate = candidates[idx]
                    break
        
        # 如果找不到，尝试默认取第一个（如果用户没说名字）
        if not target_candidate and candidates:
             if not target_name: 
                 target_candidate = candidates[0]
        
        if not target_candidate:
            state['reply'] = f"抱歉，我不确定您指的是哪位。请告诉我名字，或者先让我为您推荐几位嘉宾。"
            return state
            
        print(f"🕵️ [DeepDive] 深入分析: {target_candidate['nickname']}")
        
        # 2. 准备数据
        uid = ObjectId(target_candidate['id'])
        profile_doc = self.db.db["users_profile"].find_one({"user_id": uid})
        persona_doc = self.db.users_persona.find_one({"user_id": uid})
        
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
                "basic_info": f"{calc_age(self.db.users_basic.find_one({'_id':uid}).get('birthday'))}岁, {self.db.users_basic.find_one({'_id':uid}).get('city')}, {persona_doc.get('persona', {}).get('occupation')}",
                "personality": str(profile_doc.get('personality_profile', {})),
                "values": str(profile_doc.get('values_profile', {})),
                "love_style": str(profile_doc.get('love_style_profile', {})),
                "chat_evidence": chat_evidence
            })
            state['reply'] = res.content
        except Exception as e:
            print(f"   ❌ Deep dive failed: {e}")
            state['reply'] = "哎呀，分析这位嘉宾时出了点小差错，请稍后再试。"
            
        return state