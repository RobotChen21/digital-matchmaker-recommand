# -*- coding: utf-8 -*-
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from bson import ObjectId

from app.core.container import container
from app.common.models.state import MatchmakingState
from app.core.utils.format_utils import format_history
from app.services.ai.workflows.recommendation.state import DeepDiveOutput


class DeepDiveNode:
    def __init__(self):
        self.db = container.db
        self.chroma = container.chroma
        self.llm_intent = container.get_llm("intent") # temperature=0
        self.llm_chat = container.get_llm("chat")    # temperature=0.7
        self.profile_service = container.profile_service

        # 1. 实体识别/指代消解 Chain
        self.target_parser = PydanticOutputParser(pydantic_object=DeepDiveOutput)
        self.target_extractor_chain = (
            ChatPromptTemplate.from_template(
                """你是一个聪明的助手。请结合【对话历史】和【候选人列表】，识别用户当前想询问的是哪一位候选人。
                
                【候选人列表】:
                {candidate_names}
                
                【对话历史】:
                {chat_history}
                
                【用户输入】: {user_input}
                
                【任务】:
                1. 识别目标：用户可能使用姓名、代词（他、她、这个人）、序数词（第一个、最后一位）。
                2. 解析：请将这些表达**解析为列表中标准的姓名**。
                3. 如果用户没有指明目标，且历史记录中也没有明确目标，请尝试返回上一轮的目标姓名。
                
                输出JSON: {format_instructions}"""
            ) | self.llm_intent | self.target_parser
        )
        
        # 2. 深度分析回答 Chain
        self.deep_answer_chain = (
            ChatPromptTemplate.from_template(
                """你是一位资深的心理咨询师兼红娘。用户对候选人【{name}】很感兴趣，正在询问详情或追求建议。
                
                【用户问题】: {user_input}
                
                【候选人详细档案 (自我介绍)】:
                {candidate_profile_summary}
                
                【过往聊天记录精选 (Evidence)】:
                {chat_evidence}
                
                请结合画像和聊天记录，深入分析并回答用户的问题。
                要求：
                1. 语气知心、专业、独到。
                2. 不要仅仅复述信息，要给出你的专业见解。
                3. 如果用户问的是“怎么追/怎么相处”，请重点分析性格匹配度并给出具体建议。
                """
            ) | self.llm_chat
        )

    def deep_dive(self, state: MatchmakingState):
        """处理深度询问意图"""
        # --- 第一阶段: 指代消解 (谁是目标?) ---
        candidates = state.get('final_candidates', [])
        cand_names = []
        for c in candidates:
            if isinstance(c, dict):
                name = c.get('nickname') or c.get('name')
            else:
                name = getattr(c, 'nickname', None) or getattr(c, 'name', None)
            if name:
                cand_names.append(name)
        cand_names_str = f"[{', '.join(cand_names)}]" if cand_names else "(无当前候选人)"

        history_str = format_history(state.get('messages', []))

        target_name = None
        try:
            res = self.target_extractor_chain.invoke({
                "user_input": state['current_input'],
                "chat_history": history_str,
                "candidate_names": cand_names_str,
                "format_instructions": self.target_parser.get_format_instructions()
            })
            target_name = res.target_person
            print(f"🕵️ [DeepDive] 消解结果: {target_name} (理由: {res.reason})")
        except Exception as e:
            print(f"   ❌ 指代消解失败: {e}")
            target_name = state.get('last_target_person') # 退回到上一个

        # --- 第二阶段: 锁定对象档案 ---
        target_candidate = None
        if target_name:
            for c in candidates:
                c_name = (c.get('nickname') if isinstance(c, dict) else getattr(c, 'nickname', ''))
                if c_name == target_name:
                    target_candidate = c
                    break
        
        # 记录最后一次的目标
        if target_name:
            state['last_target_person'] = target_name

        # 兜底: 依然找不到，生成反问
        if not target_candidate:
            print("   ⚠️ 未找到目标用户，触发反问")
            state['reply'] = f"抱歉，我不确定您指的是哪位。请告诉我具体的名字，或者您可以说“第一个”、“第二个”。"
            return state
            
        print(f"   -> 锁定目标: {target_candidate.get('nickname')}")
        
        # --- 第三阶段: 获取深度信息并回复 ---
        uid = ObjectId(target_candidate['id'])
        profile_doc = self.db.db["users_profile"].find_one({"user_id": uid}) or {}
        basic_doc = self.db.users_basic.find_one({'_id':uid}) or {}
        
        # 生成画像摘要 (使用带缓存的新方法)
        candidate_profile_summary = self.profile_service.get_profile_summary_with_cache(
            basic_doc, 
            profile_doc, 
            self.db.profile
        )

        # 检索聊天记录 (Evidence)
        query = state['current_input']
        docs = self.chroma.retrieve_related_context(
            query, 
            user_id=target_candidate['id'], 
            k=3, 
            filter={"dialogue_type": {"$in": ["onboarding", "social"]}}
        )
        chat_evidence = "\n".join([d.page_content for d in docs]) if docs else "暂无相关聊天记录"
        
        # 生成回复
        try:
            res = self.deep_answer_chain.invoke({
                "name": target_candidate['nickname'],
                "user_input": state['current_input'],
                "candidate_profile_summary": candidate_profile_summary,
                "chat_evidence": chat_evidence
            })
            state['reply'] = res.content
        except Exception as e:
            print(f"   ❌ 回答生成失败: {e}")
            state['reply'] = "哎呀，分析这位嘉宾时出了点小差错，请稍后再试。"
            
        return state
