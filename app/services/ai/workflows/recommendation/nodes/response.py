# -*- coding: utf-8 -*-
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.core.config import settings
from utils.env_utils import API_KEY, BASE_URL
from app.common.models.state import MatchmakingState
from app.services.ai.workflows.recommendation.state import EvidenceOutput

class ResponseNode:
    def __init__(self, chroma_manager):
        self.chroma = chroma_manager
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            temperature=0.4,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        
        self.evidence_parser = PydanticOutputParser(pydantic_object=EvidenceOutput)
        self.evidence_chain = (
            ChatPromptTemplate.from_template(
                """你是一个证据分析师。
                请阅读以下聊天记录片段，判断其中是否包含证明用户【{query}】的证据。
                
                注意：
                1. 只关注【{candidate_nickname}】自己说的话 (User或Role为该候选人)。
                2. 忽略红娘或对方说的话。
                
                【聊天片段】:
                {raw_text}
                
                如果有证据，请用【第三人称】简练概括（例如："她曾提到自己每年冬天都会去崇礼滑雪"）。
                如果没有，summary为空。
                
                输出JSON: {format_instructions}"""
            ) | self.llm | self.evidence_parser
        )

        self.response_chain = (
            ChatPromptTemplate.from_template(
                """你是一位金牌红娘。请根据候选人信息和证据，向用户推荐这几位嘉宾。
                
                【用户需求】: {user_input}
                
                【候选人列表】:
                {candidates_info}
                
                【要求】:
                1. 语气热情、专业、真诚。
                2. 对每位嘉宾，请结合 **画像信息** 和 **证据 (Evidence)** 进行推荐。
                3. 如果有证据 (Evidence)，请务必引用，例如："特别是他之前提到过..."。
                4. 最后鼓励用户进一步了解。
                
                请直接输出推荐语。"""
            ) | self.llm
        )
        
        # [NEW] 失败分析 Chain
        self.failure_chain = (
            ChatPromptTemplate.from_template(
                """你是一位经验丰富的红娘。用户提出了要求，但我们库里目前找不到合适的人选（即使放宽条件尝试后也失败了）。
                
                【用户需求】: {user_input}
                【最后一次尝试的硬性条件】: {hard_filters}
                
                请给用户写一段回复：
                1. 诚恳道歉。
                2. 分析可能的原因（比如条件太严苛、库太小等）。
                3. **给出2-3个具体的建议**，告诉用户可以尝试调整哪些条件（例如：“要不试试放宽身高的要求？”）。
                
                语气要温柔、体贴，不要让用户感到挫败。"""
            ) | self.llm
        )

    def evidence_hunting(self, state: MatchmakingState):
        """Step 4.5: 证据搜寻与智能总结"""
        candidates = state.get('final_candidates', [])
        query = state.get('semantic_query') or state.get('current_input')
        
        if not candidates: return state
        
        print(f"🕵️ [Evidence] 为 {len(candidates)} 位候选人搜寻证据: '{query}'")
        
        for candidate in candidates:
            try:
                # 1. 检索: 只查对话记录
                search_filter = {
                    "$and": [
                        {"user_id": candidate['id']},
                        {"dialogue_type": {"$in": ["onboarding", "social"]}}
                    ]
                }
                docs = self.chroma.retrieve_related_context(query, user_id=candidate['id'], k=2, filter=search_filter)
                
                if docs:
                    # 拼接 raw text
                    raw_text = "\n".join([d.page_content for d in docs])
                    
                    # 2. 总结
                    print(f"   -> Analyzing raw text for {candidate['nickname']}...")
                    res = self.evidence_chain.invoke({
                        "query": query,
                        "raw_text": raw_text,
                        "candidate_nickname": candidate['nickname'], 
                        "format_instructions": self.evidence_parser.get_format_instructions()
                    })
                    
                    if res.has_evidence and res.evidence_summary:
                        candidate['evidence'] = res.evidence_summary
                        print(f"   ✅ Evidence Found: {res.evidence_summary}")
                    else:
                        candidate['evidence'] = "(无直接证据)"
                        print(f"   -> No valid evidence found in chat for {candidate['nickname']}.")
                else:
                    candidate['evidence'] = "(暂无相关聊天记录)"
                    print(f"   -> No chat records found for {candidate['nickname']}.")
                    
            except Exception as e:
                print(f"   ❌ Evidence failed for {candidate['nickname']}: {e}")
                candidate['evidence'] = ""

        state['final_candidates'] = candidates
        return state

    def generate_response(self, state: MatchmakingState):
        """Step 5: 生成回复"""
        candidates = state.get('final_candidates', [])
        current_gender = state.get('current_user_gender')
        
        # [Safety Check] 性别双重校验 (防止脏数据导致同性推荐)
        valid_candidates = []
        for c in candidates:
            # 简单逻辑: 必须是异性
            if current_gender == 'male' and c.get('gender') == 'male':
                print(f"   ⚠️ 剔除性别不符候选人: {c.get('nickname')} ({c.get('gender')})")
                continue
            if current_gender == 'female' and c.get('gender') == 'female':
                print(f"   ⚠️ 剔除性别不符候选人: {c.get('nickname')} ({c.get('gender')})")
                continue
            valid_candidates.append(c)
            
        candidates = valid_candidates
        
        if not candidates:
            # [NEW] 智能失败回复
            print("🤖 [Response] 搜索失败，生成建议...")
            try:
                res = self.failure_chain.invoke({
                    "user_input": state['current_input'],
                    "hard_filters": state.get('hard_filters', {})
                })
                state['reply'] = res.content
            except Exception as e:
                state['reply'] = "哎呀，即使放宽了要求，我还是没能为您找到合适的嘉宾。咱们要不试试别的条件？"
        else:
            candidates_info = ""
            for i, c in enumerate(candidates):
                evidence_str = f"(证据: {c['evidence']})" if c['evidence'] and "无" not in c['evidence'] else ""
                candidates_info += f"{i+1}. {c['summary']} {evidence_str}\n"
            
            print("🤖 [Response] 正在生成推荐语...")
            try:
                res = self.response_chain.invoke({
                    "user_input": state['current_input'],
                    "candidates_info": candidates_info
                })
                state['reply'] = res.content
            except Exception as e:
                 print(f"   ❌ 生成失败: {e}")
                 state['reply'] = "为您找到以下嘉宾:\n" + candidates_info

        print(f"🤖 [Response Done]: {state['reply'][:50]}...")
        
        # [NEW] 更新已见过的候选人列表 (用于"换一批"功能)
        seen = state.get('seen_candidate_ids', [])
        if seen is None: seen = [] # 防御性编程
        
        for c in candidates:
            if c['id'] not in seen:
                seen.append(c['id'])
        state['seen_candidate_ids'] = seen
        
        return state