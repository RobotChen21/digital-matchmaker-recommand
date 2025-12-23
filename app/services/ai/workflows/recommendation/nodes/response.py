# -*- coding: utf-8 -*-
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.core.container import container
from app.common.models.state import MatchmakingState
from app.services.ai.workflows.recommendation.state import EvidenceOutput

class ResponseNode:
    def __init__(self):
        self.chroma = container.chroma
        self.llm = container.get_llm("reason") # temperature=0.4
        
        self.evidence_parser = PydanticOutputParser(pydantic_object=EvidenceOutput)
        self.evidence_chain = (
            ChatPromptTemplate.from_template(
                """你是一个敏锐的红娘证据分析师。
                请阅读以下聊天记录片段，判断其中是否包含证明该嘉宾符合用户需求【{query}】的线索或证据。
                
                【重要标准】:
                1. **部分证明即有效**: 用户可能提了很多要求，只要这段对话能证明其中**任何一点**（比如证明了性格随和，或者证明了喜欢画画），就认为“有证据”。
                2. **关注核心**: 重点挖掘性格、兴趣、三观、生活细节等“软性”证据。
                3. **身份校验**: 只关注【{candidate_nickname}】（Role为该候选人或User）自己说的话。
                
                【聊天片段】:
                {raw_text}
                
                如果有证据，请用【第三人称】生动总结（例如："她在聊天中提到自己每周都会去画室写生，可见确实非常热爱美术"）。
                如果没有找到任何相关线索，summary为空。
                
                输出JSON: {format_instructions}"""
            ) | self.llm | self.evidence_parser
        )

        self.response_chain = (
            ChatPromptTemplate.from_template(
                """你是一位眼光毒辣、情商极高的金牌红娘。请根据用户需求，为他/她隆重介绍以下几位精选嘉宾。
                
                【用户心愿】: {user_input}
                
                【精选嘉宾列表】:
                {candidates_info}
                
                【推荐策略】:
                1. **保留标题格式**: 请**原封不动**地使用我提供的嘉宾标题（例如：晨曦（30岁，178cm...）），**严禁**修改标题里的内容或格式，也不要加“第一位”这种前缀。
                2. **拒绝报菜名**: 不要枯燥地罗列身高体重，要挖掘嘉宾的**闪光点**和**与用户的契合点**。
                3. **巧妙使用证据**: 如果嘉宾有具体的聊天记录证据 (Evidence)，请自然地融入推荐语中，佐证他/她的真实性格。
                   - ❌ 差: "证据显示他喜欢滑雪。"
                   - ✅ 优: "而且惊喜的是，他是个户外达人，之前还提到每年冬天都会去崇礼滑雪，这和您爱运动的性格简直绝配！"
                4. **差异化推荐**: 如果有多位嘉宾，请突出他们各自不同的气质（例如：一位是稳重的学霸，另一位是阳光的大男孩）。
                5. **行动号召**: 最后用一句温暖的话鼓励用户发起互动，例如“想先了解哪一位？我可以帮您详细介绍。”
                
                请直接输出推荐语，每位嘉宾的介绍之间请空一行，保持排版舒适。"""
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
        # 优化：剔除硬指标，只搜寻性格、兴趣、价值观相关的语义证据
        query = state.get('semantic_query') or state.get('current_input')
        
        if not candidates: return state
        
        print(f"🕵️ [Evidence] 为 {len(candidates)} 位候选人搜寻证据: '{query}'")
        
        for candidate in candidates:
            try:
                # 确保 ID 是字符串格式
                cid_str = str(candidate['id'])
                
                # 1. 检索: 只查对话记录
                search_filter = {
                    "$and": [
                        {"user_id": cid_str},
                        {"dialogue_type": {"$in": ["onboarding", "social"]}}
                    ]
                }
                docs = self.chroma.retrieve_related_context(query, user_id=cid_str, k=2, filter=search_filter)
                
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

    def _get_bmi_label(self, c: dict) -> str:
        """根据身高体重计算 BMI 并返回体态标签"""
        try:
            h = c.get('height')
            w = c.get('weight')
            if not h or not w: return "体态未知"
            bmi = w / ((h / 100) ** 2)
            if bmi < 18.5: return "纤细"
            if bmi < 24: return "匀称"
            if bmi < 28: return "丰满"
            return "魁梧"
        except:
            return "体态未知"

    def generate_response(self, state: MatchmakingState):
        """Step 5: 生成回复"""
        candidates = state.get('final_candidates', [])
        
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
                # 1. 构造详细头部: 名字（年龄，身高，体态，城市）
                bmi_label = self._get_bmi_label(c)
                h_str = f"{c.get('height')}cm" if c.get('height') else "身高未知"
                
                # [Fix] 年龄兜底
                age_val = c.get('age')
                age_str = f"{age_val}岁" if age_val else "年龄保密"
                
                header = f"{c.get('nickname')}（{age_str}，{h_str}，{bmi_label}，{c.get('city')}）"
                
                # 2. 获取 summary 内容 (去掉原有的 名字（年龄）前缀，防止重复)
                # 假设 c['summary'] 是 "名字（年龄） —— 标题..."
                # 如果 summary 已经包含名字，我们尝试清理一下，或者直接拼接
                summary_body = c.get('summary', '')
                if " —— " in summary_body:
                    summary_body = summary_body.split(" —— ", 1)[-1]
                elif " -- " in summary_body:
                    summary_body = summary_body.split(" -- ", 1)[-1]
                
                evidence_str = f"(证据: {c['evidence']})" if c['evidence'] and "无" not in c['evidence'] else ""
                candidates_info += f"{i+1}. {header} —— {summary_body} {evidence_str}\n"
            
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
        return state