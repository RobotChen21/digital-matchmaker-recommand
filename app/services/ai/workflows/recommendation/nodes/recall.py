# -*- coding: utf-8 -*-
from app.common.models.state import MatchmakingState
from app.core.container import container

class RecallNode:
    def __init__(self):
        self.chroma = container.chroma

    def semantic_recall(self, state: MatchmakingState):
        """Step 3: 语义召回 (只搜画像摘要)"""
        candidates = state['hard_candidate_ids']
        query = state['semantic_query']
        
        print(f"🧠 [Semantic] 向量检索 (画像摘要): '{query}' within {len(candidates)} users")
        
        if not candidates or not query:
            print("   ⚠️ 跳过语义召回 (无候选人或无关键词)")
            state['semantic_candidate_ids'] = candidates[:10] 
            return state
            
        try:
            # 过滤条件: 必须是 profile_summary 类型
            search_filter = {
                "$and": [
                    {"data_type": "profile_summary"},
                    {"user_id": {"$in": candidates}}
                ]
            }
            
            results = self.chroma.vector_db.similarity_search_with_score(
                query,
                k=20,
                filter=search_filter
            )
            
            semantic_ids = []
            seen = set()
            for doc, score in results:
                uid = doc.metadata.get('user_id')
                if uid and uid not in seen:
                    semantic_ids.append(uid)
                    seen.add(uid)
            
            state['semantic_candidate_ids'] = semantic_ids
            print(f"   -> 召回: {len(semantic_ids)} 人")
            
        except Exception as e:
            print(f"   ❌ 向量检索失败: {e}")
            state['semantic_candidate_ids'] = candidates[:10]

        return state
