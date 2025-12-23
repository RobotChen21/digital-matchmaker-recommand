# -*- coding: utf-8 -*-
from app.common.models.state import MatchmakingState
from app.core.container import container

class RecallNode:
    def __init__(self):
        self.chroma = container.chroma
        self.es_manager = container.es # <--- 从容器获取

    def semantic_recall(self, state: MatchmakingState):
        """Step 3: 语义召回 (混合检索)"""
        candidates = state['hard_candidate_ids']
        query = state['semantic_query']
        
        print(f"🧠 [Recall] ES 混合检索: '{query}' within {len(candidates)} users")
        
        if not candidates or not query:
            print("   ⚠️ 跳过召回 (无候选人或无关键词)")
            state['semantic_candidate_ids'] = candidates[:10] 
            return state
            
        try:
            # 1. 准备向量 (复用 Chroma 的 embedding 逻辑)
            query_vector = self.chroma.embeddings_model.embed_query(query)
            
            # 2. 执行 Hybrid Search
            # 过滤条件: 只在 L1 过滤后的候选人中搜 (ID 过滤)
            # 必须传 filters，否则可能召回全是 L1 范围外的人，导致最终结果为空
            filters = {"user_id": candidates}
            
            results = self.es_manager.hybrid_search(
                query_text=query,
                query_vector=query_vector,
                top_k=50, # 稍微放大召回数量，因为后面还要 RRF
                filters=filters
            )
            
            # 3. 结果整理
            semantic_ids = [res['user_id'] for res in results]
            
            # [Debug] 打印 ES 原始得分情况
            print("   🔍 [Recall Debug] Top 5 ES Scores:")
            for i, r in enumerate(results[:5]):
                print(f"      {i+1}. ID: {r.get('user_id')} | Score: {r.get('score'):.4f} | Tags: {r.get('tags')}")

            # 如果 ES 没搜到足够的人 (比如 L1 过滤太严了)，用原有的候选人垫底
            if len(semantic_ids) < 5:
                print("   ⚠️ ES 召回结果较少，尝试合并 L1 候选人...")
                for cid in candidates:
                    if cid not in semantic_ids:
                        semantic_ids.append(cid)
                    if len(semantic_ids) >= 15: break

            state['semantic_candidate_ids'] = semantic_ids[:20]
            print(f"   -> 召回: {len(semantic_ids)} 人 (来自 ES Hybrid Search)")
            
        except Exception as e:
            print(f"   ❌ ES 检索失败: {e}，尝试退回到 Chroma...")
            # 兜底逻辑: 使用 Chroma 纯语义搜索
            try:
                search_filter = {"user_id": {"$in": candidates}}
                results = self.chroma.vector_db.similarity_search(query, k=15, filter=search_filter)
                state['semantic_candidate_ids'] = [doc.metadata.get('user_id') for doc in results]
            except:
                state['semantic_candidate_ids'] = candidates[:10]

        return state
