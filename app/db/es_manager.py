# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch, helpers
from app.core.config import settings

logger = logging.getLogger(__name__)

class ESManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ESManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.es_url = settings.database.es_url
        self.index_name = settings.database.es_index_name
        self.es_vector_dims = settings.llm.vector_dims
        
        # 连接 ES (假设开发环境已关闭 Security，生产环境需配置 basic_auth)
        # 强制指定 scheme 为 http，且不传递任何 SSL 参数，防止客户端自动升级
        target_url = self.es_url.replace("https://", "http://")
        
        self.client = Elasticsearch(
            hosts=[target_url],
            request_timeout=30
        )
        
        if self.client.ping():
            logger.info(f"✅ Successfully connected to Elasticsearch at {self.es_url}")
        else:
            logger.warning(f"❌ Failed to connect to Elasticsearch at {self.es_url}")
            
        self._initialized = True

    def create_index_if_not_exists(self):
        """
        创建索引 Mapping。
        """
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index '{self.index_name}' already exists.")
            return

        mapping = {
            "mappings": {
                "properties": {
                    "user_id": { "type": "keyword" },
                    
                    # --- 硬指标 (Keyword) ---
                    "gender": { "type": "keyword" },
                    "city": { "type": "keyword" },
                    "age": { "type": "integer" },
                    
                    # --- 混合检索字段 ---
                    # 1. Tags: 半结构化标签 (如 "本科", "独生子") -> 关键词匹配
                    "tags": { 
                        "type": "text", 
                        "analyzer": "ik_smart", 
                        "fields": { "keyword": { "type": "keyword" } }
                    },
                    
                    # 2. Text: 完整画像文本 -> 全文检索 (BM25)
                    "profile_text": { 
                        "type": "text", 
                        "analyzer": "ik_smart"
                    },
                    
                    # 3. Vector: 语义向量 -> KNN 搜索
                    "profile_vector": {
                        "type": "dense_vector",
                        "dims": self.es_vector_dims,
                        "index": True,
                        "similarity": "cosine" # 余弦相似度
                    }
                }
            }
        }
        
        try:
            self.client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"✅ Created index '{self.index_name}' with hybrid mapping.")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")

    def index_user(self, user_id: str, profile_data: Dict[str, Any], vector: List[float]):
        """
        索引单个用户
        """
        doc = {
            "user_id": user_id,
            "gender": profile_data.get("gender"),
            "city": profile_data.get("city"),
            "age": profile_data.get("age"),
            "tags": profile_data.get("tags", ""), # 字符串，如 "本科 程序员 独生子"
            "profile_text": profile_data.get("profile_text", ""),
            "profile_vector": vector
        }
        try:
            self.client.index(index=self.index_name, id=user_id, document=doc)
            # logger.debug(f"Indexed user {user_id}")
        except Exception as e:
            logger.error(f"Error indexing user {user_id}: {e}")

    def bulk_index_users(self, actions: List[Dict[str, Any]]):
        """
        批量索引 (用于初始化数据)
        actions format: [{"_id": "...", "_source": {...}}, ...]
        """
        try:
            success, failed = helpers.bulk(self.client, actions, index=self.index_name)
            logger.info(f"Bulk indexed {success} documents. Failed: {failed}")
        except Exception as e:
            logger.error(f"Bulk index failed: {e}")

    def hybrid_search(self, 
                      query_text: str, 
                      query_vector: List[float], 
                      top_k: int = 20, 
                      filters: Optional[Dict] = None) -> List[Dict]:
        """
        核心方法：混合检索 (Manual RRF Implementation)
        [Fix] 在应用层手动实现 RRF，以绕过 ES Basic License 不支持 rank 参数的限制。
        """
        # --- 构造过滤条件 (共享) ---
        must_clauses = []
        if filters:
             for k, v in filters.items():
                 if isinstance(v, list):
                     must_clauses.append({"terms": {k: v}})
                 else:
                     must_clauses.append({"term": {k: v}})
        
        filter_query = {"bool": {"must": must_clauses}} if must_clauses else None

        # --- 1. 执行 KNN 搜索 (Vector) ---
        knn_hits = []
        try:
            knn_res = self.client.search(
                index=self.index_name,
                knn={
                    "field": "profile_vector",
                    "query_vector": query_vector,
                    "k": top_k * 2, # 多取一些用于融合
                    "num_candidates": 100,
                    "filter": filter_query # 向量搜索也能带 filter
                },
                size=top_k * 2,
                _source=["user_id", "tags", "gender", "age", "city"]
            )
            knn_hits = knn_res.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.error(f"KNN search failed: {e}")

        # --- 2. 执行 Text 搜索 (BM25) ---
        text_hits = []
        try:
            keyword_query = {
                "bool": {
                    "must": must_clauses,
                    "should": [
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["tags^3", "profile_text"], 
                                "type": "best_fields"
                            }
                        }
                    ]
                }
            }
            text_res = self.client.search(
                index=self.index_name,
                query=keyword_query,
                size=top_k * 2,
                _source=["user_id", "tags", "gender", "age", "city"]
            )
            text_hits = text_res.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.error(f"Text search failed: {e}")

        # --- 3. 应用 RRF 融合 (Reciprocal Rank Fusion) ---
        # Formula: score = 1 / (k + rank)
        rrf_k = 60
        scores = {}
        doc_map = {}

        # 处理 KNN 结果
        for rank, hit in enumerate(knn_hits):
            uid = hit["_source"]["user_id"]
            doc_map[uid] = hit["_source"]
            scores[uid] = scores.get(uid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # 处理 Text 结果
        for rank, hit in enumerate(text_hits):
            uid = hit["_source"]["user_id"]
            if uid not in doc_map:
                doc_map[uid] = hit["_source"]
            scores[uid] = scores.get(uid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # --- 4. 排序并返回 ---
        sorted_uids = sorted(scores.keys(), key=lambda u: scores[u], reverse=True)
        final_results = []
        
        for uid in sorted_uids[:top_k]:
            final_results.append({
                "user_id": uid,
                "score": scores[uid],
                "tags": doc_map[uid].get("tags"),
                "city": doc_map[uid].get("city")
            })
            
        return final_results
