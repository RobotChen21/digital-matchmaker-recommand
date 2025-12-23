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
        核心方法：混合检索 (BM25 + KNN + RRF)
        """
        # 1. 构建 KNN 查询 (语义向量)
        knn_query = {
            "field": "profile_vector",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": 100,
            "boost": 0.5 # 语义分权重
        }

        # 2. 构建 Keyword 查询 (BM25)
        # 重点关注 tags (硬性标签) 和 profile_text (整体描述)
        must_clauses = []
        if filters:
             # 这里可以把 Mongo 的 L1 结果作为 filter 传进来，或者直接在 ES 做简单过滤
             # 暂时只处理简单 Term 过滤
             for k, v in filters.items():
                 must_clauses.append({"term": {k: v}})

        keyword_query = {
            "bool": {
                "must": must_clauses,
                "should": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["tags^3", "profile_text"], # tags 权重 x3
                            "type": "best_fields",
                            "boost": 0.5
                        }
                    }
                ]
            }
        }

        # 3. 执行混合检索 (ES 8.x 标准写法)
        try:
            response = self.client.search(
                index=self.index_name,
                knn=knn_query,
                query=keyword_query,
                size=top_k,
                # RRF (Reciprocal Rank Fusion)
                rank={
                    "rrf": {
                        "window_size": 50,
                        "rank_constant": 20
                    }
                },
                _source=["user_id", "tags", "gender", "age", "city"] # 只返回关键字段
            )
            
            hits = response.get("hits", {}).get("hits", [])
            results = []
            for hit in hits:
                results.append({
                    "user_id": hit["_source"]["user_id"],
                    "score": hit["_score"], # RRF score
                    "tags": hit["_source"].get("tags"),
                    "city": hit["_source"].get("city")
                })
            return results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
