# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Optional, List, Dict
from bson import ObjectId
from pymongo import DESCENDING

from app.core.container import container

class SessionService:
    def __init__(self):
        self.db = container.db

    def create_session(self, user_id: str, title: str = "新对话") -> str:
        """创建一个新的会话"""
        session_doc = {
            "user_id": user_id,
            "title": title,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "is_active": True,
            "latest_state": {}, # 初始为空状态
            "messages": []      # 初始化消息列表
        }
        res = self.db.chat_sessions.insert_one(session_doc)
        return str(res.inserted_id)

    def get_session(self, session_id: str, user_id: str) -> Optional[Dict]:
        """获取会话详情"""
        if not ObjectId.is_valid(session_id):
            return None
        return self.db.chat_sessions.find_one({
            "_id": ObjectId(session_id), 
            "user_id": user_id,
            "is_active": True
        })

    def get_user_last_session(self, user_id: str) -> Optional[Dict]:
        """获取用户最近的一个活跃会话"""
        return self.db.chat_sessions.find_one(
            {"user_id": user_id, "is_active": True},
            sort=[("updated_at", DESCENDING)]
        )

    def update_session_state(self, session_id: str, new_state: Dict):
        """更新会话的 LangGraph State 快照"""
        # 我们只保存关键字段，避免存入过大的临时数据
        # 这里定义需要持久化的字段白名单
        keys_to_persist = [
            "seen_candidate_ids", 
            "last_search_criteria", 
            "last_target_person", 
            "match_policy",
            "semantic_query",
            "intent",
            "search_count"
        ]
        
        state_subset = {k: new_state.get(k) for k in keys_to_persist if k in new_state}
        
        # 对于 final_candidates，如果太长，我们可以只存 ID 列表，或者简化存
        # 这里为了演示，暂时全存（假设数量不多）
        if "final_candidates" in new_state:
             # 转换为纯 dict 存储 (如果是 Pydantic 对象)
             cands = new_state["final_candidates"]
             if cands and isinstance(cands, list):
                 state_subset["final_candidates"] = cands

        self.db.chat_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$set": {
                    "latest_state": state_subset,
                    "updated_at": datetime.now()
                }
            }
        )

    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """添加一条聊天记录 (直接存入 Session 文档)"""
        
        msg_obj = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        self.db.chat_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$push": {"messages": msg_obj},
                "$set": {"updated_at": datetime.now()}
            }
        )

    def get_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """获取聊天历史 (通过 slice 获取最后 N 条，天然按时间正序)"""
        # MongoDB $slice 接受负数，表示取最后 N 个
        doc = self.db.chat_sessions.find_one(
            {"_id": ObjectId(session_id)},
            {"messages": {"$slice": -limit}} 
        )
        if not doc or "messages" not in doc:
            return []
        
        return doc["messages"]
