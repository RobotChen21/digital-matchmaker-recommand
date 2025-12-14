# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient

class MongoDBManager:
    """MongoDB 数据库管理器"""

    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        
        # Collections
        self.users_basic = self.db["users_basic"]
        self.users_persona = self.db["users_persona"]
        self.onboarding_dialogues = self.db["users_onboarding_dialogues"]
        self.chat_records = self.db["chat_records"]
        self.profile = self.db["users_profile"]
        self.users_auth = self.db["users_auth"]
        self.users_states = self.db["users_states"] # 状态表 (注意：迁移脚本里用的是 user_states，这里保持一致)
        
        # Indexes
        # 确保 account 唯一
        self.users_auth.create_index("account", unique=True)

    def insert_user_with_persona(self, user_data: Dict[str, Any],
                                 persona_data: Dict[str, Any]) -> ObjectId:
        """插入用户基础信息和性格种子"""
        # 插入基础信息
        user_basic = {k: v for k, v in user_data.items() if k != "persona_seed"}
        user_basic["created_at"] = datetime.now()
        result = self.users_basic.insert_one(user_basic)
        user_id = result.inserted_id

        # 插入性格种子(单独存储,用于生成对话)
        persona_doc = {
            "user_id": user_id,
            "persona": persona_data,
            "created_at": datetime.now()
        }
        self.users_persona.insert_one(persona_doc)

        return user_id

    def create_auth_user(self, account: str, password_hash: str, user_id: ObjectId):
        """创建认证用户"""
        auth_data = {
            "account": account,
            "password_hash": password_hash,
            "user_id": user_id,
            "created_at": datetime.now()
        }
        self.users_auth.insert_one(auth_data)

    def get_auth_user_by_account(self, account: str) -> Optional[Dict]:
        """根据账号查找认证信息"""
        return self.users_auth.find_one({"account": account})

    def get_user_with_persona(self, user_id: ObjectId) -> Tuple[Dict, Dict]:
        """获取用户信息和性格种子"""
        user_basic = self.users_basic.find_one({"_id": user_id})
        persona_doc = self.users_persona.find_one({"user_id": user_id})
        persona = persona_doc["persona"] if persona_doc else {}
        return user_basic, persona

    def insert_onboarding_dialogue(self, user_id: ObjectId, messages: List[Dict]):
        """插入 onboarding 对话"""
        dialogue_data = {
            "user_id": user_id,
            "messages": messages,
            "updated_at": datetime.now()
        }
        self.onboarding_dialogues.insert_one(dialogue_data)

    def insert_chat_record(self, user_id: ObjectId, partner_id: ObjectId,
                           messages: List[Dict]):
        """插入聊天记录"""
        chat_data = {
            "user_id": user_id,
            "partner_id": partner_id,
            "messages": messages,
            "created_at": datetime.now()
        }
        self.chat_records.insert_one(chat_data)
