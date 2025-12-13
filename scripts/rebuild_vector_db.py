# -*- coding: utf-8 -*-
import sys
import os
import time
from datetime import datetime

# 将项目根目录添加到 python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.core.config import settings
from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import ChromaManager

def main():
    print("🚀 开始重建向量数据库索引 (MongoDB -> ChromaDB)...")
    if not settings:
        print("❌ 配置加载失败")
        return

    # 1. 初始化资源
    print(f"🔧 连接 MongoDB: {settings.database.db_name}")
    db_manager = MongoDBManager(settings.database.mongo_uri, settings.database.db_name)
    
    print(f"🔧 连接 ChromaDB: {settings.database.chroma_persist_dir}")
    chroma_manager = ChromaManager(
        settings.database.chroma_persist_dir,
        settings.database.chroma_collection_name
    )

    # 2. 处理 Onboarding 对话
    print("\n📦 [1/2] 正在处理 Onboarding 对话...")
    cursor_onboarding = db_manager.onboarding_dialogues.find({"messages": {"$not": {"$size": 0}}})
    total_onboarding = db_manager.onboarding_dialogues.count_documents({"messages": {"$not": {"$size": 0}}})
    
    count_onboarding = 0
    for record in cursor_onboarding:
        user_id = record['user_id']
        messages = record['messages']
        
        try:
            chroma_manager.add_conversation_chunks(
                str(user_id),
                messages,
                "onboarding",
                window_size=settings.rag.window_size,
                overlap=settings.rag.overlap
            )
            count_onboarding += 1
            if count_onboarding % 10 == 0:
                print(f"   已处理 {count_onboarding}/{total_onboarding} 个用户...")
        except Exception as e:
            print(f"❌ 用户 {user_id} Onboarding 向量化失败: {e}")

    # 3. 处理社交聊天记录
    print("\n📦 [2/2] 正在处理社交聊天记录...")
    cursor_chat = db_manager.chat_records.find({"messages": {"$not": {"$size": 0}}})
    total_chat = db_manager.chat_records.count_documents({"messages": {"$not": {"$size": 0}}})
    
    count_chat = 0
    for record in cursor_chat:
        user_id = record['user_id']
        partner_id = record['partner_id']
        messages = record['messages']
        
        try:
            # 社交对话属于双方，所以要为双方都建立索引
            # 注意：add_conversation_chunks 内部通过 metadata={"user_id": ...} 来区分
            
            # 为 User A 索引
            chroma_manager.add_conversation_chunks(
                str(user_id),
                messages,
                "social",
                window_size=settings.rag.window_size,
                overlap=settings.rag.overlap
            )
            
            # 为 User B 索引
            chroma_manager.add_conversation_chunks(
                str(partner_id),
                messages,
                "social",
                window_size=settings.rag.window_size,
                overlap=settings.rag.overlap
            )
            
            count_chat += 1
            if count_chat % 10 == 0:
                print(f"   已处理 {count_chat}/{total_chat} 场对话...")
                
        except Exception as e:
            print(f"❌ 聊天记录 {record['_id']} 向量化失败: {e}")

    print("\n" + "="*60)
    print("🎉 向量数据库重建完成!")
    print(f"   - Onboarding 对话: {count_onboarding}/{total_onboarding}")
    print(f"   - 社交聊天记录: {count_chat}/{total_chat}")

if __name__ == "__main__":
    main()
