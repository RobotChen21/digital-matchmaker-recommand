# -*- coding: utf-8 -*-
import sys
import os
from datetime import datetime
from bson import ObjectId

# 添加项目根目录到 Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.core.config import settings
from app.db.mongo_manager import MongoDBManager

def main():
    print("🚀 开始初始化 user_states 表 (基于 users_basic)...")
    if not settings:
        print("❌ 配置加载失败")
        return

    db_manager = MongoDBManager(settings.database.mongo_uri, settings.database.db_name)
    
    users_basic = db_manager.users_basic
    user_states = db_manager.db["users_states"] # 新表

    # 可选：先清空新表，防止重复 (开发阶段)
    user_states.drop()
    print("✅ 已清空旧的 user_states 集合")

    cursor = users_basic.find({})
    total_migrated = 0
    
    for user in cursor:
        user_id = user["_id"]
        is_completed = user.get("is_completed", True)
        
        state_doc = {
            "user_id": user_id,
            "is_onboarding_completed": is_completed,
            "updated_at": datetime.now()
        }
        
        try:
            user_states.insert_one(state_doc)
            total_migrated += 1
        except Exception as e:
            print(f"❌ 用户 {user_id} 状态迁移失败: {e}")
            
    print("\n" + "="*60)
    print(f"🎉 状态表初始化完成！共迁移 {total_migrated} 条用户状态。")
    print(f"   集合名称: user_states")
    print(f"   字段示例: {{'user_id': ObjectId(...), 'is_onboarding_completed': True/False, 'updated_at': ...}})")

if __name__ == "__main__":
    main()
