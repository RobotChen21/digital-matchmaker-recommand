# -*- coding: utf-8 -*-
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, Any
from bson import ObjectId

# 添加项目根目录到 Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from langchain_openai import ChatOpenAI
from app.core.config import settings
from utils.env_utils import API_KEY, BASE_URL
from app.db.mongo_manager import MongoDBManager
from app.services.ai.agents.extractors import (
    PersonalityExtractor, InterestExtractor, ValuesExtractor,
    LifestyleExtractor, LoveStyleExtractor, RiskExtractor,
    EducationExtractor, OccupationExtractor, FamilyExtractor,
    DatingPrefExtractor
)

def format_dialogue(messages):
    """格式化对话记录"""
    text = []
    for msg in messages:
        role = "AI红娘" if msg['role'] == 'ai' else "用户"
        content = msg['content']
        text.append(f"{role}: {content}")
    return "\n".join(text)

def remove_none_fields(data):
    """递归移除字典中值为 None 的字段"""
    if isinstance(data, dict):
        return {k: remove_none_fields(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [remove_none_fields(item) for item in data if item is not None]
    else:
        return data

def main():
    print("🚀 开始批量初始化用户画像...")
    if not settings:
        print("❌ 配置加载失败")
        return

    # 1. 初始化资源
    db_manager = MongoDBManager(settings.database.mongo_uri, settings.database.db_name)
    llm = ChatOpenAI(
        model=settings.llm.model_name,
        temperature=0.1, # 提取任务保持低温度
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    # 实例化所有 Agent
    agents = {
        "personality_profile": PersonalityExtractor(llm),
        "interest_profile": InterestExtractor(llm),
        "values_profile": ValuesExtractor(llm),
        "lifestyle_profile": LifestyleExtractor(llm),
        "love_style_profile": LoveStyleExtractor(llm),
        "risk_profile": RiskExtractor(llm),
        "education_profile": EducationExtractor(llm),
        "occupation_profile": OccupationExtractor(llm),
        "family_profile": FamilyExtractor(llm),
        "dating_preferences": DatingPrefExtractor(llm),
    }

    # 2. 获取所有有 Onboarding 记录的用户
    # 这里我们只处理有对话记录的用户
    cursor = db_manager.onboarding_dialogues.find({"messages": {"$not": {"$size": 0}}})
    total_users = db_manager.onboarding_dialogues.count_documents({"messages": {"$not": {"$size": 0}}})
    
    print(f"📊 发现 {total_users} 个待处理用户...")
    processed_count = 0
    skipped_count = 0

    for record in cursor:
        user_id = record['user_id']
        user_basic = db_manager.users_basic.find_one({"_id": user_id})
        nickname = user_basic.get('nickname', 'Unknown')
        
        # 3. 检查是否已存在画像 (简单的幂等性)
        # 如果 users_profile 表里已经有这个 user_id，且不想强制覆盖，就跳过
        existing_profile = db_manager.db["users_profile"].find_one({"user_id": user_id})
        if existing_profile:
            print(f"⏭️  用户 [{nickname}] 已有画像，跳过...")
            skipped_count += 1
            continue

        print(f"\n⚡ [{processed_count + 1}/{total_users}] 正在提取用户: {nickname} (ID: {user_id})")
        
        dialogue_text = format_dialogue(record['messages'])
        
        # 构建 UserProfile 对象的数据字典
        profile_data = {
            "user_id": user_id, # 保持 ObjectId 类型
            "updated_at": datetime.now()
        }

        # 4. 依次调用所有 Agent
        for field_name, agent in agents.items():
            # print(f"   ...运行 {field_name} agent")
            try:
                result = agent.extract(dialogue_text)
                if result:
                    # 将 Pydantic 模型转为 dict
                    profile_data[field_name] = result.model_dump()
                else:
                    profile_data[field_name] = None
            except Exception as e:
                print(f"   ❌ {field_name} 提取失败: {e}")
                profile_data[field_name] = None
        
        # 5. 清洗数据 (移除 None 值以符合 MongoDB Schema)
        cleaned_data = remove_none_fields(profile_data)

        # 6. 存入数据库
        try:
            db_manager.db["users_profile"].update_one(
                {"user_id": user_id},
                {"$set": cleaned_data},
                upsert=True
            )
            print(f"   ✅ 画像保存成功!")
            processed_count += 1
            
        except Exception as e:
             print(f"   ❌ 保存数据库失败: {e}")

    print("\n" + "="*60)
    print(f"🏁 批量初始化完成!")
    print(f"   - 总扫描: {total_users}")
    print(f"   - 新增/更新: {processed_count}")
    print(f"   - 跳过: {skipped_count}")

if __name__ == "__main__":
    main()
