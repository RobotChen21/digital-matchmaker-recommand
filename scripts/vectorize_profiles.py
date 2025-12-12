# -*- coding: utf-8 -*-
import sys
import os
from datetime import datetime

# 添加项目根目录到 Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.core.config import settings
from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import EnhancedChromaManager
from app.services.ai.agents.profile_manager import ProfileService
from langchain_core.documents import Document

def calculate_birth_year(birthday_val):
    """
    计算出生年份，兼容 str (YYYY-MM-DD) 和 datetime 对象。
    """
    if not birthday_val: return None
    try:
        if isinstance(birthday_val, datetime):
            return birthday_val.year
        elif isinstance(birthday_val, str):
            return int(birthday_val.split('-')[0])
        else:
            return None
    except:
        return None

def main():
    print("🚀 开始将用户画像向量化 (Profile -> Vector DB)...")
    if not settings:
        print("❌ 配置加载失败")
        return

    # 1. Init
    db_manager = MongoDBManager(settings.database.mongo_uri, settings.database.db_name)
    chroma_manager = EnhancedChromaManager(
        settings.database.chroma_persist_dir,
        settings.database.chroma_collection_name
    )

    # 2. Fetch Users
    cursor = db_manager.users_basic.find({})
    total = db_manager.users_basic.count_documents({})
    
    print(f"📊 发现 {total} 个用户，开始处理...")
    
    documents = []
    count = 0
    
    for basic in cursor:
        user_id = basic['_id']
        
        # 查 Profile
        profile_doc = db_manager.db["users_profile"].find_one({"user_id": user_id})
        if not profile_doc:
            print(f"   ⚠️ 用户 {basic['nickname']} 暂无画像数据，跳过详细摘要")
            profile = {}
        else:
            profile = profile_doc
            
        # 生成摘要 (调用公共方法)
        summary_text = ProfileService.generate_profile_summary(basic, profile)
        
        # 提取更多元数据
        metadata_to_add = {
            "user_id": str(user_id),
            "gender": basic.get('gender', 'unknown'), 
            "data_type": "profile_summary", 
            "city": basic.get('city', 'unknown'), 
            "timestamp": str(datetime.now())
        }
        
        height = basic.get('height')
        if height is not None:
            metadata_to_add['height'] = height
            
        birth_year = calculate_birth_year(basic.get('birthday'))
        if birth_year is not None:
            metadata_to_add['birth_year'] = birth_year

        # 构造 Document
        doc = Document(
            page_content=summary_text,
            metadata=metadata_to_add
        )
        documents.append(doc)
        count += 1
        
        if len(documents) >= 10: 
            chroma_manager.vector_db.add_documents(documents)
            print(f"   ✅ 已存入 {count}/{total} 个画像向量")
            documents = []

    if documents:
        chroma_manager.vector_db.add_documents(documents)
        print(f"   ✅ 已存入 {count}/{total} 个画像向量")

    print("\n🎉 画像向量化完成！现在您可以基于画像进行语义检索了。\n")

if __name__ == "__main__":
    main()
