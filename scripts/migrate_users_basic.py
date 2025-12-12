# -*- coding: utf-8 -*-
import sys
import os
import random
from datetime import datetime, date
from bson import ObjectId

# 添加项目根目录到 Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.core.config import settings
from app.db.mongo_manager import MongoDBManager

def calculate_age(birthday_dt):
    """根据 datetime 对象计算年龄"""
    if not isinstance(birthday_dt, datetime):
        return None
    today = datetime.now()
    return today.year - birthday_dt.year - ((today.month, today.day) < (birthday_dt.month, birthday_dt.day))

def generate_random_weight(gender, height):
    """根据性别和身高生成一个相对合理的体重"""
    if gender == 'male':
        # 男生，身高-105左右，浮动10kg
        base_weight = height - 105 if height else 70
        return random.randint(max(40, base_weight - 10), base_weight + 10) # 确保不低于40kg
    elif gender == 'female':
        # 女生，身高-110左右，浮动10kg
        base_weight = height - 110 if height else 55
        return random.randint(max(30, base_weight - 10), base_weight + 10) # 确保不低于30kg
    return random.randint(50, 80) # 默认值

def main():
    print("🚀 开始数据迁移：users_basic -> users_basic_v2 ...")
    if not settings:
        print("❌ 配置加载失败")
        return

    db_manager = MongoDBManager(settings.database.mongo_uri, settings.database.db_name)
    
    old_collection = db_manager.db["users_basic"]
    new_collection = db_manager.db["users_basic_v2"]

    new_collection.drop()
    print("✅ 已清空旧的 users_basic_v2 集合（如果存在）")

    cursor = old_collection.find({})
    total_migrated = 0
    
    for old_doc in cursor:
        try:
            new_doc = old_doc.copy()
            
            # 1. 处理 birthday: str -> datetime 对象
            birthday_str = new_doc.get("birthday")
            if birthday_str and isinstance(birthday_str, str):
                try:
                    # 使用 datetime.strptime 转换，并存储为日期对象
                    new_doc["birthday"] = datetime.strptime(birthday_str, "%Y-%m-%d")
                except ValueError:
                    print(f"   ⚠️ 用户 {new_doc.get('_id')} birthday 格式错误: {birthday_str}，跳过生日转换。")
                    new_doc["birthday"] = None # 格式错误则清除，避免后续报错
            elif not isinstance(new_doc.get("birthday"), datetime):
                new_doc["birthday"] = None # 如果不是字符串也不是datetime，也清除

            # 不再新增 age 字段，因为可以实时计算
            
            # 2. 随机生成 weight 字段 (如果缺失或类型不对)
            if "weight" not in new_doc or not isinstance(new_doc["weight"], (int, float)):
                gender = new_doc.get("gender")
                height = new_doc.get("height")
                new_doc["weight"] = generate_random_weight(gender, height)

            # 插入新集合
            new_collection.insert_one(new_doc)
            total_migrated += 1
            
        except Exception as e:
            print(f"❌ 迁移用户 {old_doc.get('_id', '未知')} 失败: {e}")
            
    print("\n" + "="*60)
    print(f"🎉 数据迁移完成！成功迁移 {total_migrated} 条记录到 users_basic_v2。")

if __name__ == "__main__":
    main()