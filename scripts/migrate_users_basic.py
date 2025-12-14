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

def calculate_age(birthday_dt: date): # 明确类型为 date
    """根据 date 对象计算年龄"""
    if not isinstance(birthday_dt, date):
        return None
    today = date.today()
    return today.year - birthday_dt.year - ((today.month, today.day) < (birthday_dt.month, birthday_dt.day))

def generate_random_weight(gender, height):
    """根据性别和身高生成一个相对合理的体重"""
    if gender == 'male':
        base_weight = height - 105 if height else 70
        return random.randint(max(40, base_weight - 10), base_weight + 10)
    elif gender == 'female':
        base_weight = height - 110 if height else 55
        return random.randint(max(30, base_weight - 10), base_weight + 10)
    return random.randint(50, 80)

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
            
            # 1. 处理 birthday: str -> datetime 对象 (BSON Date)
            birthday_val = new_doc.get("birthday")
            if isinstance(birthday_val, str):
                try:
                    # 必须存为 datetime，因为 PyMongo 不支持直接存 date
                    new_doc["birthday"] = datetime.strptime(birthday_val, "%Y-%m-%d")
                except ValueError:
                    print(f"   ⚠️ 用户 {new_doc.get('_id')} birthday 格式错误: {birthday_val}，设置为默认日期。")
                    new_doc["birthday"] = datetime(2000, 1, 1)
            elif isinstance(birthday_val, date) and not isinstance(birthday_val, datetime):
                # 如果是 date 但不是 datetime，转为 datetime
                new_doc["birthday"] = datetime(birthday_val.year, birthday_val.month, birthday_val.day)
            elif not isinstance(birthday_val, datetime):
                new_doc["birthday"] = datetime(2000, 1, 1) # 默认值

            # 2. 随机生成 weight 字段 (如果缺失或类型不对)
            if "weight" not in new_doc or not isinstance(new_doc["weight"], (int, float)):
                gender = new_doc.get("gender")
                height = new_doc.get("height")
                new_doc["weight"] = generate_random_weight(gender, height)

            new_collection.insert_one(new_doc)
            total_migrated += 1
            
        except Exception as e:
            print(f"❌ 迁移用户 {old_doc.get('_id', '未知')} 失败: {e}")
            
    print("\n" + "="*60)
    print(f"🎉 数据迁移完成！成功迁移 {total_migrated} 条记录到 users_basic_v2。")

if __name__ == "__main__":
    main()
