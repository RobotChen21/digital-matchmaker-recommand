# -*- coding: utf-8 -*-
import sys
import os
import json

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
    text = []
    for msg in messages:
        role = "AI红娘" if msg['role'] == 'ai' else "用户"
        content = msg['content']
        text.append(f"{role}: {content}")
    return "\n".join(text)

def main():
    print("🚀 初始化全维度画像提取测试...")
    if not settings:
        print("❌ 配置加载失败")
        return

    db_manager = MongoDBManager(settings.database.mongo_uri, settings.database.db_name)
    
    # 使用较低的 temperature 保证提取的客观性和格式稳定性
    llm = ChatOpenAI(
        model=settings.llm.model_name,
        temperature=0.1, 
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    # 实例化所有 Agent
    agents = {
        "🧠 性格分析": PersonalityExtractor(llm),
        "🎯 兴趣爱好": InterestExtractor(llm),
        "💎 价值观": ValuesExtractor(llm),
        "🏃 生活方式": LifestyleExtractor(llm),
        "❤️ 恋爱风格": LoveStyleExtractor(llm),
        "⚠️ 风险评估": RiskExtractor(llm),
        "🎓 教育背景": EducationExtractor(llm),
        "💼 职业背景": OccupationExtractor(llm),
        "🏠 家庭背景": FamilyExtractor(llm),
        "💑 择偶偏好": DatingPrefExtractor(llm),
    }

    # 获取用户 (随机抽取一个)
    print("\n🔍 正在随机查找适合测试的用户...")
    pipeline = [
        {"$match": {"messages": {"$not": {"$size": 0}}}}, # 筛选有消息的
        {"$sample": {"size": 1}} # 随机抽一个
    ]
    cursor = db_manager.onboarding_dialogues.aggregate(pipeline)
    onboarding_record = next(cursor, None)

    if not onboarding_record:
        print("❌ 数据库中没有找到 Onboarding 对话记录。")
        return

    user_id = onboarding_record['user_id']
    user_basic, persona_seed_data = db_manager.get_user_with_persona(user_id)
    
    print(f"✅ 选中用户: {user_basic['nickname']} (ID: {user_id})")
    print("-" * 60)
    
    dialogue_text = format_dialogue(onboarding_record['messages'])
    print(f"📜 对话长度: {len(onboarding_record['messages'])} 条消息")
    
    # 循环执行所有 Agent
    full_profile = {}
    
    print("\n🚀 开始多 Agent 协同工作...")
    for name, agent in agents.items():
        print(f"\n⚡ [{name}] 正在分析...")
        try:
            result = agent.extract(dialogue_text)
            if result:
                # 打印非空字段
                data = result.model_dump(exclude_none=True)
                print(json.dumps(data, indent=2, ensure_ascii=False))
                full_profile[name] = data
            else:
                print("   (未提取到有效信息)")
        except Exception as e:
            print(f"   ❌ 出错: {e}")

    print("\n" + "="*60)
    print("🏁 分析完成！")
    
    # 简单对比原始 Persona (如果有)
    if persona_seed_data:
        print("\n🌱 [原始 Persona 种子参考]")
        print(json.dumps(persona_seed_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()