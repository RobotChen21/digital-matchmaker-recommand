# -*- coding: utf-8 -*-
import sys
import os
import random
from pymongo.errors import OperationFailure # 导入 OperationFailure

# 添加项目根目录到 Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.core.config import settings
from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import EnhancedChromaManager
from app.services.ai.workflows.recommendation import RecommendationWorkflow

def main():
    print("🚀 启动交互式红娘推荐系统 (CLI Mode)...")
    print("输入 'q' 或 'quit' 退出")
    
    # 1. Init Dependencies
    db_manager = MongoDBManager(settings.database.mongo_uri, settings.database.db_name)
    chroma_manager = EnhancedChromaManager(
        settings.database.chroma_persist_dir,
        settings.database.chroma_collection_name
    )
    
    # 2. Init Workflow
    workflow = RecommendationWorkflow(db_manager, chroma_manager)
    app = workflow.build_graph()
    
    # 3. Pick a random user as 'me' (优先尝试 $sample，失败回退)
    me = None
    try:
        # 尝试使用 $sample (MongoDB 3.2+ 支持)
        me = db_manager.users_basic.aggregate([{"$sample": {"size": 1}}]).next()
    except OperationFailure as e:
        print(f"⚠️ MongoDB $sample 操作失败: {e}. 回退到兼容模式随机抽取。")
        # 回退到兼容模式 (skip/limit)
        user_count = db_manager.users_basic.count_documents({})
        if user_count > 0:
            random_index = random.randint(0, user_count - 1)
            me = db_manager.users_basic.find().skip(random_index).limit(1).next()
    except StopIteration: # aggregate().next() 如果没找到文档会抛出 StopIteration
        me = None

    if not me:
        print("❌ 数据库没用户，请先运行生成脚本！")
        return
        
    my_id = str(me['_id'])
    print(f"\n👤 您当前的身份: {me.get('nickname')} ({me.get('gender')}, {me.get('city')}, {me.get('birthday')})")
    print("-" * 50)
    
    # Context Memory
    last_candidates = []
    last_target = None
    
    while True:
        try:
            user_input = input("\n💬 您: ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("👋 再见！")
                break
                
            if not user_input.strip(): continue
            
            # Construct State
            state = {
                "user_id": my_id,
                "current_input": user_input,
                "messages": [], 
                "search_count": 0,
                "final_candidates": last_candidates,
                "last_target_person": last_target 
            }
            
            # Invoke
            print("⏳ 红娘正在思考...")
            final_state = app.invoke(state)
            
            # Output
            reply = final_state.get('reply')
            intent = final_state.get('intent')
            
            print(f"🤖 红娘 ({intent}): {reply}")
            
            # Update Context
            if intent == "search_candidate" and final_state.get('final_candidates'):
                last_candidates = final_state.get('final_candidates')
                print(f"   (已记忆 {len(last_candidates)} 位候选人)")
            
            if intent == "deep_dive":
                current_target = final_state.get('target_person_name')
                if current_target:
                    last_target = current_target
                    print(f"   (已锁定目标: {last_target})")
                
        except Exception as e:
            print(f"❌ 出错了: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()