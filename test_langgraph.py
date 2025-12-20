import os
import random
import sys

from bson import ObjectId # 导入 ObjectId
from pymongo.errors import OperationFailure

# 添加项目根目录到 Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from app.core.container import container

import asyncio

async def main():
    print("您好！欢迎进入红娘助手调试模式 (CLI Mode)...")
    print("输入 'q' 或 'quit' 退出。")

    # 1. Init Dependencies
    db_manager = container.db

    # 2. Init Workflow
    app = container.recommendation_app

    # 3. Pick user (随便找个ID试运行)
    target_user_id = "693ebdc20196b88668259955"
    me = db_manager.users_basic.find_one({"_id": ObjectId(target_user_id)})

    if not me:
        print(f"找不到指定用户 {target_user_id} ，尝试随机选取一名用户...")
        try:
            # 尝试随机抽取 $sample (MongoDB 3.2+ 语义)
            me = db_manager.users_basic.aggregate([{"$sample": {"size": 1}}]).next()
        except OperationFailure as e:
            print(f"MongoDB $sample 聚合失败: {e}. 回退到传统方式...")        
            # 回退到传统方式 (skip/limit)
            user_count = db_manager.users_basic.count_documents({})
            if user_count > 0:
                random_index = random.randint(0, user_count - 1)
                me = db_manager.users_basic.find().skip(random_index).limit(1).next()
        except StopIteration: # aggregate().next() 如果没数据会抛出 StopIteration
            me = None

    if not me:
        print("数据库里没找到任何用户，请先运行初始化脚本或手动添加数据。")
        return

    my_id = str(me['_id'])
    print(f"\n当前模拟登录用户: {me.get('nickname')} ({me.get('gender')}, {me.get('city')}, {me.get('birthday')})")
    print("-" * 50)

    # Context Memory
    last_candidates = []
    last_target = None
    seen_ids = []
    last_criteria = {} # 持久化上一次的搜索意图

    while True:
        try:
            user_input = input("\n用户: ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("退出对话。")
                break

            if not user_input.strip(): continue

            # Construct State
            # 每次都要传入之前的 candidates 和 target 维持记忆
            state = {
                "user_id": my_id,
                "current_input": user_input,
                "messages": [],
                "search_count": 0,
                "final_candidates": last_candidates,
                "last_target_person": last_target,
                "seen_candidate_ids": seen_ids, # 已经看过的ID
                "last_search_criteria": last_criteria # 已经提取过的意图
            }

            # Invoke
            print("正在处理请求...")
            final_state = await app.ainvoke(state)

            # Output
            reply = final_state.get('reply')
            intent = final_state.get('intent')

            print(f"红娘 ({intent}): {reply}")

            # Update Context
            if intent == "search_candidate":
                # 保存搜索条件用于翻页/换一批
                last_criteria = {
                    "hard_filters": final_state.get("hard_filters"),
                    "semantic_query": final_state.get("semantic_query"),
                    "match_policy": final_state.get("match_policy")
                }

                if final_state.get('final_candidates'):
                    last_candidates = final_state.get('final_candidates')
                    print(f"   (更新了 {len(last_candidates)} 个候选人到上下文)")

                # 保存已看过的ID。
                if final_state.get('seen_candidate_ids'):
                    seen_ids = final_state.get('seen_candidate_ids')
                    print(f"   (当前去重池共有 {len(seen_ids)} 个ID)")

            if intent == "deep_dive":
                current_target = final_state.get('target_person_name')
                if current_target:
                    last_target = current_target
                    print(f"   (当前深度探索目标: {last_target})")

        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())