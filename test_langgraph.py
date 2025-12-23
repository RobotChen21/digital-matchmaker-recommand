# -*- coding: utf-8 -*-
import os
import random
import sys
import asyncio
from bson import ObjectId
from pymongo.errors import OperationFailure

# 添加项目根目录到 Path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.core.container import container

async def main():
    print("========================================")
    print("   红娘助手调试模式 (DB Session Mode)   ")
    print("========================================")
    print("输入 'q' 或 'quit' 退出")

    # 1. Init Dependencies
    db_manager = container.db
    session_service = container.session_service
    app = container.recommendation_app

    # 2. Pick user (随便找个ID试运行)
    target_user_id = "693ebdc20196b88668259955"
    try:
        me = db_manager.users_basic.find_one({"_id": ObjectId(target_user_id)})
    except:
        me = None

    if not me:
        print(f"找不到指定用户 {target_user_id} ，尝试随机选取...")
        try:
            # 尝试随机抽取
            cursor = db_manager.users_basic.aggregate([{"$sample": {"size": 1}}])
            me = cursor.next()
        except (OperationFailure, StopIteration):
            # 回退到传统方式
            user_count = db_manager.users_basic.count_documents({})
            if user_count > 0:
                random_index = random.randint(0, user_count - 1)
                me = db_manager.users_basic.find().skip(random_index).limit(1).next()

    if not me:
        print("❌ 错误：数据库里没找到任何用户，请先添加数据ảng")
        return

    user_id = str(me['_id'])
    print(f"\n👤 当前模拟用户: {me.get('nickname')} ({me.get('gender')}, {me.get('city')})")
    print(f"🆔 User ID: {user_id}")

    # 3. Create Session (Server-side)
    session_id = session_service.create_session(user_id, title="CLI调试会话")
    print(f"✅ 会话已创建: {session_id}")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n用户: ").strip()
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("👋 退出对话")
                break

            if not user_input: continue

            # --- A. 保存用户消息 ---
            session_service.add_message(session_id, "user", user_input)

            # --- B. 加载最新状态与历史 (Restore State & History) ---
            current_session = session_service.get_session(session_id, user_id)
            if not current_session:
                print("❌ 会话丢失！")
                break
                
            latest_state = current_session.get("latest_state", {})
            # 获取最近 10 条历史记录用于上下文记忆
            history_msgs = session_service.get_history(session_id, limit=20)
            
            # [关键修正] 剔除最后一条消息(即当前用户刚刚输入的消息)
            # 因为它已经通过 'current_input' 字段独立传入了，避免在 Prompt 中出现双重重复。
            if history_msgs and history_msgs[-1]['role'] == 'user' and history_msgs[-1]['content'] == user_input:
                history_msgs = history_msgs[:-1]
            
            # --- C. 构造 LangGraph 输入 ---
            input_state = {
                "user_id": user_id,
                "current_input": user_input,
                "messages": history_msgs, # 注入历史记录，实现记忆！
                "search_count": 0,
                
                # 从 DB 恢复的关键状态
                "seen_candidate_ids": latest_state.get("seen_candidate_ids", []),
                "final_candidates": latest_state.get("final_candidates", []),
                "last_target_person": latest_state.get("last_target_person"),
                "last_search_criteria": latest_state.get("last_search_criteria", {}),
                "hard_filters": latest_state.get("hard_filters", {})
            }

            print("⏳ 正在思考...")
            
            # --- D. 执行 Workflow ---
            final_state = await app.ainvoke(input_state)

            # --- E. 提取结果 & 打印 ---
            reply = final_state.get('reply')
            intent = final_state.get('intent')
            print(f"红娘 ({intent}): {reply}")

            if intent == "search_candidate":
                cands = final_state.get('final_candidates', [])
                print(f"   -> 推荐了 {len(cands)} 人")

            if intent == "deep_dive":
                target = final_state.get('last_target_person')
                print(f"   -> 深度探索: {target}")

            # --- F. 持久化更新状态 (Persist State) ---
            session_service.update_session_state(session_id, final_state)
            session_service.add_message(session_id, "ai", str(reply), metadata={"intent": intent})

        except Exception as e:
            print(f"❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
