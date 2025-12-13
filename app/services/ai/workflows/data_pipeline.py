import random
from typing import List
from bson import ObjectId

from app.core.config import settings
from app.core.llm import get_llm
from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import ChromaManager
from app.services.ai.workflows.user_init import UserInitializationService
from app.services.ai.agents.chat_bot import PersonaBasedChatGenerator
from app.services.ai.tools.termination import DialogueTerminationManager

class EnhancedDataGenerationPipeline:
    """增强版数据生成主流程 (重构后: 原子化初始化 + 社交网络构建)"""

    def __init__(self):
        if settings is None:
            raise ValueError("❌ 错误: 配置文件未成功加载，无法初始化 Pipeline。")

        # 初始化 LLM
        self.llm_ai = get_llm(temperature=settings.llm.temperature_ai)
        self.llm_user = get_llm(temperature=settings.llm.temperature_user)

        # 初始化数据库
        self.db_manager = MongoDBManager(settings.database.mongo_uri, settings.database.db_name)
        self.chroma_manager = ChromaManager(
            settings.database.chroma_persist_dir,
            settings.database.chroma_collection_name
        )

        # 1. 原子化初始化服务 (负责: 用户生成 -> 红娘对话 -> 画像提取 -> 向量化)
        self.init_service = UserInitializationService(
            self.db_manager, 
            self.chroma_manager, 
            self.llm_ai, 
            self.llm_user
        )

        # 2. 社交聊天生成器 (负责: 用户间对话)
        # 需要单独的 Termination Manager
        self.termination_manager = DialogueTerminationManager(self.llm_ai)
        self.chat_gen = PersonaBasedChatGenerator(self.llm_user, self.termination_manager)

    def run_full_pipeline(self):
        print("🚀 开始生产级数据生成流程 (Atomic Mode)...")
        print("=" * 70)

        # Step 1: 原子化生成用户 (Loop)
        target_new_users = settings.generation.num_users
        print(f"\n📦 Step 1: 计划生成 {target_new_users} 名完整用户 (原子化流程)")
        
        new_user_ids = []
        for i in range(target_new_users):
            print(f"\n--- 处理第 {i+1}/{target_new_users} 个用户 ---")
            try:
                uid = self.init_service.create_and_onboard_single_user()
                new_user_ids.append(uid)
            except Exception as e:
                print(f"❌ 用户生成流程失败: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✅ Step 1 完成，成功初始化 {len(new_user_ids)} 名用户。")

        # Step 2: 社交聊天 (Social Chat)
        # 这一步依赖于"用户池"，所以必须在所有用户生成完之后进行
        print("\n💑 Step 2: 构建社交网络 (用户间对话)")
        
        # 获取最新的用户列表（包含刚刚生成的）
        # 注意: 这里我们只让"新生成的用户"去尝试匹配，或者让"全库用户"随机匹配
        # 策略: 让本次新增的用户，每人尝试匹配几个对象 (新老混合)
        
        all_users_data = list(self.db_manager.users_basic.find({}))
        new_users_data = [u for u in all_users_data if u["_id"] in new_user_ids]
        
        print(f"   - 本次新增用户: {len(new_users_data)} 人")
        print(f"   - 全库用户池: {len(all_users_data)} 人")

        if len(all_users_data) < 2:
            print("⚠️ 用户不足2人，无法进行社交聊天生成。")
            return

        CHATS_PER_NEW_USER = 3
        generated_count = 0
        
        for new_user in new_users_data:
            my_id = new_user["_id"]
            my_gender = new_user.get("gender")
            my_name = new_user.get("nickname")
            
            # 简单的异性筛选逻辑
            potential_partners = [
                u for u in all_users_data 
                if u.get("gender") != my_gender and u["_id"] != my_id
            ]
            
            if not potential_partners:
                print(f"   ⚠️ {my_name} 没找到异性对象，跳过")
                continue
            
            # 随机选人
            num_to_chat = min(len(potential_partners), CHATS_PER_NEW_USER)
            partners = random.sample(potential_partners, num_to_chat)
            
            for partner in partners:
                # 查重: 避免重复生成同一对
                existing_chat = self.db_manager.chat_records.find_one({
                    "$or": [
                        {"user_id": my_id, "partner_id": partner["_id"]},
                        {"user_id": partner["_id"], "partner_id": my_id}
                    ]
                })
                
                if existing_chat:
                    continue
                
                print(f"   💬 生成对话: [{my_name}] ↔ [{partner['nickname']}]")
                try:
                    chat_history = self.chat_gen.generate_chat_pair(
                        my_id,
                        partner["_id"],
                        self.db_manager,
                        settings.generation.min_chat_messages,
                        settings.generation.max_chat_messages
                    )
                    
                    # 社交对话向量化 (即时处理)
                    if chat_history:
                        self.chroma_manager.add_conversation_chunks(
                            str(my_id), chat_history, "social", settings.rag.window_size, settings.rag.overlap
                        )
                        self.chroma_manager.add_conversation_chunks(
                            str(partner["_id"]), chat_history, "social", settings.rag.window_size, settings.rag.overlap
                        )
                    
                    generated_count += 1
                except Exception as e:
                    print(f"   ❌ 聊天生成失败: {e}")

        print(f"\n✨ Step 2 完成，实际生成了 {generated_count} 场社交对话。")
        print("=" * 70)
        print("🎉 所有任务执行完毕。")

    def _build_vector_db(self, user_ids: List[ObjectId]):
        # 此方法已弃用，向量化逻辑已集成到各个阶段中
        pass