# -*- coding: utf-8 -*-
import random
from typing import List
from bson import ObjectId
from langchain_openai import ChatOpenAI

from config.settings import settings
from utils.env_utils import API_KEY, BASE_URL
from database.mongo_manager import MongoDBManager
from database.chroma_manager import EnhancedChromaManager
from services.user_generator import VirtualUserGenerator
from services.onboarding_generator import TurnByTurnOnboardingGenerator
from services.chat_generator import PersonaBasedChatGenerator

class EnhancedDataGenerationPipeline:
    """增强版数据生成主流程"""

    def __init__(self):
        if settings is None:
            raise ValueError("❌ 错误: 配置文件未成功加载，无法初始化 Pipeline。")

        # 初始化 LLM
        self.llm_ai = ChatOpenAI(
            model=settings.llm.model_name, # <-- 使用配置的模型名称
            temperature=settings.llm.temperature_ai,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        self.llm_user = ChatOpenAI(
            model=settings.llm.model_name, # <-- 使用配置的模型名称
            temperature=settings.llm.temperature_user,
            api_key=API_KEY,
            base_url=BASE_URL,
        )

        # 初始化数据库
        self.db_manager = MongoDBManager(settings.database.mongo_uri, settings.database.db_name)
        self.chroma_manager = EnhancedChromaManager(
            settings.database.chroma_persist_dir,
            settings.database.chroma_collection_name
        )

        # 初始化终止管理器
        try:
            from services.termination_service import DialogueTerminationManager
            self.termination_manager = DialogueTerminationManager(self.llm_ai)
            print("✅ 智能对话终止系统已启用")
        except:
            self.termination_manager = None
            print("⚠️  智能对话终止系统未启用")

        # 初始化生成器
        self.user_gen = VirtualUserGenerator(self.llm_user)
        self.onboarding_gen = TurnByTurnOnboardingGenerator(
            self.llm_ai, self.llm_user, self.termination_manager
        )
        self.chat_gen = PersonaBasedChatGenerator(self.llm_user, self.termination_manager)

    def run_full_pipeline(self):
        print("🚀 开始生产级数据生成流程...")
        print("=" * 70)

        # Step 1: 生成用户
        print("\n📝 Step 1: 生成虚拟用户")
        user_ids = self.user_gen.generate_batch(
            settings.generation.num_users,
            self.db_manager
        )

        # Step 2: Onboarding
        print("\n💬 Step 2: Turn-by-turn 生成 AI 红娘对话")
        for user_id in user_ids:
            self.onboarding_gen.generate_for_user(
                user_id,
                self.db_manager,
                settings.generation.min_onboarding_turns,
                settings.generation.max_onboarding_turns
            )

        # Step 3: Social Chat (新老混合)
        print("\n💑 Step 3: Persona-based 生成用户间聊天 (新老混合蜘蛛网)")
        
        new_users_data = list(self.db_manager.users_basic.find({"_id": {"$in": user_ids}}))
        all_users_data = list(self.db_manager.users_basic.find({}))
        
        print(f"   - 本次新增用户: {len(new_users_data)} 人")
        print(f"   - 全库用户池: {len(all_users_data)} 人")

        CHATS_PER_NEW_USER = 3
        generated_count = 0
        
        for new_user in new_users_data:
            my_id = new_user["_id"]
            my_gender = new_user.get("gender")
            my_name = new_user.get("nickname")
            
            potential_partners = [
                u for u in all_users_data 
                if u.get("gender") != my_gender and u["_id"] != my_id
            ]
            
            if not potential_partners:
                print(f"   ⚠️ {my_name} 没找到异性对象，跳过")
                continue
            
            num_to_chat = min(len(potential_partners), CHATS_PER_NEW_USER)
            partners = random.sample(potential_partners, num_to_chat)
            
            for partner in partners:
                existing_chat = self.db_manager.chat_records.find_one({
                    "$or": [
                        {"user_id": my_id, "partner_id": partner["_id"]},
                        {"user_id": partner["_id"], "partner_id": my_id}
                    ]
                })
                
                if existing_chat:
                    continue
                
                print(f"   💬 生成: 新用户 [{my_name}] ↔ [{'老' if partner['_id'] not in user_ids else '新'}] 用户 [{partner['nickname']}]")
                self.chat_gen.generate_chat_pair(
                    my_id,
                    partner["_id"],
                    self.db_manager,
                    settings.generation.min_chat_messages,
                    settings.generation.max_chat_messages
                )
                generated_count += 1
        
        print(f"   ✨ 实际生成了 {generated_count} 场新老混合聊天")

        # Step 4: Vector DB
        print("\n🔍 Step 4: 构建向量数据库")
        self._build_vector_db(user_ids)
        print("\n✨ 数据生成完成!")
        print("=" * 70)

    def _build_vector_db(self, user_ids: List[ObjectId]):
        for user_id in user_ids:
            # Onboarding
            onboarding = self.db_manager.onboarding_dialogues.find_one({"user_id": user_id})
            if onboarding:
                self.chroma_manager.add_conversation_chunks(
                    str(user_id),
                    onboarding["messages"],
                    "onboarding",
                    settings.rag.window_size,
                    settings.rag.overlap
                )
            # Chats
            chats = self.db_manager.chat_records.find({"user_id": user_id})
            for chat in chats:
                self.chroma_manager.add_conversation_chunks(
                    str(user_id),
                    chat["messages"],
                    "social",
                    settings.rag.window_size,
                    settings.rag.overlap
                )