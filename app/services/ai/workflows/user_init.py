# -*- coding: utf-8 -*-
from datetime import datetime, date
from bson import ObjectId
from langchain_core.documents import Document

from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import EnhancedChromaManager
from app.services.ai.agents.user_factory import VirtualUserGenerator
from app.services.ai.workflows.onboarding import TurnByTurnOnboardingGenerator
from app.services.ai.agents.profile_manager import ProfileService
from app.services.ai.tools.termination import DialogueTerminationManager
from app.core.config import settings
from app.core.llm import get_llm

class UserInitializationService:
    """
    用户初始化编排服务 (Atomic Service)
    职责：原子化地执行 [生成用户 -> 红娘对话 -> 提取画像] 这一完整流程。
    """

    def __init__(self, db_manager: MongoDBManager, chroma_manager: EnhancedChromaManager, llm_ai: ChatOpenAI, llm_user: ChatOpenAI):
        self.db_manager = db_manager
        self.chroma_manager = chroma_manager
        
        # 初始化各个子服务
        self.termination_manager = DialogueTerminationManager(llm_ai)
        self.user_gen = VirtualUserGenerator(llm_user)
        self.onboarding_gen = TurnByTurnOnboardingGenerator(llm_ai, llm_user, self.termination_manager)
        self.profile_service = ProfileService(llm_ai) # 使用 AI 模型 (通常用能力强的那个) 进行分析

    def create_and_onboard_single_user(self) -> ObjectId:
        """
        执行单个用户的完整生命周期初始化。
        包含自动回滚机制：如果中途失败，自动清理已生成的脏数据。
        
        Returns:
            user_id: 生成并处理完成的用户 ID
        """
        print("\n" + "="*50)
        print("🚀 [Atomic] 开始初始化新用户流程...")
        
        user_id = None

        try:
            # 1. 生成用户 (Generate)
            print("  1️⃣ 生成虚拟用户基础信息...")
            user_obj = self.user_gen.generate_user()
            
            # 存入 MongoDB (User Basic)
            user_data_for_mongo = user_obj.model_dump(exclude_none=True)
            # 确保 birthday 是 date 对象
            if isinstance(user_data_for_mongo.get("birthday"), str):
                try:
                    user_data_for_mongo["birthday"] = date.fromisoformat(user_data_for_mongo["birthday"])
                except ValueError:
                    user_data_for_mongo["birthday"] = date(2000,1,1) # 默认值或报错
            
            persona_dict = user_data_for_mongo.pop("persona_seed") # 分离 Persona 种子
            user_id = self.db_manager.insert_user_with_persona(user_data_for_mongo, persona_dict)
            print(f"     ✅ 用户创建成功: {user_obj.nickname} (ID: {user_id})")

            # 2. 红娘对话 (Onboarding)
            print("  2️⃣ 开启 AI 红娘 Onboarding 对话...")
            conversation_history = self.onboarding_gen.generate_for_user(
                user_id,
                self.db_manager,
                min_turns=settings.generation.min_onboarding_turns,
                max_turns=settings.generation.max_onboarding_turns
            )
            print(f"     ✅ 对话结束，共 {len(conversation_history)} 条消息")

            # 3. 提取画像 (Profile Extraction)
            print("  3️⃣ 实时分析对话提取画像...")
            
            profile_data = {}
            if conversation_history:
                dialogue_text = self.profile_service.format_dialogue_for_llm(conversation_history)
                profile_data = self.profile_service.extract_from_dialogue(dialogue_text)
                
                profile_data["user_id"] = user_id
                profile_data["updated_at"] = datetime.now()
                
                self.db_manager.db["users_profile"].update_one(
                    {"user_id": user_id},
                    {"$set": profile_data},
                    upsert=True
                )
                print("     ✅ 画像提取并保存完毕")
            else:
                print("     ⚠️ 对话为空，跳过画像提取")

            # 3.5 向量化画像 (Profile Vectorization) [NEW]
            print("  3️⃣.5️⃣ 向量化用户画像 (Profile Vectorization)...")
            if profile_data:
                basic_info = user_obj.model_dump() # 从原始生成对象获取
                
                summary_text = ProfileService.generate_profile_summary(basic_info, profile_data)
                
                # 准备元数据
                metadata = {
                    "user_id": str(user_id),
                    "gender": basic_info.get('gender', 'unknown'), 
                    "data_type": "profile_summary", 
                    "city": basic_info.get('city', 'unknown'), 
                    "timestamp": str(datetime.now())
                }
                
                if basic_info.get('height'):
                    metadata['height'] = basic_info.get('height')
                
                # 计算 birth_year (从 date 对象提取)
                if isinstance(basic_info.get('birthday'), date):
                    metadata['birth_year'] = basic_info.get('birthday').year
                elif isinstance(basic_info.get('birthday'), str): # 兼容旧数据
                    try: metadata['birth_year'] = int(basic_info.get('birthday').split('-')[0])
                    except: pass

                doc = Document(page_content=summary_text, metadata=metadata)
                self.chroma_manager.vector_db.add_documents([doc])
                print("     ✅ 画像摘要已存入向量数据库")

            # 4. 向量化对话 (Onboarding Vectorization)
            print("  4️⃣ 存入向量数据库 (Onboarding RAG)...")
            if conversation_history:
                 self.chroma_manager.add_conversation_chunks(
                    str(user_id),
                    conversation_history,
                    "onboarding",
                    window_size=settings.rag.window_size,
                    overlap=settings.rag.overlap
                )
                 print("     ✅ 对话向量索引构建完成")

            print(f"✨ 用户 [{user_obj.nickname}] 初始化流程全部完成!")
            return user_id

        except Exception as e:
            print(f"❌ 初始化过程中断，正在回滚(删除)用户数据: {user_id}")
            if user_id:
                try:
                    self.db_manager.users_basic.delete_one({"_id": user_id})
                    self.db_manager.users_persona.delete_one({"user_id": user_id})
                    self.db_manager.onboarding_dialogues.delete_one({"user_id": user_id})
                    self.db_manager.db["users_profile"].delete_one({"user_id": user_id})
                    self.db_manager.chat_records.delete_many({"user_id": user_id})
                    print("     ✅ 脏数据清理完成")
                except Exception as cleanup_error:
                    print(f"     ⚠️ 清理脏数据失败: {cleanup_error}")
            
            raise e # 重新抛出异常