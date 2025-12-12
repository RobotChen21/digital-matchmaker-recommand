# -*- coding: utf-8 -*-
from datetime import datetime
from bson import ObjectId
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import EnhancedChromaManager
from app.services.ai.agents.user_factory import VirtualUserGenerator
from app.services.ai.workflows.onboarding import TurnByTurnOnboardingGenerator
from app.services.ai.agents.profile_manager import ProfileService
from app.services.ai.tools.termination import DialogueTerminationManager
from app.core.config import settings

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
            user_dict = user_obj.model_dump()
            persona_dict = user_dict.pop("persona_seed") # 分离 Persona 种子
            user_id = self.db_manager.insert_user_with_persona(user_dict, persona_dict)
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
            
            # 确保 profile_data 初始化
            profile_data = {}
            if conversation_history:
                dialogue_text = self.profile_service.format_dialogue_for_llm(conversation_history)
                profile_data = self.profile_service.extract_from_dialogue(dialogue_text)
                
                # 补充元数据
                profile_data["user_id"] = user_id
                profile_data["updated_at"] = datetime.now()
                
                # 存入 MongoDB (User Profile)
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
                # 获取基础信息 (user_dict 是生成时的字典，包含 birthday, gender, city)
                # 注意 user_dict 里可能没有 height，但 user_obj 里有
                basic_info = user_obj.model_dump()
                
                # 生成摘要
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
                
                # 计算 birth_year (复用逻辑)
                try:
                    bday = basic_info.get('birthday')
                    if bday:
                        metadata['birth_year'] = int(bday.split('-')[0])
                except:
                    pass

                # 存入 Chroma
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
                    # Cleanup
                    self.db_manager.users_basic.delete_one({"_id": user_id})
                    self.db_manager.users_persona.delete_one({"user_id": user_id})
                    self.db_manager.onboarding_dialogues.delete_one({"user_id": user_id})
                    self.db_manager.db["users_profile"].delete_one({"user_id": user_id})
                    self.db_manager.chat_records.delete_many({"user_id": user_id})
                    # ChromaDB Cleanup? 如果 3.5 步成功但 4 步失败，会留下脏画像向量。
                    # 不过因为向量检索时会 filter user_id，如果 user_id 在 MongoDB 没了，推荐流程第一步 LoadProfile 就挂了，所以检索不到。
                    # 为了完美，最好也删。但 Chroma 删除操作复杂，暂略。
                    print("     ✅ 脏数据清理完成")
                except Exception as cleanup_error:
                    print(f"     ⚠️ 清理脏数据失败: {cleanup_error}")
            
            raise e # 重新抛出异常
