# -*- coding: utf-8 -*-
from datetime import datetime, date
from bson import ObjectId
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import ChromaManager
from app.services.ai.agents.user_factory import VirtualUserGenerator
from app.services.ai.agents.profile_manager import ProfileService
from app.services.ai.tools.termination import DialogueTerminationManager
from app.core.config import settings


class UserInitializationService:
    """
    用户初始化编排服务 (Atomic Service)
    职责：原子化地执行 [生成用户 -> 红娘对话 -> 提取画像] 这一完整流程。
    """

    def __init__(self, db_manager: MongoDBManager, chroma_manager: ChromaManager, llm_ai: ChatOpenAI, llm_user: ChatOpenAI):
        self.db_manager = db_manager
        self.chroma_manager = chroma_manager
        
        # 初始化各个子服务
        self.termination_manager = DialogueTerminationManager(llm_ai)
        self.user_gen = VirtualUserGenerator(llm_user)
        self.profile_service = ProfileService(llm_ai) # 使用 AI 模型 (通常用能力强的那个) 进行分析

    def finalize_user_onboarding(self, user_id: str) -> bool:
        """
        [原子操作块]
        当用户完成 Onboarding 对话后调用。
        负责：
        1. 读取全量对话
        2. 提取画像 -> 存库
        3. 向量化画像 -> 存库
        4. 向量化对话 -> 存库
        5. 标记用户为 is_completed=True
        """
        print(f"🚀 [Finalize] 开始处理用户 {user_id} 的最终画像与向量化...")
        uid = ObjectId(user_id)
        
        try:
            # 0. [幂等性保障] 先清理该用户已有的向量数据，防止重试导致重复积压
            # 注意：这会删除该用户的所有画像摘要和对话记录向量
            print(f"   🧹 清理用户 {uid} 的旧向量数据...")
            try:
                self.chroma_manager.vector_db.delete(where={"user_id": str(uid)})
            except Exception as e:
                # 如果是第一次生成，可能没有数据，delete 可能会(视版本而定)报错或不做任何事
                # 这里的 catch 是为了稳健，防止因为"没东西删"而报错
                print(f"   ⚠️ 清理向量数据时(可能无数据): {e}")

            # 1. 读取对话
            dialogue_record = self.db_manager.onboarding_dialogues.find_one({"user_id": uid})
            if not dialogue_record or not dialogue_record.get('messages'):
                print("   ❌ 未找到对话记录")
                return False
            
            messages = dialogue_record['messages']
            
            # 2. [优化] 直接从数据库读取最新的画像 (已经在 OnboardingNode 中增量提取并保存了)
            # 不再重复进行全量提取，节省 Token 并避免数据覆盖风险
            print("   📸 读取已有的全量画像...")
            profile_data = self.db_manager.db["users_profile"].find_one({"user_id": uid}) or {}
            
            # 3. 向量化画像
            print("   🧠 向量化画像...")
            user_basic = self.db_manager.users_basic.find_one({"_id": uid})
            summary_text = ProfileService.generate_profile_summary(user_basic, profile_data)
            
            metadata = {
                "user_id": str(user_id),
                "gender": user_basic.get('gender', 'unknown'), 
                "data_type": "profile_summary", 
                "city": user_basic.get('city', 'unknown'), 
                "timestamp": str(datetime.now())
            }
            # 补充元数据
            if user_basic.get('height'): metadata['height'] = user_basic.get('height')
            if isinstance(user_basic.get('birthday'), date): metadata['birth_year'] = user_basic.get('birthday').year
            elif isinstance(user_basic.get('birthday'), str): 
                try: metadata['birth_year'] = int(user_basic.get('birthday').split('-')[0])
                except: pass

            doc = Document(page_content=summary_text, metadata=metadata)
            self.chroma_manager.vector_db.add_documents([doc])
            
            # 4. 向量化对话
            print("   💬 向量化对话记录...")
            self.chroma_manager.add_conversation_chunks(
                str(user_id),
                messages,
                "onboarding",
                window_size=settings.rag.window_size,
                overlap=settings.rag.overlap
            )
            
            # 5. 标记完成
            self.db_manager.users_basic.update_one(
                {"_id": uid},
                {"$set": {"is_completed": True}}
            )
            self.db_manager.users_states.update_one( # [NEW] 更新状态表
                {"user_id": uid},
                {"$set": {"is_onboarding_completed": True, "updated_at": datetime.now()}},
                upsert=True
            )
            print("   ✅ 用户初始化最终完成！")
            return True
            
        except Exception as e:
            print(f"   ❌ Finalize 失败: {e}")
            import traceback
            traceback.print_exc()
            # 失败策略: 回滚状态? 或者让用户重试?
            # 暂时保持 is_completed=False，用户下次还可以继续或者重试
            return False
