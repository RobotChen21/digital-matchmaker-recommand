# -*- coding: utf-8 -*-
from datetime import datetime, date
from bson import ObjectId
from langchain_core.documents import Document

from app.core.container import container
# from app.services.ai.agents.user_factory import VirtualUserGenerator # Module missing
from app.services.ai.agents.profile_manager import ProfileService
# from app.services.ai.tools.termination import DialogueTerminationManager # Removed
from app.core.config import settings

class UserInitializationService:
    """
    用户初始化编排服务 (Atomic Service)
    职责：原子化地执行 [生成用户 -> 红娘对话 -> 提取画像] 这一完整流程。
    """

    def __init__(self):
        self.db_manager = container.db
        self.chroma_manager = container.chroma
        
        # 初始化各个子服务
        self.llm_ai = container.get_llm("chat")
        self.llm_user = container.get_llm("chat")
        
        self.termination_manager = container.termination_manager
        self.profile_service = container.profile_service

    def create_and_onboard_single_user(self) -> ObjectId:
        """
        [后台脚本用] 执行单个用户的完整生命周期初始化。
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
                    user_data_for_mongo["birthday"] = date(2000,1,1)
            
            persona_dict = user_data_for_mongo.pop("persona_seed") 
            user_id = self.db_manager.insert_user_with_persona(user_data_for_mongo, persona_dict)
            print(f"     ✅ 用户创建成功: {user_obj.nickname} (ID: {user_id})")

            # 2. 红娘对话 (Onboarding)
            print("  2️⃣ 开启 AI 红娘 Onboarding 对话...")
            # 这里的 Onboarding Generator 会生成一整套对话并存入 DB
            conversation_history = self.onboarding_gen.generate_for_user(
                user_id,
                self.db_manager,
                min_turns=settings.generation.min_onboarding_turns,
                max_turns=settings.generation.max_onboarding_turns
            )
            print(f"     ✅ 对话结束，共 {len(conversation_history)} 条消息")

            # 3. 提取与向量化 (调用复用的 finalize 逻辑)
            # 注意: 这里 finalize 会读取 DB 里的对话。generate_for_user 已经存了。
            # 但 finalize 也会尝试读取 users_profile。
            # 之前的逻辑是: 生成脚本是"一次性提取"。
            # 现在的 finalize 逻辑假设 users_profile 已经增量提取了。
            # 矛盾点: 生成脚本 (TurnByTurn) 并没有增量提取逻辑！它只存了对话。
            # 所以，对于生成脚本，我们需要先"全量提取"，再"finalize"。
            
            print("  3️⃣ 提取全量画像 (Batch Mode)...")
            dialogue_text = self.profile_service.format_dialogue_for_llm(conversation_history)
            profile_data = self.profile_service.extract_from_dialogue(dialogue_text)
            
            profile_data["user_id"] = user_id
            profile_data["updated_at"] = datetime.now()
            self.db_manager.db["users_profile"].update_one(
                {"user_id": user_id},
                {"$set": profile_data},
                upsert=True
            )
            
            # 现在可以调用 finalize 了 (它会负责向量化和标记)
            success = self.finalize_user_onboarding(str(user_id))
            if not success:
                raise Exception("Finalization failed.")

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
                    self.db_manager.users_states.delete_one({"user_id": user_id})
                    print("     ✅ 脏数据清理完成")
                except Exception as cleanup_error:
                    print(f"     ⚠️ 清理脏数据失败: {cleanup_error}")
            
            raise e 

    def finalize_user_onboarding(self, user_id: str) -> bool:
        """
        [原子操作块]
        当用户完成 Onboarding 对话后调用。
        负责：
        1. (可选) 读取全量对话
        2. (可选) 提取画像 -> 存库 
           (注意: 现在的逻辑假设画像已经存在库里了。对于生成脚本，前面已经提了。对于实时对话，OnboardingNode已经增量提了)
        3. 向量化画像 -> 存库
        4. 向量化对话 -> 存库
        5. 标记用户为 is_completed=True
        """
        print(f"🚀 [Finalize] 开始处理用户 {user_id} 的最终向量化与标记...")
        uid = ObjectId(user_id)
        
        try:
            # 0. 清理旧向量 (幂等性)
            try:
                self.chroma_manager.vector_db.delete(where={"user_id": str(uid)})
            except:
                pass

            # 1. 读取对话 (用于向量化)
            dialogue_record = self.db_manager.onboarding_dialogues.find_one({"user_id": uid})
            if not dialogue_record:
                print("   ❌ 未找到对话记录")
                return False
            messages = dialogue_record.get('messages', [])
            
            # 2. 读取画像 (用于向量化)
            profile_data = self.db_manager.db["users_profile"].find_one({"user_id": uid}) or {}
            
            # 3. 向量化画像
            print("   🧠 向量化画像...")
            user_basic = self.db_manager.users_basic.find_one({"_id": uid})
            summary_text = self.profile_service.generate_profile_summary(user_basic, profile_data)
            
            metadata = {
                "user_id": str(user_id),
                "gender": user_basic.get('gender', 'unknown'), 
                "data_type": "profile_summary", 
                "city": user_basic.get('city', 'unknown'),
                "height": user_basic.get('height', 'unknown'),
                "weight": user_basic.get('weight', 'unknown'),
                "timestamp": str(datetime.now())
            }
            if isinstance(user_basic.get('birthday'), date): metadata['birth_year'] = user_basic.get('birthday').year
            elif isinstance(user_basic.get('birthday'), str):
                try: metadata['birth_year'] = int(user_basic.get('birthday').split('-')[0])
                except: pass

            doc = Document(page_content=summary_text, metadata=metadata)
            self.chroma_manager.vector_db.add_documents([doc])
            
            # 4. 向量化对话
            print("   💬 向量化对话记录...")
            if messages:
                self.chroma_manager.add_conversation_chunks(
                    str(user_id),
                    messages,
                    "onboarding",
                    window_size=settings.rag.window_size,
                    overlap=settings.rag.overlap
                )
            
            # 5. 标记完成 (User States)
            self.db_manager.users_states.update_one(
                {"user_id": uid},
                {"$set": {"is_onboarding_completed": True, "updated_at": datetime.now()}},
                upsert=True
            )
            # 同时也更新 Basic (兼容性)
            self.db_manager.users_basic.update_one(
                {"_id": uid},
                {"$set": {"is_completed": True}}
            )
            
            print("   ✅ 用户初始化最终完成！")
            return True
            
        except Exception as e:
            print(f"   ❌ Finalize 失败: {e}")
            import traceback
            traceback.print_exc()
            return False