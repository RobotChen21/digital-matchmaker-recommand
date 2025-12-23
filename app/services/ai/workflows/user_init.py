# -*- coding: utf-8 -*-
from datetime import datetime, date
from bson import ObjectId

from app.core.container import container
from app.core.config import settings

class UserInitializationService:
    """
    用户初始化编排服务 (Atomic Service)
    职责：原子化地执行 [生成用户 -> 红娘对话 -> 提取画像] 这一完整流程。
    """

    def __init__(self):
        self.db_manager = container.db
        self.chroma_manager = container.chroma
        self.es_manager = container.es # <--- 从容器获取
        
        # 初始化各个子服务
        self.llm_ai = container.get_llm("chat")
        self.llm_user = container.get_llm("chat")
        
        self.termination_manager = container.termination_manager
        self.profile_service = container.profile_service

    def finalize_user_onboarding(self, user_id: str) -> bool:
        """
        [原子操作块]
        当用户完成 Onboarding 对话后调用。
        负责：
        1. (可选) 读取全量对话
        2. (可选) 提取画像 -> 存库 
           (注意: 现在的逻辑假设画像已经存在库里了。对于生成脚本，前面已经提了。对于实时对话，OnboardingNode已经增量提了)
        3. 向量化画像 -> 存库 (Chroma + ES)
        4. 向量化对话 -> 存库
        5. 标记用户为 is_completed=True
        """
        print(f"🚀 [Finalize] 开始处理用户 {user_id} 的最终向量化与标记...")
        uid = ObjectId(user_id)
        
        try:
            # 0. 清理旧向量 (幂等性)
            try:
                self.chroma_manager.vector_db.delete(where={"user_id": str(uid)})
                # TODO: 以后可以考虑清理 ES，但 ES 的 index 方法本身就是覆盖式的 (Upsert)，所以不删也行
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

            # 写入 ES (新增混合检索同步)
            print("   🔍 同步到 Elasticsearch (Hybrid Search)...")
            try:
                # 提取关键词标签 (全面覆盖文本字段)
                interest_info = profile_data.get("interest_profile", {}) or {}
                tags_list = interest_info.get("tags", [])
                tags_str = " ".join(tags_list) if isinstance(tags_list, list) else ""
                
                edu_info = profile_data.get("education_profile", {}) or {}
                highest_degree = edu_info.get("highest_degree", "")
                major = edu_info.get("major", "")
                
                occ_info = profile_data.get("occupation_profile", {}) or {}
                job_title = occ_info.get("job_title", "")
                industry = occ_info.get("industry", "")
                
                fam_info = profile_data.get("family_profile", {}) or {}
                family_struct = fam_info.get("family_structure", "")
                
                life_info = profile_data.get("lifestyle_profile", {}) or {}
                smoking = life_info.get("smoking", "")
                drinking = life_info.get("drinking", "")
                exercise = life_info.get("exercise_level", "")
                
                pers_info = profile_data.get("personality_profile", {}) or {}
                mbti = pers_info.get("mbti", "")
                
                love_info = profile_data.get("love_style_profile", {}) or {}
                attachment = love_info.get("attachment_style", "")
                
                raw_keywords = [
                    tags_str, highest_degree, major, job_title, industry,
                    family_struct, smoking, drinking, exercise, mbti, attachment,
                    user_basic.get('city', '')
                ]
                keyword_tags = " ".join([str(k) for k in raw_keywords if k])
                # 获取向量 (复用 Chroma 的模型)
                vector = self.chroma_manager.embeddings_model.embed_query(summary_text)
                
                # 索引到 ES
                self.es_manager.index_user(
                    user_id=str(user_id),
                    profile_data={
                        "gender": user_basic.get("gender"),
                        "city": user_basic.get("city"),
                        "age": user_basic.get("age") or 0,
                        "tags": keyword_tags,
                        "profile_text": summary_text
                    },
                    vector=vector
                )
            except Exception as es_err:
                print(f"   ⚠️ ES 同步失败 (非致命错误): {es_err}")
            
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