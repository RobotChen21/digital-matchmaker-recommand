# -*- coding: utf-8 -*-
from datetime import datetime
from bson import ObjectId
from langchain_core.prompts import ChatPromptTemplate
from app.core.utils.dict_utils import smart_merge, flatten_dict
from app.core.container import container
from app.common.models.state import MatchmakingState

# 延迟导入以避免循环依赖
# from app.services.ai.workflows.user_init import UserInitializationService 

class OnboardingNode:
    def __init__(self):
        self.db = container.db
        self.chroma = container.chroma
        self.llm = container.get_llm("chat") # 0.7 for onboarding
        
        self.termination_manager = container.termination_manager # 使用单例
        self.profile_service = container.profile_service # 使用单例
        
        # 懒加载 UserInitializationService
        self._user_init_service = None

        # 追问 Prompt (优化版，模仿 TurnByTurnOnboardingGenerator)
        self.ask_chain = (
            ChatPromptTemplate.from_template(
                """你是一名资深的婚恋顾问，正在通过对话帮助用户建立完善的个人画像。
你的目标是温柔、耐心、高情商地引导用户说出他们的家庭、教育、工作、资产、生活方式、恋爱风格等信息，这些信息尽量收集全。
每一轮你只能提出一个或少数几个问题，让用户有充足的表达空间。
如果用户表现出抵触，你需要巧妙地安抚和引导。
切记：你是一个充满人情味、专业的红娘。

【重要指令 - 核心KPI】:
1. **必须收集全以下三个核心维度**，如果用户没提到，一定要追问，不能跳过：
   - **教育背景**: 学历 (本科/硕士/博士/专科), 学校类型 (985/211/海外/双非), 学校名称/专业
   - **工作职业**: 职位/行业, 工作风格 (996/轻松/体制内), 收入水平 (如果用户提到)。**如果是学生，请改问专业/科研情况，无需问收入/工作风格**。
   - **家庭背景**: 独生子女？兄弟姐妹？, 父母健康/职业/退休？, 家庭经济状况？, 家庭氛围/父母婚姻状况(离异/重组)?
请直接输出回复。
2. 请使用**自然口语**，就像微信聊天一样。**严禁**使用 Markdown 格式，**严禁**长篇大论，每次回复控制在 3 句话以内。
【对话历史 (最近)】:
{history}
【已收集信息暗示】:
{profile_completion_hint}
(注意：此信息可能存在延迟。如果用户刚刚在【对话历史】中回答了某项信息，请以对话历史为准，请以对话历史为准，请以对话历史为准，不要重复追问。)
"""
            ) | self.llm
        )
        
        # 完结撒花 Prompt
        self.finish_chain = (
            ChatPromptTemplate.from_template(
                """你是一名红娘。用户的信息已经采集完毕了！
                
                【用户画像】: {current_profile_summary}
                
                请对用户表示感谢，并引导他开始寻找对象。
                语气温暖、期待。"""
            ) | self.llm
        )

    def _get_init_service(self):
        if not self._user_init_service:
            from app.services.ai.workflows.user_init import UserInitializationService
            # 复用 llm 实例
            self._user_init_service = UserInitializationService()
        return self._user_init_service

    async def process(self, state: MatchmakingState):
        """处理 Onboarding 逻辑"""
        print("📝 [Onboarding] 实时对话处理...")
        
        user_id = state['user_id']
        current_input = state['current_input']
        uid = ObjectId(user_id)
        
        # ⚠️ 注意: PyMongo 是同步的，在 async 函数中会阻塞 loop。
        # 在生产环境中应使用 Motor 或 run_in_executor。这里暂时保持同步调用。
        user_basic = self.db.users_basic.find_one({"_id": ObjectId(user_id)})
        
        # 1. 实时保存用户输入
        user_msg = {"role": "user", "content": current_input, "timestamp": datetime.now()}
        self.db.onboarding_dialogues.update_one(
            {"user_id": uid},
            {"$push": {"messages": user_msg}},
            upsert=True
        )
        
        # 2. 读取完整历史
        record = self.db.onboarding_dialogues.find_one({"user_id": uid})
        history_list = record.get('messages', []) if record else []

        full_profile = self.db.profile.find_one({"user_id": uid}) or {} # 先读当前的
        
        # [Strategy] 预先生成 Hint，确保如果不进 batch 更新逻辑，后续步骤也有值可用
        # ProfileService 内部可能有 LLM 调用，建议也改为 async，但为了最小改动，这里先同步执行
        profile_completion_hint = self.profile_service.generate_profile_completion_hint(profile=full_profile)

        # 逻辑改为：每当用户说了 3 句话 (即积累了约 3 轮对话)，触发一次提取
        user_msg_count = sum(1 for m in history_list if m['role'] == 'user')
        
        if user_msg_count > 0 and user_msg_count % 4 == 0:
            print(f"   🔄 触发增量画像提取 (用户已发言 {user_msg_count} 次)...")
            # 取最近的 6 条消息作为上下文 (User + AI)
            recent_batch = history_list[-10:]
            # 格式化对话
            dialogue_text = self.profile_service.format_dialogue_for_llm(recent_batch)
            # 提取 (CPU bound + Network bound)
            extracted_data = self.profile_service.extract_from_dialogue(dialogue_text)
            
            # 更新 DB (Smart Merge)
            if extracted_data:
                # 过滤空值
                update_payload = {k: v for k, v in extracted_data.items() if v}
                
                if update_payload:
                    print(f"   -> 提取到新信息 (Before Merge): {list(update_payload.keys())}")
                    
                    # [FIX] 使用智能合并：列表追加，标量覆盖
                    # 直接修改内存中的 full_profile
                    smart_merge(full_profile, update_payload)
                    
                    # 准备写入 DB 的数据
                    # 我们不仅要写入 update_payload 的 key，还要写入它们合并后的最终值 (因为 full_profile 已经被 modify 了)
                    # 为了安全，重新 flatten 一次 full_profile 中涉及到 update_payload 的部分，或者直接 save 整个 documents
                    # 考虑到并发风险低，直接 set 修改过的字段的最终值
                    
                    final_update_set = {}
                    # 重新从 full_profile 提取最终值，构造 $set
                    # 这里有一个技巧：因为 smart_merge 已经更新了 nested dict，
                    # 我们可以简单地把 update_payload 顶层 key 对应的 full_profile 值写回去
                    # 或者更细粒度一点。为了处理 list append，最简单的是把涉及到的 顶层 key 整个覆盖回去。
                    
                    for top_key in update_payload.keys():
                        final_update_set[top_key] = full_profile[top_key]
                        
                    final_update_set["updated_at"] = datetime.now()
                    
                    self.db.profile.update_one(
                        {"user_id": uid},
                        {"$set": final_update_set},
                        upsert=True
                    )
                    print(f"   -> 增量合并并更新了字段: {list(final_update_set.keys())}")
                    
                    # [FIX] 画像更新了，重新生成 Hint 以便 Termination Check 使用最新数据
                    profile_completion_hint = self.profile_service.generate_profile_completion_hint(profile=full_profile)

            # 4. 判断是否完成
            min_conversational_turns_for_check = 3
            if len(history_list) >= min_conversational_turns_for_check * 2:
                should_terminate, signal = self.termination_manager.should_terminate_onboarding(
                    # 传递生成的 hint text
                    profile_completion_hint,
                    history_list, min_conversational_turns=30, max_turns=50
                )
            else:
                should_terminate = False
                signal = None
        
            if should_terminate:
                print(f"   ✅ 检测到信息采集完成: {signal.explanation}")

                # 5. 原子化结算
                success = self._get_init_service().finalize_user_onboarding(user_id)

                if success:
                    # 读取最新画像用于结束语

                    full_profile = self.db.profile.find_one({"user_id": uid}) or {} # 重新读一次确保最新
                    current_profile_summary_text = self.profile_service.generate_profile_summary(user_basic, full_profile)

                    # [ASYNC CHANGE] 使用 ainvoke
                    res = await self.finish_chain.ainvoke({"current_profile_summary": current_profile_summary_text})
                    reply = res.content

                    ai_msg = {"role": "ai", "content": reply, "timestamp": datetime.now()}
                    self.db.onboarding_dialogues.update_one({"user_id": uid}, {"$push": {"messages": ai_msg}})

                    state['reply'] = reply
                    return state
                else:
                    print("   ❌ 结算失败，回退到继续追问")
        
        # 6. 继续追问
        print("   ⏳ 继续追问...")
        
        # profile_completion_hint 已经在上面生成了，直接用

        history_for_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in history_list[-10:]]) # 限制 History 长度
        
        print(f"   💡 [Debug] Hint used for prompt: {profile_completion_hint}")

        # [ASYNC CHANGE] 使用 ainvoke
        res = await self.ask_chain.ainvoke({
            "profile_completion_hint": profile_completion_hint, # 传递 hint
            "history": history_for_prompt
        })
        reply = res.content
        
        # 保存 AI 回复
        ai_msg = {"role": "ai", "content": reply, "timestamp": datetime.now()}
        self.db.onboarding_dialogues.update_one({"user_id": uid}, {"$push": {"messages": ai_msg}})
        
        state['reply'] = reply
        return state