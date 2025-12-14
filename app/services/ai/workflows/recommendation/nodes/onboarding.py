# -*- coding: utf-8 -*-
from datetime import datetime
from bson import ObjectId
from langchain_core.prompts import ChatPromptTemplate

from app.core.llm import get_llm
from app.common.models.state import MatchmakingState
from app.services.ai.tools.termination import DialogueTerminationManager
from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import ChromaManager
from app.services.ai.agents.profile_manager import ProfileService # for summary

# 延迟导入以避免循环依赖
# from app.services.ai.workflows.user_init import UserInitializationService 

class OnboardingNode:
    def __init__(self, db_manager: MongoDBManager, chroma_manager: ChromaManager):
        self.db = db_manager
        self.chroma = chroma_manager
        self.llm = get_llm(temperature=0.3) # Onboarding AI 温度稍高，更人性化
        
        self.termination_manager = DialogueTerminationManager(self.llm)
        self.profile_service = ProfileService(self.llm) # 初始化 ProfileService
        
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
1. 请使用**自然口语**，就像微信聊天一样。**严禁**使用 Markdown 格式，**严禁**长篇大论，每次回复控制在 3 句话以内。
2. **必须收集全以下三个核心维度**，如果用户没提到，一定要追问，不能跳过：
   - **教育背景**: 学历 (本科/硕士/博士/专科), 学校类型 (985/211/海外/双非), 学校名称/专业
   - **工作职业**: 职位/行业, 工作风格 (996/轻松/体制内), 收入水平 (如果用户提到)
   - **家庭背景**: 独生子女？兄弟姐妹？, 父母健康/职业/退休？, 家庭经济状况？, 家庭氛围/父母婚姻状况(离异/重组)?
3. 其他维度 (兴趣、性格) 可以自然穿插提问。

【对话历史】:
{history}

请直接输出回复。"""
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
            # 复用 llm 实例，这里需要两个 llm，所以传 self.llm 两次 (ai/user)
            self._user_init_service = UserInitializationService(self.db, self.chroma, self.llm, self.llm)
        return self._user_init_service

    def process(self, state: MatchmakingState):
        """处理 Onboarding 逻辑"""
        # NOTE:
        # 当前 onboarding 完成判定基于对话历史（而非 profile 完整度）
        # profile 仅在 finalize 阶段一次性生成

        print("📝 [Onboarding] 实时对话处理...")
        
        user_id = state['user_id']
        current_input = state['current_input']
        uid = ObjectId(user_id)
        
        # 1. 实时保存用户输入
        user_msg = {"role": "user", "content": current_input, "timestamp": datetime.now()}
        self.db.onboarding_dialogues.update_one(
            {"user_id": uid},
            {"$push": {"messages": user_msg}},
            upsert=True
        )
        
        # 2. 读取完整历史 (用于检测和上下文)
        record = self.db.onboarding_dialogues.find_one({"user_id": uid})
        history_list = record.get('messages', []) if record else []
        
        # 3. 判断是否完成
        min_conversational_turns_for_check = 3 # 用户回答 3 次后开始
        if len(history_list) >= min_conversational_turns_for_check * 2: # 至少 6 条消息
            should_terminate, signal = self.termination_manager.should_terminate_onboarding(
                history_list, min_turns=15, max_turns=30
            )
        else:
            should_terminate = False
            signal = None 
        
        if should_terminate:
            print(f"   ✅ 检测到信息采集完成: {signal.explanation}")
            
            # 5. 原子化结算 (Extract -> Save -> Vectorize)
            success = self._get_init_service().finalize_user_onboarding(user_id)
            user_basic = self.db.users_basic.find_one({"_id": uid}) or {}
            current_profile = self.db.profile.find_one({"user_id": uid}) or {}

            if success:
                # 生成结束语并保存
                current_profile_summary_text = ProfileService.generate_profile_summary(user_basic, current_profile)
                res = self.finish_chain.invoke({"current_profile_summary": current_profile_summary_text})
                reply = res.content
                
                # 保存 AI 回复
                ai_msg = {"role": "ai", "content": reply, "timestamp": datetime.now()}
                self.db.onboarding_dialogues.update_one(
                    {"user_id": uid},
                    {"$push": {"messages": ai_msg}}
                )
                state['reply'] = reply
                return state
            else:
                print("   ❌ 结算失败，回退到继续追问")
        
        # 6. 继续追问 (如果未完成或结算失败)
        print("   ⏳ 继续追问...")

        history_for_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in history_list[-6:]])
        res = self.ask_chain.invoke({
            "history": history_for_prompt
        })
        reply = res.content
        
        # 保存 AI 回复
        ai_msg = {"role": "ai", "content": reply, "timestamp": datetime.now()}
        self.db.onboarding_dialogues.update_one(
            {"user_id": uid},
            {"$push": {"messages": ai_msg}}
        )
        
        state['reply'] = reply
        return state