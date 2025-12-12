# -*- coding: utf-8 -*-
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime
from bson import ObjectId
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.db.mongo_manager import MongoDBManager
from app.services.ai.tools.termination import DialogueTerminationManager

class TurnByTurnOnboardingGenerator:
    """分回合 AI 红娘 Onboarding 对话生成器"""

    def __init__(self, llm_ai: ChatOpenAI, llm_user: ChatOpenAI, termination_manager: DialogueTerminationManager):
        self.llm_ai = llm_ai
        self.llm_user = llm_user
        self.termination_manager = termination_manager

        # AI 红娘的 Prompt
        self.ai_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名资深的婚恋顾问，正在通过对话帮助用户建立完善的个人画像。
你的目标是温柔、耐心、高情商地引导用户说出他们的家庭、教育、工作、资产、生活方式、恋爱风格等信息，这些信息尽量收集全。
每一轮你只能提出一个或少数几个问题，让用户有充足的表达空间。
如果用户表现出抵触，你需要巧妙地安抚和引导。
切记：你是一个充满人情味的红娘，不是冷冰冰的机器人。

【已收集信息暗示】:
{persona_hint}

【对话历史】:
{conversation_history}"""),
            MessagesPlaceholder(variable_name="chat_history"), # 聊天历史
            ("human", "{user_message}") # 用户的最新消息
        ])
        self.ai_chain = self.ai_prompt | self.llm_ai

        # 用户的 Prompt (基于 Persona 种子进行回复)
        self.user_prompt = ChatPromptTemplate.from_messages([
            ("system", """你现在扮演一个真实的交友用户，正在与一位 AI 红娘交流，红娘旨在了解你的个人情况以建立个人档案。
请根据你的人设 {user_persona}，真实、自然地回答红娘提出的问题。
你的目标是尽可能详细、真诚地展示自己，但同时也要有普通人的矜持和对隐私的保护。
可以提出自己的疑问或反问红娘。
你目前处于对话的第 {turn} 轮。
【对话历史】:
{conversation_history}""" ),
            MessagesPlaceholder(variable_name="chat_history"), # 聊天历史
            ("human", "{user_message}") # 红娘的最新消息
        ])
        self.user_chain = self.user_prompt | self.llm_user

    def generate_for_user(
                          self,
                          user_id: ObjectId,
                          db_manager: MongoDBManager,
                          min_turns: int = 12,
                          max_turns: int = 30) -> List[Dict]:
        """为指定用户生成一轮 AI 红娘 Onboarding 对话"""
        user_basic, persona_data = db_manager.get_user_with_persona(user_id)
        user_persona_str = self._format_persona(persona_data)
        
        conversation_history = []
        chat_history_lc = [] # LangChain 格式的聊天历史
        
        ai_response_content = "你好，我是你的专属红娘，很高兴认识你！我们来聊聊，我会帮你打造一份最完美的个人档案，帮你找到心仪的另一半。我们先从你的基本情况聊起好吗？"

        for turn in range(max_turns):
            print(f"  回合 {turn + 1}")
            
            # 1. AI 红娘发言
            ai_message_entry = {
                "role": "ai",
                "content": ai_response_content,
                "timestamp": datetime.now()
            }
            conversation_history.append(ai_message_entry)
            chat_history_lc.append(AIMessage(content=ai_response_content))
            print(f"    AI红娘: {ai_response_content}")

            # 2. 判断是否终止对话 (在用户回复前判断，或者在红娘提问后判断)
            # 在红娘提出问题后，用户回答前，判断当前对话是否可以结束
            if self.termination_manager:
                should_terminate, signal = self.termination_manager.should_terminate_onboarding(
                    conversation_history, min_turns, max_turns
                )
                if should_terminate:
                    print(f"  ✅ Onboarding 终止: {signal.reason} (置信度: {signal.confidence:.2f}) - {signal.explanation}")
                    break

            # 3. 用户回复
            user_response = self.user_chain.invoke({
                "user_persona": user_persona_str,
                "turn": turn + 1,
                "conversation_history": self._format_history(conversation_history),
                "chat_history": chat_history_lc,
                "user_message": ai_response_content # 这里是红娘对用户说的
            })
            user_message_content = user_response.content.strip()
            
            user_message_entry = {
                "role": "user",
                "content": user_message_content,
                "timestamp": datetime.now()
            }
            conversation_history.append(user_message_entry)
            chat_history_lc.append(HumanMessage(content=user_message_content))
            print(f"    用户: {user_message_content}")

            # 4. AI红娘根据用户回复准备下一轮提问
            ai_response_obj = self.ai_chain.invoke({
                "persona_hint": user_persona_str,
                "conversation_history": self._format_history(conversation_history),
                "chat_history": chat_history_lc,
                "user_message": user_message_content # 这里是用户对红娘说的
            })
            ai_response_content = ai_response_obj.content.strip()
            
        # 存储最终对话记录
        db_manager.insert_onboarding_dialogue(user_id, conversation_history)
        print(f"  ✨ Onboarding 对话生成完成，共 {len(conversation_history)} 条消息。")
        return conversation_history

    def _format_persona(self, persona_data: Dict[str, Any]) -> str:
        """将 persona 字典格式化为 LLM 可读的字符串"""
        if not persona_data:
            return "无详细人设"
        
        formatted_str = []
        for key, value in persona_data.items():
            if isinstance(value, list):
                formatted_str.append(f"{key}: {', '.join(value)}")
            elif isinstance(value, dict):
                formatted_str.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            else:
                formatted_str.append(f"{key}: {value}")
        return "\n".join(formatted_str)

    def _format_history(self, conversation: List[Dict]) -> str:
        """将对话历史格式化为 LLM 可读的字符串"""
        if not conversation:
            return "(对话刚开始)"
        
        formatted_history = []
        for msg in conversation[-10:]:
            role = "AI红娘" if msg["role"] == "ai" else "用户"
            formatted_history.append(f"{role}: {msg['content']}")
        return "\n".join(formatted_history)