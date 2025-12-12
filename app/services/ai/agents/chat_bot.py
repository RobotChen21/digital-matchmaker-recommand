# -*- coding: utf-8 -*-
import random
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from bson import ObjectId
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.common.models.user import VirtualUser
from app.db.mongo_manager import MongoDBManager
from app.services.ai.tools.termination import DialogueTerminationManager

class PersonaBasedChatGenerator:
    """基于角色的社交聊天生成器"""

    def __init__(self, llm_user: ChatOpenAI, termination_manager: DialogueTerminationManager):
        self.llm_user = llm_user
        self.termination_manager = termination_manager

        # 用户 A 的 Prompt
        self.user_a_prompt = ChatPromptTemplate.from_messages([
            ("system", """你现在扮演一个真实的交友用户，请根据你的性格种子和与对方的对话历史进行交流。
你的目标是像真实用户一样自然地聊天，可以分享日常，表达情感，也可以提问。
聊天时请注意保持你的 {user_a_persona} 人设。
不要过度热情，也不要过于冷淡，像普通人一样有来有回。
请在聊天中自然地展现你的兴趣、价值观、生活方式等。
你正在和 {partner_name} 聊天。

{conversation_history}"""),
            MessagesPlaceholder(variable_name="chat_history"), # 聊天历史
            ("human", "{user_message}") # 用户A的新消息
        ])
        self.user_a_chain = self.user_a_prompt | self.llm_user

        # 用户 B 的 Prompt (与 A 类似，只是角色互换)
        self.user_b_prompt = ChatPromptTemplate.from_messages([
            ("system", """你现在扮演一个真实的交友用户，请根据你的性格种子和与对方的对话历史进行交流。
你的目标是像真实用户一样自然地聊天，可以分享日常，表达情感，也可以提问。
聊天时请注意保持你的 {user_b_persona} 人设。
不要过度热情，也不要过于冷淡，像普通人一样有来有回。
请在聊天中自然地展现你的兴趣、价值观、生活方式等。
你正在和 {partner_name} 聊天。

{conversation_history}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_message}")
        ])
        self.user_b_chain = self.user_b_prompt | self.llm_user


    def generate_chat_pair(self, # <--- 补上了 self
                           user_a_id: ObjectId,
                           user_b_id: ObjectId,
                           db_manager: MongoDBManager,
                           min_messages: int = 40,
                           max_messages: int = 60) -> List[Dict]:
        """
        生成两个用户之间的聊天记录。
        """
        user_a_basic, user_a_persona_data = db_manager.get_user_with_persona(user_a_id)
        user_b_basic, user_b_persona_data = db_manager.get_user_with_persona(user_b_id)

        user_a_persona_str = self._format_persona(user_a_persona_data)
        user_b_persona_str = self._format_persona(user_b_persona_data)

        chat_history = []
        full_conversation_for_termination = [] # 用于判断终止
        
        current_speaker = random.choice([user_a_id, user_b_id])
        last_message_content = "你好，很高兴认识你！" # 初始开场白

        for turn in range(max_messages):
            sender_id = current_speaker
            receiver_id = user_b_id if current_speaker == user_a_id else user_a_id
            
            sender_basic = user_a_basic if sender_id == user_a_id else user_b_basic
            sender_persona = user_a_persona_str if sender_id == user_a_id else user_b_persona_str
            sender_chain = self.user_a_chain if sender_id == user_a_id else self.user_b_chain

            partner_basic = user_b_basic if receiver_id == user_b_id else user_a_basic
            partner_name = partner_basic['nickname']

            # 判断是否需要终止对话
            if self.termination_manager:
                should_terminate, signal = self.termination_manager.should_terminate_social_chat(
                    full_conversation_for_termination, min_messages, max_messages
                )
                if should_terminate:
                    print(f"  ✅ 聊天自然结束: {signal.reason} (置信度: {signal.confidence:.2f})")
                    break
            
            # 生成消息
            response = sender_chain.invoke({
                "user_a_persona": sender_persona if sender_id == user_a_id else None,
                "user_b_persona": sender_persona if sender_id == user_b_id else None,
                "partner_name": partner_name,
                "conversation_history": self._format_chat_history(full_conversation_for_termination),
                "chat_history": chat_history,
                "user_message": last_message_content
            })
            
            new_message_content = response.content.strip()
            
            # 存储消息
            message_entry = {
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "content": new_message_content,
                "timestamp": datetime.now()
            }
            full_conversation_for_termination.append(message_entry)
            chat_history.append(HumanMessage(content=last_message_content) if sender_id != current_speaker else AIMessage(content=last_message_content))
            chat_history.append(AIMessage(content=new_message_content) if sender_id == current_speaker else HumanMessage(content=new_message_content))

            last_message_content = new_message_content
            current_speaker = receiver_id # 交换发言人

        # 存储最终聊天记录
        db_manager.insert_chat_record(user_a_id, user_b_id, full_conversation_for_termination)
        return full_conversation_for_termination

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

    def _format_chat_history(self, conversation: List[Dict]) -> str:
        """将对话历史格式化为 LLM 可读的字符串"""
        if not conversation:
            return "(对话刚开始)"
        
        formatted_history = []
        for msg in conversation[-10:]:
            sender_role = "用户A" if str(msg["sender_id"]) == "current_user_id_placeholder" else "用户B"
            formatted_history.append(f"{sender_role}: {msg['content']}")
        return "\n".join(formatted_history)
