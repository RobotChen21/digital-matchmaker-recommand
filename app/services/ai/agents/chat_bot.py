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
