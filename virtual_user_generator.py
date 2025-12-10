"""
AI Virtual User & Dialogue Data Generation System v2.0 (Production Grade)
生产级虚拟用户与对话数据生成系统

核心改进:
1. Turn-by-turn 多轮交互式对话生成
2. Persona-based 持久化性格模拟
3. 完整的用户画像种子 (persona_seed)
4. 正确的消息结构 (sender_id)
5. 结构化向量存储 + metadata
6. Conversation window chunking for RAG
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from bson import ObjectId
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from pymongo import MongoClient
import chromadb

from dialogue_termination_system import DialogueTerminationManager
from env_utils import BASE_URL, API_KEY


# ============================================================================
# 配置部分 (Configuration)
# ============================================================================

class Config:
    """系统配置"""
    # MongoDB 配置
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "digital_matchmaker"

    TEMPERATURE_USER = 0.9  # 用户生成温度更高,更随机
    TEMPERATURE_AI = 0.7  # AI 红娘更稳定

    # Chroma 配置 - 单一向量库
    CHROMA_PERSIST_DIR = "./chroma_db"
    CHROMA_COLLECTION_NAME = "dating_app_dialogues"

    # 生成数量配置
    NUM_USERS = 5
    MIN_ONBOARDING_TURNS = 12
    MAX_ONBOARDING_TURNS = 30
    MIN_CHAT_MESSAGES = 40
    MAX_CHAT_MESSAGES = 60

    # RAG Chunking 配置
    CONVERSATION_WINDOW_SIZE = 5  # 每个 chunk 包含 5 条消息
    CONVERSATION_OVERLAP = 2  # 窗口重叠 2 条


# ============================================================================
# 数据模型 (Enhanced Data Models)
# ============================================================================

class PersonaSeed(BaseModel):
    """用户性格种子 - LLM 生成对话的依据"""
    personality_traits: List[str] = Field(description="性格特征: 外向/内向/乐观/谨慎等")
    occupation: str = Field(description="职业")
    occupation_detail: str = Field(description="职业细节描述")
    interests: List[str] = Field(description="兴趣爱好")
    relationship_history: str = Field(description="感情经历")
    family_background: str = Field(description="家庭背景")
    values: List[str] = Field(description="价值观")
    communication_style: str = Field(description="沟通风格: 直接/委婉/幽默/严肃")
    emotional_stability: str = Field(description="情绪稳定性: 高/中/低")
    response_speed: str = Field(description="回复速度倾向: 快/中/慢")
    ideal_partner: str = Field(description="理想型描述")


class VirtualUser(BaseModel):
    """虚拟用户基础信息模型"""
    nickname: str = Field(description="用户昵称")
    gender: str = Field(description="性别: male/female/other")
    birthday: str = Field(description="生日,格式: YYYY-MM-DD")
    height: int = Field(description="身高(cm)")
    city: str = Field(description="城市")
    self_intro_raw: str = Field(description="自我介绍,不超过40字")
    persona_seed: PersonaSeed = Field(description="性格种子(不存入基础表)")


# ============================================================================
# MongoDB 连接 (Enhanced Database)
# ============================================================================

class MongoDBManager:
    """MongoDB 数据库管理器 - 增强版"""

    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.users_basic = self.db["users_basic"]
        self.users_persona = self.db["users_persona"]  # 新增: 性格种子表
        self.onboarding_dialogues = self.db["users_onboarding_dialogues"]
        self.chat_records = self.db["chat_records"]

    def insert_user_with_persona(self, user_data: Dict[str, Any],
                                 persona_data: Dict[str, Any]) -> ObjectId:
        """插入用户基础信息和性格种子"""
        # 插入基础信息
        user_basic = {k: v for k, v in user_data.items() if k != "persona_seed"}
        user_basic["created_at"] = datetime.now()
        result = self.users_basic.insert_one(user_basic)
        user_id = result.inserted_id

        # 插入性格种子(单独存储,用于生成对话)
        persona_doc = {
            "user_id": user_id,
            "persona": persona_data,
            "created_at": datetime.now()
        }
        self.users_persona.insert_one(persona_doc)

        return user_id

    def get_user_with_persona(self, user_id: ObjectId) -> Tuple[Dict, Dict]:
        """获取用户信息和性格种子"""
        user_basic = self.users_basic.find_one({"_id": user_id})
        persona_doc = self.users_persona.find_one({"user_id": user_id})
        persona = persona_doc["persona"] if persona_doc else {}
        return user_basic, persona

    def insert_onboarding_dialogue(self, user_id: ObjectId, messages: List[Dict]):
        """插入 onboarding 对话"""
        dialogue_data = {
            "user_id": user_id,
            "messages": messages,
            "updated_at": datetime.now()
        }
        self.onboarding_dialogues.insert_one(dialogue_data)

    def insert_chat_record(self, user_id: ObjectId, partner_id: ObjectId,
                           messages: List[Dict]):
        """插入聊天记录 - 增强版消息结构"""
        chat_data = {
            "user_id": user_id,
            "partner_id": partner_id,
            "messages": messages,
            "created_at": datetime.now()
        }
        self.chat_records.insert_one(chat_data)


# ============================================================================
# Module 1: 虚拟用户生成器 (User Generator with Persona)
# ============================================================================

class VirtualUserGenerator:
    """虚拟用户生成器 - 包含完整性格种子"""

    def __init__(self, llm):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=VirtualUser)

        self.prompt = ChatPromptTemplate.from_template(
            """请创建一个真实可信的虚拟人物,包括基础信息和完整的性格种子。

要求:
- 自我介绍不超过 40 字,自然真实
- 性格特征要具体且一致
- 职业描述要详细真实
- 兴趣爱好要多样化
- 感情经历要合理
- 沟通风格要明确
- 不要说"我是AI"或任何虚拟相关词汇

请输出纯 JSON 格式,不要任何解释或 Markdown 标记。

{format_instructions}

请生成一个真实的中国用户画像。"""
        )

    def generate_user(self) -> VirtualUser:
        """生成一个虚拟用户"""
        chain = self.prompt | self.llm

        response = chain.invoke({
            "format_instructions": self.parser.get_format_instructions()
        })

        # 解析 JSON
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()

        user_dict = json.loads(content)
        return VirtualUser(**user_dict)

    def generate_batch(self, count: int, db_manager: MongoDBManager) -> List[ObjectId]:
        """批量生成用户并存入数据库"""
        user_ids = []

        for i in range(count):
            try:
                user = self.generate_user()
                user_dict = user.model_dump()
                persona_dict = user_dict.pop("persona_seed")

                user_id = db_manager.insert_user_with_persona(user_dict, persona_dict)
                user_ids.append(user_id)
                print(f"✅ 用户 {i + 1}/{count} 生成成功: {user.nickname} (ID: {user_id})")
            except Exception as e:
                print(f"❌ 用户 {i + 1} 生成失败: {e}")

        return user_ids


# ============================================================================
# Module 2: Turn-by-Turn Onboarding 对话生成器
# ============================================================================

class TurnByTurnOnboardingGenerator:
    """Turn-by-turn 交互式 Onboarding 对话生成器 (支持智能终止)"""

    def __init__(self, llm_ai, llm_user,
                 termination_manager: Optional['DialogueTerminationManager'] = None):
        self.llm_ai = llm_ai  # AI 红娘 LLM
        self.llm_user = llm_user  # 虚拟用户 LLM
        self.termination_manager = termination_manager  # 终止管理器

        # AI 红娘的 Prompt
        self.ai_prompt = ChatPromptTemplate.from_template(
            """你是一位温柔、专业且敏锐的 AI 红娘，正在与新用户进行首次深度访谈。

当前对话历史:
{conversation_history}

🔍 你的核心任务是收集用户的【六大核心画像】信息：
1. **教育背景**：学历(本科/硕士等)、学校层次、专业。
2. **工作职业**：行业、职位、工作强度(是否忙碌)、收入水平大致范围。
3. **家庭背景**：
   - 基础结构：是否独生子女、兄弟姐妹情况。
   - 父母状况：父母身体是否健康、父母职业/退休情况。
   - 资产/经济：家庭经济条件（如房产情况、父母养老是否有压力等）。
4. **生活方式**：作息习惯、烟酒情况、社交频率、兴趣爱好。
5. **恋爱画像**：恋爱风格(粘人/独立)、依恋类型、过往情感经历。
6. **约会偏好**：理想型要求、绝对不能接受的点(Dealbreakers)、期望结婚时间。

💡 提问策略：
- **查漏补缺**：请检查对话历史，优先询问【尚未涉及】或【信息模糊】的板块。
- **高情商探寻**：涉及家庭资产和父母情况时，请务必礼貌委婉。例如，通过“和父母住一起吗”来侧面了解房产，或“父母退休生活丰富吗”来侧面了解经济压力。
- **自然过渡**：话题之间要流畅衔接，避免像查户口一样生硬。

用户的隐藏性格特征(你不知道,但要通过对话探索):
{persona_hint}

现在是第 {turn} 轮对话，请生成 AI 红娘的下一句回复(只输出内容):"""
        )

        # 虚拟用户的 Prompt
        self.user_prompt = ChatPromptTemplate.from_template(
            """你是一个真实的用户,正在与 AI 红娘聊天。

你的基础信息:
- 昵称: {nickname}
- 性别: {gender}
- 年龄: {age}
- 城市: {city}

你的性格特征:
{persona}

对话历史:
{conversation_history}

AI 红娘刚才说: {ai_message}

请根据你的性格特征回复(注意:要自然、有情绪、可能犹豫、可能跳跃话题)。
只输出你的回复内容,不要解释:"""
        )

    def generate_dialogue(self, user_basic: Dict, persona: Dict,
                          min_turns: int = 8, max_turns: int = 20) -> List[Dict]:
        """生成 turn-by-turn 对话 (支持智能终止)"""

        conversation_history = []
        messages = []

        # 计算年龄
        birthday = datetime.strptime(user_basic["birthday"], "%Y-%m-%d")
        age = (datetime.now() - birthday).days // 365

        # 格式化 persona
        persona_text = self._format_persona(persona)
        persona_hint = f"性格: {', '.join(persona.get('personality_traits', []))}"

        current_time = datetime.now()

        for turn in range(max_turns):
            # 1. AI 红娘提问
            if turn == 0:
                ai_message = "你好呀,我是你的专属红娘小助手~很高兴认识你!能先简单聊聊你自己吗?"
            else:
                ai_chain = self.ai_prompt | self.llm_ai
                ai_response = ai_chain.invoke({
                    "conversation_history": self._format_history(conversation_history),
                    "persona_hint": persona_hint,
                    "turn": turn + 1
                })
                ai_message = ai_response.content.strip()

            conversation_history.append({"role": "ai", "content": ai_message})
            messages.append({
                "role": "ai",
                "content": ai_message,
                "timestamp": current_time
            })
            current_time += timedelta(minutes=1)

            # 2. 用户回复
            user_chain = self.user_prompt | self.llm_user
            user_response = user_chain.invoke({
                "nickname": user_basic["nickname"],
                "gender": user_basic["gender"],
                "age": age,
                "city": user_basic["city"],
                "persona": persona_text,
                "conversation_history": self._format_history(conversation_history[:-1]),
                "ai_message": ai_message
            })
            user_message = user_response.content.strip()

            conversation_history.append({"role": "user", "content": user_message})
            messages.append({
                "role": "user",
                "content": user_message,
                "timestamp": current_time
            })
            current_time += timedelta(minutes=random.randint(1, 3))

            # 3. 智能终止检测
            if self.termination_manager and turn >= min_turns - 1:
                should_end, signal = self.termination_manager.should_terminate_onboarding(
                    conversation_history,
                    min_turns=min_turns,
                    max_turns=max_turns
                )

                if should_end:
                    print(f"  ⚡ 对话提前结束: {signal.reason} (置信度: {signal.confidence:.2f})")
                    print(f"     {signal.explanation}")

                    # 添加礼貌结束语
                    if signal.reason in ["user_hesitant", "user_tired"]:
                        closing = "好的,那我们今天就先聊到这里吧~有什么想聊的随时来找我!"
                    elif signal.reason == "info_collected":
                        closing = "好的!我已经对你有了基本了解,后续我会为你推荐合适的人选~"
                    else:
                        closing = "今天聊得很开心!期待下次和你聊天~"

                    messages.append({
                        "role": "ai",
                        "content": closing,
                        "timestamp": current_time
                    })
                    break

        return messages

    def _format_persona(self, persona: Dict) -> str:
        """格式化 persona 为文本"""
        lines = []
        for key, value in persona.items():
            if isinstance(value, list):
                lines.append(f"{key}: {', '.join(value)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        if not history:
            return "(对话刚开始)"

        lines = []
        for msg in history[-6:]:  # 只保留最近 6 条
            role = "AI" if msg["role"] == "ai" else "用户"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def generate_for_user(self, user_id: ObjectId, db_manager: MongoDBManager,
                          min_turns: int = 8, max_turns: int = 20):
        """为指定用户生成 turn-by-turn onboarding 对话 (支持智能终止)"""
        user_basic, persona = db_manager.get_user_with_persona(user_id)

        if not user_basic:
            print(f"❌ 用户 {user_id} 不存在")
            return

        try:
            messages = self.generate_dialogue(user_basic, persona, min_turns, max_turns)
            db_manager.insert_onboarding_dialogue(user_id, messages)
            print(f"✅ {user_basic['nickname']} 的 onboarding 对话生成成功 ({len(messages)} 条消息)")
        except Exception as e:
            print(f"❌ {user_basic['nickname']} 的对话生成失败: {e}")


# ============================================================================
# Module 3: Persona-based Social Chat Generator
# ============================================================================

class PersonaBasedChatGenerator:
    """基于 Persona 的社交对话生成器 (支持智能终止)"""

    def __init__(self, llm,
                 termination_manager: Optional['DialogueTerminationManager'] = None):
        self.llm = llm
        self.termination_manager = termination_manager

        self.user_prompt = ChatPromptTemplate.from_template(
            """你是用户 {nickname},正在与 {partner_nickname} 聊天。

你的性格特征:
{persona}

对话历史:
{conversation_history}

{partner_nickname} 刚才说: {partner_message}

请根据你的性格特征回复(要自然、有情绪、符合你的沟通风格)。
只输出回复内容:"""
        )

    def generate_chat(self, user_a_data: Tuple[Dict, Dict],
                      user_b_data: Tuple[Dict, Dict],
                      min_messages: int = 20, max_messages: int = 60) -> List[Dict]:
        """生成两个用户之间的 persona-based 聊天 (支持智能终止)"""

        user_a_basic, persona_a = user_a_data
        user_b_basic, persona_b = user_b_data

        conversation_history = []
        messages = []
        current_time = datetime.now()

        # 第一条消息: 用户A 主动打招呼
        first_message = self._generate_greeting(user_a_basic["nickname"], persona_a)

        conversation_history.append({
            "sender": user_a_basic["_id"],
            "content": first_message
        })
        messages.append({
            "sender_id": user_a_basic["_id"],
            "receiver_id": user_b_basic["_id"],
            "content": first_message,
            "timestamp": current_time
        })
        current_time += self._get_response_delay(persona_b)

        # 交替生成对话
        for i in range(1, max_messages):
            # 确定当前发言者
            is_a_turn = (i % 2 == 1)

            if is_a_turn:
                current_user = user_b_basic
                current_persona = persona_b
                partner_user = user_a_basic
                last_message = conversation_history[-1]["content"]
            else:
                current_user = user_a_basic
                current_persona = persona_a
                partner_user = user_b_basic
                last_message = conversation_history[-1]["content"]

            # 生成回复
            response = self._generate_response(
                current_user, current_persona,
                partner_user, last_message,
                conversation_history
            )

            conversation_history.append({
                "sender": current_user["_id"],
                "content": response
            })
            messages.append({
                "sender_id": current_user["_id"],
                "receiver_id": partner_user["_id"],
                "content": response,
                "timestamp": current_time
            })

            # 根据性格决定回复延迟
            current_time += self._get_response_delay(current_persona)

            # 智能终止检测
            if self.termination_manager and i >= min_messages:
                should_end, signal = self.termination_manager.should_terminate_social_chat(
                    messages,
                    min_messages=min_messages,
                    max_messages=max_messages
                )

                if should_end:
                    print(f"  ⚡ 聊天自然结束: {signal.reason} (置信度: {signal.confidence:.2f})")
                    print(f"     {signal.explanation}")
                    break

        return messages

    def _generate_greeting(self, nickname: str, persona: Dict) -> str:
        """生成打招呼消息"""
        style = persona.get("communication_style", "")

        if "幽默" in style:
            greetings = ["Hi~看到你的资料很有趣!", "嗨!终于匹配到你了😊", "Hello~"]
        elif "直接" in style:
            greetings = ["你好", "Hi", "在吗?"]
        else:
            greetings = ["你好呀~", "Hi,很高兴认识你", "嗨~"]

        return random.choice(greetings)

    def _generate_response(self, user: Dict, persona: Dict,
                           partner: Dict, partner_message: str,
                           history: List[Dict]) -> str:
        """根据 persona 生成回复"""

        chain = self.user_prompt | self.llm

        response = chain.invoke({
            "nickname": user["nickname"],
            "partner_nickname": partner["nickname"],
            "persona": self._format_persona(persona),
            "conversation_history": self._format_history(history),
            "partner_message": partner_message
        })

        return response.content.strip()

    def _format_persona(self, persona: Dict) -> str:
        """格式化 persona"""
        lines = []
        for key, value in persona.items():
            if isinstance(value, list):
                lines.append(f"{key}: {', '.join(value)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        if not history:
            return "(对话刚开始)"

        lines = []
        for msg in history[-8:]:
            lines.append(f"消息: {msg['content']}")
        return "\n".join(lines)

    def _get_response_delay(self, persona: Dict) -> timedelta:
        """根据 persona 获取回复延迟"""
        speed = persona.get("response_speed", "中")

        if speed == "快":
            minutes = random.randint(1, 3)
        elif speed == "慢":
            minutes = random.randint(5, 10)
        else:
            minutes = random.randint(2, 5)

        return timedelta(minutes=minutes)

    def generate_chat_pair(self, user_a_id: ObjectId, user_b_id: ObjectId,
                           db_manager: MongoDBManager,
                           min_msgs: int = 20, max_msgs: int = 60):
        """为两个用户生成聊天 (支持智能终止)"""
        user_a_data = db_manager.get_user_with_persona(user_a_id)
        user_b_data = db_manager.get_user_with_persona(user_b_id)

        try:
            messages = self.generate_chat(user_a_data, user_b_data, min_msgs, max_msgs)
            db_manager.insert_chat_record(user_a_id, user_b_id, messages)
            print(f"✅ {user_a_data[0]['nickname']} ↔ {user_b_data[0]['nickname']} "
                  f"聊天生成成功 ({len(messages)} 条)")
        except Exception as e:
            print(f"❌ 聊天生成失败: {e}")


# ============================================================================
# Chroma 向量数据库管理 - 单库多过滤器设计
# ============================================================================

class EnhancedChromaManager:
    """增强版 Chroma 管理器 - 单库 + metadata 过滤"""

    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """获取或创建集合"""
        try:
            return self.client.get_collection(name=self.collection_name)
        except:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Dating app dialogues with metadata"}
            )

    def add_conversation_chunks(self, user_id: str, messages: List[Dict],
                                dialogue_type: str, window_size: int = 5,
                                overlap: int = 2):
        """使用滑动窗口添加对话块"""
        chunks = self._create_conversation_windows(messages, window_size, overlap)

        documents = []
        metadatas = []
        ids = []

        for idx, chunk in enumerate(chunks):
            # 构建文档文本
            doc_text = self._format_chunk(chunk)
            documents.append(doc_text)

            # 构建 metadata
            metadata = {
                "user_id": user_id,
                "dialogue_type": dialogue_type,
                "chunk_index": idx,
                "timestamp": chunk[0].get("timestamp", datetime.now()).isoformat(),
                "num_messages": len(chunk)
            }

            # 如果是社交聊天,添加 sender 信息
            if dialogue_type == "social" and "sender_id" in chunk[0]:
                metadata["sender_id"] = str(chunk[0]["sender_id"])

            metadatas.append(metadata)
            ids.append(f"{user_id}_{dialogue_type}_{idx}")

        # 批量添加
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ 为用户 {user_id} 添加 {len(documents)} 个对话窗口 ({dialogue_type})")

    def _create_conversation_windows(self, messages: List[Dict],
                                     window_size: int, overlap: int) -> List[List[Dict]]:
        """创建滑动窗口切片"""
        windows = []
        step = window_size - overlap

        for i in range(0, len(messages), step):
            window = messages[i:i + window_size]
            if len(window) >= 2:  # 至少2条消息才有意义
                windows.append(window)

        return windows

    def _format_chunk(self, chunk: List[Dict]) -> str:
        """格式化对话窗口为文本"""
        lines = []
        for msg in chunk:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def query_user_dialogues(self, user_id: str, query_text: str,
                             dialogue_type: Optional[str] = None,
                             n_results: int = 5) -> Dict:
        """查询用户对话"""
        where_filter = {"user_id": user_id}
        if dialogue_type:
            where_filter["dialogue_type"] = dialogue_type

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        return results


# ============================================================================
# 主流程 (Enhanced Pipeline)
# ============================================================================

class EnhancedDataGenerationPipeline:
    """增强版数据生成主流程 (支持智能终止)"""

    def __init__(self, config: Config):
        self.config = config

        # 初始化两个 LLM (不同温度)
        # self.llm_ai = Tongyi(
        #     model="qwen-plus",  # 或 qwen-turbo, qwen-plus
        #     model_kwargs={"temperature": config.TEMPERATURE_AI},
        #     api_key=API_KEY,
        # )
        #
        # self.llm_user = Tongyi(
        #     model="deepseek-v3.1",  # 或 qwen-turbo, qwen-plus
        #     api_key=API_KEY,
        # )

        self.llm_ai = ChatOpenAI(
            model="deepseek-v3.1",
            temperature=config.TEMPERATURE_AI,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        self.llm_user = ChatOpenAI(
            model="deepseek-v3.1",
            temperature=config.TEMPERATURE_USER,
            api_key=API_KEY,
            base_url=BASE_URL
        )

        # 初始化数据库
        self.db_manager = MongoDBManager(config.MONGO_URI, config.DB_NAME)
        self.chroma_manager = EnhancedChromaManager(
            config.CHROMA_PERSIST_DIR,
            config.CHROMA_COLLECTION_NAME
        )

        # 初始化终止管理器
        # 导入终止系统(如果存在)
        try:
            from dialogue_termination_system import DialogueTerminationManager
            self.termination_manager = DialogueTerminationManager(self.llm_ai)
            print("✅ 智能对话终止系统已启用")
        except:
            self.termination_manager = None
            print("⚠️  智能对话终止系统未启用(使用固定轮数)")

        # 初始化生成器
        self.user_gen = VirtualUserGenerator(self.llm_user)
        self.onboarding_gen = TurnByTurnOnboardingGenerator(
            self.llm_ai, self.llm_user, self.termination_manager
        )
        self.chat_gen = PersonaBasedChatGenerator(self.llm_user, self.termination_manager)

    def run_full_pipeline(self):
        """运行完整数据生成流程"""
        print("🚀 开始生产级数据生成流程...")
        print("=" * 70)

        # Step 1: 生成虚拟用户(带完整 persona)
        print("\n📝 Step 1: 生成虚拟用户(包含性格种子)")
        user_ids = self.user_gen.generate_batch(
            self.config.NUM_USERS,
            self.db_manager
        )

        # Step 2: Turn-by-turn 生成 onboarding 对话
        print("\n💬 Step 2: Turn-by-turn 生成 AI 红娘对话")
        for user_id in user_ids:
            self.onboarding_gen.generate_for_user(
                user_id,
                self.db_manager,
                self.config.MIN_ONBOARDING_TURNS,
                self.config.MAX_ONBOARDING_TURNS
            )

        # Step 3: Persona-based 生成用户间聊天 (新老用户混合配对)
        print("\n💑 Step 3: Persona-based 生成用户间聊天 (新老混合蜘蛛网)")
        
        # 1. 获取本次生成的新用户 (New Users)
        new_users_data = list(self.db_manager.users_basic.find({"_id": {"$in": user_ids}}))
        
        # 2. 获取数据库中已有的老用户 (Existing Users) - 可以限制数量，比如最近活跃的 100 人
        # 这里为了演示，我们获取全量用户，实际生产中应加 limit 或按活跃度排序
        all_users_data = list(self.db_manager.users_basic.find({}))
        
        print(f"   - 本次新增用户: {len(new_users_data)} 人")
        print(f"   - 全库用户池: {len(all_users_data)} 人")

        # 3. 为每个新用户匹配聊天对象 (可能是新用户，也可能是老用户)
        # 设定目标: 每个新用户至少要聊 2-3 场
        CHATS_PER_NEW_USER = 3
        
        generated_count = 0
        
        for new_user in new_users_data:
            my_id = new_user["_id"]
            my_gender = new_user.get("gender")
            my_name = new_user.get("nickname")
            
            # 在全库中寻找异性 (排除自己)
            potential_partners = [
                u for u in all_users_data 
                if u.get("gender") != my_gender and u["_id"] != my_id
            ]
            
            if not potential_partners:
                print(f"   ⚠️ {my_name} ({my_gender}) 没找到异性对象，跳过")
                continue
            
            # 随机抽取 N 个对象
            num_to_chat = min(len(potential_partners), CHATS_PER_NEW_USER)
            partners = random.sample(potential_partners, num_to_chat)
            
            for partner in partners:
                # 检查是否已经聊过 (避免重复生成)
                # 注意: 需要双向检查 (A,B) 或 (B,A)
                existing_chat = self.db_manager.chat_records.find_one({
                    "$or": [
                        {"user_id": my_id, "partner_id": partner["_id"]},
                        {"user_id": partner["_id"], "partner_id": my_id}
                    ]
                })
                
                if existing_chat:
                    # 已经聊过了，跳过
                    continue
                
                # 生成聊天
                print(f"   💬 生成: 新用户 [{my_name}] ↔ [{'老' if partner['_id'] not in user_ids else '新'}] 用户 [{partner['nickname']}]")
                self.chat_gen.generate_chat_pair(
                    my_id,
                    partner["_id"],
                    self.db_manager,
                    self.config.MIN_CHAT_MESSAGES,
                    self.config.MAX_CHAT_MESSAGES
                )
                generated_count += 1
        
        print(f"   ✨ 实际生成了 {generated_count} 场新老混合聊天")

        # Step 4: 构建单一向量数据库(window chunking)
        print("\n🔍 Step 4: 构建向量数据库(使用窗口切片)")
        self._build_vector_db(user_ids)

        print("\n✨ 数据生成完成!")
        print("=" * 70)

    def _build_vector_db(self, user_ids: List[ObjectId]):
        """为所有用户构建向量数据库"""
        for user_id in user_ids:
            user_basic, persona = self.db_manager.get_user_with_persona(user_id)

            # 添加 onboarding 对话
            onboarding = self.db_manager.onboarding_dialogues.find_one({"user_id": user_id})
            if onboarding:
                self.chroma_manager.add_conversation_chunks(
                    str(user_id),
                    onboarding["messages"],
                    "onboarding",
                    self.config.CONVERSATION_WINDOW_SIZE,
                    self.config.CONVERSATION_OVERLAP
                )

            # 添加聊天记录
            chats = self.db_manager.chat_records.find({"user_id": user_id})
            for chat in chats:
                self.chroma_manager.add_conversation_chunks(
                    str(user_id),
                    chat["messages"],
                    "social",
                    self.config.CONVERSATION_WINDOW_SIZE,
                    self.config.CONVERSATION_OVERLAP
                )


# ============================================================================
# 使用示例 (Usage Example)
# ============================================================================

if __name__ == "__main__":
    # 创建配置
    config = Config()

    # 创建流程
    pipeline = EnhancedDataGenerationPipeline(config)

    # 运行完整流程
    pipeline.run_full_pipeline()

    # 示例: 查询向量数据库
    print("\n" + "=" * 70)
    print("🔍 向量查询示例")
    print("=" * 70)

    # 假设查询第一个用户的对话
    # user_id = "some_user_id"
    # results = pipeline.chroma_manager.query_user_dialogues(
    #     user_id=user_id,
    #     query_text="工作和职业",
    #     dialogue_type="onboarding",
    #     n_results=3
    # )
    # print(results)