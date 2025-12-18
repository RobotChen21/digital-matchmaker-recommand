# -*- coding: utf-8 -*-
import json
from typing import Dict, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.common.models.termination import TerminationReason, TerminationSignal
from app.services.ai.agents.profile_manager import ProfileService # 导入 ProfileService
from app.common.models.profile import REQUIRED_PROFILE_DIMENSIONS # 导入公共常量

class HesitancyDetector:
    """检测用户是否不想继续对话"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=TerminationSignal)

        self.prompt = ChatPromptTemplate.from_template(
            """你是一个心理分析专家,专门分析用户的对话意愿。

对话历史(最近5轮):
{conversation_history}

用户最新回复:
"{user_message}"

请分析用户是否显示出以下信号:
1. 犹豫/敷衍: 回复很短、"嗯"、"还好"、"随便"
2. 回避: 不愿深入话题、转移话题、回答含糊
3. 疲惫: "有点累了"、"改天聊"、回复间隔变长
4. 不耐烦: "就这样吧"、"差不多了"、语气变冷
5. 明确拒绝: "不想说"、"不太方便"、"下次再聊"

请输出 JSON 格式,不要任何解释或 Markdown 标记。

{format_instructions}"""
        )
    
    def detect(self, user_message: str, conversation_history: List[Dict]) -> TerminationSignal:
        chain = self.prompt | self.llm
        response = chain.invoke({
            "user_message": user_message,
            "conversation_history": self._format_history(conversation_history),
            "format_instructions": self.parser.get_format_instructions()
        })
        return self._parse_response(response.content)
    
    def _format_history(self, history: List[Dict]) -> str:
        lines = []
        for msg in history[-5:]:
            role = "AI" if msg.get("role") == "ai" else "用户"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)
    
    def _parse_response(self, content: str) -> TerminationSignal:
        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            data = json.loads(content)
            return TerminationSignal(**data)
        except json.JSONDecodeError as e:
            print(f"❌ HesitancyDetector JSON parsing failed: {e}")
            print(f"   Original content: {content}")
            return TerminationSignal(should_terminate=False, reason=None, confidence=0.0, explanation=f"JSON解析失败: {e}")
        except Exception as e:
            print(f"❌ HesitancyDetector general parsing failed: {e}")
            print(f"   Original content: {content}")
            return TerminationSignal(should_terminate=False, reason=None, confidence=0.0, explanation=f"解析失败: {e}")


class InfoCompletenessDetector:
    """检测 Onboarding 信息是否收集完成"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=TerminationSignal)
        # 精细化需要收集的维度 (仅供 LLM 参考，实际判断基于 profile_summary)
        self.required_dimensions_for_prompt = REQUIRED_PROFILE_DIMENSIONS
        
        self.prompt = ChatPromptTemplate.from_template(
            """你是 AI 红娘的数据质量官，正在评估用户画像数据是否已充分收集。

【用户当前画像概要】:
{profile_completion_hint}

【必须收集的核心维度 - 完整列表】:
{required_dimensions}

请严格评估:
1. 哪些维度已经**充分收集**? (例如: 明确知道是硕士、985、程序员、年薪50w+、独生子女、父母退休健康、家庭富裕)
2. 哪些维度**还缺失**或**太模糊**? (例如: 只说了有弟弟，没说父母情况; 或者只说了工作，没说收入范围)
3. 是否可以结束访谈?

【评估标准】:
- **最重要原则**：如果【用户当前画像概要】中已经明确提到 "核心画像已完善" 或类似表述，请直接判为 `should_terminate=True`，不要再纠结细节。
- **Hard Constraints (仅在画像概要未明确说完善时检查)**:
  1. **教育背景**: 学历, 学校类型 (985/211/海外/双非)
  2. **工作职业**: 职位/行业。**特殊例外**：如果用户是**学生/在读**，则【工作风格】和【收入水平】**不需要**收集。
  3. **家庭背景**: 兄弟姐妹情况, 父母大致情况 (健康/职业)。**例外**：如果用户已概括描述家庭和谐/经济良好，不需要死抠每一个细节。
- 只有当以上所有强制维度都已相对清晰，允许结束访谈。
- 其他非强制核心维度 (如兴趣、价值观、恋爱观) 尽量收集，但不卡流程。

请输出 JSON 格式,不要任何解释或 Markdown 标记。

{format_instructions}"""
        )
    
    def detect(self, profile_completion_hint_text: str) -> TerminationSignal:
        """
        根据结构化画像摘要来判断是否结束 Onboarding。
        直接接收外部生成好的 hint text，避免重复调用 LLM。
        """
        chain = self.prompt | self.llm
        response = chain.invoke({
            "profile_completion_hint": profile_completion_hint_text,
            "required_dimensions": "\n".join(self.required_dimensions_for_prompt),
            "format_instructions": self.parser.get_format_instructions()
        })
        return self._parse_response(response.content)
    
    def _format_conversation(self, conversation: List[Dict]) -> str:
        lines = []
        for msg in conversation:
            role = "AI" if msg.get("role") == "ai" else "用户"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)
    
    def _parse_response(self, content: str) -> TerminationSignal:
        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            data = json.loads(content)
            return TerminationSignal(should_terminate=data["should_terminate"], reason=data.get("reason"), confidence=data["confidence"], explanation=data.get("explanation", ""))
        except json.JSONDecodeError as e:
            print(f"❌ InfoCompletenessDetector JSON parsing failed: {e}")
            print(f"   Original content: {content}")
            return TerminationSignal(should_terminate=False, reason=None, confidence=0.0, explanation=f"JSON解析失败: {e}")
        except Exception as e:
            print(f"❌ InfoCompletenessDetector general parsing failed: {e}")
            print(f"   Original content: {content}")
            return TerminationSignal(should_terminate=False, reason=None, confidence=0.0, explanation=f"解析失败: {e}")


class DialogueTerminationManager:
    """综合管理对话终止逻辑"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.hesitancy_detector = HesitancyDetector(llm)
        self.info_detector = InfoCompletenessDetector(llm)
    
    def should_terminate_onboarding(self, profile_completion_hint_text: str, full_conversation: List[Dict], min_conversational_turns: int = 8, max_turns: int = 30) -> Tuple[bool, TerminationSignal]:
        # min_conversational_turns 用来确保至少聊了几句才结束，避免开场白就说全了
        
        # 对话轮数检查 (确保聊了一段时间)
        num_turns = len(full_conversation) // 2
        if num_turns >= max_turns:
            return True, TerminationSignal(should_terminate=True, reason=TerminationReason.MAX_TURNS, confidence=1.0, explanation=f"达到最大轮数 {max_turns}")
        if num_turns < min_conversational_turns:
            return False, TerminationSignal(should_terminate=False, reason=None, confidence=1.0, explanation=f"对话不足 {min_conversational_turns} 轮")
            
        # 优先判断用户是否不想聊了
        if len(full_conversation) >= 2:
            last_user_msg = None
            for msg in reversed(full_conversation):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
            if last_user_msg:
                hesitancy_signal = self.hesitancy_detector.detect(last_user_msg, full_conversation)
                if hesitancy_signal.should_terminate and hesitancy_signal.confidence > 0.7:
                    return True, hesitancy_signal
        
        # 检查信息完整度 (主要逻辑)
        # 直接使用传入的 hint text 进行判断
        info_signal = self.info_detector.detect(profile_completion_hint_text)
        if info_signal.should_terminate and info_signal.confidence > 0.8:
            return True, info_signal
            
        return False, TerminationSignal(should_terminate=False, reason=None, confidence=0.0, explanation="继续收集信息")