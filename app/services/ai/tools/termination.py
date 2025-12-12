# -*- coding: utf-8 -*-
import json
from typing import Dict, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.common.models.termination import TerminationReason, TerminationSignal

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
        if not history:
            return "(对话刚开始)"
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
        self.required_dimensions = [
            "教育背景 (学历/学校)",
            "工作职业 (行业/忙碌程度)",
            "家庭背景 (独生/父母/资产情况)",
            "兴趣爱好 (具体的活动/投入程度)",
            "核心价值观 (家庭观/事业观/金钱观)",
            "生活方式 (烟酒/作息)",
            "恋爱风格 (依恋类型/粘人程度)",
            "约会偏好 (理想型/雷点)"
        ]
        
        self.prompt = ChatPromptTemplate.from_template(
            """你是 AI 红娘的数据质量官，正在评估是否已经收集到足够的用户画像数据。

必须收集的核心维度:
{required_dimensions}

完整对话记录:
{full_conversation}

请严格评估:
1. 哪些维度已经**充分收集**? (例如: 明确知道是本科、不抽烟、独生子女、家庭经济无负担)
2. 哪些维度**还缺失**或**太模糊**? (例如: 只说了有弟弟，没说父母情况; 或者只说了工作，没说收入范围)
3. 是否可以结束访谈?

评估标准:
- 必须覆盖至少 6/8 个核心维度。
- 对于"缺失"的维度，必须是用户明确拒绝回答或无法获取，否则应继续询问。

请输出 JSON 格式,不要任何解释或 Markdown 标记。

{format_instructions}"""
        )
    
    def detect(self, full_conversation: List[Dict], min_turns: int = 8) -> TerminationSignal:
        if len(full_conversation) < min_turns * 2:
            return TerminationSignal(should_terminate=False, reason=None, confidence=1.0, explanation=f"对话不足 {min_turns} 轮")
        
        chain = self.prompt | self.llm
        response = chain.invoke({
            "required_dimensions": ", ".join(self.required_dimensions),
            "full_conversation": self._format_conversation(full_conversation),
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
            return TerminationSignal(should_terminate=data["should_terminate"], reason=data.get("reason"), confidence=data["confidence"], explanation=data["explanation"])
        except json.JSONDecodeError as e:
            print(f"❌ InfoCompletenessDetector JSON parsing failed: {e}")
            print(f"   Original content: {content}")
            return TerminationSignal(should_terminate=False, reason=None, confidence=0.0, explanation=f"JSON解析失败: {e}")
        except Exception as e:
            print(f"❌ InfoCompletenessDetector general parsing failed: {e}")
            print(f"   Original content: {content}")
            return TerminationSignal(should_terminate=False, reason=None, confidence=0.0, explanation=f"解析失败: {e}")


class NaturalEndDetector:
    """检测社交对话是否自然结束"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=TerminationSignal)

        self.prompt = ChatPromptTemplate.from_template(
            """你是对话分析专家,判断两个人的聊天是否到了自然结束点。

最近对话(最后8条):
{recent_conversation}

完整对话统计:
- 总消息数: {total_messages}
- 持续时间: {duration}

请分析是否出现以下信号:
1. 话题耗尽: 开始重复、无新话题、沉默增多
2. 礼貌结束: "今天聊得很开心"、"改天再聊"、"要去忙了"
3. 约定后续: "那我们周末见"、"加个微信吧"
4. 自然收尾: 相互告别、对话完整闭环
5. 冷场: 连续简短回复、"嗯嗯"、"好的"

注意: 20条消息以下不应该结束(还在热聊期)

请输出 JSON 格式,不要任何解释或 Markdown 标记。

{format_instructions}"""
        )
    
    def detect(self, full_conversation: List[Dict], min_messages: int = 20) -> TerminationSignal:
        if len(full_conversation) < min_messages:
            return TerminationSignal(should_terminate=False, reason=None, confidence=1.0, explanation=f"消息不足 {min_messages} 条")
        
        duration = self._calculate_duration(full_conversation)
        chain = self.prompt | self.llm
        response = chain.invoke({
            "recent_conversation": self._format_recent(full_conversation),
            "total_messages": len(full_conversation),
            "duration": duration,
            "format_instructions": self.parser.get_format_instructions()
        })
        return self._parse_response(response.content)
    
    def _format_recent(self, conversation: List[Dict]) -> str:
        lines = []
        for msg in conversation[-8:]:
            sender = f"用户{msg.get('sender_id', 'A')}"
            lines.append(f"{sender}: {msg.get('content', '')}")
        return "\n".join(lines)
    
    def _calculate_duration(self, conversation: List[Dict]) -> str:
        if not conversation or len(conversation) < 2:
            return "刚开始"
        first_ts = conversation[0].get("timestamp")
        last_ts = conversation[-1].get("timestamp")
        if first_ts and last_ts:
            if isinstance(first_ts, str):
                from datetime import datetime
                try:
                    first_ts = datetime.fromisoformat(first_ts)
                    last_ts = datetime.fromisoformat(last_ts)
                except:
                    pass
            try:
                duration = last_ts - first_ts
                minutes = duration.total_seconds() / 60
                return f"{int(minutes)} 分钟"
            except:
                return "计算错误"
        return "未知"
    
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
            print(f"❌ NaturalEndDetector JSON parsing failed: {e}")
            print(f"   Original content: {content}")
            return TerminationSignal(should_terminate=False, reason=None, confidence=0.0, explanation=f"JSON解析失败: {e}")
        except Exception as e:
            print(f"❌ NaturalEndDetector general parsing failed: {e}")
            print(f"   Original content: {content}")
            return TerminationSignal(should_terminate=False, reason=None, confidence=0.0, explanation=f"解析失败: {e}")


class DialogueTerminationManager:
    """综合管理对话终止逻辑"""
    
    def __init__(self, llm):
        self.hesitancy_detector = HesitancyDetector(llm)
        self.info_detector = InfoCompletenessDetector(llm)
        self.natural_end_detector = NaturalEndDetector(llm)
    
    def should_terminate_onboarding(self, conversation: List[Dict], min_turns: int, max_turns: int) -> Tuple[bool, TerminationSignal]:
        num_turns = len(conversation) // 2
        if num_turns >= max_turns:
            return True, TerminationSignal(should_terminate=True, reason=TerminationReason.MAX_TURNS, confidence=1.0, explanation=f"达到最大轮数 {max_turns}")
        if num_turns < min_turns:
            return False, TerminationSignal(should_terminate=False, reason=None, confidence=1.0, explanation=f"未达到最小轮数 {min_turns}")
        
        if len(conversation) >= 2:
            last_user_msg = None
            for msg in reversed(conversation):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
            if last_user_msg:
                hesitancy_signal = self.hesitancy_detector.detect(last_user_msg, conversation)
                if hesitancy_signal.should_terminate and hesitancy_signal.confidence > 0.7:
                    return True, hesitancy_signal
        
        info_signal = self.info_detector.detect(conversation, min_turns)
        if info_signal.should_terminate and info_signal.confidence > 0.8:
            return True, info_signal
            
        return False, TerminationSignal(should_terminate=False, reason=None, confidence=0.0, explanation="继续收集信息")
    
    def should_terminate_social_chat(self, conversation: List[Dict], min_messages: int, max_messages: int) -> Tuple[bool, TerminationSignal]:
        if len(conversation) >= max_messages:
            return True, TerminationSignal(should_terminate=True, reason=TerminationReason.MAX_TURNS, confidence=1.0, explanation=f"达到最大消息数 {max_messages}")
        if len(conversation) < min_messages:
            return False, TerminationSignal(should_terminate=False, reason=None, confidence=1.0, explanation=f"未达到最小消息数 {min_messages}")
        
        natural_signal = self.natural_end_detector.detect(conversation, min_messages)
        if natural_signal.should_terminate and natural_signal.confidence > 0.7:
            return True, natural_signal
            
        if len(conversation) >= 1:
            last_msg = conversation[-1].get("content", "")
            hesitancy_signal = self.hesitancy_detector.detect(last_msg, conversation)
            if hesitancy_signal.should_terminate and hesitancy_signal.confidence > 0.8:
                return True, hesitancy_signal
        
        return False, TerminationSignal(should_terminate=False, reason=None, confidence=0.0, explanation="继续聊天")