"""
智能对话结束系统 (Intelligent Dialogue Termination System)

支持自然的对话结束条件:
1. 用户不想继续(hesitancy detection)
2. AI 判断信息收集完成
3. 聊天自然结束(话题耗尽)
4. 情绪信号(疲惫/不耐烦)
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json


# ============================================================================
# 对话结束条件枚举
# ============================================================================

class TerminationReason(str, Enum):
    """对话结束原因"""
    USER_HESITANT = "user_hesitant"           # 用户犹豫/不想继续
    INFO_COLLECTED = "info_collected"         # 信息收集完成
    NATURAL_END = "natural_end"               # 自然结束
    USER_TIRED = "user_tired"                 # 用户疲惫
    TOPIC_EXHAUSTED = "topic_exhausted"       # 话题耗尽
    MAX_TURNS = "max_turns_reached"           # 达到最大轮数
    USER_REQUEST = "user_request_end"         # 用户主动要求结束


class TerminationSignal(BaseModel):
    """对话结束信号"""
    should_terminate: bool = Field(description="是否应该结束对话")
    reason: Optional[TerminationReason] = Field(description="结束原因")
    confidence: float = Field(description="置信度 0-1")
    explanation: str = Field(description="判断依据")


# ============================================================================
# 用户犹豫检测器 (User Hesitancy Detector)
# ============================================================================

class HesitancyDetector:
    """检测用户是否不想继续对话"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
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

请输出 JSON 格式(不要任何解释):
{{
  "should_terminate": true/false,
  "reason": "user_hesitant/user_tired/user_request_end/null",
  "confidence": 0.0-1.0,
  "explanation": "判断依据"
}}""")
    
    def detect(self, user_message: str, conversation_history: List[Dict]) -> TerminationSignal:
        """检测用户是否想结束对话"""
        
        chain = self.prompt | self.llm
        
        response = chain.invoke({
            "user_message": user_message,
            "conversation_history": self._format_history(conversation_history)
        })
        
        # 解析响应
        return self._parse_response(response.content)
    
    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        if not history:
            return "(对话刚开始)"
        
        lines = []
        for msg in history[-5:]:  # 只看最近5轮
            role = "AI" if msg.get("role") == "ai" else "用户"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)
    
    def _parse_response(self, content: str) -> TerminationSignal:
        """解析 LLM 响应"""
        try:
            # 清理 JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            return TerminationSignal(**data)
        except:
            # 默认不结束
            return TerminationSignal(
                should_terminate=False,
                reason=None,
                confidence=0.0,
                explanation="解析失败,继续对话"
            )


# ============================================================================
# 信息完整度检测器 (Information Completeness Detector)
# ============================================================================

class InfoCompletenessDetector:
    """检测 Onboarding 信息是否收集完成"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # 必须收集的信息维度
        self.required_dimensions = [
            "教育背景 (学历/学校)",
            "工作职业 (行业/忙碌程度)",
            "家庭背景 (独生/父母/资产情况)",
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
- 必须覆盖至少 5/6 个核心维度。
- 对于"缺失"的维度，必须是用户明确拒绝回答或无法获取，否则应继续询问。

请输出 JSON 格式(不要任何解释):
{{
  "should_terminate": true/false,
  "reason": "info_collected/null",
  "confidence": 0.0-1.0,
  "explanation": "已收集: [...] / 缺失: [...]",
  "collected_dimensions": ["教育", "工作", ...],
  "missing_dimensions": ["家庭资产", ...]
}}""")
    
    def detect(self, full_conversation: List[Dict], min_turns: int = 8) -> TerminationSignal:
        """检测信息是否收集完成"""
        
        # 基本检查: 至少对话 min_turns 轮
        if len(full_conversation) < min_turns * 2:  # *2 因为每轮有 AI 和 user
            return TerminationSignal(
                should_terminate=False,
                reason=None,
                confidence=1.0,
                explanation=f"对话不足 {min_turns} 轮,继续收集"
            )
        
        chain = self.prompt | self.llm
        
        response = chain.invoke({
            "required_dimensions": ", ".join(self.required_dimensions),
            "full_conversation": self._format_conversation(full_conversation)
        })
        
        return self._parse_response(response.content)
    
    def _format_conversation(self, conversation: List[Dict]) -> str:
        """格式化完整对话"""
        lines = []
        for msg in conversation:
            role = "AI" if msg.get("role") == "ai" else "用户"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)
    
    def _parse_response(self, content: str) -> TerminationSignal:
        """解析响应"""
        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            return TerminationSignal(
                should_terminate=data["should_terminate"],
                reason=data.get("reason"),
                confidence=data["confidence"],
                explanation=data["explanation"]
            )
        except:
            return TerminationSignal(
                should_terminate=False,
                reason=None,
                confidence=0.0,
                explanation="解析失败,继续收集"
            )


# ============================================================================
# 社交对话自然结束检测器 (Natural Conversation End Detector)
# ============================================================================

class NaturalEndDetector:
    """检测社交对话是否自然结束"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
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

请输出 JSON 格式(不要任何解释):
{{
  "should_terminate": true/false,
  "reason": "natural_end/topic_exhausted/null",
  "confidence": 0.0-1.0,
  "explanation": "判断依据"
}}""")
    
    def detect(self, full_conversation: List[Dict], min_messages: int = 20) -> TerminationSignal:
        """检测是否自然结束"""
        
        # 基本检查: 至少 min_messages 条
        if len(full_conversation) < min_messages:
            return TerminationSignal(
                should_terminate=False,
                reason=None,
                confidence=1.0,
                explanation=f"消息不足 {min_messages} 条,还在热聊期"
            )
        
        # 计算持续时间
        duration = self._calculate_duration(full_conversation)
        
        chain = self.prompt | self.llm
        
        response = chain.invoke({
            "recent_conversation": self._format_recent(full_conversation),
            "total_messages": len(full_conversation),
            "duration": duration
        })
        
        return self._parse_response(response.content)
    
    def _format_recent(self, conversation: List[Dict]) -> str:
        """格式化最近对话"""
        lines = []
        for msg in conversation[-8:]:
            sender = f"用户{msg.get('sender_id', 'A')}"
            lines.append(f"{sender}: {msg.get('content', '')}")
        return "\n".join(lines)
    
    def _calculate_duration(self, conversation: List[Dict]) -> str:
        """计算对话持续时间"""
        if not conversation or len(conversation) < 2:
            return "刚开始"
        
        first_ts = conversation[0].get("timestamp")
        last_ts = conversation[-1].get("timestamp")
        
        if first_ts and last_ts:
            duration = last_ts - first_ts
            minutes = duration.total_seconds() / 60
            return f"{int(minutes)} 分钟"
        
        return "未知"
    
    def _parse_response(self, content: str) -> TerminationSignal:
        """解析响应"""
        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            return TerminationSignal(**data)
        except:
            return TerminationSignal(
                should_terminate=False,
                reason=None,
                confidence=0.0,
                explanation="解析失败,继续对话"
            )


# ============================================================================
# 综合对话终止管理器 (Dialogue Termination Manager)
# ============================================================================

class DialogueTerminationManager:
    """综合管理对话终止逻辑"""
    
    def __init__(self, llm):
        self.hesitancy_detector = HesitancyDetector(llm)
        self.info_detector = InfoCompletenessDetector(llm)
        self.natural_end_detector = NaturalEndDetector(llm)
    
    def should_terminate_onboarding(
        self, 
        conversation: List[Dict],
        min_turns: int = 8,
        max_turns: int = 20
    ) -> Tuple[bool, TerminationSignal]:
        """判断 onboarding 对话是否应该结束"""
        
        # 1. 检查最大轮数
        num_turns = len(conversation) // 2
        if num_turns >= max_turns:
            return True, TerminationSignal(
                should_terminate=True,
                reason=TerminationReason.MAX_TURNS,
                confidence=1.0,
                explanation=f"达到最大轮数 {max_turns}"
            )
        
        # 2. 检查最小轮数
        if num_turns < min_turns:
            return False, TerminationSignal(
                should_terminate=False,
                reason=None,
                confidence=1.0,
                explanation=f"未达到最小轮数 {min_turns}"
            )
        
        # 3. 检查用户犹豫
        if len(conversation) >= 2:
            last_user_msg = None
            for msg in reversed(conversation):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
            
            if last_user_msg:
                hesitancy_signal = self.hesitancy_detector.detect(
                    last_user_msg, 
                    conversation
                )
                
                if hesitancy_signal.should_terminate and hesitancy_signal.confidence > 0.7:
                    return True, hesitancy_signal
        
        # 4. 检查信息完整度
        info_signal = self.info_detector.detect(conversation, min_turns)
        
        if info_signal.should_terminate and info_signal.confidence > 0.8:
            return True, info_signal
        
        # 默认继续
        return False, TerminationSignal(
            should_terminate=False,
            reason=None,
            confidence=0.0,
            explanation="继续收集信息"
        )
    
    def should_terminate_social_chat(
        self,
        conversation: List[Dict],
        min_messages: int = 20,
        max_messages: int = 60
    ) -> Tuple[bool, TerminationSignal]:
        """判断社交聊天是否应该结束"""
        
        # 1. 检查最大消息数
        if len(conversation) >= max_messages:
            return True, TerminationSignal(
                should_terminate=True,
                reason=TerminationReason.MAX_TURNS,
                confidence=1.0,
                explanation=f"达到最大消息数 {max_messages}"
            )
        
        # 2. 检查最小消息数
        if len(conversation) < min_messages:
            return False, TerminationSignal(
                should_terminate=False,
                reason=None,
                confidence=1.0,
                explanation=f"未达到最小消息数 {min_messages}"
            )
        
        # 3. 检查自然结束
        natural_signal = self.natural_end_detector.detect(conversation, min_messages)
        
        if natural_signal.should_terminate and natural_signal.confidence > 0.7:
            return True, natural_signal
        
        # 4. 检查用户疲惫(检查最后一条消息)
        if len(conversation) >= 1:
            last_msg = conversation[-1].get("content", "")
            hesitancy_signal = self.hesitancy_detector.detect(last_msg, conversation)
            
            if hesitancy_signal.should_terminate and hesitancy_signal.confidence > 0.8:
                return True, hesitancy_signal
        
        # 默认继续
        return False, TerminationSignal(
            should_terminate=False,
            reason=None,
            confidence=0.0,
            explanation="继续聊天"
        )


# ============================================================================
# 使用示例
# ============================================================================

def demo_termination_detection():
    """演示终止检测功能"""
    
    from langchain_openai import ChatOpenAI
    import os
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    manager = DialogueTerminationManager(llm)
    
    print("=" * 80)
    print("🎯 对话终止检测系统演示")
    print("=" * 80)
    
    # 示例 1: 用户犹豫
    print("\n场景 1: 用户显示犹豫信号")
    print("-" * 80)
    conversation = [
        {"role": "ai", "content": "能聊聊你的感情经历吗?"},
        {"role": "user", "content": "嗯...这个...不太想说..."}
    ]
    
    should_end, signal = manager.should_terminate_onboarding(conversation, min_turns=1)
    print(f"是否结束: {should_end}")
    print(f"原因: {signal.reason}")
    print(f"置信度: {signal.confidence}")
    print(f"说明: {signal.explanation}")
    
    # 示例 2: 信息收集完成
    print("\n场景 2: 信息收集完整")
    print("-" * 80)
    conversation = [
        {"role": "ai", "content": "你的工作是什么?"},
        {"role": "user", "content": "我是产品经理,在互联网公司工作"},
        {"role": "ai", "content": "平时有什么兴趣爱好?"},
        {"role": "user", "content": "喜欢跑步、看电影、旅行"},
        {"role": "ai", "content": "理想的另一半是什么样的?"},
        {"role": "user", "content": "希望对方独立、有趣、三观契合"},
        {"role": "ai", "content": "家庭情况如何?"},
        {"role": "user", "content": "独生子女,父母退休了"},
        {"role": "ai", "content": "你觉得感情中最重要的是什么?"},
        {"role": "user", "content": "我觉得真诚和沟通最重要"}
    ]
    
    should_end, signal = manager.should_terminate_onboarding(conversation, min_turns=4)
    print(f"是否结束: {should_end}")
    print(f"原因: {signal.reason}")
    print(f"置信度: {signal.confidence}")
    print(f"说明: {signal.explanation}")
    
    # 示例 3: 社交聊天自然结束
    print("\n场景 3: 社交对话自然结束")
    print("-" * 80)
    from datetime import datetime, timedelta
    
    base_time = datetime.now()
    conversation = []
    
    messages_content = [
        "Hi,看到你也喜欢摄影!",
        "对呀,不过我是业余的哈哈",
        "没事,我也是业余的~你一般拍什么?",
        "风景为主,偶尔拍人像",
        # ... 假设中间有很多对话
        "今天聊得很开心!",
        "我也是!那我们周末一起去拍照吧?",
        "好啊!那就这样说定了~",
        "嗯嗯,到时候联系!",
        "好的,拜拜~",
        "拜拜!"
    ]
    
    for i, content in enumerate(messages_content):
        conversation.append({
            "sender_id": f"user_{i % 2}",
            "content": content,
            "timestamp": base_time + timedelta(minutes=i*3)
        })
    
    # 填充到至少 20 条
    while len(conversation) < 20:
        conversation.insert(-4, {
            "sender_id": "user_0",
            "content": "聊天内容...",
            "timestamp": base_time + timedelta(minutes=len(conversation)*3)
        })
    
    should_end, signal = manager.should_terminate_social_chat(conversation, min_messages=15)
    print(f"是否结束: {should_end}")
    print(f"原因: {signal.reason}")
    print(f"置信度: {signal.confidence}")
    print(f"说明: {signal.explanation}")
    
    print("\n" + "=" * 80)
    print("✨ 演示完成!")


if __name__ == "__main__":
    demo_termination_detection()
