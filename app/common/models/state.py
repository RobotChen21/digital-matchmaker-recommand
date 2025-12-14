# -*- coding: utf-8 -*-
from typing import TypedDict, List, Dict, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator

class MatchmakingState(TypedDict):
    """
    红娘推荐系统的状态对象 (State)
    这个字典在 LangGraph 的各个节点之间传递，存储所有的上下文信息。
    """
    # 1. 对话上下文
    messages: Annotated[List[BaseMessage], operator.add] # 聊天历史
    current_input: str        # 用户当前的最新输入
    user_id: str              # 当前交互的用户 ID

    # 2. 意图与条件
    intent: Optional[str]     # 识别出的意图
    hard_filters: Dict        # 硬性过滤条件
    semantic_query: str       # 语义检索关键词
    match_policy: Dict        # 匹配策略 (从 IntentOutput.match_policy 转换而来)
    
    # 3. 召回结果
    hard_candidate_ids: List[str]      # 硬性筛选出的候选人 ID 列表
    semantic_candidate_ids: List[str]  # 语义检索出的候选人 ID 列表 (Top N)
    
    # 4. 最终结果
    final_candidates: List[Dict]       # 最终精排后的候选人完整信息 (Top 3)
    reply: str                         # 最终给用户的回复文本
    
    # 5. 控制标志 & 上下文记忆
    search_count: int                  # 搜索次数 (防止无限循环)
    error_msg: Optional[str]           # 错误信息
    target_person_name: Optional[str]  # 当前正在深度探索的目标名字
    last_target_person: Optional[str]  # 上一轮深度探索的目标名字 (用于指代消解)
    seen_candidate_ids: List[str]      # [NEW] 已经推荐过的候选人 ID 列表 (用于"换一批"排除)
    last_search_criteria: Optional[Dict] # [NEW] 上一轮的搜索条件 (用于"换一批"继承: hard_filters, semantic_query, match_policy)
    
    # 6. 用户画像上下文
    current_user_gender: Optional[str] # 当前用户性别
    current_user_summary: Optional[str] # 当前用户基础信息+画像摘要
    current_user_profile: Optional[Dict] # [NEW] 当前用户完整画像结构体