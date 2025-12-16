# -*- coding: utf-8 -*-
from langgraph.graph import StateGraph, END
from app.common.models.state import MatchmakingState
from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import ChromaManager
from bson import ObjectId

# Import Nodes
from .nodes.intent import IntentNode
from .nodes.filter import FilterNode
from .nodes.recall import RecallNode
from .nodes.ranking import RankingNode
from .nodes.response import ResponseNode
from .nodes.deep_dive import DeepDiveNode
from .nodes.onboarding import OnboardingNode # New
#TODO 需要将聊天对话进行定期或者定量更新画像然后存入向量数据库(不是无上限，也是定量的)
class RecommendationGraphBuilder:
    def __init__(self, db_manager: MongoDBManager, chroma_manager: ChromaManager):
        # 初始化各个 Node 类
        self.intent_node = IntentNode(db_manager)
        self.filter_node = FilterNode(db_manager)
        self.recall_node = RecallNode(chroma_manager)
        self.ranking_node = RankingNode(db_manager)
        self.response_node = ResponseNode(chroma_manager)
        self.deep_dive_node = DeepDiveNode(db_manager, chroma_manager)
        self.onboarding_node = OnboardingNode(db_manager, chroma_manager) # New
        self.db = db_manager

    def check_search_results(self, state: MatchmakingState) -> str:
        count = len(state.get('hard_candidate_ids', []))
        search_attempts = state.get('search_count', 0)
        
        if count > 0:
            return "semantic"
        elif search_attempts < 2:
            return "refine" 
        else:
            return "response"
    
    def route_intent(self, state: MatchmakingState) -> str:
        intent = state.get('intent')
        if intent == "search_candidate":
            return "hard_filter"
        elif intent == "deep_dive":
            return "deep_dive"
        else:
            return "chitchat"

    def check_profile_status(self, state: MatchmakingState) -> str:
        """
        检查用户画像是否完善，决定是否进入 Onboarding。
        逻辑：依赖 users_states.is_onboarding_completed 标志位。
        """
        user_id = state['user_id']
        try:
            uid = ObjectId(user_id)
            state_doc = self.db.users_states.find_one({"user_id": uid})
            
            if state_doc and state_doc.get("is_onboarding_completed"):
                return "intent" # 已完成 Finalize -> 正常流程
            else:
                return "onboarding" # 未完成 Finalize -> 强制 Onboarding
            
        except Exception as e:
            print(f"Error checking profile status: {e}")
            return "onboarding" # 默认去 onboarding

    def build(self):
        workflow = StateGraph(MatchmakingState)

        # Add Nodes
        workflow.add_node("load_profile", self.intent_node.load_profile)
        
        workflow.add_node("onboarding", self.onboarding_node.process) # New
        
        workflow.add_node("intent", self.intent_node.analyze_intent)
        workflow.add_node("chitchat", self.intent_node.chitchat)
        
        workflow.add_node("hard_filter", self.filter_node.hard_filter)
        workflow.add_node("refine_query", self.filter_node.refine_query)
        
        workflow.add_node("semantic_recall", self.recall_node.semantic_recall)
        workflow.add_node("ranking", self.ranking_node.ranking)
        
        workflow.add_node("evidence_hunting", self.response_node.evidence_hunting)
        workflow.add_node("response", self.response_node.generate_response)
        
        workflow.add_node("deep_dive", self.deep_dive_node.deep_dive)

        # Edges
        workflow.set_entry_point("load_profile")
        
        # [NEW] Profile Check Router
        workflow.add_conditional_edges(
            "load_profile",
            self.check_profile_status,
            {
                "intent": "intent",
                "onboarding": "onboarding"
            }
        )
        
        # Onboarding 结束直接 END，等待用户下一次输入
        workflow.add_edge("onboarding", END)
        
        # Intent Router
        workflow.add_conditional_edges(
            "intent",
            self.route_intent,
            {
                "hard_filter": "hard_filter", 
                "chitchat": "chitchat",
                "deep_dive": "deep_dive"
            }
        )
        
        # Hard Filter Loop
        workflow.add_conditional_edges(
            "hard_filter",
            self.check_search_results,
            {
                "semantic": "semantic_recall", 
                "refine": "refine_query",
                "response": "response" 
            }
        )
        
        workflow.add_edge("refine_query", "hard_filter")
        
        workflow.add_edge("semantic_recall", "ranking")
        workflow.add_edge("ranking", "evidence_hunting") 
        workflow.add_edge("evidence_hunting", "response") 
        
        workflow.add_edge("response", END)
        workflow.add_edge("chitchat", END)
        workflow.add_edge("deep_dive", END)

        return workflow.compile()
