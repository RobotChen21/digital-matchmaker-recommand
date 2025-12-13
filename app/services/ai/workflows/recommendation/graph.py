# -*- coding: utf-8 -*-
from langgraph.graph import StateGraph, END
from app.common.models.state import MatchmakingState
from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import ChromaManager

# Import Nodes
from .nodes.intent import IntentNode
from .nodes.filter import FilterNode
from .nodes.recall import RecallNode
from .nodes.ranking import RankingNode
from .nodes.response import ResponseNode
from .nodes.deep_dive import DeepDiveNode # New

class RecommendationGraphBuilder:
    def __init__(self, db_manager: MongoDBManager, chroma_manager: ChromaManager):
        # 初始化各个 Node 类
        self.intent_node = IntentNode(db_manager)
        self.filter_node = FilterNode(db_manager)
        self.recall_node = RecallNode(chroma_manager)
        self.ranking_node = RankingNode(db_manager)
        self.response_node = ResponseNode(chroma_manager)
        self.deep_dive_node = DeepDiveNode(db_manager, chroma_manager)

    def check_search_results(self, state: MatchmakingState) -> str:
        """检查 Hard Filter 的结果，决定是否循环"""
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

    def build(self):
        workflow = StateGraph(MatchmakingState)

        # Add Nodes
        workflow.add_node("load_profile", self.intent_node.load_profile)
        workflow.add_node("intent", self.intent_node.analyze_intent)
        workflow.add_node("chitchat", self.intent_node.chitchat)
        
        workflow.add_node("hard_filter", self.filter_node.hard_filter)
        workflow.add_node("refine_query", self.filter_node.refine_query)
        
        workflow.add_node("semantic_recall", self.recall_node.semantic_recall)
        workflow.add_node("ranking", self.ranking_node.ranking)
        
        workflow.add_node("evidence_hunting", self.response_node.evidence_hunting)
        workflow.add_node("response", self.response_node.generate_response)
        
        workflow.add_node("deep_dive", self.deep_dive_node.deep_dive) # New Node

        # Edges
        workflow.set_entry_point("load_profile")
        workflow.add_edge("load_profile", "intent")
        
        # Intent Router (Updated)
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
        
        # Main Flow
        workflow.add_edge("semantic_recall", "ranking")
        workflow.add_edge("ranking", "evidence_hunting") 
        workflow.add_edge("evidence_hunting", "response") 
        
        workflow.add_edge("response", END)
        workflow.add_edge("chitchat", END)
        workflow.add_edge("deep_dive", END) # Deep Dive 直接结束

        return workflow.compile()