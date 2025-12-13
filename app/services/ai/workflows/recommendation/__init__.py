# -*- coding: utf-8 -*-
from app.db.mongo_manager import MongoDBManager
from app.db.chroma_manager import ChromaManager
from .graph import RecommendationGraphBuilder

class RecommendationWorkflow:
    """
    Facade for the Recommendation Workflow.
    Maintains backward compatibility with existing code.
    """
    def __init__(self, db_manager: MongoDBManager, chroma_manager: ChromaManager):
        self.builder = RecommendationGraphBuilder(db_manager, chroma_manager)

    def build_graph(self):
        return self.builder.build()
