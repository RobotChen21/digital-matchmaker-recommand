# -*- coding: utf-8 -*-
from .graph import RecommendationGraphBuilder

class RecommendationWorkflow:
    """
    Facade for the Recommendation Workflow.
    """
    def __init__(self):
        # 不再需要传入 db_manager 和 chroma_manager
        self.builder = RecommendationGraphBuilder()

    def build_graph(self):
        return self.builder.build()