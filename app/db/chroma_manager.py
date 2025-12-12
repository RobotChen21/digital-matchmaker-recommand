# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter # 更新后的导入路径
from langchain_core.documents import Document

class EnhancedChromaManager:
    """增强型的 ChromaDB 管理器，支持对话分块和检索"""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "default_collection"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # 从配置中加载嵌入模型名称
        from app.core.config import settings # 在这里局部导入settings，避免循环引用
        self.embeddings_model = HuggingFaceEmbeddings(model_name=settings.llm.chroma_embedding_model)
        
        self.vector_db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings_model,
            persist_directory=self.persist_directory
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 默认块大小
            chunk_overlap=100, # 默认重叠
            length_function=len,
            add_start_index=True,
        )

    def add_conversation_chunks(self,
                                user_id: str,
                                messages: List[Dict],
                                dialogue_type: str, # "onboarding" or "social"
                                window_size: int = 5,
                                overlap: int = 2):
        """
        将对话消息切分为带有滑动窗口的块，并添加到向量数据库。
        每个块包含 `window_size` 条消息，相邻块重叠 `overlap` 条消息。
        """
        if not messages:
            return

        documents = []
        # 构建滑动窗口
        for i in range(0, len(messages), window_size - overlap):
            window = messages[i : i + window_size]
            if not window:
                continue
            
            # 将窗口内的消息拼接成一个文本块
            # 兼容两种消息格式: Onboarding (有role) 和 Social Chat (有sender_id)
            context_text = "\n".join([
                f"{msg.get('role', msg.get('sender_id', 'Unknown'))}: {msg.get('content', '')}" 
                for msg in window
            ])
            
            # 创建 Document，包含元数据
            doc = Document(
                page_content=context_text,
                metadata={
                    "user_id": user_id,
                    "dialogue_type": dialogue_type,
                    "start_message_index": i,
                    "end_message_index": i + len(window) - 1,
                    "timestamp": str(window[0].get('timestamp', datetime.now())) # 确保转为字符串
                }
            )
            documents.append(doc)
        
        if documents:
            # 清理旧的文档，避免重复
            self.vector_db.delete(
                ids=[d.metadata['user_id'] + "_" + d.metadata['dialogue_type'] + "_" + str(d.metadata['start_message_index']) for d in documents]
            )
            self.vector_db.add_documents(documents)
            # self.vector_db.persist() # 新版本自动持久化，无需手动调用
            print(f"✅ ChromaDB: 为用户 {user_id} 添加 {len(documents)} 条 {dialogue_type} 对话块。")

    def retrieve_related_context(self, query: str, user_id: str = None, k: int = 5, filter: Dict = None) -> List[Document]:
        """
        从向量数据库中检索与查询相关的文档。
        支持 user_id 快捷过滤，也支持自定义 filter 字典。
        """
        search_filter = filter if filter else {}
        
        # 如果传入了 user_id，合并到 filter 中
        if user_id:
            if search_filter:
                # 如果已经有 filter，使用 $and 组合
                search_filter = {"$and": [search_filter, {"user_id": user_id}]}
            else:
                search_filter = {"user_id": user_id}

        results = self.vector_db.similarity_search_with_score(
            query=query,
            k=k,
            filter=search_filter
        )
        
        # results 是 (Document, score) 元组的列表
        return [doc for doc, score in results]

from datetime import datetime # 导入 datetime 以避免 NameError 
