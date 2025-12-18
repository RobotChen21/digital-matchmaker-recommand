# -*- coding: utf-8 -*-
from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId

from app.api.schemas.chat_dto import ChatRequest, ChatResponse, CandidateDTO, ChatContext
from app.api.v1.endpoints.auth import get_current_user_id
from app.core.container import container # 引入容器

router = APIRouter()

# --- Helpers ---
def serialize_mongo_obj(obj):
    """递归将 ObjectId 转换为字符串"""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {k: serialize_mongo_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_mongo_obj(i) for i in obj]
    return obj

@router.post("/message", response_model=ChatResponse)
async def chat_with_matchmaker(
    request: ChatRequest,
    user_id: str = Depends(get_current_user_id) 
):
    """
    与 AI 红娘对话接口
    """
    ctx = request.context
    
    # 构造初始状态
    initial_state = {
        "user_id": user_id, 
        "current_input": request.message,
        "messages": [], 
        "search_count": 0,
        
        "seen_candidate_ids": ctx.seen_candidate_ids,
        "final_candidates": ctx.last_candidates,
        "last_target_person": ctx.last_target_person,
        "last_search_criteria": ctx.last_search_criteria
    }

    try:
        # 从容器获取 app
        app = container.recommendation_app
        final_state = app.invoke(initial_state)
        
        candidates_data = final_state.get("final_candidates", [])
        intent = final_state.get("intent", "unknown")
        
        final_candidates_dtos = []
        for c in candidates_data:
            dto = CandidateDTO(
                id=c.get('id', ''),
                nickname=c.get('nickname', '未知'),
                gender=c.get('gender', 'unknown'),
                age=c.get('age', 0),
                city=c.get('city', ''),
                summary=c.get('summary', ''),
                evidence=c.get('evidence', '')
            )
            final_candidates_dtos.append(dto)
            
        # 构造新的 Context
        cleaned_last_criteria = serialize_mongo_obj(final_state.get("last_search_criteria", {}))
        
        new_ctx = ChatContext(
            seen_candidate_ids=final_state.get("seen_candidate_ids", []),
            last_candidates=serialize_mongo_obj(candidates_data if intent == 'search_candidate' else ctx.last_candidates),
            last_target_person=final_state.get("last_target_person"),
            last_search_criteria=cleaned_last_criteria
        )
        
        return ChatResponse(
            reply=final_state.get("reply", "系统暂时无法处理您的请求"),
            intent=intent,
            final_candidates=final_candidates_dtos,
            new_context=new_ctx,
            debug_info={
                "semantic_query": final_state.get("semantic_query"),
                "hard_filters": serialize_mongo_obj(final_state.get("hard_filters"))
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI 处理出错: {str(e)}")
