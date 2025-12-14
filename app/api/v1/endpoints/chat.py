# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException, Depends
from app.services.ai.workflows.recommendation import RecommendationWorkflow
from app.api.schemas.chat_dto import ChatRequest, ChatResponse, CandidateDTO, ChatContext
from app.api.v1.endpoints.auth import get_current_user_id # 引入依赖

router = APIRouter()

rec_workflow = RecommendationWorkflow()
rec_app = rec_workflow.build_graph()

@router.post("/message", response_model=ChatResponse)
async def chat_with_matchmaker(
    request: ChatRequest,
    user_id: str = Depends(get_current_user_id) # 从 Token 获取 ID
):
    """
    与 AI 红娘对话接口 (需要 Bearer Token)
    """
    ctx = request.context
    
    # 构造初始状态
    initial_state = {
        "user_id": user_id, # 使用 Token 中的 ID
        "current_input": request.message,
        "messages": [], 
        "search_count": 0,
        
        "seen_candidate_ids": ctx.seen_candidate_ids,
        "final_candidates": ctx.last_candidates,
        "last_target_person": ctx.last_target_person,
        "last_search_criteria": ctx.last_search_criteria
    }

    try:
        final_state = rec_app.invoke(initial_state)
        
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
            
        new_ctx = ChatContext(
            seen_candidate_ids=final_state.get("seen_candidate_ids", []),
            last_candidates=candidates_data if intent == 'search_candidate' else ctx.last_candidates,
            last_target_person=final_state.get("last_target_person"),
            last_search_criteria=final_state.get("last_search_criteria", {})
        )
        
        return ChatResponse(
            reply=final_state.get("reply", "系统暂时无法处理您的请求"),
            intent=intent,
            final_candidates=final_candidates_dtos,
            new_context=new_ctx,
            debug_info={
                "semantic_query": final_state.get("semantic_query"),
                "hard_filters": final_state.get("hard_filters")
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI 处理出错: {str(e)}")