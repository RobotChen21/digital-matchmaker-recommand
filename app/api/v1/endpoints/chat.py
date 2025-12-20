# -*- coding: utf-8 -*-
import json
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query, status
from bson import ObjectId

from app.api.schemas.chat_dto import ChatRequest, ChatResponse, CandidateDTO, ChatContext
from app.api.v1.endpoints.auth import get_current_user_id
from app.core.container import container # 引入容器
from app.core.security import decode_access_token

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

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...)
):
    """
    WebSocket 实时对话接口
    URL: ws://host/api/v1/chat/ws?token=<access_token>
    """
    # 1. 验证 Token
    payload = decode_access_token(token)
    if not payload or not payload.get("sub"):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    user_id = payload.get("sub")
    await websocket.accept()
    
    try:
        while True:
            # 2. 接收消息
            data = await websocket.receive_text()
            try:
                request_data = json.loads(data)
                # 兼容直接发送字符串或完整 JSON 对象
                if isinstance(request_data, str):
                    current_msg = request_data
                    ctx_dict = {} # 默认空上下文
                else:
                    current_msg = request_data.get("message", "")
                    ctx_dict = request_data.get("context", {})
            except json.JSONDecodeError:
                current_msg = data
                ctx_dict = {}

            # 3. 构造 Context 对象 (简单处理，容错)
            # 注意: 这里尽量模拟 ChatRequest 的结构，但允许部分缺失
            initial_state = {
                "user_id": user_id, 
                "current_input": current_msg,
                "messages": [], 
                "search_count": 0,
                
                "seen_candidate_ids": ctx_dict.get("seen_candidate_ids", []),
                "final_candidates": ctx_dict.get("last_candidates", []), # 注意 key 映射
                "last_target_person": ctx_dict.get("last_target_person"),
                "last_search_criteria": ctx_dict.get("last_search_criteria")
            }
            
            # 4. 流式执行 LangGraph
            app = container.recommendation_app
            
            final_output = None
            
            # 这里的 config 可以传 run_name 等
            async for event in app.astream_events(initial_state, version="v1"):
                kind = event["event"]
                
                # 捕获 LLM 的流式输出 (Token)
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        await websocket.send_json({
                            "type": "token",
                            "content": chunk.content
                        })
                
                # 捕获链结束，尝试提取最终状态
                elif kind == "on_chain_end":
                    output = event["data"].get("output")
                    if isinstance(output, dict):
                        # 简单的启发式判断：如果包含 reply 且包含 user_id，很可能是最终状态
                        # 或者判断 key 是否覆盖了我们关心的字段
                        if "reply" in output and "intent" in output:
                            final_output = output
            
            # 发送最终结果 (Context 更新)
            if final_output:
                # 构造类似 ChatResponse 的结构供前端更新 Context
                # 需要序列化 ObjectId
                
                # 提取 CandidateDTO
                candidates_data = final_output.get("final_candidates", [])
                intent = final_output.get("intent", "unknown")
                final_candidates_dtos = []
                for c in candidates_data:
                    dto = {
                        "id": c.get('id', ''),
                        "nickname": c.get('nickname', '未知'),
                        "gender": c.get('gender', 'unknown'),
                        "age": c.get('age', 0),
                        "city": c.get('city', ''),
                        "summary": c.get('summary', ''),
                        "evidence": c.get('evidence', '')
                    }
                    final_candidates_dtos.append(dto)

                result_payload = {
                    "intent": intent,
                    "final_candidates": final_candidates_dtos,
                    "new_context": {
                        "seen_candidate_ids": serialize_mongo_obj(final_output.get("seen_candidate_ids", [])),
                        "last_candidates": serialize_mongo_obj(candidates_data if intent == 'search_candidate' else ctx_dict.get("last_candidates", [])),
                        "last_target_person": serialize_mongo_obj(final_output.get("last_target_person")),
                        "last_search_criteria": serialize_mongo_obj(final_output.get("last_search_criteria", {}))
                    },
                    "debug_info": {
                         "semantic_query": final_output.get("semantic_query"),
                         "hard_filters": serialize_mongo_obj(final_output.get("hard_filters"))
                    }
                }
                await websocket.send_json({"type": "result", "data": result_payload})
            
            # 尝试发送一个结束标记
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        print(f"Client #{user_id} disconnected")
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except:
            pass

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
        final_state = await app.ainvoke(initial_state)
        
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
