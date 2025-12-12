from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.ai.workflows.recommendation import RecommendationWorkflow

router = APIRouter()

# 初始化 Workflow (单例模式)
# 注意: 在生产环境中，这应该放在 startup 事件中初始化，或者使用依赖注入
rec_workflow = RecommendationWorkflow()
rec_app = rec_workflow.build_graph()

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    intent: str
    debug_info: dict

@router.post("/message", response_model=ChatResponse)
async def chat_with_matchmaker(request: ChatRequest):
    """
    与 AI 红娘对话接口
    """
    # 构造初始状态
    initial_state = {
        "user_id": request.user_id,
        "current_input": request.message,
        "messages": [], # 这里暂为空，实际应从数据库加载历史
        "search_count": 0
    }

    try:
        # 运行 LangGraph
        # invoke 会运行图直到结束节点
        final_state = rec_app.invoke(initial_state)
        
        return ChatResponse(
            reply=final_state.get("reply", "系统暂时无法处理您的请求"),
            intent=final_state.get("intent", "unknown"),
            debug_info={
                "semantic_query": final_state.get("semantic_query"),
                "hard_filters": final_state.get("hard_filters")
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI 处理出错: {str(e)}")