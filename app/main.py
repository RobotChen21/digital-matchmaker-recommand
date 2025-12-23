from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import chat, users, auth
from app.core.container import container

# --- Lifespan (生命周期) 管理 ---
# 替代旧版的 @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [Startup] 启动时执行
    print("🚀 Application starting up...")
    try:
        # 自动检查 ES 索引是否存在
        container.es.create_index_if_not_exists()
        print("✅ Elasticsearch index check passed.")
    except Exception as e:
        print(f"⚠️ Startup Warning: ES Index check failed: {e}")
        
    yield # --- 应用运行中 ---

    # [Shutdown] 关闭时执行
    print("🛑 Application shutting down...")

# --- App 实例化 ---
app = FastAPI(
    title="Digital Matchmaker API",
    description="Backend service for the Digital Matchmaker system",
    version="0.2.0",
    lifespan=lifespan  # 注入生命周期管理
)

# --- 中间件 ---
# 配置 CORS (允许前端跨域访问，支持 JWT Header)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生产环境建议改为具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], # 允许 Authorization 头通过
)

# --- 路由注册 ---
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])

@app.get("/")
def root():
    return {"message": "Welcome to Digital Matchmaker API. Visit /docs for documentation."}
