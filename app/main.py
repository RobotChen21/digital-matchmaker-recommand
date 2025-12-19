from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import chat, users, auth

app = FastAPI(
    title="Digital Matchmaker API",
    description="Backend service for the Digital Matchmaker system",
    version="0.2.0"
)

# ... (CORS middleware) ...
# 注册路由
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])

@app.get("/")
def root():
    return {"message": "Welcome to Digital Matchmaker API. Visit /docs for documentation."}