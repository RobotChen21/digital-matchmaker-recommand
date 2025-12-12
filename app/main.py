from fastapi import FastAPI
from app.api.v1 import chat

app = FastAPI(
    title="Digital Matchmaker API",
    description="Backend service for the Digital Matchmaker system",
    version="0.1.0"
)

# 注册路由
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

@app.get("/")
def root():
    return {"message": "Welcome to Digital Matchmaker API. Visit /docs for documentation."}
