# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from app.api.schemas.auth_dto import LoginRequest, Token
from app.db.mongo_manager import MongoDBManager
from app.core.config import settings
from app.core.security import verify_password, create_access_token, decode_access_token

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

def get_db():
    return MongoDBManager(settings.database.mongo_uri, settings.database.db_name)
#TODO 实现拦截器逻辑
@router.post("/login", response_model=Token)
async def login(request: LoginRequest):
    db = get_db()
    
    # 1. 查用户
    user = db.get_auth_user_by_account(request.username)
    if not user:
        raise HTTPException(status_code=401, detail="账号或密码错误")
        
    # 2. 验密码
    if not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="账号或密码错误")
        
    # 3. 发 Token (sub 存 user_id)
    user_id = str(user["user_id"])
    access_token = create_access_token(subject=user_id)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user_id
    )

# 依赖注入：获取当前用户ID
async def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="无效的 Token")
    
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Token 缺少身份信息")
        
    return user_id
