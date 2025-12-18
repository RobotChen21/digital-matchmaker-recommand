# -*- coding: utf-8 -*-
from datetime import datetime, date
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends
from app.api.schemas.user_dto import UserRegisterRequest, UserRegisterResponse, UserProfileUpdate, UserProfileResponse
from app.core.container import container
from app.core.security import get_password_hash
from app.api.v1.endpoints.auth import get_current_user_id

router = APIRouter()

@router.post("/register", response_model=UserRegisterResponse)
async def register_account(request: UserRegisterRequest):
    """
    Step 1: 注册账号 (创建账号 + 关联空的用户档案)
    """
    db = container.db
    
    if db.get_auth_user_by_account(request.account):
        raise HTTPException(status_code=400, detail="该账号已被注册")

    try:
        # 1. 创建一个空的 user_basic 记录
        # 只初始化必要的空字段或默认值
        empty_basic = {
            "created_at": datetime.now(),
            "nickname": f"用户{request.account[-4:]}", # 默认昵称
            "gender": "unknown", # 必须给个默认值，避免后续流程报错
            "city": "未知",
            "birthday": datetime(2000, 1, 1), # 默认生日
            "height": 178,
            "weight": 70,
            "self_intro_raw": "unknow",
        }
        result = db.users_basic.insert_one(empty_basic)
        user_id = result.inserted_id
        
        # 1.5 初始化用户状态
        db.users_states.insert_one({
            "user_id": user_id,
            "is_onboarding_completed": False,
            "updated_at": datetime.now()
        })
        
        # 2. 创建 auth 记录
        pwd_hash = get_password_hash(request.password)
        db.create_auth_user(request.account, pwd_hash, user_id)
        
        return UserRegisterResponse(
            user_id=str(user_id),
            message="账号注册成功！请登录并完善资料。"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")

@router.put("/profile", response_model=UserProfileResponse)
async def update_profile(
    request: UserProfileUpdate,
    user_id: str = Depends(get_current_user_id)
):
    """
    Step 3: 完善/更新个人资料 (需要登录)
    """
    db = container.db
    
    update_data = request.dict() # Pydantic 会自动把 date 类型传过来
    
    # Mongo 不支持直接存 date，转为 datetime
    if update_data.get("birthday"):
        bday = update_data["birthday"]
        update_data["birthday"] = datetime(bday.year, bday.month, bday.day)
    
    # is_completed 应该由 Onboarding 流程决定，此处仅更新基础信息
    # update_data["is_completed"] = True 
    update_data["updated_at"] = datetime.now()
    
    try:
        db.users_basic.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        # 顺便初始化 persona (如果不存在)
        if not db.users_persona.find_one({"user_id": ObjectId(user_id)}):
             default_persona = {
                "user_id": ObjectId(user_id),
                "persona": {"occupation": "未知"},
                "created_at": datetime.now()
            }
             db.users_persona.insert_one(default_persona)

        return UserProfileResponse(
            user_id=user_id,
            is_profile_completed=True,
            **request.dict() # 这里 request.dict() 里的 birthday 是 date 类型
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存资料失败: {str(e)}")

@router.get("/me", response_model=UserProfileResponse)
async def get_my_profile(user_id: str = Depends(get_current_user_id)):
    """
    获取当前用户信息 (用于前端侧边栏展示)
    """
    db = container.db
    user = db.users_basic.find_one({"_id": ObjectId(user_id)})
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    # 获取状态
    state_doc = db.users_states.find_one({"user_id": ObjectId(user_id)})
    is_completed = state_doc.get("is_onboarding_completed", False) if state_doc else False
    
    # birthday 从 MongoDB 读出就是 datetime.datetime 类型，Pydantic 会自动处理
    # 如果是 date 类型，Pydantic 也会自动序列化为 YYYY-MM-DD 字符串
        
    return UserProfileResponse(
        user_id=str(user["_id"]),
        nickname=user.get("nickname", "未设置"),
        gender=user.get("gender", "male"), 
        birthday=user.get("birthday", date(2000, 1, 1)), # 提供一个默认的 date 对象
        city=user.get("city", "未设置"),
        height=user.get("height", 0),
        weight=user.get("weight", 0),
        self_intro=user.get("self_intro_raw", ""),
        is_profile_completed=is_completed
    )