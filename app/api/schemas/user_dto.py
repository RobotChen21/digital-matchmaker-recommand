# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date # 导入 date 类型

# 1. 账号注册
class UserRegisterRequest(BaseModel):
    account: str = Field(description="登录账号 (手机号/邮箱)")
    password: str = Field(description="登录密码")

class UserRegisterResponse(BaseModel):
    user_id: str
    message: str

# 2. 资料完善
class UserProfileUpdate(BaseModel):
    nickname: str = Field(description="昵称")
    gender: Literal["male", "female"] = Field(description="性别")
    birthday: date = Field(description="出生日期 YYYY-MM-DD") # 强制为 date 类型
    city: str = Field(description="所在城市")
    height: int = Field(description="身高 (cm)")
    weight: int = Field(description="体重 (kg)")
    self_intro: Optional[str] = Field("", description="自我介绍")

# 3. 资料获取
class UserProfileResponse(BaseModel): # 直接继承 BaseModel，而非 UserProfileUpdate，方便扩展
    user_id: str
    nickname: str
    gender: str
    birthday: date # 确保是 date 类型
    city: str
    height: int
    self_intro: str
    weight: int
    is_profile_completed: bool = Field(False, description="资料是否已完善")
