# -*- coding: utf-8 -*-
from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str # 这里对应 account (手机/邮箱)
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str # 为了方便前端，顺便返回 user_id
