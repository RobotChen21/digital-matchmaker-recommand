# -*- coding: utf-8 -*-
import json
import random
from typing import List
from bson import ObjectId
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.common.models.user import VirtualUser
from app.db.mongo_manager import MongoDBManager

class VirtualUserGenerator:
    """虚拟用户生成器"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=VirtualUser)

        # 随机种子库：用于打破 LLM 的生成惯性
        self.diversity_seeds = [
            "职业: 程序员, 性格: 内向宅", "职业: 教师, 性格: 温柔顾家",
            "职业: 自由摄影师, 性格: 浪漫随性", "职业: 金融分析师, 性格: 理性精英",
            "职业: 护士, 性格: 细心体贴", "职业: 创业者, 性格: 冒险野心",
            "职业: 公务员, 性格: 稳重踏实", "职业: 健身教练, 性格: 阳光外向",
            "职业: 或是艺术家, 性格: 敏感独特", "职业: 销售经理, 性格: 社交达人",
            "爱好: 极限运动", "爱好: 古典音乐", "爱好: 二次元动漫", "爱好: 烹饪烘焙",
            "生活状态: 刚毕业迷茫期", "生活状态: 事业上升期忙碌", "生活状态: 享受生活佛系"
        ]

        self.prompt = ChatPromptTemplate.from_template(
            """请创建一个真实可信的虚拟人物,包括基础信息和完整的性格种子。

要求:
- 自我介绍不超过 40 字,自然真实
- 性格特征要具体且一致
- 职业描述要详细真实
- 兴趣爱好要多样化
- 感情经历要合理
- 沟通风格要明确
- 不要说"我是AI"或任何虚拟相关词汇

【多样性指令】:
请基于以下设定进行发散创作（不要完全照搬，而是作为灵感来源）：
{diversity_hint}

请输出纯 JSON 格式,不要任何解释或 Markdown 标记。

{format_instructions}

请生成一个真实的中国用户画像。"""
        )

    def generate_user(self) -> VirtualUser:
        """生成一个虚拟用户"""
        # 随机抽取 1-2 个特征组合作为灵感种子
        seed = ", ".join(random.sample(self.diversity_seeds, k=2))
        
        chain = self.prompt | self.llm
        response = chain.invoke({
            "format_instructions": self.parser.get_format_instructions(),
            "diversity_hint": seed
        })
        
        content = response.content.strip()
        
        # 尝试使用 PydanticOutputParser 直接解析
        try:
            return self.parser.parse(content)
        except Exception as e:
            print(f"⚠️ Parser failed, trying manual cleanup: {e}")
            # 手动清洗重试
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            try:
                # strict=False 允许控制字符
                user_dict = json.loads(content, strict=False)
                return VirtualUser(**user_dict)
            except Exception as e2:
                print(f"❌ Final parsing failed. Content preview: {content[:100]}...")
                raise e2

    def generate_batch(self, count: int, db_manager: MongoDBManager) -> List[ObjectId]:
        """批量生成用户并存入数据库"""
        user_ids = []
        for i in range(count):
            try:
                user = self.generate_user()
                user_dict = user.model_dump()
                persona_dict = user_dict.pop("persona_seed")

                user_id = db_manager.insert_user_with_persona(user_dict, persona_dict)
                user_ids.append(user_id)
                print(f"✅ 用户 {i + 1}/{count} 生成成功: {user.nickname} (ID: {user_id})")
            except Exception as e:
                print(f"❌ 用户 {i + 1} 生成失败: {e}")
        return user_ids