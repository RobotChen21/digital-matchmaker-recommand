# -*- coding: utf-8 -*-
from bson import ObjectId
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.core.config import settings
from app.core.llm import get_llm
from app.common.models.state import MatchmakingState
from app.services.ai.workflows.recommendation.state import FilterOutput, RefineOutput

class FilterNode:
    def __init__(self, db_manager):
        self.db = db_manager
        self.llm = get_llm(temperature=0)
        
        self.filter_parser = PydanticOutputParser(pydantic_object=FilterOutput)
        self.filter_chain = (
            ChatPromptTemplate.from_template(
                """你是 MongoDB 查询专家。根据用户描述和当前用户画像，生成 MongoDB 查询语句。
                
                当前用户: {user_profile_summary}
                用户需求: {user_input}
                
                【硬性规则 - 重要!】:
                1. **绝对不要**包含 `gender` 或 `sex` 字段。
                2. **只允许**筛选以下字段: `city`, `height`.
                3. 年龄请提取 `age_min` 和 `age_max`。
                4. 身材请提取 `bmi_min` 和 `bmi_max`，参考以下映射表：
                   - "很瘦/骨感" -> bmi_max=18.5
                   - "瘦/苗条/纤细" -> bmi_max=20
                   - "不胖/匀称/标准" -> bmi_min=18.5, bmi_max=24
                   - "微胖/丰满/有肉/壮实" -> bmi_min=24, bmi_max=28
                   - "胖/大码" -> bmi_min=28
                   - "不要太瘦" -> bmi_min=18.5
                   - "不要胖的" -> bmi_max=24
                   - "不要太胖" -> bmi_max=28 (包含微胖)
                
                例如:
                - "我要找上海的" -> {{"mongo_query": {{"city": "上海"}}}}
                - "25-30岁，微胖也可以" -> {{"age_min": 25, "age_max": 30, "bmi_max": 28}}
                - "找个瘦一点的" -> {{"bmi_max": 20}}
                
                输出JSON: {format_instructions}"""
            ) | self.llm | self.filter_parser
        )

        self.refine_parser = PydanticOutputParser(pydantic_object=RefineOutput)
        self.refine_chain = (
            ChatPromptTemplate.from_template(
                """你是一个聪明的红娘。用户之前的要求太高了，导致数据库里找不到人。
                请你根据之前的失败查询，生成一个新的、稍微**放宽**一点的要求描述。
                
                【失败的查询描述】: {current_input}
                【失败的 Mongo 条件】: {hard_filters}
                
                【策略】:
                - 扩大年龄范围
                - 放宽身材要求
                - 移除地理位置限制
                
                请输出放宽后的自然语言描述。
                输出JSON: {format_instructions}"""
            ) | self.llm | self.refine_parser
        )

    def hard_filter(self, state: MatchmakingState):
        """Step 2: 硬性筛选"""
        print(f"🔍 [HardFilter] 生成条件 (第 {state.get('search_count', 0) + 1} 次尝试)...")
        
        try:
            res = self.filter_chain.invoke({
                "user_profile_summary": state.get('current_user_summary', 'Unknown'),
                "user_input": state['current_input'],
                "format_instructions": self.filter_parser.get_format_instructions()
            })
            query = res.mongo_query if res.mongo_query else {}
            
            # 1. 处理年龄区间
            age_min = res.age_min
            age_max = res.age_max
            if age_min or age_max:
                current_year = datetime.now().year
                if age_max:
                    max_birth_year = current_year - age_max
                    min_birthday = datetime(max_birth_year, 1, 1)
                    query["birthday"] = {"$gte": min_birthday}
                if age_min:
                    min_birth_year = current_year - age_min
                    max_birthday = datetime(min_birth_year, 12, 31)
                    if "birthday" in query:
                        query["birthday"]["$lte"] = max_birthday
                    else:
                        query["birthday"] = {"$lte": max_birthday}

            # 2. 处理 BMI 过滤 ($expr)
            bmi_min = res.bmi_min
            bmi_max = res.bmi_max
            if bmi_min or bmi_max:
                # BMI = weight / (height/100)^2
                bmi_expr = {
                    "$divide": ["$weight", {"$pow": [{"$divide": ["$height", 100]}, 2]}]
                }
                
                expr_conds = []
                if bmi_min:
                    expr_conds.append({"$gte": [bmi_expr, bmi_min]})
                if bmi_max:
                    expr_conds.append({"$lte": [bmi_expr, bmi_max]})
                
                if expr_conds:
                    if len(expr_conds) == 1:
                        query["$expr"] = expr_conds[0]
                    else:
                        query["$expr"] = {"$and": expr_conds}
                
                print(f"   -> Calculated BMI filter: {bmi_min}-{bmi_max}")

            print(f"   -> LLM Query (before gender): {query}")
            
            # 3. 强制注入性别筛选
            current_gender = state.get('current_user_gender')
            target_gender = None
            if current_gender:
                cg = current_gender.lower()
                if cg == 'female': target_gender = 'male'
                elif cg == 'male': target_gender = 'female'
            
            if target_gender:
                query['gender'] = target_gender

            # 4. 排除自己 和 排除已见过的候选人 ("换一批")
            exclude_ids = [ObjectId(state['user_id'])]
            
            seen_ids = state.get('seen_candidate_ids', [])
            if seen_ids:
                print(f"   -> Excluding {len(seen_ids)} previously seen candidates.")
                for sid in seen_ids:
                    try:
                        exclude_ids.append(ObjectId(sid))
                    except:
                        pass
            
            query["_id"] = {"$nin": exclude_ids}
            
            print(f"   -> Final Mongo Query: {query}")

            cursor = self.db.users_basic.find(query, {"_id": 1}).limit(50)
            candidate_ids = [str(doc['_id']) for doc in cursor]
            
            state['hard_filters'] = query
            state['hard_candidate_ids'] = candidate_ids
            print(f"   -> 命中: {len(candidate_ids)} 人")
            
        except Exception as e:
            print(f"   ❌ 筛选失败: {e}")
            state['hard_candidate_ids'] = []
            
        return state

    def refine_query(self, state: MatchmakingState):
        """Step 2.5: 自修正节点"""
        print("🔄 [Refine] 结果为空，尝试放宽条件...")
        
        try:
            res = self.refine_chain.invoke({
                "current_input": state['current_input'],
                "hard_filters": state.get('hard_filters', {}),
                "format_instructions": self.refine_parser.get_format_instructions()
            })
            
            print(f"   -> 修正策略: {res.reason}")
            print(f"   -> 新查询: {res.relaxed_query}")
            
            state['current_input'] = res.relaxed_query
            state['search_count'] = state.get('search_count', 0) + 1
            
        except Exception as e:
            print(f"   ❌ 修正失败: {e}")
            state['search_count'] = 99
            
        return state
