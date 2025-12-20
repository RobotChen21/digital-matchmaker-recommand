# -*- coding: utf-8 -*-
from bson import ObjectId
from datetime import datetime, date
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.core.container import container
from app.common.models.state import MatchmakingState
from app.services.ai.workflows.recommendation.state import FilterOutput, RefineOutput

class FilterNode:
    def __init__(self):
        self.db = container.db
        self.llm = container.get_llm("intent") # Filter 需要严谨，复用 intent 配置
        
        self.filter_parser = PydanticOutputParser(pydantic_object=FilterOutput)
        self.filter_chain = (
            ChatPromptTemplate.from_template(
                """你是信息提取专家。请从用户描述中提取硬性筛选条件。
                
                用户需求: {user_input}
                
                【提取规则】:
                1. **City**: 提取提到的所有城市，输出为字符串列表。如 "上海或杭州" -> ["上海", "杭州"]。
                2. **Height**: 提取身高范围(cm)。如 "1米8以上" -> height_min=180。
                3. **Age**: 提取年龄范围。如 "25到30岁" -> age_min=25, age_max=30；"大于20岁" -> age_min=20。
                4. **BMI**: 根据描述提取BMI范围。
                   - "很瘦/骨感" -> bmi_max=18.5
                   - "瘦/苗条/纤细" -> bmi_max=20
                   - "不胖/匀称/标准" -> bmi_min=18.5, bmi_max=24
                   - "微胖/丰满/有肉/壮实" -> bmi_min=24, bmi_max=28
                   - "胖/大码" -> bmi_min=28
                例如:
                - "我要找上海或苏州的" -> {{"city": ["上海", "苏州"]}}
                - "25-30岁，175以上" -> {{"age_min": 25, "age_max": 30, "height_min": 175}}
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
                "user_input": state['current_input'],
                "format_instructions": self.filter_parser.get_format_instructions()
            })
            
            # 手动构建 Mongo Query
            query = {}

            # 1. City (List -> $in)
            if res.city:
                # 如果只有一个城市且不是列表（兼容旧习惯），转为列表
                cities = res.city if isinstance(res.city, list) else [res.city]
                if cities:
                    query["city"] = {"$in": cities}

            # 2. Height
            if res.height_min or res.height_max:
                h_query = {}
                if res.height_min: h_query["$gte"] = res.height_min
                if res.height_max: h_query["$lte"] = res.height_max
                query["height"] = h_query

            # 3. BMI (动态计算: weight / (height/100)^2)
            if res.bmi_min or res.bmi_max:
                # BMI = weight_kg / (height_m ^ 2)
                # MongoDB aggregation syntax within $expr
                bmi_calc = {
                    "$divide": [
                        "$weight", 
                        {"$pow": [{"$divide": ["$height", 100]}, 2]}
                    ]
                }
                
                expr_conditions = []
                if res.bmi_min:
                    expr_conditions.append({"$gte": [bmi_calc, res.bmi_min]})
                if res.bmi_max:
                    expr_conditions.append({"$lte": [bmi_calc, res.bmi_max]})
                
                if expr_conditions:
                    if "$expr" not in query:
                        query["$expr"] = {"$and": expr_conditions}
                    else:
                        # 如果已有 $expr (虽然目前不太可能)，需要合并
                        if "$and" not in query["$expr"]:
                             query["$expr"] = {"$and": [query["$expr"]] + expr_conditions}
                        else:
                             query["$expr"]["$and"].extend(expr_conditions)
            
            # 4. 处理年龄区间
            age_min = res.age_min
            age_max = res.age_max
            if age_min or age_max:
                current_year = datetime.now().year
                if age_max:
                    max_birth_year = current_year - age_max
                    # PyMongo requires datetime.datetime, not datetime.date
                    min_birthday = datetime(max_birth_year, 1, 1)
                    query["birthday"] = {"$gte": min_birthday}
                    print(f"   -> Calculated birthday min: {min_birthday.strftime('%Y-%m-%d')}")
                if age_min:
                    min_birth_year = current_year - age_min
                    max_birthday = datetime(min_birth_year, 12, 31)
                    if "birthday" in query:
                        query["birthday"]["$lte"] = max_birthday
                    else:
                        query["birthday"] = {"$lte": max_birthday}
                    print(f"   -> Calculated birthday max: {max_birthday.strftime('%Y-%m-%d')}")
            
            print(f"   -> Constructed Query (before gender): {query}")
            
            # 2. 强制注入性别筛选
            current_gender = state.get('current_user_gender')
            target_gender = None
            if current_gender:
                cg = current_gender.lower()
                if cg == 'female': target_gender = 'male'
                elif cg == 'male': target_gender = 'female'
            
            if target_gender:
                query['gender'] = target_gender

            # 3. 排除自己 和 排除已见过的候选人 ("换一批")
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