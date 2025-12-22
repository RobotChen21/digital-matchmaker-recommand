# -*- coding: utf-8 -*-
from bson import ObjectId
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.core.container import container
from app.common.models.state import MatchmakingState
from app.services.ai.workflows.recommendation.state import FilterOutput, RefineOutput
from app.core.utils.cal_utils import calc_age

class FilterNode:
    def __init__(self):
        self.db = container.db
        self.llm = container.get_llm("intent") # Filter 需要严谨，复用 intent 配置
        
        self.filter_parser = PydanticOutputParser(pydantic_object=FilterOutput)
        self.filter_chain = (
            ChatPromptTemplate.from_template(
                """你是红娘推荐系统的搜索解析中枢。请从【用户需求】中一次性提取**硬性过滤条件(Mongo)**和**语义检索关键词(ES)**。
                
                【当前用户信息】:
                {user_info}
                
                【用户需求】: {user_input}
                
                # 任务一：提取硬性过滤条件 (Mongo)
                1. **City**: 提取提到的所有城市，输出为字符串列表。
                   - 如 "上海或杭州" -> ["上海", "杭州"]。
                   - 如果用户说"找老乡/同城"，请参考用户信息中的城市。
                   - **相对位置处理**: 如果用户说 "周边", "附近", "隔壁城市" (如 "找周边的", "离我不远的"):
                     请读取【当前用户信息】里的**City**作为中心点，列出该城市周围 100-200km 范围内的 3-5 个主要城市名称。
                   - **模糊区域处理**: 如果用户说 "江浙沪", "大湾区" 等，请展开为该区域的核心城市列表。
                   - **严禁**输出 "周边", "附近" 等模糊后缀。
                2. **Height**: 提取身高范围(cm)。如 "1米8以上" -> height_min=180。
                   - 如果用户说"比我高"，请参考用户身高计算。
                3. **Age**: 提取年龄范围。如 "25到30岁" -> age_min=25, age_max=30；"大于20岁" -> age_min=20。
                   - 如果用户说"比我大"，"和我差不多"(上下3岁)，请参考用户年龄。
                4. **BMI**: 根据描述提取BMI范围。
                   - "很瘦/骨感" -> bmi_max=18.5
                   - "瘦/苗条/纤细" -> bmi_max=20
                   - "不胖/匀称/标准" -> bmi_min=18.5, bmi_max=24
                   - "微胖/丰满/有肉/壮实" -> bmi_min=24, bmi_max=28
                   - "胖/大码" -> bmi_min=28
                
                # 任务二：提取语义关键词 (ES Hybrid Search)
                请从用户需求中提取**所有**关于理想对象的描述词（关键词），用空格分隔。比如以下
                1. **教育与职业**: 学历(硕士/985/学校名)、专业、职位(程序员/经理)、行业、收入水平
                2. **家庭背景**: 成员状况(独生子女/有兄弟姐妹)、父母职业、经济条件。
                3. **生活方式**: 运动习惯、社交偏好、烟酒情况(不抽烟/偶尔喝酒)。
                4. **性格与三观**: MBTI/人格特质(温柔/开朗/内向)、价值观偏好。
                5. **情感与兴趣**: 恋爱风格(依恋类型/恋爱语言)、兴趣标签(滑雪/看书)。
                **提取范围**：      
                - 包括但不限于：学历要求、职业特征、家庭状况、性格特质、生活习惯、兴趣爱好、三观倾向等。 
                
                **唯一排除项**：
                - 请**不要**包含：City, Age, Height, Gender (这些已在任务一处理)。
                
                **Examples**:
                - "找杭州的985程序员，1米75以上" 
                  -> City=["杭州"], Height_min=175, Keywords="985 程序员"
                - "我要找个工作稳定的独生女，父母有退休金，不抽烟" 
                  -> Keywords="工作稳定 独生女 父母有退休金 不抽烟"
                
                输出JSON: {format_instructions}"""
            ) | self.llm | self.filter_parser
        )

        self.refine_parser = PydanticOutputParser(pydantic_object=RefineOutput)
        self.refine_chain = (
            ChatPromptTemplate.from_template(
                """你是一个聪明的红娘。用户之前的要求太高了，导致数据库里找不到人。
                请你根据之前的失败查询，生成一个新的、稍微**放宽**一点的结构化筛选条件。
                
                【失败的查询描述】: {current_input}
                【失败的 Mongo 条件】: {hard_filters}
                【当前的语义关键词】: {current_keywords}
                【放宽策略】:
                1. **只放宽硬性指标**：扩大年龄范围、放宽身高/BMI、移除地理位置限制。
                2.**严格保留关键词**：请**原封不动**地保留【当前的语义关键词】到输出中，除非它本身包含明显的冲突（如既要胖又要瘦），否则不要修改或删除关键词。
                
                请输出：
                1. `criteria`: 放宽后的完整结构化条件 (FilterOutput)。注意 `keywords` 字段要填入保留的关键词。
                2. `relaxed_query_str`: 放宽后的自然语言描述。
                3. `reason`: 解释理由。
                
                输出JSON: {format_instructions}"""
            ) | self.llm | self.refine_parser
        )

    def hard_filter(self, state: MatchmakingState):
        """Step 2: 统一提取 (Hard Filters + Semantic Keywords)"""
        print(f"🔍 [Filter] 提取条件 (Intent: {state.get('intent')})...")
        
        # --- 判断意图类型 & 检查是否有预设条件 ---
        is_refresh = (state.get('intent') == 'refresh_candidate')
        last_criteria = state.get('last_search_criteria')
        refined_criteria = state.get('refined_criteria') # [NEW] 来自 RefineNode 的结构化修正
        
        # 初始化 res (FilterOutput 对象或类似字典)
        res = None
        
        # --- 场景 A: 换一批 (Refresh) ---
        if is_refresh and last_criteria:
            print("   🔄 触发[换一批]: 继承上一轮搜索条件")
            query = last_criteria.get('hard_filters', {}).copy()
            semantic_query = last_criteria.get('semantic_query', "")
            # 保持 seen_ids 不变
            
        # --- 场景 B: 结构化修正 (Refine Loop) ---
        elif refined_criteria:
            print("   🔧 触发[自修正]: 使用 RefineNode 提供的结构化条件 (跳过提取)")
            # 直接使用 Pydantic 模型还原对象
            try:
                res = FilterOutput(**refined_criteria)
                state['refined_criteria'] = None # 消费完即毁
                state['seen_candidate_ids'] = [] # 修正条件后视为新搜索
            except Exception as e:
                print(f"   ❌ 还原修正条件失败: {e}")
                # 降级处理：视为新搜索
                pass
            
        # --- 场景 C: 新搜索 (Search Candidate) ---
        else:
            if is_refresh: print("   ⚠️ 用户请求换一批但无历史/修正条件，视为新搜索")
            
            state['seen_candidate_ids'] = []
            
            # LLM 提取
            user_basic = state.get('current_user_basic', {})
            user_age = calc_age(user_basic.get('birthday')) if user_basic.get('birthday') else "未知"
            user_info_str = (f"性别: {user_basic.get('gender', '未知')}, 年龄: {user_age}, "
                             f"身高: {user_basic.get('height', '未知')}cm, 体重: {user_basic.get('weight', '未知')}kg, "
                             f"城市: {user_basic.get('city', '未知')}")
            try:
                res = self.filter_chain.invoke({
                    "user_input": state['current_input'],
                    "user_info": user_info_str,
                    "format_instructions": self.filter_parser.get_format_instructions()
                })
            except Exception as e:
                print(f"   ❌ 筛选解析失败: {e}")
                state['hard_candidate_ids'] = []
                return state

        # --- 如果 res 存在 (场景 B 或 C)，则构建 Mongo Query ---
        if res:
            query = {}
            # City
            if res.city:
                cities = res.city if isinstance(res.city, list) else [res.city]
                if cities: query["city"] = {"$in": cities}
            # Height
            if res.height_min or res.height_max:
                h_query = {}
                if res.height_min: h_query["$gte"] = res.height_min
                if res.height_max: h_query["$lte"] = res.height_max
                query["height"] = h_query
            # BMI
            if res.bmi_min or res.bmi_max:
                bmi_calc = {"$divide": ["$weight", {"$pow": [{"$divide": ["$height", 100]}, 2]}]}
                expr = []
                if res.bmi_min: expr.append({"$gte": [bmi_calc, res.bmi_min]})
                if res.bmi_max: expr.append({"$lte": [bmi_calc, res.bmi_max]})
                if expr:
                    if "$expr" not in query: query["$expr"] = {"$and": expr}
                    else: query["$expr"]["$and"].extend(expr)
            # Age
            age_min = res.age_min
            age_max = res.age_max
            if age_min or age_max:
                now = datetime.now().year
                if age_max: query["birthday"] = {"$gte": datetime(now - age_max, 1, 1)}
                # 注意处理 min/max 的覆盖问题，这里简化处理
                if age_min: 
                    target = datetime(now - age_min, 12, 31)
                    if "birthday" in query: query["birthday"]["$lte"] = target
                    else: query["birthday"] = {"$lte": target}
            
            # Gender (强制)
            cg = state.get('current_user_basic', {}).get('gender', '').lower()
            if cg == 'female': query['gender'] = 'male'
            elif cg == 'male': query['gender'] = 'female'

            semantic_query = res.keywords
            
            # 保存 Criteria
            state['last_search_criteria'] = {
                "hard_filters": query.copy(),
                "semantic_query": semantic_query
            }

        # --- 通用逻辑: 排除 ID ---
        exclude_ids = [ObjectId(state['user_id'])]
        for sid in state.get('seen_candidate_ids', []):
            try: exclude_ids.append(ObjectId(sid))
            except: pass
        
        # 合并 _id
        if "_id" not in query: query["_id"] = {"$nin": exclude_ids}

        # --- 执行查询 ---
        print(f"   -> Hard Filter: {query}")
        print(f"   -> Semantic Keywords: '{semantic_query}'")
        
        try:
            cursor = self.db.users_basic.find(query, {"_id": 1}).limit(200)
            candidate_ids = [str(doc['_id']) for doc in cursor]
            
            state['hard_filters'] = query
            state['semantic_query'] = semantic_query
            state['hard_candidate_ids'] = candidate_ids
            print(f"   -> 命中(Mongo): {len(candidate_ids)} 人")
            
        except Exception as e:
            print(f"   ❌ Mongo 查询失败: {e}")
            state['hard_candidate_ids'] = []

        return state

    def refine_query(self, state: MatchmakingState):
        """Step 2.5: 自修正节点"""
        print("🔄 [Refine] 结果为空，尝试放宽条件...")
        try:
            res = self.refine_chain.invoke({
                "current_input": state['current_input'],
                "hard_filters": state.get('hard_filters', {}),
                "current_keywords": state.get('semantic_query', ""),  # [NEW] 传入当前关键词
                "format_instructions": self.refine_parser.get_format_instructions()
            })
            
            print(f"   -> 修正策略: {res.reason}")
            print(f"   -> 新查询(展示): {res.relaxed_query_str}")
            
            # [关键] 将结构化条件直接存入 state，供 hard_filter 消费
            state['refined_criteria'] = res.criteria.model_dump()
            state['current_input'] = res.relaxed_query_str # 更新展示文本
            state['search_count'] = state.get('search_count', 0) + 1
            
        except Exception as e:
            print(f"   ❌ 修正失败: {e}")
            state['search_count'] = 99
            
        return state
