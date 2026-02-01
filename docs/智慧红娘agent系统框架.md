# 智慧红娘 Agent 系统框架 (Digital Matchmaker System Framework)

### **一、系统整体架构概览 (FastAPI + LangGraph + Mongo + ES + Chroma)**

*   **分层架构**：严格遵循 Controller-Service-Repository 模式，代码模块化，职责分离。
*   **核心驱动**：LangGraph 作为大脑，编排复杂的推荐逻辑。
*   **数据层**：
    *   **MongoDB**：存储用户基础信息、完整画像、鉴权数据及系统状态。
    *   **Elasticsearch**：存储候选人索引，负责高性能的**混合检索 (Vector + Text)**。
    *   **ChromaDB**：存储历史对话记录的向量，用于 **RAG (Evidence Hunting)** 证据召回。
*   **LLM 驱动**：大量使用 LLM 进行意图识别、数据提取、文本生成、内容总结。
*   **安全认证**：实现了 JWT 认证机制，保护 API 接口。

### **二、用户生命周期流程 (User Onboarding Flow)**

1.  **账号注册 (`POST /api/v1/users/register`)**
    *   **输入**：`account` (手机/邮箱), `password`。
    *   **动作**：
        1.  哈希 `password`，将 `account` 和 `password_hash` 存储在 `users_auth` 集合。
        2.  同时在 `users_basic` 集合中创建一个**占位**文档（包含 `user_id`, `created_at`, `is_completed=False`, 以及默认的昵称、性别、生日等），以确保后续流程能找到该用户的记录。
    *   **产出**：`user_id`。

2.  **登录 (`POST /api/v1/auth/login`)**
    *   **输入**：`username` (account), `password`。
    *   **动作**：验证密码。
    *   **产出**：JWT `access_token` (包含 `user_id`)。

3.  **完善资料 (`PUT /api/v1/users/profile`)**
    *   **输入**：`nickname`, `gender`, `birthday` (Date), `city`, `height`, `self_intro`。
    *   **鉴权**：请求需携带 JWT Token。
    *   **动作**：更新 `users_basic` 集合中该 `user_id` 对应的记录，填充真实的用户资料。但**不修改 `is_completed` 状态** (仍为 `False`)。

4.  **实时信息采集 (Onboarding) - 核心**
    *   **触发**：用户登录后进入聊天界面，首次发送消息，且 `users_basic.is_completed` 为 `False`。
    *   **核心机制 (Hint-Driven & Multi-Agent)**：
        1.  **动态指引 (Hint-Driven Strategy)**：
            *   红娘 Agent 本身不维护复杂的全局状态，而是依赖 `ProfileService` 实时生成的 `profile_completion_hint`。
            *   每次回复前，系统会对比当前画像与 `REQUIRED_PROFILE_DIMENSIONS`，生成一段自然语言提示（如：“已收集学历，缺少家庭背景，建议追问父母情况”）。
            *   红娘 Agent 接收此 Hint 作为系统指令，结合**最近 10 条对话历史**（Sliding Window），生成极具针对性的追问，确保对话不跑题、不遗漏。
        2.  **多 Agent 并行提取 (Map-Reduce)**：
            *   **触发时机**：每积累约 3-4 轮对话，触发一次增量提取。
            *   **并行执行**：`ProfileService` 调度 **10+ 个细分领域的垂直 Agent**（如 `EducationExtractor`, `PersonalityExtractor`, `ValuesExtractor` 等）并行工作。
            *   **逻辑**：每个 Agent 只关注自己领域的字段（如教育 Agent 只看学历学校）。
            *   **聚合**：所有 Agent 的提取结果通过 `smart_merge` 策略智能合并到 MongoDB 的 `users_profile` 中（列表追加，标量覆盖）。
        3.  **记忆压缩与状态同步**：
            *   **短期记忆**：Prompt 中只保留最近 N 条原始对话，避免 Context Window 爆炸。
            *   **长期记忆**：通过不断更新的结构化 Profile 和由此生成的 Hint，让红娘“记住”了用户很久之前说过的信息（因为 Hint 会告诉它“已收集”）。
        4.  **动态终止判定**：
            *   `TerminationManager` 依据生成的 Hint（分析画像完整度）和对话轮数阈值，判断是否结束采集。
    *   **最终一致原子块**：
        *   采集完成后，调用 `UserInitializationService` 进行收尾：
            *   **画像摘要生成**：LLM 将结构化 JSON 转写为一段 350 字的生动第三人称自我介绍。
            *   **向量化入库**：将画像摘要写入 ChromaDB (`profile_summary`)。
            *   **状态翻转**：`users_basic.is_completed` -> `True`。

### **三、核心推荐工作流 (LangGraph Agent)**

当用户的 `is_completed` 为 `True` 时，系统进入推荐模式。当前的推荐引擎采用了 **ES 混合检索 (Hybrid Search) + RRF 融合 + 心理学精排** 的架构，远比传统的漏斗模型复杂和精准。

#### **1. 感知与意图路由 (`IntentNode` & `ChitchatNode`)**
*   **全量画像加载 (`LoadProfile`)**：
    *   每次会话开始时，系统加载用户的完整画像 (`user_basic` + `user_profile`) 到 State。
    *   **缓存优化**：内置智能缓存机制，仅当画像更新时间晚于摘要生成时间时，才重新调用 LLM 生成 `Profile Summary`，大幅降低 Token 消耗。
*   **意图识别 (`IntentNode`)**：
    *   LLM 分析用户输入与上下文，将意图精准分类为 4 种：
        *   **`search_candidate`**：发起新搜索或修改条件（如“找个 180 的”）。
        *   **`refresh_candidate`**：基于当前条件翻页（如“换一批”、“再看看”）。
        *   **`deep_dive`**：对特定候选人感兴趣，想深入了解。
        *   **`chitchat`**：闲聊、打招呼或情感咨询。
*   **情感顾问模式 (`ChitchatNode`)**：
    *   如果意图判定为 `chitchat`，系统不会简单的敷衍。
    *   红娘切换为**“资深情感顾问”**模式，结合用户的画像摘要，提供有针对性的恋爱建议、个人提升指导或情绪安抚，保持专业且温暖的“人设”。
*   **意图路由 (Routing Logic)**：
    *   根据识别结果，LangGraph 将流程分发到不同分支：
        *   **`search_candidate` / `refresh_candidate`** ➜ 进入 **`FilterNode`**，启动核心推荐链路 (Filter -> Recall -> Rank)。
        *   **`deep_dive`** ➜ 进入 **`DeepDiveNode`**，进行多轮深度问答。
        *   **`chitchat`** ➜ 进入 **`ChitchatNode`**，回复后直接结束当前 Turn。

#### **2. 智能过滤与自修正 (`FilterNode` & `RefineNode`)**
*   **双模式提取**：LLM 从用户口语中一次性提取两类信号：
    *   **硬性指标 (Hard Filters)**：映射为 MongoDB 查询 (如 `city="杭州"`, `age >= 25`, `bmi < 24`)。
    *   **语义关键词 (Keywords)**：提取为 ES 检索词 (如 "985 程序员 温柔 顾家")。
*   **Refine Loop (自修正闭环)**：
    *   如果 Hard Filter 导致 MongoDB 命中结果为 0，系统不会直接返回失败。
    *   **自动降级策略**：进入 `RefineNode`，LLM 分析失败原因，生成一个**结构化的放宽策略**（如：主动扩大年龄范围 ±3 岁，移除次要的地域限制），保留核心语义关键词，重新发起搜索。

#### **3. 混合召回 Hybrid Search & RRF (`RecallNode`)**
这是系统的核心检索引擎，利用 **Elasticsearch** 同时进行两路召回，并使用 **RRF (Reciprocal Rank Fusion)** 算法进行结果融合：
*   **向量通路 (Vector Path - KNN)**：
    *   使用 Embedding 模型将用户的语义需求转化为向量。
    *   在 ES 中对 `profile_vector` 字段进行 KNN 近邻搜索，捕捉“气味相投”、“感觉对”等隐性特征。
*   **文本通路 (Text Path - BM25)**：
    *   使用提取的 `keywords` 对 `tags` (结构化标签) 和 `profile_text` (画像文本) 进行 BM25 全文检索。
    *   确保“985”、“独生子”、“不抽烟”等显性特征的精确匹配。
*   **RRF 融合 (Application Layer)**：
    *   系统在应用层手动实现了 RRF 算法 (`score = 1 / (k + rank)`)。
    *   将向量和文本两路召回的排名进行加权融合，生成最终的 Top N 候选人列表。这避免了单向量检索“查准率低”和单关键字检索“查全率低”的问题。

#### **4. 语义匹配度精排 (`RankingNode`)**
对 RRF 召回的 Top 30 候选人进行语义匹配度评分，计算 `Compatibility Score`：
*   **MBTI 匹配逻辑**：
    *   **同频加分**：MBTI 完全一致 (+10分)。
    *   **互补加分**：E/I 维度相反但认知功能（后三位）相同（如 ENFP & INFP）给予更高权重 (+15分)，符合心理学“互补吸引”理论。
*   **生活方式共鸣**：烟酒习惯的一致性（尤其是“不抽烟”的强匹配）。
*   **兴趣交叉 (Tags Intersection)**：计算双方兴趣标签的重合度。
*   **综合得分**：`Final Score = Base Rank Score (from RRF) + Psychological Score`。

#### **5. 证据与回复 (`ResponseNode`)**
*   **Evidence Hunting**：对最终 Top 3，LLM 深入其 ChromaDB 的对话历史中寻找“证据”。
*   **可解释性推荐**：生成的回复不仅推荐人，还会带上理由（如：“推荐王先生，不仅因为他也喜欢滑雪（兴趣共鸣），而且他曾在对话中提到‘希望能找个一起去崇礼的伴’（RAG 证据）”）。

#### **6. 深度交互 (`DeepDiveNode`) - 上下文感知与持久化**
当用户对某位候选人产生兴趣并进一步追问（如：“她性格怎么样？”，“怎么追他？”）时，系统进入 `DeepDiveNode`。
该节点具有极强的**上下文感知能力**，支持多轮连续追问。

*   **智能指代消解 (Contextual Entity Resolution)**：
    *   **问题**：用户常使用代词（“她”、“这个人”）或序数词（“第一个”）进行指代。
    *   **解决**：系统运行一个专门的 `TargetExtractorChain`，结合**对话历史**、**当前候选人列表**和**用户输入**，精准解析用户指的是谁。
*   **状态持久化 (State Persistence)**：
    *   为了支持连续对话，`MatchmakingState` 中专门维护了 `last_target_person` 字段。
    *   **上下文切换**：
        *   **显式切换**：如果用户直接问“那第二个呢？”或“王小姐怎么样？”，指代消解模块提取新目标并更新状态。
        *   **隐式追问**：如果用户问“她喜欢什么？”，系统利用持久化的目标信息快速锁定上一轮讨论对象，实现无缝的多轮深度交互。
*   **双层信息检索**：
    *   **静态画像**：加载候选人缓存的 `Profile Summary`，获取全貌。
    *   **动态证据 (RAG)**：针对具体问题（如“恋爱观”），实时在 ChromaDB 中检索该候选人的过往对话记录 (`onboarding` 或 `social` 记录)。
*   **专家级咨询**：
    *   最终，红娘 Agent 扮演心理咨询师角色，结合**画像事实**和**对话细节**，给出既有事实依据又具情感洞察的回答（例如：“虽然她资料写比较宅，但从对话看，她其实很渴望有人带她去户外...”）。

### **四、性能与稳定性优化 (Performance & Robustness)**

针对 LLM 应用常见的延迟高、不稳定性问题，系统在画像提取环节进行了深度优化。

#### **多 Agent 并行提取与容错 (Parallel Extraction & Fault Tolerance)**
*   **场景**：`ProfileService` 需要同时调度 10+ 个垂直领域的 Extractor（教育、职业、性格、家庭等）来处理对话日志。
*   **技术选型对比**：
    *   **LangChain 原生并行 (`RunnableParallel`)**：虽然 LCEL 提供了优雅的并行构建方式，但其默认行为往往是 **Fail-Fast** 的。即如果并行的 10 条链中有 1 条因为 JSON 解析失败抛出异常，整个并行任务可能会中断，或者需要极其复杂的 fallback 逻辑配置。
    *   **Python 线程池 (`ThreadPoolExecutor`)**：我们选择了更底层的多线程方案。
*   **设计哲学 (Best-Effort Strategy)**：
    *   **隔离性与鲁棒性**：我们不仅是为了快（将耗时压缩至最慢的一个 Agent），更是为了**局部失败容忍**。在显式线程池中，每个 Agent 的执行都被独立的 `try-except` 保护。
    *   **结果**：如果“价值观提取器”崩溃了，系统只会记录一条 Warning 并跳过该字段，**绝不会拖累其他 9 个正常的提取器**。这种“能跑多少是多少”的 Best-Effort 策略，对于容错率要求极高的生产环境至关重要。

### **五、工程化与设计模式 (Engineering & Design Patterns)**

本项目采用了一些经典的软件工程设计模式，以确保系统的可维护性、扩展性和性能。

#### **1. 轻量级 IOC 容器 (Manual IOC Container)**
*   **核心组件**：`app.core.container.AppContainer`
*   **单例模式 (Singleton)**：
    *   容器本身是全局单例，确保 `MongoDBManager`, `ESManager`, `ChromaManager` 等数据库连接和核心服务 (`ProfileService`) 在应用生命周期内只被初始化一次。
*   **懒加载 (Lazy Loading)**：
    *   所有服务属性均使用 `@property` 装饰器实现延迟初始化。依赖只有在第一次被调用时才会被创建，有效减少了冷启动时间。
*   **依赖注入与解耦**：
    *   业务节点（如 `OnboardingNode`）不再直接 `new` 对象，而是从 `container` 请求依赖。这消除了模块间的强耦合，也解决了复杂的循环依赖问题（通过在方法内部局部导入）。

#### **2. LLM 工厂模式 (LLM Factory)**
*   **上下文感知配置**：
    *   `get_llm(type)` 方法充当工厂，根据任务场景分发不同配置的 LLM 实例：
        *   **`"intent"` (Temp=0.0)**：严谨模式，用于意图识别、JSON 提取、指代消解。
        *   **`"reason"` (Temp=0.4)**：平衡模式，用于逻辑推演、画像摘要生成、专业建议。
        *   **`"chat"` (Temp=0.7)**：创造模式，用于拟人化闲聊、共情安抚、Onboarding 追问。
*   **统一资源管理**：
    *   集中管理 API Key、Base URL 和 Model Name，方便在不同环境（Dev/Prod）或不同模型供应商（OpenAI/DeepSeek/Qwen）之间一键切换。

### **六、ReAct 与 CoT 思维链的应用 (ReAct & CoT Application)**

本项目不仅仅是简单的 LLM 调用，而是深度融合了 Agentic Workflow 的核心设计思想。

#### **1. CoT (Chain of Thought) 在画像提取中的应用**
*   **场景**：对于性格打分、价值观判断等主观性极强的任务，直接让 LLM 输出 0.8 分往往会导致“幻觉”或随机性。
*   **实现**：在 `app/services/ai/agents/extractors.py` 中，所有提取 Agent（特别是 `ValuesExtractor`, `RiskExtractor`）都被强制要求遵循 **"Reasoning First"** 原则。
*   **Prompt 示例**：“【思维链要求】价值观是非常隐性的。**必须优先填充 'reasoning' 字段**。请结合用户对未来规划的描述、对选择的取舍，详细推导为什么给这个分数。”
*   **效果**：强制 LLM 先生成一段逻辑推演（Thinking Process），再输出最终的 JSON 结果。实验表明，这种 CoT 机制显著提升了画像数据的准确性和可解释性。

#### **2. ReAct (Reason + Act) 在推荐流中的闭环**
整个 LangGraph 的工作流本质上是一个巨大的 ReAct 循环：
*   **感知与规划 (Reason)**：`IntentNode` 分析用户输入，判断意图是搜索、深挖还是闲聊。
*   **行动 (Act)**：系统根据意图调用相应的工具节点（如 `FilterNode`, `DeepDiveNode`）。
*   **观察与修正 (Observation & Correction)**：
    *   当 `FilterNode` 执行搜索行动后，如果发现命中人数为 0（Observation）。
    *   系统不会直接报错，而是进入 `RefineNode` 进行二次思考（Reason）：分析是哪个条件太苛刻了？
    *   然后制定新的行动计划（Act）：生成放宽后的查询条件，并重新执行搜索。
    *   这是一个经典的 **Self-Correction** 模式，赋予了 Agent 自主解决问题的能力。

### **七、关键特性与亮点 (Key Features & Highlights)**

#### **1. 原子化状态收尾 (Atomic User Finalization)**
*   **实现位置**：`app.services.ai.workflows.user_init.UserInitializationService`
*   **跨库一致性**：Onboarding 结束时，系统执行一个类似分布式事务的原子操作，确保数据在三个数据库间的一致性：
    1.  **MongoDB**：更新用户状态 (`is_completed=True`)。
    2.  **Elasticsearch**：构建倒排索引和向量索引，支持后续的混合检索。
    3.  **ChromaDB**：将历史对话切片向量化，作为未来推荐时的 RAG 证据。
*   **意义**：保证用户一旦结束对话，立即可被系统检索和推荐，无数据延迟。

#### **2. 双重智能终止判定 (Dual-Logic Termination)**
*   **实现位置**：`app.services.ai.tools.termination.DialogueTerminationManager`
*   **策略**：系统不再仅仅依赖简单的“对话轮数”，而是引入了双 LLM 判决机制：
    *   **任务导向 (`InfoCompletenessDetector`)**：实时评估画像完整度，核心维度（教育/职业/家庭）收集全了才放行。
    *   **高情商导向 (`HesitancyDetector`)**：实时检测用户情绪。如果用户表现出“疲惫”、“敷衍”或“不耐烦”，系统会果断提前结束采集，优先保护用户体验，而非强行完成任务。

#### **3. 会话快照与持久化 (Session Snapshotting)**
*   **实现位置**：`app.services.session_service.SessionService`
*   **存储结构**：在 MongoDB 中以 `session_id` 为核心，聚合存储 `messages` (完整对话列表) 和 `latest_state` (状态快照)。
*   **元信息机制 (State as Metadata)**：
    *   `latest_state` 被视为会话的**“元信息”**，包含了 `seen_candidate_ids` (去重记录)、`last_search_criteria` (搜索条件)、`last_target_person` (指代目标) 等关键变量。
*   **价值**：
    *   **上下文唤醒**：状态不再仅仅存在于内存中。当用户通过 `session_id` 重新进入对话时，系统能瞬间“唤醒”之前的搜索进度和讨论目标。
    *   **业务连贯性**：得益于这种“元信息”设计，用户即使在几天后发送“换一批”，系统依然能精准排除已阅候选人，并基于之前的筛选逻辑继续工作，提供了极致的业务连贯性。

### **八、多源数据协同与生命周期管理 (Data Coordination & Lifecycle)**

系统采用 **“MongoDB 主库 + ES/Chroma 索引库”** 的架构，明确了“持久化层”与“计算检索层”的职责边界。

#### **1. 三库协同架构 (Triple-DB Coordination)**
*   **MongoDB (The Source of Truth - 大本营)**：
    *   **职责**：数据的最终归宿。存储**全量**的用户画像、**无限**的聊天记录列表、用户鉴权信息及 Session 状态。
    *   **地位**：所有数据以 Mongo 为准。即使向量库崩坏，也可以随时从 Mongo 重建索引。
*   **Elasticsearch (The Search Engine - 候选人索引)**：
    *   **职责**：高性能混合检索。
    *   **ID 过滤与冗余存储**：
        *   虽然 `FilterNode` 目前主要传递 `user_id` 列表给 ES 进行范围圈定，但我们依然在 ES 中冗余存储了 `city`, `age`, `gender`, `tags` 等基础信息。
        *   **设计考量**：这些文本数据占用空间极小，但赋予了 ES 独立进行复杂过滤的能力，方便后期扩展（如纯 ES 过滤）及 Debug。
    *   **检索内容**：存储 `profile_vector` (画像向量) 和 `tags/profile_text` (画像文本)，支持 Vector + BM25 并行查询。
*   **ChromaDB (The Evidence Store - 证据索引)**：
    *   **职责**：RAG 证据召回。
    *   **Meta 信息**：存储 `user_id`, `dialogue_type` (来源), `timestamp`。这是实施分级生命周期管理的关键。

#### **2. 分级生命周期策略 (Tiered Lifecycle Management)**
通过 `metadata.dialogue_type` 字段，系统对不同类型的对话数据实施了差异化的保留策略：
*   **`onboarding` (永久保留 - Permanent)**：
    *   这是用户与红娘进行的初始画像采集对话。它是用户“数字孪生”的基石，**永不删除/切片**，确保系统始终能完整回溯用户的核心设定。
*   **`social` / 红娘对话 (用户管理 - Persistent)**：
    *   类似 ChatGPT 的会话历史。除非用户主动删除会话，否则这部分红娘与用户的交互记录会一直保留索引，以便红娘“记得”之前的建议和用户的反馈。
*   **用户间对话 (滑动窗口 - Sliding Window)**：
    *   (针对未来扩展的用户互聊场景) 采用**时间滑动窗口**策略。只索引最近 N 天或 N 轮的对话，用于捕捉即时上下文，避免高频闲聊导致索引库无限膨胀。
*   **智能切片参数详解 (Smart Chunking Rationale)**：
    *   **元信息锚点**：每个向量块 (Chunk) 的 metadata 中都包含 `session_id` 或 `source_id`，这不仅用于过滤，更是**回溯 MongoDB 主数据**的锚点，确保每一条检索到的证据都能找到原始出处。
    *   **Size=5 (窗口大小)**：设置为 5 条消息，通常覆盖约 2.5 轮对话（用户-红娘-用户-红娘-用户）。这足以包含一个完整的“提问-回答-追问”语义闭环，避免断章取义。
    *   **Overlap=2 (重叠步长)**：设置为 2 条消息。确保相邻的 Chunk 之间有足够的语义重叠，防止关键信息（如话题转换的过渡句）被切断在两个块的缝隙中，保证了上下文的平滑过渡。

### **九、未来规划与演进 (Future Roadmap & MCP Evolution)**

#### **1. 基于 MCP 的动态地理空间匹配 (Dynamic Geospatial Matching via MCP)**
为了进一步提升推荐的精准度与实效性，我们计划引入 **Model Context Protocol (MCP)** 理念，重构地理位置服务模块。

*   **背景与痛点 (Pain Point)**：
    *   当前系统的城市匹配依赖 LLM 的静态知识库（幻觉风险）。例如用户说“我在上海郊区，别给我推江苏的人”，LLM 可能很难精准界定哪些卫星城（如昆山、花桥）属于江苏从而进行物理排除。
    *   缺乏**实时地理感知**能力，无法处理“推荐离我 5km 以内的异性”这类基于 LBS 的动态需求。

*   **解决方案 (The Agentic Solution)**：
    *   **标准化连接 (MCP Integration)**：将高德地图/百度地图封装为标准的 **MCP Server**，向 Agent 暴露 `get_nearby_cities(lat, lon, radius)` 等通用工具接口，而非硬编码 API。
    *   **ReAct 动态推理流**：
        1.  **感知 (Sense)**：获取用户实时 GPS，解析语义需求（“上海周边，但不要江苏”）。
        2.  **行动 I (Act)**：Agent 自主决定初始搜索半径（如 20km），调用 MCP 工具获取覆盖城市/区域列表。
        3.  **推理与过滤 (Reason)**：Agent 结合语义约束（“Exclude Jiangsu”）对返回的城市列表进行清洗，自动剔除昆山、太仓等行政区划属于江苏的城市。
        4.  **自适应调整 (Self-Correction)**：如果过滤后剩余候选城市过少，Agent **自动决策**扩大半径至 50km 并重试，直到获取足够的地理候选池。

*   **技术价值**：
    *   这是一个 **“大模型 + 外部动态上下文”** 的应用场景。通过 MCP 协议，我们让 Agent 具备了“像人一样查地图、看边界、做决策”的能力，实现了从“基于文本匹配”到“基于时空推理”的跨越。
    *   **详细方案**：关于 MCP 的具体技术实现细节，请参考文档 **[基于 MCP 的地理位置服务实现](./地理位置的MCP的实现.md)**。
