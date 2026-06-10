# 智慧红娘 Agent 模型蒸馏详细实现方案：基于“双脑”架构与 10 大维度画像

## 1. 核心目标：提示词内化与工程解耦
本方案旨在将当前系统中依赖的 **20+ 个复杂 Prompt 模板** 通过蒸馏技术固化到 7B/8B 级小模型中，实现“零 Prompt 启动”。通过 **Multi-LoRA** 技术，在同一个推理引擎中同时运行“逻辑抽取（左脑）”和“拟人对话（右脑）”两个脑区。

---

## 2. “左脑”逻辑抽取区 (Left Brain - Precision Extraction)
**训练目标**：100% 遵循指令，准确提取 JSON，具备极强的格式约束能力。
**对应维度**：涵盖画像系统的 10 大核心维度。

### 2.1 任务映射表 (Task Mapping)
| 模块名称 | 对应代码位置 | 蒸馏目标 | 负采样策略 |
| :--- | :--- | :--- | :--- |
| **性格提取** | `PersonalityExtractor` | 提取 Big5 权重与 MBTI | 用户胡言乱语时输出 `mbti: null` |
| **兴趣提取** | `InterestExtractor` | 提取爱好标签与热度权重 | 识别非爱好类废话 |
| **三观分析** | `ValuesExtractor` | 提取金钱观、家庭观等 | 区分用户玩笑话与真实价值观 |
| **生活方式** | `LifestyleExtractor` | 提取烟酒、作息、家务偏好 | 识别模糊表述 |
| **恋爱风格** | `LoveStyleExtractor` | 提取依恋类型、约会偏好 | 区分对过往的吐槽与核心风格 |
| **风险预警** | `RiskExtractor` | 识别负债、疾病、性格雷点 | 具备极高的查准率 (Recall) |
| **教育背景** | `EducationExtractor` | 提取学校名、专业、学历 | 别名归一化 (如“五道口” -> “清华”) |
| **职业背景** | `OccupationExtractor` | 提取职位、行业、收入区间 | 识别虚假夸大或模糊表述 |
| **家庭背景** | `FamilyExtractor` | 提取原生家庭关系、氛围 | 提取情感色彩而非纯事实 |
| **择偶偏好** | `DatingPrefExtractor` | 提取硬性过滤与软性偏好 | 识别“宁缺毋滥”等情绪化表达 |
| **意图识别** | `IntentNode` | 4 类意图：Search, Refresh, Dive, Chat | 识别极其细微的语义转换 |
| **完备度评估** | `TerminationManager` | 判断画像是否足以支撑匹配 (Boolean) | 区分“已回答”与“有效回答” |
| **证据搜寻** | `EvidenceHunting` | 从原始对话中挖掘推荐理由 | 排除无意义的日常废话 |

### 2.2 数据生成流水线 (Extraction SFT Data)
1.  **种子构造**：利用现有的 30 个 `users_persona` 作为原始语料。
2.  **Teacher 扩增**：调用 DeepSeek-v3 对每个维度进行 **Self-Instruct** 场景化模拟。
    *   *示例 Prompt*：“模拟一段用户关于‘金钱观’的回复，语气要口语化，包含一点干扰信息，然后提取 JSON。”
3.  **多维度并发**：在训练集中，将不同维度的提取任务混在一起，训练模型通过 `instruction` 自动切换脑区。

---

## 3. “右脑”拟人对话区 (Right Brain - Emotional EQ)
**训练目标**：高情商、口语化、严禁 AI 味、具备强烈的“红娘”人设。

### 3.1 任务映射表 (Task Mapping)
| 模块名称 | 对应代码位置 | 核心人设要求 | DPO 优化点 (Rejected 样本) |
| :--- | :--- | :--- | :--- |
| **互动式访谈** | `OnboardingNode` | 温柔、引导性强、不逼问 | **拒绝**：机械的一问一答、长篇大论 |
| **推荐开场白** | `recommend_chain` | 眼光毒辣、隆重介绍、突出亮点 | **拒绝**：复读画像标签、缺乏情感波动的陈述 |
| **安慰与拒绝** | `not_found_chain` | 温体、体贴、给用户希望 | **拒绝**：生硬的“没找到”、逻辑苍白的道歉 |
| **深度追问** | `DeepDiveNode` | 敏锐、像闺蜜一样深入挖掘 | **拒绝**：指代消解错误、重复提问 |
| **通用咨询** | `chitchat_chain` | 专业、知性、有边界感 | **拒绝**：油腻称呼（亲、宝贝）、空洞套话 |

### 3.2 偏好对齐策略 (DPO Construction)
针对右脑，不仅要通过 SFT 学会“红娘怎么说话”，更要通过 DPO 学会“红娘不该怎么说话”：
*   **Chosen (高情商)**：引用用户之前的对话细节作为证据，语气有起伏。
*   **Rejected (低情商)**：虽然逻辑正确，但语气生冷，像个说明书。
    *   *对比示例*：
        *   Chosen: “哎呀，刚才听你说喜欢猫，我特意给你留意了，小美家里也养了只布偶，你们肯定聊得来！”
        *   Rejected: “根据您的画像，小美也喜欢宠物，匹配度较高，建议认识。”

---

## 4. 关键实现节点 (Milestones)

### 4.1 全量数据蒸馏引擎：`training/scripts/gen_distill_data.py`
该脚本是整个工程的数据源泉，实现了从业务逻辑到训练数据的全自动转化：
1.  **全维度覆盖**：通过实例化项目中 10 大维度的 `Extractor` 类，确保每一个字段的提取逻辑都与生产环境 100% 对齐。
2.  **Self-Instruct 增强**：利用 Teacher 模型基于 `training/seeds/persona_seeds.json` 模拟长段、带有情绪和噪音的用户微信对话，大幅提升 `input` 的真实感。
3.  **TaskID 指令化**：为每条数据注入 `TaskID` 前缀（如 `TaskID: IntentNode`），支持模型在同一 LoRA 适配器下精准切换逻辑子区。
4.  **DPO 构造**：通过 `Adversarial Prompting` 诱导生成包含技术黑话、去拟人化的 Rejected 样本，与高情商的 Chosen 样本形成极致风格对比。

### 4.2 训练配置与流水线：`training/config/` & `training/scripts/run_train.sh`
1.  **SFT 阶段**：`qwen_sft.yaml` 配置了 1024 截断长度、2e-4 学习率及 LoRA 全量参数微调。
2.  **DPO 阶段**：`qwen_dpo.yaml` 专注于风格对齐。

### 4.3 物理样本证据：`training/data/`
1.  **`extraction_samples.jsonl`**：内化了画像分析师和 10 大维度提取逻辑。
2.  **`chat_samples.jsonl`**：内化了金牌红娘的沟通技巧与深度追问逻辑。
3.  **`dpo_samples.jsonl`**：建立了系统人设的红线，让模型明确区分“专业红娘”与“冷血机器人”。

### 4.4 部署架构：vLLM + Multi-LoRA
*   **共享基座**：Qwen2.5-7B-Instruct。
*   **路由逻辑**：在 `app/core/llm.py` 中，根据业务节点（提取 vs 对话）动态分发请求至 `extraction_adapter` 或 `chat_adapter`，实现显存节省 80% 且零切换延迟。

