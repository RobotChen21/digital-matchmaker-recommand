# 红娘 Agent：从通用大模型到专用小模型蒸馏指南

## 1. 核心理念：提示词内化 (Prompt Internalization)

**现状 (Prompt Engineering)**：
依赖 DeepSeek-v3/GPT-4，通过几百个 Token 的 `System Prompt` 强行约束模型行为。
*   缺点：昂贵、首字延迟高、上下文窗口被 Prompt 挤占、稳定性依赖大模型版本。

**目标 (Model Distillation)**：
将 `System Prompt` 的规则“内化”为小模型的权重。
*   优点：零 Prompt 启动、极速响应、数据隐私安全、极低推理成本。

---

## 2. 架构拆解：双脑模型策略 (The Two-Brain Strategy)

为了在资源（显存）与效果之间取得平衡，我们将任务拆解为两类，通过 **Multi-LoRA** 技术在同一个基座模型上运行。

| 特性 | **Left Brain (逻辑/抽取)** | **Right Brain (拟人/聊天)** |
| :--- | :--- | :--- |
| **对应代码** | `extractors.py` (画像提取), `intent.py` (意图分类) | `response.py` (回复生成), `onboarding.py` |
| **核心能力** | 严谨、遵循指令、JSON 格式化 | 共情、口语化、角色扮演、多轮逻辑 |
| **训练目标** | **准确率 (Accuracy)** | **人类偏好 (Human Preference/EQ)** |
| **训练方法** | **SFT (监督微调) Only** | **SFT (风格) -> DPO (对齐)** |
| **推荐基座** | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct / Llama-3-8B |

---

## 3. 第一阶段：数据收集 (Data Engine)

利用现有的 DeepSeek-v3 作为 "Teacher"，收集高质量数据。

### 3.1 埋点日志 (The Logger)
在 `app/core/llm.py` 或各 Agent 的入口处添加日志记录，保存：
1.  **Raw Input**: 用户的原始输入。
2.  **Context**: 当时的对话历史（Messages）。
3.  **Teacher Output**: DeepSeek 生成的完美结果。

### 3.2 Left Brain 数据构建 (Instruction-Input-Output)
*   **目标**：教会模型“看到这段话，就提取这个 JSON”。
*   **Prompt 处理**：训练数据中的 Input **不包含** 冗长的 System Prompt，只有简短指令。
*   **关键点**：必须包含 **负采样 (Negative Sampling)**，即用户说废话时，模型输出全 `null` 的 JSON。

**数据样本 (JSONL):**
```json
// 正样本
{"instruction": "提取画像JSON", "input": "我28岁，喜欢滑雪", "output": "{\"age\": 28, \"hobbies\": [\"滑雪\"]}"}
// 负样本 (防止幻觉)
{"instruction": "提取画像JSON", "input": "哈哈哈哈笑死我了", "output": "{\"age\": null, \"hobbies\": []}"}
```

### 3.3 Right Brain 数据构建 (Conversation & Preference)
* **SFT 数据**：多轮对话格式，去除“作为AI助手”的废话，保留“红娘”口吻，减少自我认知的提示词，学会红娘的基本业务逻辑，学会了输出格式。

*   **DPO 数据 (三元组)**：用于训练高情商，改变说话语气的，减少幻觉与安全对齐（比如不该说什么）。
    
    *   **Prompt**: 用户的问题。
    *   **Chosen (好回答)**: 语气委婉、有理有据、引导性强。
    *   **Rejected (坏回答)**: 机械、生硬、单纯复读、不仅没用还惹人生气。
    
    针对 DPO 训练中所需的配对数据，采用 **“风格对比 + 鲁棒性增强”** 双维策略构建高质量**负样本**（**Rejected** Data）：
    
    #### 1. 弱模型采样 (Weak Model Sampling) —— *构建能力级差*
    
    - **原理**：利用未经 RLHF/SFT 的 **Base 基座模型** 或 **旧版本模型 (Legacy Checkpoints)** 进行推理。
    - **效果**：生成的回答通常逻辑正确但缺乏“人性”，语气机械、啰嗦或过于官方。
    - **应用**：天然构建出 **“高情商红娘 (Chosen) vs 呆板 AI (Rejected)”** 的鲜明对比，迫使模型学习更生动的表达风格。
    
    #### 2. 负向 Prompt 诱导 (Adversarial Prompting) —— *构建风格对立*
    
    - **原理**：通过 Prompt Engineering 故意诱导模型生成“错误风格”的回答（如指令：“请扮演一个冷漠、不耐烦的客服”）。
    - **效果**：生成逻辑通顺但 **情感色彩错误**（冷漠、粗鲁、复读机式）的负例。
    - **应用**：专门用于 **Right Brain (拟人对话)** 训练，让模型精准识别并远离“低情商”话术，强化“红娘人设”的对齐。
    
    #### 3. 规则扰动 (Rule-based Perturbation) —— *构建鲁棒性边界*
    
    - **原理**：对高质量的 Chosen 数据进行人工破坏，包括 **语义截断、随机重复、格式/语法破坏**（如破坏 JSON 闭合）。
    - **效果**：制造结构性错误或非指令遵循的负例。
    - **应用**：主要用于 **Left Brain (逻辑抽取)** 任务，提升模型对指令依从性（Instruction Following）和输出格式（Format constraints）的鲁棒性。

**DPO 样本示例:**

```json
{
  "prompt": "这个男生太矮了，我不喜欢。",
  "chosen": "哎呀亲，身高确实是硬指标。不过这小伙子才华横溢，是设计总监呢。咱们通过见面了解下内在？如果实在介意，姐再给你挑个高个的。",
  "rejected": "收到。已为您过滤身高低于 175 的候选人。正在搜索新用户。"
}
```

---

## 4. 第二阶段：训练流程 (The Pipeline)

推荐工具：**LLaMA-Factory** (支持 WebUI 和 CLI，集成 SFT/DPO/LoRA)。

### 4.1 Left Brain 训练 (SFT)
*   **Dataset**: `extraction_dataset` (约 1k - 5k 条)
*   **Hyperparams**：
    *   Learning Rate: 2e-4
    *   Epochs: 3
    *   LoRA Rank: 16
*   **Output**: `adapter_extraction` (约 50MB)

### 4.2 Right Brain 训练 (SFT + DPO)
1.  **Step 1: SFT (学会说话)**
    *   Dataset: `chat_sft_dataset` (约 10k 条)
    *   Output: `adapter_chat_sft`
2.  **Step 2: DPO (学会高情商)**
    *   Base Model: 加载 `adapter_chat_sft` 后的模型。
    *   Dataset: `dpo_preference_dataset` (约 1k - 2k 对)
    *   **Output**: `adapter_hongniang_final`

---

## 5. 第三阶段：生产部署 (Production)

### 5.1 架构：vLLM + Multi-LoRA
在没有 Multi-LoRA 之前，如果想通过一个模型做两件事（提取信息 + 拟人对话），有两个笨办法：

- **笨办法 A（两台机器）：** 部署两个独立的微调大模型（Model A 和 Model B）。**缺点：** 显存占用翻倍（比如 70B 模型，两份就要占 280GB+ 显存），你的 8x4090 可能就不够用了。
- **笨办法 B（来回切换）：** 只部署一个，但每次要用不同功能时，把旧权重卸载，加载新权重。**缺点：** 延迟极高，用户要等几秒钟加载，体验极差。

**现在的方案 (vLLM Multi-LoRA)：**

- **Shared Backbone (共享骨架)：** 你的基座模型（Base Model）只需要在 8 张卡上加载**一次**。
- **Hot-swappable Adapters (热插拔适配器)：** 你的 LoRA 权重非常小（几十 MB），可以同时把 LoRA A（逻辑抽取）和 LoRA B（拟人对话）都加载在显存里。
- **Concurrent Batching (并发批处理)：** **（这是最骚的操作）** vLLM 可以**在一个 Batch 里同时处理不同任务**！
  - 请求 1 进来：指定用 LoRA A。
  - 请求 2 进来：指定用 LoRA B。
  - vLLM 会在推理时，动态地把 LoRA A 的权重加给请求 1，把 LoRA B 的权重加给请求 2，**同时计算**。

“在基础设施层面，我们利用 **Tensor Parallelism (TP=8)** 技术，将基座模型切分部署在 8 张 RTX 4090 上，构建了统一的推理底座。

针对不同的业务场景，我们采用了 **vLLM 的 Multi-LoRA Serving** 机制。我并没有为每个任务单独部署模型，而是将‘逻辑抽取’和‘拟人对话’作为轻量级的 **LoRA Adapter** 挂载在同一个基座上。

这样做的好处是：**显存利用率最大化**（192GB 显存只需要存一份基座权重），并且实现了**零切换延迟**（Zero Overhead Switching）。在同一个 Batch 中，我可以同时响应需要结构化抽取的 API 请求和需要情感陪伴的用户对话请求，极大地提升了吞吐量（Throughput）。”

### 5.2 启动命令
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /models/DeepSeek-V3-Base \
    --tensor-parallel-size 8 \
    --enable-lora \
    --lora-modules extraction_adapter=/models/lora_extraction_v1 chat_adapter=/models/lora_chat_v2 \
    --port 8000
```

### 5.3 代码集成 (`app/core/llm.py`)
在代码层面，通过修改 `model` 参数路由到不同的 LoRA 适配器。

```python
# 假设这是您的配置类
class AppConfig:
    VLLM_API_BASE = "http://localhost:8000/v1"
    VLLM_API_KEY = "EMPTY"  # 本地部署通常不需要 Key
    
    # 这里的名字必须对应 vLLM 启动命令中的 --lora-modules 别名
    MODEL_NAME_EXTRACTION = "extraction_adapter"  
    MODEL_NAME_CHAT = "chat_adapter"

config = AppConfig()

class LLMFactory:
    """
    负责生产不同脑区的 LLM 实例
    """
    
    @staticmethod
    def get_left_brain_llm() -> ChatOpenAI:
        """
        【左脑 - 逻辑与抽取】
        配置：
        1. model: 指向 extraction_adapter
        2. temperature: 0 (强制由 LoRA 权重主导逻辑，减少随机性)
        3. max_tokens: 通常较短，只需要结构化数据
        """
        return ChatOpenAI(
            openai_api_base=config.VLLM_API_BASE,
            openai_api_key=config.VLLM_API_KEY,
            model=config.MODEL_NAME_EXTRACTION,  # <--- 关键点：切换 LoRA
            temperature=0,                       # <--- 关键点：绝对理性
            max_tokens=512,
            # model_kwargs={"stop": ["\n}"]}     # 可选：可以添加停止词防止废话
        )

    @staticmethod
    def get_right_brain_llm() -> ChatOpenAI:
        """
        【右脑 - 情感与对话】
        配置：
        1. model: 指向 chat_adapter
        2. temperature: 0.7 - 0.9 (增加创造性和情感丰富度)
        """
        return ChatOpenAI(
            openai_api_base=config.VLLM_API_BASE,
            openai_api_key=config.VLLM_API_KEY,
            model=config.MODEL_NAME_CHAT,        # <--- 关键点：切换 LoRA
            temperature=0.8,                     # <--- 关键点：感性波动
            streaming=True                       # 对话通常需要流式输出
        )
```

---

## 6. 路线图总结

1.  **冷启动**:
    *   继续使用 DeepSeek API。
    *   **部署 Log 模块**，收集数据。
    *   人工审查 Log，构建初始评估集 (Golden Set)。
2.  **原型验证**:
    *   清洗 500 条抽取数据，训练 `Left Brain`。
    *   对比 DeepSeek 和 Qwen-LoRA 在抽取任务上的 JSON 准确率。
3.  **风格迁移**:
    *   清洗对话数据，训练 `Right Brain` SFT。
    *   上线灰度测试，收集用户点赞/点踩 (构建 DPO 数据)。
4.  **完全进化**:
    *   运行 DPO 训练。
    *   全量切换至本地 vLLM，DeepSeek 退役为“数据标注员”。
