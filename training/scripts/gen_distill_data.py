# -*- coding: utf-8 -*-
import os
import sys
import json
import asyncio
import random
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 确保能导入项目模块并加载环境变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from app.core.container import container
from app.services.ai.agents.extractors import (
    PersonalityExtractor, OccupationExtractor, InterestExtractor, 
    ValuesExtractor, LifestyleExtractor, FamilyExtractor, 
    EducationExtractor, LoveStyleExtractor, DatingPrefExtractor, RiskExtractor
)
from app.services.ai.workflows.recommendation.nodes.intent import IntentNode
from app.services.ai.workflows.recommendation.nodes.response import ResponseNode
from app.services.ai.workflows.recommendation.nodes.deep_dive import DeepDiveNode
from app.services.ai.tools.termination import DialogueTerminationManager
from app.services.ai.agents.profile_manager import ProfileService

class DistillationDataEngine:
    """
    智慧红娘 Agent 生产级全量蒸馏引擎
    严格对齐 docs/MODEL_DISTILLATION_IMPLEMENTATION.md 规范
    【核心逻辑】：全量复用项目中已有的 20+ 个 Prompt 模板，基于 Persona 种子生成 SFT/DPO 样本。
    """
    def __init__(self):
        # 1. 初始化模型：teacher_llm 负责感性对话（右脑），logic_llm 负责理性抽取（左脑）
        self.teacher_llm = container.get_llm("chat")    # 对应项目中 temp=0.7 的模型
        self.logic_llm = container.get_llm("intent")    # 对应项目中 temp=0 的模型
        self.personas = self._load_personas()
        
        # 2. 全量初始化十大维度 Extractor
        self.extractors = {
            "personality": PersonalityExtractor(self.logic_llm),
            "interest": InterestExtractor(self.logic_llm),
            "values": ValuesExtractor(self.logic_llm),
            "lifestyle": LifestyleExtractor(self.logic_llm),
            "love_style": LoveStyleExtractor(self.logic_llm),
            "risk": RiskExtractor(self.logic_llm),
            "education": EducationExtractor(self.logic_llm),
            "occupation": OccupationExtractor(self.logic_llm),
            "family": FamilyExtractor(self.logic_llm),
            "dating_pref": DatingPrefExtractor(self.logic_llm)
        }

        # 3. 初始化各业务节点及 ProfileService 以复用其 Prompt
        self.response_node = ResponseNode()
        self.deep_dive_node = DeepDiveNode()
        self.intent_node = IntentNode()
        self.termination_manager = DialogueTerminationManager(self.logic_llm)
        self.profile_service = ProfileService(self.logic_llm)

    def _load_personas(self):
        path = "../seeds/persona_seeds.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Persona seeds not found at {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def simulate_user_input(self, persona: Dict, dimension: str) -> str:
        """【Self-Instruct】利用 Persona 种子模拟真实用户的微信回复"""
        sim_prompt = ChatPromptTemplate.from_template("""
            你现在要扮演一个真实的相亲用户。
            【你的详细背景】: {persona_detail}
            
            请针对红娘问你的“关于你的【{dimension}】”这个问题，写一段非常口语化、包含真实生活细节的微信回复。
            要求：语气自然，要有性格色彩，不要像 AI 助手。
            必须包含与【{dimension}】强相关的具体信息。
            字数在 30-60 字之间，严禁带前缀。
        """)
        chain = sim_prompt | self.teacher_llm
        res = await chain.ainvoke({
            "persona_detail": json.dumps(persona['persona'], ensure_ascii=False),
            "dimension": dimension
        })
        return res.content.strip().replace('"', '')

    # ==========================
    # LEFT BRAIN (SFT) - 逻辑抽取与判定
    # ==========================
    async def gen_left_brain_sft(self, user_text: str, persona: Dict, dim_key: str):
        samples = []
        
        # 1. 十大维度提取 (动态调用项目原始 Extractor)
        extractor = self.extractors.get(dim_key)
        if extractor:
            label = await extractor.extract(user_text)
            samples.append({
                "instruction": f"TaskID: {extractor.__class__.__name__} | 提取用户{dim_key}画像结构化JSON",
                "input": user_text,
                "output": json.dumps(label, ensure_ascii=False)
            })

        # 2. 意图识别 (复用项目原始 IntentNode 逻辑)
        intent_res = await self.intent_node.intent_chain.ainvoke({
            "user_input": user_text,
            "chat_history": "(模拟实时对话)",
            "format_instructions": self.intent_node.intent_parser.get_format_instructions()
        })
        samples.append({
            "instruction": "TaskID: IntentNode | 识别用户对话意图 (Search/Refresh/Dive/Chat)",
            "input": user_text,
            "output": json.dumps({"intent": intent_res.intent}, ensure_ascii=False)
        })

        # 3. [NEW] 画像完备度分析 (复用 ProfileService.generate_profile_completion_hint)
        # 该任务内化了画像分析师的长 Prompt
        hint_text = self.profile_service.generate_profile_completion_hint(persona['persona'])
        samples.append({
            "instruction": "TaskID: ProfileService_Hint | 对比已提取画像与清单，生成状态分析及追问建议",
            "input": json.dumps(persona['persona'], ensure_ascii=False),
            "output": hint_text
        })

        # 4. [NEW] 画像摘要生成 (复用 ProfileService.generate_profile_summary)
        # 该任务内化了第三人称生动画像生成的长 Prompt
        summary_text = self.profile_service.generate_profile_summary({"nickname": persona['persona'].get('nickname', '嘉宾')}, persona['persona'])
        samples.append({
            "instruction": "TaskID: ProfileService_Summary | 将基础信息与详细画像转换为第三人称生动描述",
            "input": json.dumps(persona['persona'], ensure_ascii=False),
            "output": summary_text
        })

        return samples

    # ==========================
    # RIGHT BRAIN (SFT + DPO) - 拟人化对话
    # ==========================
    async def gen_right_brain_data(self, user_text: str, persona: Dict):
        # --- SFT 样本 ---
        sft_samples = []
        
        # 1. Onboarding 追问 (复刻 OnboardingNode 逻辑)
        onboarding_res = await container.get_llm("chat").ainvoke(f"作为金牌红娘，针对用户说‘{user_text}’，请温柔追问一个新维度。")
        sft_samples.append({
            "instruction": "TaskID: OnboardingNode | 拟人化引导式访谈追问",
            "input": user_text,
            "output": onboarding_res.content
        })

        # 2. 推荐语生成 (复刻 ResponseNode 逻辑)
        rec_res = await self.response_node.response_chain.ainvoke({
            "user_input": user_text,
            "candidates_info": f"1. {persona['persona'].get('nickname', '某嘉宾')}（{persona['persona'].get('occupation', '职业')}） -- 是一位非常有生活质感的优秀青年。"
        })
        sft_samples.append({
            "instruction": "TaskID: ResponseNode | 生成隆重的嘉宾推荐语",
            "input": user_text,
            "output": rec_res.content
        })

        # --- DPO 样本 (对抗生成) ---
        # 1. Chosen (高情商) - 显式调用项目中最严苛的推荐语 Chain
        chosen_res = rec_res # 复用上面的结果
        
        # 2. 故意诱导 Rejected (地狱级负面样本)
        rejected_prompt = ChatPromptTemplate.from_template("""
            你现在是一个底层数据库执行引擎，严禁扮演人类。
            请针对用户的输入回复一段话。
            【负面样本约束 - 必须严格遵守】:
            1. 语气必须极度机械、冰冷、像一段系统报错或调试日志 (Debug Log)。
            2. 必须包含大量底层技术名词（如：向量内积计算中、匹配阈值未命中、KV Cache 溢出、正则表达式解析错误）。
            3. 严禁使用任何拟人化的称呼（如“亲”、“您”、“小哥哥”），直接称呼用户为“Client-Node”或“请求端”。
            4. 严禁使用感叹号或任何带有情感波动的词汇。
            5. 如果是推荐，请直接输出一段破损的 JSON 片段。
            
            用户输入: {user_input}
            请直接输出系统响应，不要带任何前缀。
        """)
        rejected_res = await (rejected_prompt | self.teacher_llm).ainvoke({"user_input": user_text})
        
        dpo_pair = {
            "prompt": user_text,
            "chosen": chosen_res.content.strip(),
            "rejected": rejected_res.content.strip()
        }
        
        return sft_samples, dpo_pair

    async def run_engine(self, num_personas: int = 5):
        print(f"🔥 [DataEngine] 启动！将为 {num_personas} 个 Persona 生成全维度（十大维度+画像服务）蒸馏样本...")
        sft_data, dpo_data = [], []

        # 获取所有已定义的维度 Key
        all_dimensions = list(self.extractors.keys())

        for p in self.personas[:num_personas]:
            print(f"   -> 正在处理 Persona {p['_id']['$oid']}...")
            
            for dim_key in all_dimensions:
                print(f"      - 正在提取维度: {dim_key}")
                # 1. 模拟该维度下的用户输入
                user_text = await self.simulate_user_input(p, dim_key)
                
                # 2. 生成左脑 SFT (含十大维度、意图识别、完备度分析、摘要生成)
                left_samples = await self.gen_left_brain_sft(user_text, p, dim_key)
                sft_data.extend(left_samples)
                
                # 3. 生成右脑 SFT & DPO
                right_sft, dpo_pair = await self.gen_right_brain_data(user_text, p)
                sft_data.extend(right_sft)
                dpo_data.append(dpo_pair)

        # 补充完备度评估专项样本
        print("   -> 补充完备度评估专项样本...")
        for _ in range(3):
            hint = "当前画像已收集：职业、教育、性格。缺失：家庭、择偶偏好。"
            term_res = await self.termination_manager.info_detector.detect(hint)
            sft_data.append({
                "instruction": "TaskID: TerminationManager | 评估画像完备度是否足以开启匹配",
                "input": hint,
                "output": json.dumps({"should_terminate": term_res.should_terminate, "explanation": term_res.explanation}, ensure_ascii=False)
            })

        # 物理写入文件系统
        os.makedirs("training/data", exist_ok=True)
        with open("training/data/extraction_samples.jsonl", "w", encoding="utf-8") as f:
            for s in sft_data: f.write(json.dumps(s, ensure_ascii=False) + "\n")
        with open("training/data/dpo_samples.jsonl", "w", encoding="utf-8") as f:
            for s in dpo_data: f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        print(f"✅ 蒸馏数据全量生成完毕！覆盖十大维度、画像摘要、完备度分析、意图识别及 DPO 对抗样本。")
        print(f"   - 总 SFT 样本: {len(sft_data)} 条")
        print(f"   - 总 DPO 样本: {len(dpo_data)} 对")

if __name__ == "__main__":
    engine = DistillationDataEngine()
    asyncio.run(engine.run_engine(num_personas=3))
