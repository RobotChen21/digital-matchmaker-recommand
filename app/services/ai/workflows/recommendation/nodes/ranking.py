# -*- coding: utf-8 -*-
from bson import ObjectId
from datetime import datetime, date
from app.common.models.state import MatchmakingState
from app.core.utils.cal_utils import calc_age


class RankingNode:
    def __init__(self, db_manager):
        self.db = db_manager

    def ranking(self, state: MatchmakingState):
        """Step 4: 精排 (Layer 2: Rule Scoring)"""
        top_ids = state.get('semantic_candidate_ids', [])[:20] # 取 Top 20 进行精排
        policy = state.get('match_policy', {}) # 获取用户策略
        
        scored_candidates = []
        
        print(f"⚖️ [Ranking] 开始对 {len(top_ids)} 位候选人进行规则打分...")
        
        for uid in top_ids:
            # 查完整画像
            user_id_obj = ObjectId(uid)
            basic = self.db.users_basic.find_one({"_id": user_id_obj})
            profile_doc = self.db.db["users_profile"].find_one({"user_id": user_id_obj})
            # 还有 persona (职业描述在这里)
            persona_doc = self.db.users_persona.find_one({"user_id": user_id_obj})
            persona = persona_doc.get("persona", {}) if persona_doc else {}
            
            if not basic: continue
            
            score = 0
            reasons = []
            is_filtered = False # 是否被"半硬条件"一票否决
            
            # --- Rule 1: 学历打分 ---
            edu_weight = policy.get('education_weight', 0)
            if edu_weight > 0:
                edu_profile = profile_doc.get('education_profile', {}) if profile_doc else {}
                degree = edu_profile.get('highest_degree', '未知')
                school_type = edu_profile.get('school_type', '')
                target = policy.get('preferred_degree', '')
                
                # 增强匹配逻辑: 检查学历和学校类型
                match = False
                if target:
                    if target in degree: match = True # 如 "硕士" in "硕士研究生"
                    if target in school_type: match = True # 如 "985" in "985"
                
                if match:
                    points = 10 * edu_weight # 1->10, 2->20, 3->30
                    score += points
                    reasons.append(f"学历符合({degree}/{school_type}, +{points})")
                elif edu_weight == 3:
                    # 权重为 3 (Must)，不符合直接剔除
                    is_filtered = True
                    print(f"   -> 剔除 {basic['nickname']}: 学历不符 (需{target}, 实{degree}/{school_type})")
            
            # --- Rule 2: 工作打分 ---
            job_weight = policy.get('job_weight', 0)
            if job_weight > 0:
                job_target = policy.get('preferred_job', '')
                # 组合所有工作相关字段进行匹配
                occ_profile = profile_doc.get('occupation_profile', {}) if profile_doc else {}
                user_job_text = f"{persona.get('occupation', '')} {persona.get('occupation_detail', '')} {occ_profile.get('industry', '')} {occ_profile.get('job_title', '')}"
                
                if job_target and job_target in user_job_text:
                    points = 10 * job_weight
                    score += points
                    reasons.append(f"工作匹配({job_target}, +{points})")
                elif job_weight == 3:
                    is_filtered = True
                    print(f"   -> 剔除 {basic['nickname']}: 工作不符 (需{job_target})")

            # --- Rule 3: 家庭打分 ---
            family_weight = policy.get('family_weight', 0)
            if family_weight > 0:
                fam_target = policy.get('preferred_family', '')
                fam_profile = profile_doc.get('family_profile', {}) if profile_doc else {}
                
                # 组合家庭字段
                user_fam_text = f"{fam_profile.get('family_structure', '')} {fam_profile.get('parents_occupation', '')} {fam_profile.get('family_economy_level', '')}"
                
                if fam_target and fam_target in user_fam_text:
                    points = 10 * family_weight
                    score += points
                    reasons.append(f"家庭背景匹配({fam_target}, +{points})")
                elif family_weight == 3:
                    is_filtered = True
                    print(f"   -> 剔除 {basic['nickname']}: 家庭不符 (需{fam_target})")
            
            if not is_filtered:
                basic['id'] = str(basic.pop('_id'))
                basic['score'] = score
                basic['match_reasons'] = ", ".join(reasons)
                
                # 构造 summary
                age = calc_age(basic.get('birthday'))
                job = persona.get('occupation', '')
                edu = profile_doc.get('education_profile', {}).get('highest_degree', '') if profile_doc else ''
                
                basic['summary'] = f"【{basic['nickname']}】 {age}岁 {basic.get('city')} | {job} | {edu}"
                basic['evidence'] = "" # 留给 Evidence Node 填
                
                scored_candidates.append(basic)

        # 排序
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 取 Top 3
        final_candidates = scored_candidates[:3]
        state['final_candidates'] = final_candidates
        print(f"   -> 精排完成，Top 1 得分: {final_candidates[0]['score'] if final_candidates else 0}")
        
        return state
