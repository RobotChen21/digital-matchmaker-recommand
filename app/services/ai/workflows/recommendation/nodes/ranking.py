# -*- coding: utf-8 -*-
from bson import ObjectId
from app.common.models.state import MatchmakingState
from app.core.container import container
from app.core.utils.cal_utils import calc_age

class RankingNode:
    def __init__(self):
        self.db = container.db

    def _get_profile_field(self, profile, category, field, default=None):
        """Helper: 安全获取画像字段"""
        return profile.get(category, {}).get(field, default)

    def calculate_compatibility(self, user_profile, candidate_profile):
        """计算两个画像的心理匹配度"""
        score = 0
        reasons = []

        # 1. MBTI 匹配 (简单的 E/I 互补或相同策略)
        u_mbti = self._get_profile_field(user_profile, 'personality_profile', 'mbti', '')
        c_mbti = self._get_profile_field(candidate_profile, 'personality_profile', 'mbti', '')
        
        if u_mbti and c_mbti:
            u_mbti = u_mbti.upper()
            c_mbti = c_mbti.upper()
            if u_mbti == c_mbti:
                score += 10
                reasons.append(f"MBTI同频({u_mbti})")
            elif len(u_mbti) == 4 and len(c_mbti) == 4:
                # E/I 互补 (第一位不同，后三位相同，如 ENFP & INFP)
                if u_mbti[0] != c_mbti[0] and u_mbti[1:] == c_mbti[1:]:
                    score += 15
                    reasons.append(f"性格互补({u_mbti}&{c_mbti})")

        # 2. 生活方式共鸣 (Lifestyle)
        # 烟酒不分家，习惯要一致
        u_smoke = self._get_profile_field(user_profile, 'lifestyle_profile', 'smoking', '')
        c_smoke = self._get_profile_field(candidate_profile, 'lifestyle_profile', 'smoking', '')
        if u_smoke and c_smoke and u_smoke == c_smoke:
            score += 5
            # reasons.append("抽烟习惯一致") # 太细了不展示

        u_drink = self._get_profile_field(user_profile, 'lifestyle_profile', 'drinking', '')
        c_drink = self._get_profile_field(candidate_profile, 'lifestyle_profile', 'drinking', '')
        if u_drink and c_drink and u_drink == c_drink:
            score += 5

        # 3. 兴趣共鸣 (Tags Intersection)
        u_tags = set(self._get_profile_field(user_profile, 'interest_profile', 'tags', []))
        c_tags = set(self._get_profile_field(candidate_profile, 'interest_profile', 'tags', []))
        common_tags = u_tags.intersection(c_tags)
        
        if common_tags:
            points = len(common_tags) * 5
            score += points
            tags_str = ",".join(list(common_tags)[:3])
            reasons.append(f"共同爱好({tags_str})")

        return score, reasons

    def ranking(self, state: MatchmakingState):
        """Step 4: 心理学精排 (Psychological Rerank)"""
        top_ids = state.get('semantic_candidate_ids', [])[:30] # 从 ES 拿回 Top 30
        if not top_ids:
            print("⚠️ [Ranking] 无候选人可排")
            state['final_candidates'] = []
            return state
            
        print(f"⚖️ [Ranking] 心理学匹配计算 (候选人: {len(top_ids)})...")
        
        # 加载当前用户画像 (用于比对)
        current_profile = state.get('current_user_profile') or {}
        
        # [Debug] 检查当前用户画像数据
        u_mbti = self._get_profile_field(current_profile, 'personality_profile', 'mbti', '无')
        u_tags = self._get_profile_field(current_profile, 'interest_profile', 'tags', [])
        print(f"   🧐 [Ranking Debug] Current User: MBTI={u_mbti}, Tags={u_tags}")
        
        scored_candidates = []
        
        for uid in top_ids:
            # 查候选人画像
            target_uid = ObjectId(uid)
            basic = self.db.users_basic.find_one({"_id": target_uid})
            profile = self.db.db["users_profile"].find_one({"user_id": target_uid}) or {}
            
            if not basic: continue
            
            # 基础分 (来自 ES 排序的隐含分，这里简单的倒序给分)
            # 假设 top_ids 是有序的，第1名给30分，第30名给1分
            base_score = 30 - top_ids.index(uid)
            
            # 心理匹配分
            psych_score, reasons = self.calculate_compatibility(current_profile, profile)
            
            final_score = base_score + psych_score
            
            # 构造前端展示数据
            basic['id'] = str(basic.pop('_id'))
            basic['score'] = final_score
            basic['match_reasons'] = ", ".join(reasons) if reasons else "眼缘匹配"
            
            # 构造 Summary (用于 Chat 里的 context)
            age = calc_age(basic.get('birthday'))
            basic['age'] = age
            job = self._get_profile_field(profile, 'occupation_profile', 'job_title', '未知职业')
            edu = self._get_profile_field(profile, 'education_profile', 'highest_degree', '')
            basic['summary'] = f"【{basic['nickname']}】 {age}岁 {basic.get('city')} | {job} | {edu}"
            
            scored_candidates.append(basic)

        # 最终排序
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 取 Top 3 (给前端展示)
        final_candidates = scored_candidates[:3]
        state['final_candidates'] = final_candidates
        
        # [NEW] 更新已阅名单
        seen_ids = state.get('seen_candidate_ids', [])
        new_ids = [c['id'] for c in final_candidates]
        # 合并并去重
        state['seen_candidate_ids'] = list(set(seen_ids + new_ids))
        
        print(f"   -> 冠军: {final_candidates[0]['nickname']} (分: {final_candidates[0]['score']})")
        print(f"   -> 理由: {final_candidates[0]['match_reasons']}")
        print(f"   -> 已阅名单已更新 (Total: {len(state['seen_candidate_ids'])})")
        
        return state