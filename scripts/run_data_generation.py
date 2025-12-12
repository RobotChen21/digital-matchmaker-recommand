# -*- coding: utf-8 -*-
import sys
import os

# 将项目根目录添加到 python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.services.ai.workflows.data_pipeline import EnhancedDataGenerationPipeline

if __name__ == "__main__":
    print(f"🔧 Working Directory: {os.getcwd()}")
    print(f"🔧 Project Root: {project_root}")
    
    # 创建并运行流程
    pipeline = EnhancedDataGenerationPipeline()
    pipeline.run_full_pipeline()