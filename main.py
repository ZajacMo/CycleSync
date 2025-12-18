import os
import time
import sys

def run_step(step_name, script_path):
    print(f"\n{'='*50}")
    print(f"Running Step: {step_name}")
    print(f"Script: {script_path}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    ret = os.system(f"{sys.executable} {script_path}")
    end_time = time.time()
    
    if ret != 0:
        print(f"\n[ERROR] Step {step_name} failed with exit code {ret}")
        exit(ret)
    
    print(f"\n[SUCCESS] Step {step_name} completed in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    print("Starting Math Modeling Pipeline...")
    
    # 1. 数据预处理
    run_step("Data Preprocessing", "src/data_preprocessing.py")
    
    # 2. 站点聚类
    run_step("Station Clustering", "src/station_clustering.py")
    
    # 3. 需求预测
    run_step("Demand Prediction (ST-LSTM)", "src/demand_prediction.py")

    # 3.1 聚类结果可视化 (新增)
    run_step("Cluster Visualization", "src/visualize_clusters.py")
    
    # 4. 调度优化
    run_step("Optimization (VRP-PD-TW)", "src/optimization.py")
    
    # 5. 围栏布局
    run_step("Fence Optimization", "src/fence_optimization.py")
    
    print("\nAll steps completed successfully!")
    print("Results are available in data/output/")
