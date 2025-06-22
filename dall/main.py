#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DALL模型主程序
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import pickle
# 导入所需模块
import sys
import os
import multiprocessing as mp # 新增: 多进程支持
from functools import partial # 新增: partial函数支持

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上级目录（experiments目录）
parent_dir = os.path.dirname(current_dir)

# 添加上级目录到系统路径
sys.path.append(parent_dir)

# 导入模块
from dall.data_processor import DataProcessor
from dall.weight_calculator import WeightCalculator
from dall.item_predictor import ItemPredictor
from dall.residual_corrector import ResidualCorrector
from dall.evaluator import Evaluator
from dall.item_perturbation import ItemPerturbationAnalyzer
from config import CALCULATE_MODULE_CONTROLLABILITY, PERTURBATION_SIZE # 新增: 导入 PERTURBATION_SIZE

# 设置数据路径
ACOUSTIC_FEATURES_PATH = '/home/a001/xuxiao/DALL/dataset/CS-NRAC-E.csv'
QUESTIONNAIRE_PATH = '/home/a001/xuxiao/DALL/dataset/raw_info.csv'

def save_model(model, path):
    """
    保存模型到文件
    
    Parameters
    ----------
    model : object
        要保存的模型对象
    path : str
        保存路径
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到 {path}")

# 新增: 从main_test.py移植的 analyze_sample_batch 函数
def analyze_sample_batch(batch_indices, X_test_data, Y_test_pred_data, ipca_analyzer, current_perturbation_size):
    """
    分析一批样本 (移植并重命名参数以避免与外部作用域冲突)
    
    Parameters
    ----------
    batch_indices : list
        样本索引列表
    X_test_data : pandas.DataFrame
        测试集特征矩阵
    Y_test_pred_data : pandas.DataFrame
        测试集问题项预测矩阵
    ipca_analyzer : ItemPerturbationAnalyzer
        IPCA分析器实例
    current_perturbation_size : float
        扰动大小
    
    Returns
    -------
    list
        分析结果列表
    """
    batch_results_list = []
    
    for i_loop, idx_loop in enumerate(batch_indices): # 使用不同的循环变量名
        # --- IPCA DEBUG PRINT (BEGIN analyze_sample_batch) ---
        # 标记是否为第一个样本，以便在 _predict_base/_residual 中触发调试打印
        is_first_sample_for_debug = (idx_loop == X_test_data.index[0]) if X_test_data.index.size > 0 else False
        # --- IPCA DEBUG PRINT (END) ---

        if i_loop % 10 == 0: # 减少打印频率或基于需要调整
            process_id = os.getpid()
            print(f"进程 {process_id}: 分析样本 {i_loop+1}/{len(batch_indices)} (索引: {idx_loop})...")
            
        # 获取样本数据
        X_sample = X_test_data.loc[[idx_loop]]
        Y_item_sample = Y_test_pred_data.loc[[idx_loop]]
        
        # 计算原始预测
        y_base_orig = ipca_analyzer._predict_base(Y_item_sample)
        y_corr_orig = ipca_analyzer._predict_residual(X_sample, Y_item_sample)
        y_total_orig = y_base_orig + y_corr_orig
        
        # 对每个问题项进行扰动分析
        for item_col in Y_test_pred_data.columns: # 使用不同的循环变量名
            # 创建正向扰动的问题项预测
            Y_item_pos = Y_item_sample.copy()
            # --- IPCA DEBUG PRINT (BEGIN analyze_sample_batch - per item) ---
            if is_first_sample_for_debug and item_col == Y_test_pred_data.columns[0]:
                print(f"\n=== DEBUG: Main loop for sample {idx_loop}, item {item_col} ===")
                print(f"Original Y_item_sample[{item_col}]: {Y_item_sample[item_col].values[0]:.4f}")
                Y_item_pos.attrs['ipca_debug_sample'] = True
                Y_item_pos.attrs['ipca_debug_item_name'] = item_col
            # --- IPCA DEBUG PRINT (END) ---
            Y_item_pos[item_col] += current_perturbation_size
            if is_first_sample_for_debug and item_col == Y_test_pred_data.columns[0]:
                print(f"Perturbed Y_item_pos[{item_col}]: {Y_item_pos[item_col].values[0]:.4f}")

            # 创建负向扰动的问题项预测
            Y_item_neg = Y_item_sample.copy()
            # --- IPCA DEBUG PRINT (BEGIN analyze_sample_batch - per item) ---
            if is_first_sample_for_debug and item_col == Y_test_pred_data.columns[0]:
                Y_item_neg.attrs['ipca_debug_sample'] = True
                Y_item_neg.attrs['ipca_debug_item_name'] = item_col
            # --- IPCA DEBUG PRINT (END) ---
            Y_item_neg[item_col] -= current_perturbation_size
            
            # 计算基础路径调制
            y_base_pos = ipca_analyzer._predict_base(Y_item_pos)
            y_base_neg = ipca_analyzer._predict_base(Y_item_neg)
            delta_base_pos = y_base_pos - y_base_orig
            delta_base_neg = y_base_neg - y_base_orig
            
            # 计算校正路径调制
            y_corr_pos = ipca_analyzer._predict_residual(X_sample, Y_item_pos)
            y_corr_neg = ipca_analyzer._predict_residual(X_sample, Y_item_neg)
            delta_corr_pos = y_corr_pos - y_corr_orig
            delta_corr_neg = y_corr_neg - y_corr_orig
            
            # 计算总预测调制
            delta_total_pos = delta_base_pos + delta_corr_pos
            delta_total_neg = delta_base_neg + delta_corr_neg
            
            # 使用中心差分量化敏感度
            S_total = (delta_total_pos - delta_total_neg) / (2 * current_perturbation_size)
            S_base = (delta_base_pos - delta_base_neg) / (2 * current_perturbation_size)
            S_corr = (delta_corr_pos - delta_corr_neg) / (2 * current_perturbation_size)
            
            # --- IPCA DEBUG PRINT (BEGIN analyze_sample_batch - results) ---
            if is_first_sample_for_debug and item_col == Y_test_pred_data.columns[0]:
                print(f"  DEBUG Results for sample {idx_loop}, item {item_col}:")
                print(f"    y_base_orig: {y_base_orig}, y_base_pos: {y_base_pos}, y_base_neg: {y_base_neg}")
                print(f"    delta_base_pos: {delta_base_pos}, delta_base_neg: {delta_base_neg}")
                print(f"    y_corr_orig: {y_corr_orig}, y_corr_pos: {y_corr_pos}, y_corr_neg: {y_corr_neg}")
                print(f"    delta_corr_pos: {delta_corr_pos}, delta_corr_neg: {delta_corr_neg}")
                print(f"    S_total: {S_total[0] if isinstance(S_total, np.ndarray) else S_total:.4f}")
                print(f"    S_base:  {S_base[0] if isinstance(S_base, np.ndarray) else S_base:.4f}")
                print(f"    S_corr:  {S_corr[0] if isinstance(S_corr, np.ndarray) else S_corr:.4f}")
                print(f"=======================================================")
                # Resetting attrs for next calls if any, though new copies are made each time.
                if 'ipca_debug_sample' in Y_item_pos.attrs: del Y_item_pos.attrs['ipca_debug_sample']
                if 'ipca_debug_item_name' in Y_item_pos.attrs: del Y_item_pos.attrs['ipca_debug_item_name']
                if 'ipca_debug_sample' in Y_item_neg.attrs: del Y_item_neg.attrs['ipca_debug_sample']
                if 'ipca_debug_item_name' in Y_item_neg.attrs: del Y_item_neg.attrs['ipca_debug_item_name']

            # --- IPCA DEBUG PRINT (END) ---

            # 保存结果
            batch_results_list.append({
                'sample_id': idx_loop,
                'item': item_col,
                'S_total': S_total[0] if isinstance(S_total, np.ndarray) and S_total.size == 1 else S_total,
                'S_base': S_base[0] if isinstance(S_base, np.ndarray) and S_base.size == 1 else S_base,
                'S_corr': S_corr[0] if isinstance(S_corr, np.ndarray) and S_corr.size == 1 else S_corr,
            })
    
    return batch_results_list

def main():
    """
    主函数
    """
    # 使用固定的数据路径和输出目录
    acoustic_path = '/home/a001/xuxiao/DALL/dataset/CS-NRAC-E.csv'
    questionnaire_path = '/home/a001/xuxiao/DALL/dataset/raw_info.csv'
    output_dir = '/home/a001/xuxiao/DALL/results/dall'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 创建IPCA输出子目录 (如果IPCA部分会用到)
    os.makedirs(os.path.join(output_dir, 'ipca'), exist_ok=True)

    # 记录开始时间
    start_time = time.time()

    # 数据处理
    print("\n=== 数据处理 ===")
    data_processor = DataProcessor(acoustic_path, questionnaire_path)
    data_dict = data_processor.process()

    # 网络分析与权重计算
    print("\n=== 网络分析与权重计算 ===")
    weight_calculator = WeightCalculator(data_dict['Y_train'])
    item_weights, subscale_weights = weight_calculator.calculate_weights(calculate_module_controllability=CALCULATE_MODULE_CONTROLLABILITY)

    # 保存权重
    item_weights.to_csv(os.path.join(output_dir, 'item_weights.csv'))
    print(f"项目权重(CF值)已保存到 {os.path.join(output_dir, 'item_weights.csv')}")

    # 项目级预测与子量表聚合
    print("\n=== 项目级预测与子量表聚合 ===")
    item_predictor = ItemPredictor(
        data_dict['X_train'],
        data_dict['Y_train'],
        data_dict['X_val'],
        data_dict['Y_val'],
        data_dict['X_test']
    )

    # 训练项目级模型
    item_predictor.train_models()

    # 生成训练集OOF预测
    Y_oof = item_predictor.generate_oof_predictions()

    # 生成测试集预测
    Y_test_pred = item_predictor.predict() # This is Y_test_pred for IPCA

    # 结构化残差校正 (使用 item-level 数据)
    print("\n=== 结构化残差校正 (item-level) ===") # MODIFIED print
    residual_corrector_instance = ResidualCorrector(
        data_dict['X_train'],
        Y_oof,                            # MODIFIED: 使用 item-level OOF 预测
        data_dict['y_total_train'],
        item_weights,                     # MODIFIED: 使用 item-level 权重
        data_dict['X_test'],
        Y_test_pred                       # MODIFIED: 使用 item-level 测试集预测
    )

    # 训练残差校正模型
    base_estimator, residual_corrector_model = residual_corrector_instance.train()

    # 生成总分预测
    y_total_test_pred = residual_corrector_instance.predict()

    # 评估
    print("\n=== 评估 ===")
    evaluator = Evaluator(
        data_dict['Y_test'],
        Y_test_pred, # Original Y_test_pred from item_predictor
        # data_dict['Y_sub_test'], # REMOVED
        # Y_sub_test_pred, # REMOVED
        data_dict['y_total_test'],
        y_total_test_pred,
        output_dir
    )

    # 执行评估
    metrics = evaluator.evaluate()
    
    # --- 问题级扰动贡献分析 (IPCA) - 多进程版本 ---
    print("\n=== 问题级扰动贡献分析 (多进程) ===")
    
    # 1. 初始化IPCA分析器 (使用已加载/训练的模型和数据)
    ipca_analyzer_instance = ItemPerturbationAnalyzer( # Renamed
        data_dict['X_test'],
        Y_test_pred,
        base_estimator,
        residual_corrector_model,
        item_weights,
        output_dir
    )
    
    # 2. 设置多进程参数
    num_cores = mp.cpu_count()
    print(f"检测到{num_cores}个CPU核心。")
    # 使用最多num_cores-1个核心，保留一个核心给操作系统，或者根据需要调整
    num_workers = max(1, num_cores - 1 if num_cores > 1 else 1) 
    print(f"将使用{num_workers}个线程进行并行IPCA计算。")
    
    sample_indices = data_dict['X_test'].index.tolist()
    if not sample_indices:
        print("错误：没有样本可供IPCA分析。请检查X_test数据。")
        return

    batch_size = max(1, len(sample_indices) // num_workers if len(sample_indices) >= num_workers else 1)
    batches = [sample_indices[i:i + batch_size] for i in range(0, len(sample_indices), batch_size)]
    
    print(f"IPCA样本总数: {len(sample_indices)}")
    print(f"IPCA每批样本数: {batch_size}")
    print(f"IPCA批次总数: {len(batches)}")
    
    # 3. 创建部分应用函数 (partial)
    # PERTURBATION_SIZE 应从 config.py 导入
    analyze_func_partial = partial(
        analyze_sample_batch, 
        X_test_data=data_dict['X_test'],
        Y_test_pred_data=Y_test_pred,
        ipca_analyzer=ipca_analyzer_instance, 
        current_perturbation_size=PERTURBATION_SIZE # 使用导入的 PERTURBATION_SIZE
    )
    
    # 4. 执行并行分析
    print("\n开始并行IPCA分析样本...")
    all_ipca_results_list = []
    if batches: # 确保有批次可以处理
        with mp.Pool(processes=num_workers) as pool:
            # pool.map会阻塞直到所有任务完成
            list_of_lists_results = pool.map(analyze_func_partial, batches)
        
        # 合并来自所有批次的结果
        for single_batch_results in list_of_lists_results:
            all_ipca_results_list.extend(single_batch_results)
        print("并行IPCA分析完成。")
    else:
        print("没有批次数据供IPCA分析。")

    if not all_ipca_results_list:
        print("错误：IPCA分析未产生任何结果。请检查输入数据和扰动逻辑。")
         # 可以在这里提前返回或进行进一步调试
    else:
        # 5. 转换为DataFrame
        results_df = pd.DataFrame(all_ipca_results_list)
        
        # 6. 保存详细敏感度结果
        print("\n保存详细IPCA敏感度分析结果...")
        results_df.to_csv(os.path.join(output_dir, 'ipca', 'sensitivity.csv'), index=False)
        
        # 7. 计算平均敏感度和变异系数
        print("计算IPCA统计数据 (平均敏感度, CV)...")
        avg_sensitivity = results_df.groupby('item')[['S_total', 'S_base', 'S_corr']].mean()
        
        cv_stats = results_df.groupby('item')[['S_total', 'S_base', 'S_corr']].apply(
            lambda x: pd.Series({
                'CV_total': x['S_total'].std() / abs(x['S_total'].mean()) if abs(x['S_total'].mean()) > 1e-6 else np.nan, # 调整epsilon
                'CV_base': x['S_base'].std() / abs(x['S_base'].mean()) if abs(x['S_base'].mean()) > 1e-6 else np.nan,
                'CV_corr': x['S_corr'].std() / abs(x['S_corr'].mean()) if abs(x['S_corr'].mean()) > 1e-6 else np.nan
            })
        ).reset_index() #确保item是列
        
        # 合并平均敏感度和变异系数
        item_stats = pd.merge(avg_sensitivity.reset_index(), cv_stats, on='item').set_index('item')

        item_stats.to_csv(os.path.join(output_dir, 'ipca', 'item_stats.csv'))
        
        # 8. 生成图表 (使用 ItemPerturbationAnalyzer 实例的方法)
        print("\n生成IPCA分析图表...")
        if not item_stats.empty:
            # 使用新的IPCA分析方法
            ipca_analyzer_instance._create_importance_ranking_table(item_stats, results_df)
            ipca_analyzer_instance._create_dual_pathway_mechanism_plot(item_stats)
            ipca_analyzer_instance._create_consistency_matrix_heatmap(results_df, item_stats)
        else:
            print("item_stats 为空，跳过部分图表生成。")

        if not results_df.empty:
            ipca_analyzer_instance._create_sample_explanation_waterfall(results_df, item_stats)
        else:
            print("results_df 为空，跳过样本解释瀑布图生成。")

        # 新增: 生成所有样本的题目rank顺序CSV
        if not results_df.empty:
            print("\n=== 生成样本题目rank顺序CSV ===")
            ipca_analyzer_instance.generate_sample_rank_csv(results_df)
        else:
            print("results_df 为空，跳过样本rank CSV生成。")

    # 记录结束时间
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    # 临时增大扰动以测试，如果需要的话
    original_perturbation_size = PERTURBATION_SIZE
    PERTURBATION_SIZE = original_perturbation_size * 10 
    print(f"注意：PERTURBATION_SIZE 已临时增大到 {PERTURBATION_SIZE} 以进行测试。")
    
    main()
    
    # # 恢复原始扰动大小 (如果上面修改了)
    # PERTURBATION_SIZE = original_perturbation_size
    # print(f"PERTURBATION_SIZE 已恢复到 {PERTURBATION_SIZE}。")
