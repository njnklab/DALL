#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基线模型实现：SVM和XGBoost
用于与DALL模型进行对比
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse

# 定义子量表
SUBSCALES = {
    'PHQ': ['PHQ1', 'PHQ2', 'PHQ3', 'PHQ4', 'PHQ5', 'PHQ6', 'PHQ7', 'PHQ8', 'PHQ9'],
    'GAD': ['GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7'],
    'ISI': ['ISI1', 'ISI2', 'ISI3', 'ISI4', 'ISI5', 'ISI6', 'ISI7'],
    'PSS': ['PSS1', 'PSS2', 'PSS3', 'PSS4', 'PSS5', 'PSS6', 'PSS7', 'PSS8', 'PSS9', 'PSS10', 'PSS11', 'PSS12', 'PSS13', 'PSS14']
}

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 结果保存路径
RESULTS_DIR = '/home/user/xuxiao/DALL/results/baseline'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 数据路径
ACOUSTIC_FEATURES_PATH = '/home/user/xuxiao/DALL/dataset/CS-NRAC-E.csv'
QUESTIONNAIRE_PATH = '/home/user/xuxiao/DALL/dataset/raw_info.csv'

def load_data(acoustic_path, questionnaire_path, test_size=0.2, random_seed=RANDOM_SEED):
    """
    加载数据并进行预处理

    Parameters
    ----------
    acoustic_path : str
        声学特征数据路径
    questionnaire_path : str
        问卷数据路径
    test_size : float
        测试集比例
    random_seed : int
        随机种子

    Returns
    -------
    dict
        包含处理后数据的字典
    """
    print("加载声学特征数据...")
    X = pd.read_csv(acoustic_path)

    print("加载问卷数据...")
    Y = pd.read_csv(questionnaire_path)

    # 打印原始数据大小
    print(f"原始声学特征数据大小: {X.shape}")
    print(f"原始问卷数据大小: {Y.shape}")

    # 确保两个数据集有相同的ID
    common_ids = set(X['id']).intersection(set(Y['id']))
    print(f"声学特征数据和问卷数据共有 {len(common_ids)} 个共同ID")

    # 过滤数据，只保留共同ID
    X = X[X['id'].isin(common_ids)].set_index('id')
    Y = Y[Y['id'].isin(common_ids)].set_index('id')

    # 确保两个数据集的索引顺序一致
    common_indices = X.index.intersection(Y.index)
    X = X.loc[common_indices]
    Y = Y.loc[common_indices]

    print(f"过滤后声学特征数据大小: {X.shape}")
    print(f"过滤后问卷数据大小: {Y.shape}")

    # 检查是否有NaN值
    print(f"声学特征数据中的NaN值数量: {X.isna().sum().sum()}")
    print(f"问卷数据中的NaN值数量: {Y.isna().sum().sum()}")

    # 删除包含NaN值的行
    X = X.dropna()
    Y = Y.loc[X.index]

    # 删除Y中的重复索引
    if Y.index.duplicated().any():
        print(f"警告: Y.index中有重复值，删除重复行...")
        Y = Y[~Y.index.duplicated(keep='first')]

    # 确保X和Y有相同的索引
    common_indices = X.index.intersection(Y.index)
    X = X.loc[common_indices]
    Y = Y.loc[common_indices]

    print(f"处理后声学特征数据大小: {X.shape}")
    print(f"处理后问卷数据大小: {Y.shape}")

    # 减少特征数量，只保留前100个特征
    if X.shape[1] > 100:
        print(f"减少特征数量，只保留前100个特征...")
        X = X.iloc[:, :100]
        print(f"减少特征后声学特征数据大小: {X.shape}")

    # 计算子量表得分
    print("计算子量表得分...")
    Y_sub = pd.DataFrame(index=Y.index)

    for subscale_name, items in SUBSCALES.items():
        # 确保所有项目都在Y中
        valid_items = [item for item in items if item in Y.columns]
        if len(valid_items) != len(items):
            missing_items = set(items) - set(valid_items)
            print(f"警告: 子量表 {subscale_name} 缺少以下项目: {missing_items}")

        # 计算子量表得分
        Y_sub[subscale_name] = Y[valid_items].sum(axis=1)

    # 计算总得分
    print("计算总得分...")
    y_total = Y_sub.sum(axis=1)

    # 划分数据集
    print(f"划分数据集 (测试集: {test_size})...")

    # 划分出测试集
    X_train, X_test, Y_train, Y_test, Y_sub_train, Y_sub_test, y_total_train, y_total_test = train_test_split(
        X, Y, Y_sub, y_total, test_size=test_size, random_state=random_seed
    )

    print(f"数据集划分完成 - 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    return {
        'X_train': X_train, 'X_test': X_test,
        'Y_train': Y_train, 'Y_test': Y_test,
        'Y_sub_train': Y_sub_train, 'Y_sub_test': Y_sub_test,
        'y_total_train': y_total_train, 'y_total_test': y_total_test
    }

def train_and_evaluate_model(model_type, X_train, X_test, y_train, y_test, target_name):
    """
    训练并评估模型

    Parameters
    ----------
    model_type : str
        模型类型，'svr'或'xgb'或'rf'
    X_train : pandas.DataFrame
        训练集特征
    X_test : pandas.DataFrame
        测试集特征
    y_train : pandas.Series
        训练集目标
    y_test : pandas.Series
        测试集目标
    target_name : str
        目标变量名称

    Returns
    -------
    dict
        评估指标
    """
    print(f"训练{model_type}模型预测{target_name}...")

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建模型
    if model_type == 'svr':
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
    elif model_type == 'xgb':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_SEED)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 训练模型
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 返回评估指标
    return {
        'model_type': model_type,
        'target': target_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }

def run_baseline_models(data_dict, model_types=['svr', 'xgb', 'rf']):
    """
    运行基线模型

    Parameters
    ----------
    data_dict : dict
        数据字典
    model_types : list
        模型类型列表

    Returns
    -------
    dict
        评估结果
    """
    results = []

    # 对每个子量表和总分运行模型
    for model_type in model_types:
        print(f"\n运行{model_type}模型...")

        # 对每个子量表运行模型
        for subscale in data_dict['Y_sub_train'].columns:
            result = train_and_evaluate_model(
                model_type,
                data_dict['X_train'],
                data_dict['X_test'],
                data_dict['Y_sub_train'][subscale],
                data_dict['Y_sub_test'][subscale],
                subscale
            )
            results.append(result)

        # 对总分运行模型
        result = train_and_evaluate_model(
            model_type,
            data_dict['X_train'],
            data_dict['X_test'],
            data_dict['y_total_train'],
            data_dict['y_total_test'],
            'total'
        )
        results.append(result)

    return results

def save_results(results, output_dir=RESULTS_DIR, model_types=None):
    """
    保存结果

    Parameters
    ----------
    results : list
        评估结果列表
    output_dir : str
        输出目录
    model_types : list
        模型类型列表
    """
    if model_types is None:
        model_types = list(set([r['model_type'] for r in results]))

    # 创建结果DataFrame
    results_df = pd.DataFrame([
        {
            'model_type': r['model_type'],
            'target': r['target'],
            'rmse': r['rmse'],
            'mae': r['mae'],
            'r2': r['r2']
        }
        for r in results
    ])

    # 保存结果
    results_df.to_csv(os.path.join(output_dir, 'baseline_results.csv'), index=False)

    # 创建结果表格
    table_df = results_df.pivot_table(
        index='target',
        columns='model_type',
        values=['rmse', 'mae', 'r2']
    )

    # 保存表格
    table_df.to_csv(os.path.join(output_dir, 'baseline_table.csv'))

    # 生成Markdown报告
    report = ["# 基线模型评估报告\n"]

    # 添加RMSE表格
    report.append("## RMSE\n")

    # 创建表头
    header = "| 目标 |"
    separator = "| --- |"
    for model in model_types:
        model_name = "SVR" if model == "svr" else "XGBoost" if model == "xgb" else "RandomForest" if model == "rf" else model
        header += f" {model_name} |"
        separator += " --- |"

    report.append(header)
    report.append(separator)

    for target in sorted(set(results_df['target'])):
        target_results = results_df[results_df['target'] == target]
        row = f"| {target} |"

        for model in model_types:
            model_result = target_results[target_results['model_type'] == model]
            if not model_result.empty:
                row += f" {model_result['rmse'].values[0]:.4f} |"
            else:
                row += " - |"

        report.append(row)

    report.append("\n")

    # 添加MAE表格
    report.append("## MAE\n")

    # 创建表头
    header = "| 目标 |"
    separator = "| --- |"
    for model in model_types:
        model_name = "SVR" if model == "svr" else "XGBoost" if model == "xgb" else "RandomForest" if model == "rf" else model
        header += f" {model_name} |"
        separator += " --- |"

    report.append(header)
    report.append(separator)

    for target in sorted(set(results_df['target'])):
        target_results = results_df[results_df['target'] == target]
        row = f"| {target} |"

        for model in model_types:
            model_result = target_results[target_results['model_type'] == model]
            if not model_result.empty:
                row += f" {model_result['mae'].values[0]:.4f} |"
            else:
                row += " - |"

        report.append(row)

    report.append("\n")

    # 添加R²表格
    report.append("## R²\n")

    # 创建表头
    header = "| 目标 |"
    separator = "| --- |"
    for model in model_types:
        model_name = "SVR" if model == "svr" else "XGBoost" if model == "xgb" else "RandomForest" if model == "rf" else model
        header += f" {model_name} |"
        separator += " --- |"

    report.append(header)
    report.append(separator)

    for target in sorted(set(results_df['target'])):
        target_results = results_df[results_df['target'] == target]
        row = f"| {target} |"

        for model in model_types:
            model_result = target_results[target_results['model_type'] == model]
            if not model_result.empty:
                row += f" {model_result['r2'].values[0]:.4f} |"
            else:
                row += " - |"

        report.append(row)

    # 保存报告
    with open(os.path.join(output_dir, 'baseline_report.md'), 'w') as f:
        f.write('\n'.join(report))

def main():
    # 设置固定的数据路径和参数
    acoustic_path = ACOUSTIC_FEATURES_PATH
    questionnaire_path = QUESTIONNAIRE_PATH
    test_size = 0.2
    models = 'svr,xgb,rf'

    # 解析模型类型
    model_types = models.split(',')

    # 加载数据
    data_dict = load_data(acoustic_path, questionnaire_path, test_size)

    # 运行基线模型
    results = run_baseline_models(data_dict, model_types)

    # 保存结果
    save_results(results, model_types=model_types)

    print(f"\n结果已保存到 {RESULTS_DIR}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总运行时间: {end_time - start_time:.2f} 秒")
