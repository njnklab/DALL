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
# 导入所需模块
import sys
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加当前目录到系统路径
sys.path.append(current_dir)

# 导入模块
from data_processor import DataProcessor
from weight_calculator import WeightCalculator
from item_predictor import ItemPredictor
from residual_corrector import ResidualCorrector
from evaluator import Evaluator
from subscale_contribution import SubscaleContributionAnalyzer

# 设置数据路径
ACOUSTIC_FEATURES_PATH = '/home/user/xuxiao/DALL/dataset/CS-NRAC-E.csv'
QUESTIONNAIRE_PATH = '/home/user/xuxiao/DALL/dataset/raw_info.csv'

def main():
    """
    主函数
    """
    # 使用固定的数据路径和输出目录
    acoustic_path = '/home/user/xuxiao/DALL/dataset/CS-NRAC-E.csv'
    questionnaire_path = '/home/user/xuxiao/DALL/dataset/raw_info.csv'
    output_dir = '/home/user/xuxiao/DALL/results/dall'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 记录开始时间
    start_time = time.time()

    # 数据处理
    print("\n=== 数据处理 ===")
    data_processor = DataProcessor(acoustic_path, questionnaire_path)
    data_dict = data_processor.process()

    # 网络分析与权重计算
    print("\n=== 网络分析与权重计算 ===")
    weight_calculator = WeightCalculator(data_dict['Y_train'])
    item_weights, subscale_weights = weight_calculator.calculate_weights()

    # 保存权重
    item_weights.to_csv(os.path.join(output_dir, 'item_weights.csv'))
    subscale_weights.to_csv(os.path.join(output_dir, 'subscale_weights.csv'))

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
    Y_test_pred = item_predictor.predict()

    # 聚合子量表预测
    Y_sub_oof = item_predictor.aggregate_subscales(Y_oof)
    Y_sub_test_pred = item_predictor.aggregate_subscales(Y_test_pred)

    # 结构化残差校正
    print("\n=== 结构化残差校正 ===")
    residual_corrector = ResidualCorrector(
        data_dict['X_train'],
        Y_sub_oof,
        data_dict['y_total_train'],
        subscale_weights,
        data_dict['X_test'],
        Y_sub_test_pred
    )

    # 训练残差校正模型
    base_estimator, residual_corrector_model = residual_corrector.train()

    # 生成总分预测
    y_total_test_pred = residual_corrector.predict()

    # 评估
    print("\n=== 评估 ===")
    evaluator = Evaluator(
        data_dict['Y_test'],
        Y_test_pred,
        data_dict['Y_sub_test'],
        Y_sub_test_pred,
        data_dict['y_total_test'],
        y_total_test_pred,
        output_dir
    )

    # 执行评估
    metrics = evaluator.evaluate()

    # 子量表贡献调制分析
    print("\n=== 子量表贡献调制分析 ===")
    scma = SubscaleContributionAnalyzer(
        data_dict['X_test'],
        Y_sub_test_pred,
        base_estimator,
        residual_corrector_model,
        subscale_weights,
        output_dir
    )

    # 执行分析
    scma.analyze()

    # 记录结束时间
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
