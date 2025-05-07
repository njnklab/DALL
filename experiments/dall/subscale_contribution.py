#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
子量表贡献调制分析模块
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import PERTURBATION_SIZE

class SubscaleContributionAnalyzer:
    """
    子量表贡献调制分析类
    """

    def __init__(self, X_test, Y_sub_test, base_estimator, residual_corrector, subscale_weights, output_dir):
        """
        初始化子量表贡献调制分析器

        Parameters
        ----------
        X_test : pandas.DataFrame
            测试集特征矩阵
        Y_sub_test : pandas.DataFrame
            测试集子量表预测矩阵
        base_estimator : object
            基估计器
        residual_corrector : object
            残差校正模型
        subscale_weights : pandas.Series
            子量表权重
        output_dir : str
            输出目录
        """
        self.X_test = X_test
        self.Y_sub_test = Y_sub_test
        self.base_estimator = base_estimator
        self.residual_corrector = residual_corrector
        self.subscale_weights = subscale_weights
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(os.path.join(output_dir, 'scma'), exist_ok=True)

    def analyze(self, perturbation_size=PERTURBATION_SIZE):
        """
        执行子量表贡献调制分析

        Parameters
        ----------
        perturbation_size : float
            扰动大小

        Returns
        -------
        pandas.DataFrame
            敏感度分析结果
        """
        print(f"执行子量表贡献调制分析 (扰动大小: {perturbation_size})...")

        # 获取子量表列表
        subscales = self.Y_sub_test.columns

        # 初始化结果DataFrame
        results = []

        # 对每个样本进行分析
        for i, idx in enumerate(self.X_test.index):
            if i % 100 == 0:
                print(f"分析样本 {i+1}/{len(self.X_test)}...")

            # 获取样本数据
            X_sample = self.X_test.loc[[idx]]
            Y_sub_sample = self.Y_sub_test.loc[[idx]]

            # 计算原始预测
            y_base_orig = self._predict_base(Y_sub_sample)
            y_corr_orig = self._predict_residual(X_sample, Y_sub_sample)
            y_total_orig = y_base_orig + y_corr_orig

            # 对每个子量表进行扰动分析
            for subscale in subscales:
                # 创建正向扰动的子量表预测
                Y_sub_pos = Y_sub_sample.copy()
                Y_sub_pos[subscale] += perturbation_size

                # 创建负向扰动的子量表预测
                Y_sub_neg = Y_sub_sample.copy()
                Y_sub_neg[subscale] -= perturbation_size

                # 计算基础路径调制
                y_base_pos = self._predict_base(Y_sub_pos)
                y_base_neg = self._predict_base(Y_sub_neg)
                delta_base_pos = y_base_pos - y_base_orig
                delta_base_neg = y_base_neg - y_base_orig

                # 计算校正路径调制
                y_corr_pos = self._predict_residual(X_sample, Y_sub_pos)
                y_corr_neg = self._predict_residual(X_sample, Y_sub_neg)
                delta_corr_pos = y_corr_pos - y_corr_orig
                delta_corr_neg = y_corr_neg - y_corr_orig

                # 计算总预测调制
                delta_total_pos = delta_base_pos + delta_corr_pos
                delta_total_neg = delta_base_neg + delta_corr_neg

                # 使用中心差分量化敏感度
                S_total = (delta_total_pos - delta_total_neg) / (2 * perturbation_size)
                S_base = (delta_base_pos - delta_base_neg) / (2 * perturbation_size)
                S_corr = (delta_corr_pos - delta_corr_neg) / (2 * perturbation_size)

                # 保存结果
                results.append({
                    'sample_id': idx,
                    'subscale': subscale,
                    'S_total': S_total[0],
                    'S_base': S_base[0],
                    'S_corr': S_corr[0]
                })

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 保存结果
        results_df.to_csv(os.path.join(self.output_dir, 'scma', 'sensitivity.csv'), index=False)

        # 计算平均敏感度
        avg_sensitivity = results_df.groupby('subscale')[['S_total', 'S_base', 'S_corr']].mean()
        avg_sensitivity.to_csv(os.path.join(self.output_dir, 'scma', 'avg_sensitivity.csv'))

        # 可视化
        self._visualize_sensitivity(avg_sensitivity)

        print("子量表贡献调制分析完成")
        return results_df

    def _predict_base(self, Y_sub):
        """
        使用基估计器生成预测

        Parameters
        ----------
        Y_sub : pandas.DataFrame
            子量表预测矩阵

        Returns
        -------
        numpy.ndarray
            基预测
        """
        # 构建特征矩阵
        X = pd.DataFrame(index=Y_sub.index)
        for subscale in Y_sub.columns:
            if subscale in self.subscale_weights.index:
                X[f"weighted_{subscale}"] = Y_sub[subscale] * self.subscale_weights[subscale]

        # 生成预测
        return self.base_estimator.predict(X)

    def _predict_residual(self, X, Y_sub):
        """
        使用残差校正模型生成预测

        Parameters
        ----------
        X : pandas.DataFrame
            特征矩阵
        Y_sub : pandas.DataFrame
            子量表预测矩阵

        Returns
        -------
        numpy.ndarray
            残差预测
        """
        # 构建特征矩阵
        Z = pd.DataFrame(index=X.index)

        # 添加原始特征
        for col in X.columns:
            Z[f"acoustic_{col}"] = X[col]

        # 添加子量表预测
        for subscale in Y_sub.columns:
            Z[f"subscale_{subscale}"] = Y_sub[subscale]

        # 添加子量表权重
        for subscale in self.subscale_weights.index:
            if subscale in Y_sub.columns:
                Z[f"weight_{subscale}"] = self.subscale_weights[subscale]

        # 生成预测
        return self.residual_corrector.predict(Z)

    def _visualize_sensitivity(self, avg_sensitivity):
        """
        可视化敏感度

        Parameters
        ----------
        avg_sensitivity : pandas.DataFrame
            平均敏感度
        """
        # 设置样式
        sns.set(style="whitegrid")

        # 创建图形
        plt.figure(figsize=(12, 8))

        # 绘制柱状图
        ax = sns.barplot(x=avg_sensitivity.index, y='S_total', data=avg_sensitivity.reset_index(), color='blue', label='Total')
        sns.barplot(x=avg_sensitivity.index, y='S_base', data=avg_sensitivity.reset_index(), color='green', label='Base')
        sns.barplot(x=avg_sensitivity.index, y='S_corr', data=avg_sensitivity.reset_index(), color='red', label='Correction')

        # 添加标题和标签
        plt.title('Average Sensitivity by Subscale', fontsize=16)
        plt.xlabel('Subscale', fontsize=14)
        plt.ylabel('Sensitivity', fontsize=14)
        plt.legend()

        # 保存图形
        plt.savefig(os.path.join(self.output_dir, 'scma', 'avg_sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.close()
