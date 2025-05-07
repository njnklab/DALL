#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估模块
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from config import SUBSCALES

class Evaluator:
    """
    评估类
    """

    def __init__(self, Y_test, Y_test_pred, Y_sub_test, Y_sub_test_pred, y_total_test, y_total_test_pred, output_dir):
        """
        初始化评估器

        Parameters
        ----------
        Y_test : pandas.DataFrame
            测试集真实项目得分矩阵
        Y_test_pred : pandas.DataFrame
            测试集预测项目得分矩阵
        Y_sub_test : pandas.DataFrame
            测试集真实子量表得分矩阵
        Y_sub_test_pred : pandas.DataFrame
            测试集预测子量表得分矩阵
        y_total_test : pandas.Series
            测试集真实总分
        y_total_test_pred : numpy.ndarray
            测试集预测总分
        output_dir : str
            输出目录
        """
        self.Y_test = Y_test
        self.Y_test_pred = Y_test_pred
        self.Y_sub_test = Y_sub_test
        self.Y_sub_test_pred = Y_sub_test_pred
        self.y_total_test = y_total_test
        self.y_total_test_pred = y_total_test_pred
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self):
        """
        执行评估

        Returns
        -------
        dict
            评估指标
        """
        print("执行评估...")

        # 初始化评估指标
        metrics = {
            'item_level': {},
            'subscale_level': {},
            'total_level': {}
        }

        # 评估项目级预测
        print("评估项目级预测...")
        valid_items = [item for subscale_items in SUBSCALES.values() for item in subscale_items
                      if item in self.Y_test.columns and item in self.Y_test_pred.columns]

        for item in valid_items:
            y_true = self.Y_test[item]
            y_pred = self.Y_test_pred[item]

            # 计算评估指标
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)  # 计算RMSE
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            pearson_r, p_value = pearsonr(y_true, y_pred)

            # 保存评估指标
            metrics['item_level'][item] = {
                'rmse': rmse,  # 使用RMSE替代MSE
                'mae': mae,
                'r2': r2,
                'pearson_r': pearson_r,
                'p_value': p_value
            }

        # 计算项目级平均指标
        metrics['item_level']['average'] = {
            'rmse': np.mean([m['rmse'] for m in metrics['item_level'].values() if isinstance(m, dict)]),
            'mae': np.mean([m['mae'] for m in metrics['item_level'].values() if isinstance(m, dict)]),
            'r2': np.mean([m['r2'] for m in metrics['item_level'].values() if isinstance(m, dict)]),
            'pearson_r': np.mean([m['pearson_r'] for m in metrics['item_level'].values() if isinstance(m, dict)])
        }

        # 评估子量表级预测
        print("评估子量表级预测...")
        for subscale in self.Y_sub_test.columns:
            if subscale in self.Y_sub_test_pred.columns:
                y_true = self.Y_sub_test[subscale]
                y_pred = self.Y_sub_test_pred[subscale]

                # 计算评估指标
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)  # 计算RMSE
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                pearson_r, p_value = pearsonr(y_true, y_pred)

                # 保存评估指标
                metrics['subscale_level'][subscale] = {
                    'rmse': rmse,  # 使用RMSE替代MSE
                    'mae': mae,
                    'r2': r2,
                    'pearson_r': pearson_r,
                    'p_value': p_value
                }

        # 计算子量表级平均指标
        metrics['subscale_level']['average'] = {
            'rmse': np.mean([m['rmse'] for m in metrics['subscale_level'].values() if isinstance(m, dict)]),
            'mae': np.mean([m['mae'] for m in metrics['subscale_level'].values() if isinstance(m, dict)]),
            'r2': np.mean([m['r2'] for m in metrics['subscale_level'].values() if isinstance(m, dict)]),
            'pearson_r': np.mean([m['pearson_r'] for m in metrics['subscale_level'].values() if isinstance(m, dict)])
        }

        # 评估总分级预测
        print("评估总分级预测...")
        y_true = self.y_total_test
        y_pred = self.y_total_test_pred

        # 计算评估指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)  # 计算RMSE
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        pearson_r, p_value = pearsonr(y_true, y_pred)

        # 保存评估指标
        metrics['total_level'] = {
            'rmse': rmse,  # 使用RMSE替代MSE
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_r,
            'p_value': p_value
        }

        # 保存评估指标
        self._save_metrics(metrics)

        # 可视化结果
        self._visualize_results()

        # 生成评估报告
        self._generate_report(metrics)

        print("评估完成")
        return metrics

    def _save_metrics(self, metrics):
        """
        保存评估指标

        Parameters
        ----------
        metrics : dict
            评估指标
        """
        # 保存项目级指标
        item_metrics = pd.DataFrame([
            {
                'item': item,
                'rmse': m['rmse'],
                'mae': m['mae'],
                'r2': m['r2'],
                'pearson_r': m['pearson_r'],
                'p_value': m['p_value'] if 'p_value' in m else np.nan
            }
            for item, m in metrics['item_level'].items() if isinstance(m, dict)
        ])
        item_metrics.to_csv(os.path.join(self.output_dir, 'item_metrics.csv'), index=False)

        # 保存子量表级指标
        subscale_metrics = pd.DataFrame([
            {
                'subscale': subscale,
                'rmse': m['rmse'],
                'mae': m['mae'],
                'r2': m['r2'],
                'pearson_r': m['pearson_r'],
                'p_value': m['p_value'] if 'p_value' in m else np.nan
            }
            for subscale, m in metrics['subscale_level'].items() if isinstance(m, dict)
        ])
        subscale_metrics.to_csv(os.path.join(self.output_dir, 'subscale_metrics.csv'), index=False)

        # 保存总分级指标
        total_metrics = pd.DataFrame([
            {
                'metric': metric,
                'value': value
            }
            for metric, value in metrics['total_level'].items()
        ])
        total_metrics.to_csv(os.path.join(self.output_dir, 'total_metrics.csv'), index=False)

    def _visualize_results(self):
        """
        可视化结果
        """
        # 设置样式
        sns.set(style="whitegrid")

        # 可视化总分预测
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=self.y_total_test, y=self.y_total_test_pred)
        plt.plot([self.y_total_test.min(), self.y_total_test.max()],
                [self.y_total_test.min(), self.y_total_test.max()],
                'k--', lw=2)
        plt.title('Total Score: True vs Predicted', fontsize=16)
        plt.xlabel('True Total Score', fontsize=14)
        plt.ylabel('Predicted Total Score', fontsize=14)
        plt.savefig(os.path.join(self.output_dir, 'total_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 可视化子量表R²
        plt.figure(figsize=(12, 8))
        subscale_r2 = {subscale: r2_score(self.Y_sub_test[subscale], self.Y_sub_test_pred[subscale])
                      for subscale in self.Y_sub_test.columns if subscale in self.Y_sub_test_pred.columns}
        sns.barplot(x=list(subscale_r2.keys()), y=list(subscale_r2.values()))
        plt.title('Subscale R² Scores', fontsize=16)
        plt.xlabel('Subscale', fontsize=14)
        plt.ylabel('R²', fontsize=14)
        plt.savefig(os.path.join(self.output_dir, 'subscale_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 可视化项目级Pearson相关系数
        valid_items = [item for subscale_items in SUBSCALES.values() for item in subscale_items
                      if item in self.Y_test.columns and item in self.Y_test_pred.columns]

        item_pearson = {}
        for item in valid_items:
            pearson_r, _ = pearsonr(self.Y_test[item], self.Y_test_pred[item])
            item_pearson[item] = pearson_r

        # 创建热图数据
        heatmap_data = pd.Series(item_pearson).reset_index()
        heatmap_data.columns = ['item', 'pearson_r']

        # 提取子量表信息
        heatmap_data['subscale'] = heatmap_data['item'].apply(lambda x: next((s for s, items in SUBSCALES.items() if x in items), 'Unknown'))

        # 透视表
        heatmap_matrix = heatmap_data.pivot_table(index='subscale', columns='item', values='pearson_r')

        # 绘制热图
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Item-level Pearson Correlation Coefficients', fontsize=16)
        plt.savefig(os.path.join(self.output_dir, 'item_pearson_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_report(self, metrics):
        """
        生成评估报告

        Parameters
        ----------
        metrics : dict
            评估指标
        """
        # 创建报告
        report = ["# DALL模型评估报告\n"]

        # 添加总分级评估结果
        report.append("## 总分级评估结果\n")
        for metric, value in metrics['total_level'].items():
            report.append(f"- {metric}: {value:.4f}")
        report.append("\n")

        # 添加子量表级评估结果
        report.append("## 子量表级评估结果\n")
        report.append("| 子量表 | RMSE | MAE | R² | Pearson r | p值 |")
        report.append("| --- | --- | --- | --- | --- | --- |")
        for subscale, subscale_metrics in metrics['subscale_level'].items():
            if isinstance(subscale_metrics, dict) and subscale != 'average':
                report.append(f"| {subscale} | {subscale_metrics['rmse']:.4f} | {subscale_metrics['mae']:.4f} | {subscale_metrics['r2']:.4f} | {subscale_metrics['pearson_r']:.4f} | {subscale_metrics['p_value']:.4e} |")
        report.append(f"| 平均 | {metrics['subscale_level']['average']['rmse']:.4f} | {metrics['subscale_level']['average']['mae']:.4f} | {metrics['subscale_level']['average']['r2']:.4f} | {metrics['subscale_level']['average']['pearson_r']:.4f} | - |")
        report.append("\n")

        # 添加项目级评估结果摘要
        report.append("## 项目级评估结果摘要\n")
        report.append(f"- 平均RMSE: {metrics['item_level']['average']['rmse']:.4f}")
        report.append(f"- 平均MAE: {metrics['item_level']['average']['mae']:.4f}")
        report.append(f"- 平均R²: {metrics['item_level']['average']['r2']:.4f}")
        report.append(f"- 平均Pearson r: {metrics['item_level']['average']['pearson_r']:.4f}")
        report.append("\n")

        # 保存报告
        with open(os.path.join(self.output_dir, 'evaluation_report.md'), 'w') as f:
            f.write('\n'.join(report))
