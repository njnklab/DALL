#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理模块
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED, SUBSCALES, TEST_SIZE, VAL_SIZE

class DataProcessor:
    """
    数据处理类
    """

    def __init__(self, acoustic_path, questionnaire_path):
        """
        初始化数据处理器

        Parameters
        ----------
        acoustic_path : str
            声学特征数据路径
        questionnaire_path : str
            问卷数据路径
        """
        self.acoustic_path = acoustic_path
        self.questionnaire_path = questionnaire_path

    def load_data(self):
        """
        加载数据

        Returns
        -------
        tuple
            (X, Y, Y_sub, y_total)
        """
        print("加载声学特征数据...")
        X = pd.read_csv(self.acoustic_path)

        print("加载问卷数据...")
        Y = pd.read_csv(self.questionnaire_path)

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

        return X, Y, Y_sub, y_total

    def split_data(self, X, Y, Y_sub, y_total):
        """
        划分数据集

        Parameters
        ----------
        X : pandas.DataFrame
            特征矩阵
        Y : pandas.DataFrame
            项目得分矩阵
        Y_sub : pandas.DataFrame
            子量表得分矩阵
        y_total : pandas.Series
            总得分

        Returns
        -------
        dict
            包含划分后数据的字典
        """
        print(f"划分数据集 (测试集: {TEST_SIZE}, 验证集: {VAL_SIZE})...")

        # 划分出测试集
        X_train_val, X_test, Y_train_val, Y_test, Y_sub_train_val, Y_sub_test, y_total_train_val, y_total_test = train_test_split(
            X, Y, Y_sub, y_total, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        # 从训练集中划分出验证集
        val_ratio = VAL_SIZE / (1 - TEST_SIZE)
        X_train, X_val, Y_train, Y_val, Y_sub_train, Y_sub_val, y_total_train, y_total_val = train_test_split(
            X_train_val, Y_train_val, Y_sub_train_val, y_total_train_val, test_size=val_ratio, random_state=RANDOM_SEED
        )

        print(f"数据集划分完成 - 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test,
            'Y_sub_train': Y_sub_train, 'Y_sub_val': Y_sub_val, 'Y_sub_test': Y_sub_test,
            'y_total_train': y_total_train, 'y_total_val': y_total_val, 'y_total_test': y_total_test
        }

    def process(self):
        """
        处理数据

        Returns
        -------
        dict
            包含处理后数据的字典
        """
        X, Y, Y_sub, y_total = self.load_data()
        data_dict = self.split_data(X, Y, Y_sub, y_total)
        return data_dict
