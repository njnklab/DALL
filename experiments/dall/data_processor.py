#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理模块
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import RANDOM_SEED, SUBSCALES, TEST_SIZE, VAL_SIZE, OUTLIER_THRESHOLD, APPLY_PCA, PCA_VARIANCE
from .sdi_filter import SDIFilter

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
        
        # 应用SDI过滤
        if OUTLIER_THRESHOLD > 0:
            print(f"应用SDI样本过滤 (阈值: {OUTLIER_THRESHOLD})...")
            sdi_filter = SDIFilter(Y, Y_sub, SUBSCALES, OUTLIER_THRESHOLD)
            filtered_Y, filtered_Y_sub = sdi_filter.filter_samples()
            
            # 更新数据
            filtered_indices = filtered_Y.index
            X = X.loc[filtered_indices]
            Y = filtered_Y
            Y_sub = filtered_Y_sub
            y_total = Y_sub.sum(axis=1)
            
            print(f"SDI过滤后数据大小 - X: {X.shape}, Y: {Y.shape}")
        
        # 应用PCA降维（只对声学特征X进行处理）
        if APPLY_PCA:
            print(f"应用PCA降维 (保留方差比例: {PCA_VARIANCE})...")
            # 保存原始索引
            original_index = X.index
            
            # 标准化数据
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 应用PCA
            pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_SEED)
            X_pca = pca.fit_transform(X_scaled)
            
            # 获取PCA降维后的特征名称和相关信息
            n_components = pca.n_components_
            explained_variance_ratio = pca.explained_variance_ratio_.sum()
            
            # 转换回DataFrame并恢复索引
            feature_names = [f"PC{i+1}" for i in range(n_components)]
            X = pd.DataFrame(X_pca, columns=feature_names, index=original_index)
            
            print(f"PCA降维后数据大小: {X.shape}，保留了{explained_variance_ratio:.2%}的方差")
            
            # 打印各主成分解释的方差比例
            if n_components <= 20:  # 如果主成分数量不多，打印每个主成分的方差贡献
                for i, var in enumerate(pca.explained_variance_ratio_):
                    print(f"  PC{i+1}: {var:.2%}")
            else:  # 否则只打印前10个
                for i, var in enumerate(pca.explained_variance_ratio_[:10]):
                    print(f"  PC{i+1}: {var:.2%}")
                print(f"  ... 以及其他 {n_components-10} 个主成分")
        
        data_dict = self.split_data(X, Y, Y_sub, y_total)
        return data_dict
