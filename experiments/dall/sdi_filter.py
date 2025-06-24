#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于SDI（样本异质性指数）的困难样本过滤模块
"""

import numpy as np
import pandas as pd
from config import SUBSCALES, RANDOM_SEED, OUTLIER_THRESHOLD

class SDIFilter:
    """
    基于SDI（样本异质性指数）的困难样本过滤类
    """

    def __init__(self, Y, Y_sub, subscale_mapping=SUBSCALES, outlier_threshold=0.1):
        """
        初始化SDI过滤器

        Parameters
        ----------
        Y : pandas.DataFrame
            项目得分矩阵
        Y_sub : pandas.DataFrame
            子量表得分矩阵
        subscale_mapping : dict
            子量表映射
        outlier_threshold : float
            异常值阈值（0-1之间），表示要移除的样本比例
        """
        self.Y = Y
        self.Y_sub = Y_sub
        self.subscale_mapping = subscale_mapping
        self.outlier_threshold = outlier_threshold
        self.difficult_samples = None
        self.filtered_Y = None
        self.filtered_Y_sub = None

    def standardize(self, df, columns):
        """
        标准化指定列的数据

        Parameters
        ----------
        df : pandas.DataFrame
            数据框
        columns : list
            要标准化的列名列表

        Returns
        -------
        pandas.DataFrame
            标准化后的数据
        """
        return (df[columns] - df[columns].mean()) / df[columns].std()

    def categorize_severity(self, df, column, bins, labels):
        """
        对指定列进行严重程度分类

        Parameters
        ----------
        df : pandas.DataFrame
            数据框
        column : str
            要分类的列名
        bins : list
            分类边界值
        labels : list
            分类标签

        Returns
        -------
        pandas.Series
            分类结果
        """
        return pd.cut(df[column], bins=bins, labels=labels)

    def calculate_center_vector(self, df, group_column, target_columns):
        """
        计算各组的中心向量

        Parameters
        ----------
        df : pandas.DataFrame
            数据框
        group_column : str
            分组列名
        target_columns : list
            目标特征列名列表

        Returns
        -------
        dict
            各组的中心向量
        """
        groups = df.groupby(group_column)
        center_vectors = {}
        for group_name, group in groups:
            center_vector = group[target_columns].mean()
            center_vectors[group_name] = center_vector
        return center_vectors

    def calculate_distances(self, df, group_column, target_columns, center_vectors):
        """
        计算样本到其所属组中心向量的距离

        Parameters
        ----------
        df : pandas.DataFrame
            数据框
        group_column : str
            分组列名
        target_columns : list
            目标特征列名列表
        center_vectors : dict
            各组的中心向量

        Returns
        -------
        list
            距离列表
        """
        distances = []
        for index, row in df.iterrows():
            group = row[group_column]
            center_vector = center_vectors[group]
            distance = np.sqrt(((row[target_columns] - center_vector) ** 2).sum())
            distances.append(distance)
        return distances

    def normalize_distances(self, distances):
        """
        归一化距离

        Parameters
        ----------
        distances : list
            距离列表

        Returns
        -------
        numpy.ndarray
            归一化后的距离
        """
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        normalized_distances = (distances - min_dist) / (max_dist - min_dist)
        return normalized_distances

    def calculate_gdi(self, df, group_column, target_columns):
        """
        计算组异质性指数(GDI)

        Parameters
        ----------
        df : pandas.DataFrame
            数据框
        group_column : str
            分组列名
        target_columns : list
            目标特征列名列表

        Returns
        -------
        tuple
            (gdi, summary)
        """
        center_vectors = self.calculate_center_vector(df, group_column, target_columns)
        distances = self.calculate_distances(df, group_column, target_columns, center_vectors)
        normalized_distances = self.normalize_distances(distances)
        df = df.copy()
        df['distance'] = normalized_distances
        mean_distance = df.groupby(group_column)['distance'].mean()
        std_distance = df.groupby(group_column)['distance'].std()
        gdi = mean_distance + std_distance
        return gdi, df.groupby(group_column)['distance'].agg(['mean', 'std', 'min', 'max', 'count'])

    def calculate_sdi(self, gdi, counts):
        """
        计算样本异质性指数(SDI)

        Parameters
        ----------
        gdi : pandas.Series
            组异质性指数
        counts : pandas.Series
            各组的样本数

        Returns
        -------
        float
            样本异质性指数
        """
        weighted_sum = (gdi * counts).sum()
        total_count = counts.sum()
        sdi = weighted_sum / total_count
        return sdi

    def identify_difficult_samples(self):
        """
        基于SDI识别困难样本

        Returns
        -------
        pandas.Index
            困难样本的索引
        """
        print("开始基于SDI识别困难样本...")
        
        # 创建包含原始项目得分和子量表得分的工作数据框
        df_work = self.Y.copy()
        for subscale in self.Y_sub.columns:
            df_work[subscale + '_Total'] = self.Y_sub[subscale]

        # 用于存储各样本的距离得分
        all_distances = []
        all_indices = []
        
        # 对每个子量表计算距离
        for subscale, items in self.subscale_mapping.items():
            if subscale not in self.Y_sub.columns:
                continue
                
            print(f"计算子量表 {subscale} 的样本距离...")
            
            # 确保所有项目都在Y中
            valid_items = [item for item in items if item in self.Y.columns]
            if len(valid_items) < 2:  # 需要至少两个项目才能计算距离
                print(f"警告: 子量表 {subscale} 有效项目不足，跳过")
                continue

            # 标准化项目得分
            df_standardized = df_work.copy()
            df_standardized[valid_items] = self.standardize(df_work, valid_items)
            
            # 基于总分分组
            df_standardized['temp_group'] = pd.qcut(df_standardized[subscale + '_Total'], 
                                                   q=5, labels=False, duplicates='drop')
            
            # 计算各样本到其组中心的距离
            center_vectors = self.calculate_center_vector(df_standardized, 'temp_group', valid_items)
            distances = self.calculate_distances(df_standardized, 'temp_group', valid_items, center_vectors)
            
            # 存储距离和索引
            all_distances.extend(distances)
            all_indices.extend(df_standardized.index)
        
        # 如果没有有效的距离，返回空列表
        if not all_distances:
            print("警告: 无法计算任何距离，无法识别困难样本")
            self.difficult_samples = pd.Index([])
            return self.difficult_samples
            
        # 创建距离数据框
        distance_df = pd.DataFrame({
            'index': all_indices,
            'distance': all_distances
        })
        
        # 对于重复的索引，取平均距离
        distance_df = distance_df.groupby('index')['distance'].mean().reset_index()
        
        # 按距离降序排序
        distance_df = distance_df.sort_values('distance', ascending=False)
        
        # 根据阈值选择困难样本
        n_difficult = int(len(set(distance_df['index'])) * self.outlier_threshold)
        if n_difficult == 0:
            n_difficult = 1  # 至少选择一个样本
            
        difficult_samples = pd.Index(distance_df.head(n_difficult)['index'])
        
        print(f"基于SDI识别出 {len(difficult_samples)} 个困难样本 (总共 {len(self.Y)} 个样本)")
        self.difficult_samples = difficult_samples
        
        return difficult_samples

    def filter_samples(self):
        """
        过滤SDI较高的困难样本

        Returns
        -------
        tuple
            (filtered_Y, filtered_Y_sub)
        """
        if self.difficult_samples is None:
            self.identify_difficult_samples()
            
        # 过滤困难样本
        mask = ~self.Y.index.isin(self.difficult_samples)
        self.filtered_Y = self.Y[mask].copy()
        self.filtered_Y_sub = self.Y_sub[mask].copy()
        
        # 确保至少保留一些样本
        if len(self.filtered_Y) == 0:
            print("警告: SDI过滤后没有剩余样本，将使用全部原始样本")
            self.filtered_Y = self.Y.copy()
            self.filtered_Y_sub = self.Y_sub.copy()
        
        print(f"SDI过滤后剩余 {len(self.filtered_Y)} 个样本 (总共 {len(self.Y)} 个样本)")
        
        return self.filtered_Y, self.filtered_Y_sub 