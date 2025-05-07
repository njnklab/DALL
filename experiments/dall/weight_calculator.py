#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网络分析与权重计算模块
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from config import RANDOM_SEED, SUBSCALES, GRAPHICAL_LASSO_CV, GRAPHICAL_LASSO_ALPHA, MDS_ITERATIONS

class WeightCalculator:
    """
    权重计算类
    """

    def __init__(self, Y_train, subscale_mapping=SUBSCALES):
        """
        初始化权重计算器

        Parameters
        ----------
        Y_train : pandas.DataFrame
            训练集项目得分矩阵
        subscale_mapping : dict
            子量表映射
        """
        self.Y_train = Y_train
        self.subscale_mapping = subscale_mapping

    def calculate_weights(self):
        """
        计算权重

        Returns
        -------
        tuple
            (item_weights, subscale_weights)
        """
        print("计算项目权重...")

        # 获取有效项目
        valid_items = []
        for items in self.subscale_mapping.values():
            valid_items.extend([item for item in items if item in self.Y_train.columns])

        # 提取相关数据
        corr_data = self.Y_train[valid_items]

        # 计算相关系数矩阵
        corr_matrix = corr_data.corr()

        # 使用Graphical Lasso估计稀疏逆协方差矩阵
        print("使用Graphical Lasso估计稀疏逆协方差矩阵...")
        if GRAPHICAL_LASSO_CV:
            model = GraphicalLassoCV(cv=5)
        else:
            model = GraphicalLasso(alpha=GRAPHICAL_LASSO_ALPHA, random_state=RANDOM_SEED)
        model.fit(corr_data)

        # 获取精度矩阵（逆协方差矩阵）
        precision_matrix = pd.DataFrame(model.precision_, index=valid_items, columns=valid_items)

        # 构建条件依赖网络图
        print("构建条件依赖网络图...")
        graph = nx.Graph()
        for item in valid_items:
            graph.add_node(item)
        for i, item1 in enumerate(valid_items):
            for j, item2 in enumerate(valid_items):
                if i < j:
                    weight = precision_matrix.loc[item1, item2]
                    if weight != 0:
                        graph.add_edge(item1, item2, weight=abs(weight))

        # 计算最小支配集
        print(f"计算最小支配集 (重复{MDS_ITERATIONS}次)...")
        item_counts = {item: 0 for item in valid_items}

        for _ in range(MDS_ITERATIONS):
            # 使用贪心算法近似计算最小支配集
            dominating_set = self._greedy_min_dominating_set(graph)

            # 更新计数
            for item in dominating_set:
                item_counts[item] += 1

        # 计算控制频率
        item_weights = {item: count / MDS_ITERATIONS for item, count in item_counts.items()}

        # 计算子量表权重
        print("计算子量表权重...")
        subscale_weights = {}

        for subscale, items in self.subscale_mapping.items():
            valid_subscale_items = [item for item in items if item in valid_items]
            if valid_subscale_items:
                subscale_weights[subscale] = np.mean([item_weights[item] for item in valid_subscale_items])
            else:
                subscale_weights[subscale] = 0.0

        # 转换为Series
        item_weights = pd.Series(item_weights)
        subscale_weights = pd.Series(subscale_weights)

        print("权重计算完成")
        return item_weights, subscale_weights

    def _greedy_min_dominating_set(self, graph):
        """
        使用贪心算法近似计算最小支配集

        Parameters
        ----------
        graph : networkx.Graph
            网络图

        Returns
        -------
        set
            最小支配集
        """
        # 初始化
        dominating_set = set()
        dominated = {node: False for node in graph.nodes()}

        # 当还有未被支配的节点时
        while not all(dominated.values()):
            # 找到能支配最多未被支配节点的节点
            max_node = None
            max_dominated = -1

            for node in graph.nodes():
                if node in dominating_set:
                    continue

                # 计算该节点能支配的未被支配节点数量
                dominated_count = 0
                if not dominated[node]:
                    dominated_count += 1  # 节点自身

                for neighbor in graph.neighbors(node):
                    if not dominated[neighbor]:
                        dominated_count += 1

                if dominated_count > max_dominated:
                    max_node = node
                    max_dominated = dominated_count

            # 将最优节点加入支配集
            if max_node is not None:
                dominating_set.add(max_node)

                # 更新被支配状态
                dominated[max_node] = True
                for neighbor in graph.neighbors(max_node):
                    dominated[neighbor] = True

        return dominating_set
