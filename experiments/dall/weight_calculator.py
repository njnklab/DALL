#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网络分析与权重计算模块
"""

import numpy as np
import pandas as pd
import networkx as nx
import random
import itertools
from scipy import stats
import datetime
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from config import RANDOM_SEED, SUBSCALES, GRAPHICAL_LASSO_CV, GRAPHICAL_LASSO_ALPHA, MDS_ITERATIONS
import community.community_louvain as louvain


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
        self.graph = None
        self.valid_items = []
        self.all_dom_sets = []
        self.communities = None
        self.item_weights = None
        self.subscale_weights = None
        self.module_controllability = None

    def calculate_weights(self, calculate_module_controllability=True):
        """
        计算权重

        Parameters
        ----------
        calculate_module_controllability : bool
            是否计算模块可控性

        Returns
        -------
        tuple
            (item_weights, subscale_weights)
        """
        print("计算项目权重...")
        start_time = datetime.datetime.now()
        
        # 获取有效项目
        self.valid_items = []
        for items in self.subscale_mapping.values():
            self.valid_items.extend([item for item in items if item in self.Y_train.columns])

        # 提取相关数据
        corr_data = self.Y_train[self.valid_items]

        # 构建网络
        self.graph = self._construct_network(corr_data, self.valid_items)

        # 社区检测（如果louvain模块可用）
        if louvain is not None and calculate_module_controllability:
            print("执行社区检测...")
            self.communities = louvain.best_partition(self.graph)
            # 简化社区检测输出
            num_communities = max(self.communities.values()) + 1
            print(f"检测到 {num_communities} 个社区")

        # 计算最小支配集
        print(f"计算最小支配集 (重复{MDS_ITERATIONS}次)...")
        self.all_dom_sets = self._greedy_minimum_dominating_set(self.graph, MDS_ITERATIONS)
        
        # 计算控制频率（节点权重）
        self.item_weights = self._dominating_frequency(self.all_dom_sets, self.graph)
        print("控制频率(CF)值计算完成")

        # 计算子量表权重
        print("计算子量表权重...")
        self.subscale_weights = {}

        for subscale, items in self.subscale_mapping.items():
            valid_subscale_items = [item for item in items if item in self.valid_items]
            if valid_subscale_items:
                self.subscale_weights[subscale] = np.mean([self.item_weights[item] for item in valid_subscale_items])
            else:
                self.subscale_weights[subscale] = 0.0

        # 计算模块可控性（如果需要）
        if louvain is not None and calculate_module_controllability and self.communities is not None:
            # 仅在内部计算模块可控性，不输出详细信息
            self.module_controllability = self._calculate_module_controllability()

        # 转换为Series
        self.item_weights = pd.Series(self.item_weights)
        self.subscale_weights = pd.Series(self.subscale_weights)

        end_time = datetime.datetime.now()
        print(f"权重计算完成，耗时: {(end_time - start_time).total_seconds():.2f}秒")
        return self.item_weights, self.subscale_weights

    def _construct_network(self, data, columns):
        """
        使用Graphical Lasso构建网络
        
        Parameters
        ----------
        data : pandas.DataFrame
            数据
        columns : list
            列名
            
        Returns
        -------
        networkx.Graph
            网络图
        """
        # 计算相关系数矩阵
        corr_matrix = data.corr()

        # 使用Graphical Lasso估计稀疏逆协方差矩阵
        print("使用Graphical Lasso估计稀疏逆协方差矩阵...")
        if GRAPHICAL_LASSO_CV:
            model = GraphicalLassoCV(cv=5)
        else:
            model = GraphicalLasso(alpha=GRAPHICAL_LASSO_ALPHA, random_state=RANDOM_SEED)
        model.fit(data)

        # 获取精度矩阵（逆协方差矩阵）
        precision_matrix = pd.DataFrame(model.precision_, index=columns, columns=columns)
        
        # 将精度矩阵转换为偏相关矩阵
        diag_sqrt = np.sqrt(np.diag(model.precision_))
        partial_corr_matrix = -model.precision_ / np.outer(diag_sqrt, diag_sqrt)
        np.fill_diagonal(partial_corr_matrix, 1)
        
        # 预处理矩阵
        processed_matrix = self._matrix_preprocess(partial_corr_matrix)
        
        # 构建条件依赖网络图
        print("构建条件依赖网络图...")
        # 使用带有有效项目标签的图
        graph = nx.Graph()
        for i, item1 in enumerate(columns):
            graph.add_node(item1)
            for j, item2 in enumerate(columns):
                if i < j and processed_matrix[i, j] != 0:
                    graph.add_edge(item1, item2, weight=processed_matrix[i, j])
                    
        return graph

    def _matrix_preprocess(self, matrix):
        """
        预处理矩阵：去对角线和取绝对值
        
        Parameters
        ----------
        matrix : numpy.ndarray
            输入矩阵
            
        Returns
        -------
        numpy.ndarray
            处理后的矩阵
        """
        number_of_nodes = matrix.shape[1]
        matrix_result = matrix.copy()
        # 去对角线
        for i in range(0, number_of_nodes):
            matrix_result[i, i] = 0
        # 取绝对值
        matrix_result = abs(matrix_result)
        return matrix_result

    def _greedy_minimum_dominating_set(self, graph, times):
        """
        使用贪心算法近似计算最小支配集
        
        Parameters
        ----------
        graph : networkx.Graph
            网络图
        times : int
            重复计算次数
            
        Returns
        -------
        list
            所有最小支配集
        """
        min_dominating_set = []

        for time in range(times):
            graph_copy = graph.copy()
            dominating_set = []

            while graph_copy.nodes():
                # 随机选择一个节点
                node = random.choice(list(graph_copy.nodes()))
                dominating_set.append(node)
                
                # 移除该节点及其邻居
                remove_list = [node]
                for neighbor in graph_copy.neighbors(node):
                    remove_list.append(neighbor)

                for node_to_remove in remove_list:
                    if graph_copy.has_node(node_to_remove):
                        graph_copy.remove_node(node_to_remove)

            dominating_set = set(dominating_set)
            
            # 更新最小支配集列表
            if len(min_dominating_set) == 0:
                min_dominating_set.append(dominating_set)
            elif len(min_dominating_set[0]) == len(dominating_set) and dominating_set not in min_dominating_set:
                min_dominating_set.append(dominating_set)
            elif len(min_dominating_set[0]) > len(dominating_set):
                min_dominating_set.clear()
                min_dominating_set.append(dominating_set)
                
            if (time + 1) % 100 == 0:
                print(f"已完成 {time + 1} 次迭代，当前最小支配集大小: {len(min_dominating_set[0])}, 数量: {len(min_dominating_set)}")

        return min_dominating_set

    def _dominating_frequency(self, all_dom_sets, graph):
        """
        计算节点在最小支配集中出现的频率
        
        Parameters
        ----------
        all_dom_sets : list
            所有最小支配集
        graph : networkx.Graph
            网络图
            
        Returns
        -------
        dict
            节点权重字典
        """
        num_dom_set = len(all_dom_sets)
        
        # 初始化频率计数
        node_list = list(graph.nodes())
        as_dom_node_count = {node: 0 for node in node_list}
        
        # 统计每个节点出现在最小支配集中的次数
        for min_dom_set in all_dom_sets:
            for dom_node in min_dom_set:
                as_dom_node_count[dom_node] = as_dom_node_count[dom_node] + 1

        # 计算频率
        for node in as_dom_node_count:
            as_dom_node_count[node] = as_dom_node_count[node] / num_dom_set
            
        return as_dom_node_count
        
    def _calculate_module_controllability(self):
        """
        计算模块可控性
        
        Returns
        -------
        dict
            模块可控性字典
        """
        if louvain is None or self.communities is None:
            print("警告: 社区检测结果不可用，无法计算模块可控性")
            return {}
            
        # 获取社区数量
        number_of_communities = max(self.communities.values()) + 1
        
        # 按社区分组节点
        module = {i: [] for i in range(number_of_communities)}
        for node, community_index in self.communities.items():
            module[community_index].append(node)
        
        # 初始化结果字典
        average_module_controllability_result = {f"{source}_{target}": 0 
                                                for source in module 
                                                for target in module}
        
        # 对每个最小支配集计算模块可控性
        for min_dom_set in self.all_dom_sets:
            # 计算每个支配节点的控制区域（该节点及其邻居）
            dominated_area = {dom_node: set(self.graph.neighbors(dom_node)).union({dom_node}) 
                             for dom_node in min_dom_set}
            
            # 计算每个模块的控制区域（模块内所有支配节点的控制区域的并集）
            modules_control_area = {}
            for module_index, node_in_module in module.items():
                # 获取该模块内的支配节点
                dom_nodes_in_module = [node for node in node_in_module if node in min_dom_set]
                
                # 如果该模块内有支配节点，计算控制区域
                if dom_nodes_in_module:
                    control_areas = [dominated_area[node] for node in dom_nodes_in_module]
                    modules_control_area[module_index] = set().union(*control_areas)
                else:
                    modules_control_area[module_index] = set()
            
            # 计算模块间的控制能力
            temp_module_controllability_result = {}
            for module_source, control_area in modules_control_area.items():
                for module_target, target_module_area in module.items():
                    # 计算源模块对目标模块的控制能力
                    intersection = control_area.intersection(set(target_module_area))
                    controllability = len(intersection) / len(target_module_area) if target_module_area else 0
                    temp_module_controllability_result[f"{module_source}_{module_target}"] = controllability
                    
                    # 累加到平均结果
                    average_module_controllability_result[f"{module_source}_{module_target}"] += controllability
        
        # 计算平均模块可控性
        for key in average_module_controllability_result:
            average_module_controllability_result[key] /= len(self.all_dom_sets)
            
        return average_module_controllability_result
        
    def get_community_membership(self):
        """
        获取节点所属的社区
        
        Returns
        -------
        dict
            节点到社区的映射
        """
        return self.communities
        
    def get_module_controllability(self):
        """
        获取模块可控性矩阵
        
        Returns
        -------
        dict
            模块可控性矩阵
        """
        return self.module_controllability
