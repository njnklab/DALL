# model_control_weighting.py
"""
模型控制权重(Model Control Weighting)模块：
- 基于相关结构计算项目权重
- 优化特征表示
- 实现重要性驱动的抑郁项目权重分配
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union
from scipy.stats import pearsonr
import networkx as nx
import torch


class ModelControlWeighting:
    """基于相关结构的模型控制权重计算"""
    
    def __init__(self, alpha: float = 0.5, correlation_threshold: float = 0.1, device: str = "cpu"):
        """
        初始化模型控制权重计算
        
        :param alpha: 自相关和互相关的平衡因子
        :param correlation_threshold: 相关系数阈值，低于该值的相关性将被忽略
        :param device: 计算设备，可以是"cpu"或"cuda"
        """
        self.logger = logging.getLogger(__name__)
        self.alpha = alpha  # 自相关与互相关的平衡因子
        self.correlation_threshold = correlation_threshold
        self.weights = None  # 项目权重字典
        self.correlation_graph = None  # 相关性图
        self.device = torch.device(device)  # 计算设备
        self.logger.info(f"初始化模型控制权重: alpha={alpha}, correlation_threshold={correlation_threshold}, device={device}")
        
    def _compute_self_correlation(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """
        计算特征与各项目的自相关性
        
        :param X: 特征矩阵 [n_samples, n_features]
        :param y: 项目评分矩阵 [n_samples, n_items]
        :return: 每个项目的自相关性得分字典
        """
        self.logger.info("计算特征与项目的自相关性")
        
        item_self_corr = {}
        
        for item in y.columns:
            # 计算每个特征与项目的相关系数
            item_corrs = []
            for feature in X.columns:
                corr, _ = pearsonr(X[feature], y[item])
                if abs(corr) >= self.correlation_threshold:
                    item_corrs.append(abs(corr))  # 取绝对值
            
            # 如果有足够的相关特征，则使用平均值
            if item_corrs:
                item_self_corr[item] = np.mean(item_corrs)
            else:
                item_self_corr[item] = 0.0
                
        return item_self_corr
    
    def _compute_cross_correlation(self, y: pd.DataFrame) -> Dict[str, float]:
        """
        计算项目之间的互相关性
        
        :param y: 项目评分矩阵 [n_samples, n_items]
        :return: 每个项目的互相关性得分字典
        """
        self.logger.info("计算项目间的互相关性")
        
        # 计算项目间的相关系数矩阵
        corr_matrix = y.corr().abs()
        
        # 创建相关性图
        G = nx.Graph()
        
        # 添加所有项目作为节点
        for item in y.columns:
            G.add_node(item)
        
        # 添加高于阈值的相关性作为边
        for i, item1 in enumerate(y.columns):
            for j, item2 in enumerate(y.columns):
                if i < j:  # 避免重复
                    corr_val = corr_matrix.loc[item1, item2]
                    if corr_val >= self.correlation_threshold:
                        G.add_edge(item1, item2, weight=corr_val)
        
        # 计算每个节点的中心性
        centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        self.correlation_graph = G
        
        # 如果任何项目没有与其他项目足够的相关性，设置一个基础值
        for item in y.columns:
            if item not in centrality:
                centrality[item] = 0.1
        
        return centrality
    
    def compute_weights(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """
        计算模型控制权重
        
        :param X: 特征矩阵 [n_samples, n_features]
        :param y: 项目评分矩阵 [n_samples, n_items]
        :return: 项目权重字典
        """
        self.logger.info("计算模型控制权重")
        
        # 计算自相关性
        self_correlation = self._compute_self_correlation(X, y)
        
        # 计算互相关性
        cross_correlation = self._compute_cross_correlation(y)
        
        # 计算综合权重
        weights = {}
        for item in y.columns:
            # 结合自相关性和互相关性
            weights[item] = self.alpha * self_correlation.get(item, 0.0) + \
                           (1 - self.alpha) * cross_correlation.get(item, 0.0)
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for item in weights:
                weights[item] /= total_weight
        else:
            # 如果所有权重都为0，则均等分配
            equal_weight = 1.0 / len(weights)
            for item in weights:
                weights[item] = equal_weight
        
        # 存储结果
        self.weights = weights
        
        # 输出更详细的权重信息
        self.logger.info(f"计算完成，得到 {len(weights)} 个项目权重")
        
        # 输出权重统计信息
        weight_values = list(weights.values())
        self.logger.info(f"权重统计: 最小={min(weight_values):.4f}, 最大={max(weight_values):.4f}, 平均={np.mean(weight_values):.4f}, 中位数={np.median(weight_values):.4f}")
        
        # 输出所有有意义的权重（不是0或特别小的值）
        threshold = 0.01  # 设置一个较小的阈值
        significant_weights = {k: v for k, v in weights.items() if v > threshold}
        self.logger.info(f"有意义的权重项目数量: {len(significant_weights)}/{len(weights)}")
        
        # 输出前5个最高权重项目（仅用于参考）
        top_items = self.get_top_items(n=5)
        self.logger.info(f"前5个最高权重项目（仅供参考）: {top_items}")
        
        return weights
    
    def get_item_importances(self) -> Dict[str, float]:
        """
        获取项目重要性得分
        
        :return: 项目重要性字典（未归一化的原始得分）
        """
        if self.weights is None:
            self.logger.warning("尚未计算权重，返回空字典")
            return {}
        
        return self.weights
    
    def get_top_items(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        获取重要性最高的n个项目
        
        :param n: 返回的项目数量
        :return: (项目名, 权重)元组列表
        """
        if self.weights is None:
            self.logger.warning("尚未计算权重，返回空列表")
            return []
        
        # 按权重排序
        sorted_items = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        
        # 格式化权重值，保留4位小数
        formatted_items = [(item, round(weight, 4)) for item, weight in sorted_items[:n]]
        
        # 返回前n个
        return formatted_items
    
    def get_significant_items(self, threshold: float = 0.01) -> List[Tuple[str, float]]:
        """
        获取所有权重大于阈值的项目
        
        :param threshold: 权重阈值，默认为0.01
        :return: (项目名, 权重)元组列表
        """
        if self.weights is None:
            self.logger.warning("尚未计算权重，返回空列表")
            return []
        
        # 筛选权重大于阈值的项目并按权重排序
        significant_items = [(item, round(weight, 4)) for item, weight in self.weights.items() if weight > threshold]
        return sorted(significant_items, key=lambda x: x[1], reverse=True)
    
    def get_feature_item_correlations(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        获取特征与项目之间的相关性矩阵
        
        :param X: 特征矩阵 [n_samples, n_features]
        :param y: 项目评分矩阵 [n_samples, n_items]
        :return: 特征-项目相关性矩阵
        """
        self.logger.info("计算特征与项目间的相关性矩阵")
        
        # 创建相关性矩阵
        corr_matrix = pd.DataFrame(index=X.columns, columns=y.columns)
        
        # 计算每个特征与每个项目的相关系数
        for feature in X.columns:
            for item in y.columns:
                corr, _ = pearsonr(X[feature], y[item])
                corr_matrix.loc[feature, item] = corr
        
        # 输出相关性矩阵的统计信息
        abs_corr = corr_matrix.abs()
        self.logger.info(f"特征-项目相关性统计: 最小={abs_corr.min().min():.4f}, 最大={abs_corr.max().max():.4f}, 平均={abs_corr.mean().mean():.4f}")
        
        # 计算高相关性特征数量
        high_corr_count = (abs_corr > self.correlation_threshold).sum().sum()
        total_pairs = X.shape[1] * y.shape[1]
        self.logger.info(f"高相关性特征-项目对: {high_corr_count}/{total_pairs} ({high_corr_count/total_pairs*100:.2f}%)")
        
        return corr_matrix
    
    def get_top_features_for_item(self, corr_matrix: pd.DataFrame, 
                                 item: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        获取与特定项目相关性最高的特征
        
        :param corr_matrix: 特征-项目相关性矩阵
        :param item: 项目名称
        :param top_n: 返回的特征数量
        :return: (特征名, 相关性)元组列表
        """
        if item not in corr_matrix.columns:
            self.logger.warning(f"项目 {item} 不在相关性矩阵中")
            return []
        
        # 获取该项目的所有特征相关性
        item_corrs = corr_matrix[item].abs()
        
        # 按绝对相关性排序
        sorted_features = item_corrs.sort_values(ascending=False)
        
        # 返回前top_n个
        top_features = [(feature, sorted_features[feature]) 
                       for feature in sorted_features.index[:top_n]]
        
        return top_features
