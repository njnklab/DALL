# item_prediction.py
"""
项目预测模块：
- 使用注意力增强特征-项目协作分解(A-FICD)模型进行预测
- 优化项目子集选择
- 计算加权总分与个体项目预测
"""

import pandas as pd
import numpy as np
import logging
import torch
import time
import json
from typing import Dict, List, Tuple, Set, Union
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# 导入DALL框架组件
from attention_ficd import AttentionFICD
from morl import MORL
from model_control_weighting import ModelControlWeighting


class ItemPredictor:
    """项目预测器 - DALL框架的统一接口
    
    这个类整合了DALL框架的三个核心组件：
    1. 模型控制权重 (MCW) - 计算项目重要性权重
    2. 注意力增强特征-项目协作分解 (A-FICD) - 学习特征与项目的关系
    3. 多目标强化学习 (MORL) - 优化项目子集选择
    
    通过提供统一的接口，使用户可以方便地使用DALL框架进行项目预测和总分预测。
    """
    
    def __init__(self, d_model: int = 128, num_heads: int = 4, n_latent: int = 16, 
                 morl_w1: float = 0.7, morl_w2: float = 0.3, device: str = "cpu"):
        """
        初始化项目预测器
        
        :param d_model: 特征维度
        :param num_heads: 注意力头数量
        :param n_latent: 潜在因子数量
        :param morl_w1: MORL性能权重
        :param morl_w2: MORL子集大小权重
        :param device: 计算设备
        """
        self.logger = logging.getLogger(__name__)
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_latent = n_latent
        self.morl_w1 = morl_w1
        self.morl_w2 = morl_w2
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # 模型组件
        self.mcw = None  # Model Control Weighting
        self.ficd = None  # Attention-FICD
        self.morl = None  # MORL
        
        # 最优子集和权重
        self.optimal_subset = None
        self.item_weights = None
        
        self.logger.info(f"初始化ItemPredictor: d_model={d_model}, num_heads={num_heads}, "  
                         f"n_latent={n_latent}, device={self.device}")
        self.logger.info(f"MORL参数: w1={morl_w1}, w2={morl_w2}")
    
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, total_score: pd.Series = None,
            epochs: int = 100, lr: float = 0.001, correlation_threshold: float = 0.1,
            alpha: float = 0.5, optimize_subset: bool = True) -> Dict[str, float]:
        """
        训练DALL模型
        
        :param X: 特征矩阵 [n_samples, n_features]
        :param Y: 项目评分矩阵 [n_samples, n_items]
        :param total_score: 总分（可选）
        :param epochs: 训练轮数
        :param lr: 学习率
        :param correlation_threshold: 相关性阈值
        :param alpha: 自相关和互相关的平衡因子
        :param optimize_subset: 是否优化项目子集
        :return: 训练指标字典
        """
        self.logger.info(f"训练DALL模型: samples={X.shape[0]}, features={X.shape[1]}, items={Y.shape[1]}")
        self.logger.info(f"训练参数: epochs={epochs}, lr={lr}, correlation_threshold={correlation_threshold}, alpha={alpha}, optimize_subset={optimize_subset}")
        
        # 1. 计算模型控制权重
        self.logger.info("1. 计算模型控制权重 (MCW)")
        self.mcw = ModelControlWeighting(alpha=alpha, correlation_threshold=correlation_threshold, device=self.device)
        self.item_weights = self.mcw.compute_weights(X, Y)
        
        # 获取特征-项目相关性矩阵，用于后续分析
        self.feature_item_corr = self.mcw.get_feature_item_correlations(X, Y)
        
        # 获取所有有意义的权重项目（权重大于阈值）
        significant_items = self.mcw.get_significant_items(threshold=0.01)
        self.logger.info(f"有意义的权重项目数量: {len(significant_items)}/{len(Y.columns)}")
        self.logger.info(f"有意义的权重项目: {significant_items}")
        
        # 仅供参考，输出前5个项目
        top_items = self.mcw.get_top_items(n=5)
        self.logger.info(f"前5个重要项目(仅供参考): {top_items}")
        
        # 输出权重字典（格式化后）
        formatted_weights = {item: round(weight, 4) for item, weight in self.item_weights.items()}
        self.logger.info(f"MCW计算的项目权重: {json.dumps(formatted_weights, indent=2)}")
        
        # 2. 训练Attention-FICD模型
        self.logger.info("2. 训练注意力增强特征-项目协作分解 (A-FICD)")
        self.ficd = AttentionFICD(
            d_model=self.d_model,
            num_heads=self.num_heads,
            n_latent=self.n_latent,
            device=self.device,
            item_names=list(Y.columns)  # 传递项目名称列表
        )
        ficd_start_time = time.time()
        self.ficd.fit(X, Y, weights=self.item_weights, epochs=epochs, lr=lr)
        ficd_time = time.time() - ficd_start_time
        self.logger.info(f"A-FICD训练完成，耗时: {ficd_time:.2f}秒")
        
        # 输出交互矩阵信息
        if self.ficd.interaction_matrix is not None:
            interaction_shape = self.ficd.interaction_matrix.shape
            self.logger.info(f"交互矩阵形状: {interaction_shape}")
            # 计算交互矩阵的统计信息
            if isinstance(self.ficd.interaction_matrix, torch.Tensor):
                im = self.ficd.interaction_matrix.cpu().numpy()
            else:
                im = self.ficd.interaction_matrix
            self.logger.info(f"交互矩阵统计: 最小={im.min():.4f}, 最大={im.max():.4f}, 平均={im.mean():.4f}, 中位数={np.median(im):.4f}")
        
        # 3. 如果需要，使用MORL优化项目子集
        if optimize_subset and self.ficd.interaction_matrix is not None:
            self.logger.info("3. 使用多目标强化学习 (MORL) 优化项目子集")
            
            # 如果未提供总分，则使用权重计算总分
            if total_score is None:
                self.logger.info("使用权重计算总分")
                total_score = Y.multiply(pd.Series(self.item_weights)).sum(axis=1)
                self.logger.info(f"计算的总分统计: 最小={total_score.min():.4f}, 最大={total_score.max():.4f}, 平均={total_score.mean():.4f}")
            
            # 初始化MORL
            self.morl = MORL(
                interaction_matrix=self.ficd.interaction_matrix,
                item_names=list(Y.columns),
                w1=self.morl_w1,
                w2=self.morl_w2,
                device=self.device
            )
            
            # 选择最优子集
            morl_start_time = time.time()
            self.optimal_subset, subset_metrics = self.morl.select_optimal_subset(
                X=X, 
                y=total_score,
                max_episodes=50,  # 调整为适当的训练轮次
                min_subset_size=3
            )
            morl_time = time.time() - morl_start_time
            self.logger.info(f"MORL优化完成，耗时: {morl_time:.2f}秒")
            
            # 更新权重
            self.item_weights = self.morl.get_weights_from_subset(self.optimal_subset)
            
            # 记录信息
            self.logger.info(f"最优子集大小: {len(self.optimal_subset)}/{Y.shape[1]}")
            self.logger.info(f"最优子集指标: R²={subset_metrics['r2']:.4f}, 覆盖率={subset_metrics['coverage']:.4f}")
            optimal_items = [Y.columns[i] for i in self.optimal_subset]
            self.logger.info(f"最优子集项目: {optimal_items}")
            
            # 输出优化后的权重
            optimized_weights = {item: round(self.item_weights.get(item, 0.0), 4) for item in Y.columns}
            self.logger.info(f"MORL优化后的项目权重: {json.dumps(optimized_weights, indent=2)}")
        else:
            self.logger.info("跳过项目子集优化")
            # 使用所有项目
            self.optimal_subset = set(range(Y.shape[1]))
        
        # 4. 评估训练结果
        # 训练集上的评估
        train_pred = self.predict_items(X)
        
        # 计算项目级别的R²和MSE
        metrics = {}
        for item in Y.columns:
            r2 = r2_score(Y[item], train_pred[item])
            mse = mean_squared_error(Y[item], train_pred[item])
            metrics[f"{item}_r2"] = r2
            metrics[f"{item}_mse"] = mse
        
        # 如果有总分，计算总分的指标
        if total_score is not None:
            total_pred = self.predict_total(X)
            total_r2 = r2_score(total_score, total_pred)
            total_mse = mean_squared_error(total_score, total_pred)
            metrics["total_r2"] = total_r2
            metrics["total_mse"] = total_mse
            self.logger.info(f"总分预测: R²={total_r2:.4f}, MSE={total_mse:.4f}")
        
        # 计算平均指标
        avg_r2 = np.mean([metrics[f"{item}_r2"] for item in Y.columns])
        avg_mse = np.mean([metrics[f"{item}_mse"] for item in Y.columns])
        metrics["avg_r2"] = avg_r2
        metrics["avg_mse"] = avg_mse
        
        self.logger.info(f"训练完成: 平均R²={avg_r2:.4f}, 平均MSE={avg_mse:.4f}")
        return metrics
        
    def predict_items(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        预测所有项目分数
        
        :param X: 特征矩阵 [n_samples, n_features]
        :return: 预测的项目分数 [n_samples, n_items]
        """
        if self.ficd is None:
            self.logger.error("模型尚未训练")
            return pd.DataFrame(index=X.index)
        
        self.logger.info(f"预测项目分数: samples={X.shape[0]}")
        return self.ficd.predict_items(X)
    
    def predict_total(self, X: pd.DataFrame) -> pd.Series:
        """
        预测总分
        
        :param X: 特征矩阵 [n_samples, n_features]
        :return: 预测的总分 [n_samples]
        """
        if self.ficd is None or self.item_weights is None:
            self.logger.error("模型尚未训练")
            return pd.Series(index=X.index)
        
        self.logger.info("预测总分")
        return self.ficd.predict_total(X, self.item_weights)
    
    def get_item_weights(self) -> Dict[str, float]:
        """
        获取项目权重
        
        :return: 项目权重字典
        """
        if self.item_weights is None:
            self.logger.warning("尚未训练模型，返回空字典")
            return {}
            
        # 格式化权重值，保留4位小数
        formatted_weights = {item: round(weight, 4) for item, weight in self.item_weights.items()}
        return formatted_weights
    
    def get_optimal_subset(self, item_names: List[str] = None) -> List[str]:
        """
        获取最优项目子集
        
        :param item_names: 项目名称列表
        :return: 最优子集项目名称列表
        """
        if self.optimal_subset is None:
            self.logger.warning("尚未计算最优子集")
            return []
        
        if item_names is None:
            return list(self.optimal_subset)
        
        # 将索引转换为项目名称
        return [item_names[i] for i in self.optimal_subset]


def evaluate_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
    """
    评估预测结果
    
    :param y_true: 真实项目分数 [n_samples, n_items]
    :param y_pred: 预测项目分数 [n_samples, n_items]
    :return: 评估指标字典
    """
    logger = logging.getLogger(__name__)
    logger.info("评估预测结果")
    
    metrics = {}
    
    # 确保两个数据框包含相同的列
    common_cols = list(set(y_true.columns) & set(y_pred.columns))
    if len(common_cols) != len(y_true.columns):
        logger.warning(f"评估只使用{len(common_cols)}/{len(y_true.columns)}个共同列")
    
    # 计算每个项目的R²、MSE、MAE和RMSE
    for item in common_cols:
        r2 = r2_score(y_true[item], y_pred[item])
        mse = mean_squared_error(y_true[item], y_pred[item])
        mae = mean_absolute_error(y_true[item], y_pred[item])
        rmse = np.sqrt(mse)
        
        metrics[f"{item}_r2"] = r2
        metrics[f"{item}_mse"] = mse
        metrics[f"{item}_mae"] = mae
        metrics[f"{item}_rmse"] = rmse
        
        logger.info(f"  {item}: R²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # 计算平均指标
    avg_r2 = np.mean([metrics[f"{item}_r2"] for item in common_cols])
    avg_mse = np.mean([metrics[f"{item}_mse"] for item in common_cols])
    avg_mae = np.mean([metrics[f"{item}_mae"] for item in common_cols])
    avg_rmse = np.mean([metrics[f"{item}_rmse"] for item in common_cols])
    
    metrics["avg_r2"] = avg_r2
    metrics["avg_mse"] = avg_mse
    metrics["avg_mae"] = avg_mae
    metrics["avg_rmse"] = avg_rmse
    
    # 计算总分
    total_r2 = r2_score(y_true.sum(axis=1), y_pred.sum(axis=1))
    total_mse = mean_squared_error(y_true.sum(axis=1), y_pred.sum(axis=1))
    total_mae = mean_absolute_error(y_true.sum(axis=1), y_pred.sum(axis=1))
    total_rmse = np.sqrt(total_mse)
    
    metrics["total_r2"] = total_r2
    metrics["total_mse"] = total_mse
    metrics["total_mae"] = total_mae
    metrics["total_rmse"] = total_rmse
    
    logger.info(f"总体评估: 平均R²={avg_r2:.4f}, 平均MSE={avg_mse:.4f}, 平均MAE={avg_mae:.4f}, 平均RMSE={avg_rmse:.4f}")
    logger.info(f"总分评估: R²={total_r2:.4f}, MSE={total_mse:.4f}, MAE={total_mae:.4f}, RMSE={total_rmse:.4f}")
    
    return metrics