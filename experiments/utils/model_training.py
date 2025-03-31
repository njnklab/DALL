# model_training.py
"""
模型训练相关模块：
- 双层学习器框架实现 (DALL Framework)
- 语音抑郁项目预测模型
- 基于注意力的特征表示学习
- 多特征融合和动态项目优化
"""

import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


class DualLayerModel(nn.Module):
    """双层学习器基础模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1, 
                dropout: float = 0.2):
        """
        初始化双层学习器模型
        
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出维度（项目数量）
        :param dropout: Dropout比例
        """
        super().__init__()
        
        # 第一层：特征表示层
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 第二层：项目预测层
        self.item_layer = nn.Linear(hidden_dim // 2, output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        :param x: 输入特征 [batch_size, input_dim]
        :return: 项目预测 [batch_size, output_dim]
        """
        # 提取特征表示
        features = self.feature_layer(x)
        
        # 进行项目预测
        predictions = self.item_layer(features)
        
        return predictions
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征表示
        
        :param x: 输入特征 [batch_size, input_dim]
        :return: 特征表示 [batch_size, hidden_dim//2]
        """
        return self.feature_layer(x)


class AttentionLayer(nn.Module):
    """注意力层"""
    
    def __init__(self, feature_dim: int, item_dim: int, num_heads: int = 4):
        """
        初始化注意力层
        
        :param feature_dim: 特征维度
        :param item_dim: 项目维度
        :param num_heads: 注意力头数量
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.item_dim = item_dim
        self.num_heads = num_heads
        
        # 注意力模块
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 项目编码器（将项目索引转换为嵌入向量）
        self.item_encoder = nn.Embedding(item_dim, feature_dim)
        
        # 注意力输出映射
        self.output_layer = nn.Linear(feature_dim, item_dim)
    
    def forward(self, features: torch.Tensor, item_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        应用注意力机制
        
        :param features: 特征表示 [batch_size, feature_dim]
        :param item_indices: 项目索引（可选），如果未提供则考虑所有项目 [batch_size]
        :return: 注意力加权项目预测 [batch_size, item_dim]
        """
        batch_size = features.size(0)
        
        # 如果未指定项目索引，考虑所有项目
        if item_indices is None:
            item_indices = torch.arange(self.item_dim, device=features.device)
        
        # 将项目索引转换为1D形状，确保我们得到2D输出
        if item_indices.dim() > 1:
            item_indices = item_indices.reshape(-1)
        
        # 获取项目嵌入
        item_embeddings = self.item_encoder(item_indices)  # [num_items, feature_dim]
        
        # 将特征与项目嵌入进行注意力计算
        # 将特征扩展为所有批次样本
        features_expanded = features.repeat(1, 1)  # [batch_size, feature_dim]
        
        # 应用多头注意力
        attn_output, _ = self.attention(
            query=item_embeddings.unsqueeze(0).expand(batch_size, -1, -1),  # [batch_size, num_items, feature_dim]
            key=features_expanded.unsqueeze(1),  # [batch_size, 1, feature_dim]
            value=features_expanded.unsqueeze(1)   # [batch_size, 1, feature_dim]
        )
        
        # 输出预测 [batch_size, num_items, feature_dim]
        predictions = self.output_layer(attn_output)  # [batch_size, num_items, item_dim]
        
        # 确保输出形状为 [batch_size, item_dim]
        if predictions.dim() == 3:
            predictions = predictions.squeeze(1)  # 如果是 [batch_size, 1, item_dim]
            if predictions.dim() == 3:  # 如果还是3D [batch_size, num_items, item_dim]
                # 取平均值得到2D输出
                predictions = predictions.mean(dim=1)  # [batch_size, item_dim]
        
        return predictions


class DALLTrainer:
    """双层学习器训练器类"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_items: int = 21,
                 learning_rate: float = 0.001, weight_decay: float = 1e-5,
                 device: str = None):
        """
        初始化DALL训练器
        
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏层维度
        :param num_items: 项目数量
        :param learning_rate: 学习率
        :param weight_decay: 权重衰减
        :param device: 训练设备（'cpu'或'cuda'）
        """
        self.logger = logging.getLogger(__name__)
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = DualLayerModel(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=num_items
        ).to(self.device)
        
        # 初始化注意力层
        self.attention_layer = AttentionLayer(
            feature_dim=hidden_dim // 2,
            item_dim=num_items
        ).to(self.device)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.attention_layer.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 初始化训练相关参数
        self.num_items = num_items
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_attention_state = None
    
    def _prepare_data_loader(self, X: pd.DataFrame, y: pd.DataFrame, batch_size: int = 32,
                            is_train: bool = True, test_size: float = 0.2, random_state: int = 42):
        """
        准备数据加载器
        
        :param X: 特征矩阵
        :param y: 项目矩阵
        :param batch_size: 批量大小
        :param is_train: 是否训练集（如果为 True，则拆分为训练集和验证集）
        :param test_size: 测试集占比
        :param random_state: 随机种子
        :return: 训练集加载器，验证集加载器（如果 is_train 为 False，验证集为 None）
        """
        # 转换为 numpy 数组
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.float32)
        
        if is_train:
            # 拆分训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_np, y_np, test_size=test_size, random_state=random_state
            )
            
            # 创建 PyTorch TensorDataset
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            
            # 创建DataLoader
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
            
            return train_loader, val_loader
        else:
            # 创建 PyTorch TensorDataset
            test_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_np, dtype=torch.float32),
                torch.tensor(y_np, dtype=torch.float32)
            )
            
            # 创建DataLoader
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )
            
            return test_loader, None
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, num_epochs: int = 100, batch_size: int = 32,
              val_size: float = 0.2, random_state: int = 42, early_stopping: int = 10,
              item_weights: Optional[np.ndarray] = None) -> Tuple[Dict[str, List[float]], Dict]:
        """
        训练双层学习器模型
        
        :param X: 特征矩阵
        :param y: 项目矩阵
        :param num_epochs: 训练轮数
        :param batch_size: 批量大小
        :param val_size: 验证集占比
        :param random_state: 随机种子
        :param early_stopping: 早停轮数
        :param item_weights: 项目权重（可选，如果不提供则使用相等权重）
        :return: 训练历史和最佳性能指标
        """
        self.logger.info("准备数据加载器...")
        train_loader, val_loader = self._prepare_data_loader(
            X, y, batch_size, is_train=True, test_size=val_size, random_state=random_state
        )
        
        # 设置项目权重
        if item_weights is None:
            # 如果没有提供项目权重，则使用相等权重
            item_weights = np.ones(self.num_items) / self.num_items
        
        # 确保权重维度正确
        assert len(item_weights) == self.num_items, f"项目权重维度错误：{len(item_weights)} vs {self.num_items}"
        
        # 转换为 PyTorch 张量并移动到设备
        item_weights_tensor = torch.tensor(item_weights, dtype=torch.float32, device=self.device)
        
        # 设置损失函数
        mse_loss = nn.MSELoss(reduction='none')
        
        # 记录训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_item_loss': [],
            'val_item_loss': [],
            'train_total_loss': [],
            'val_total_loss': []
        }
        
        # 早停计数器
        no_improve_count = 0
        
        self.logger.info(f"开始训练，共 {num_epochs} 轮...")
        for epoch in range(num_epochs):
            # 训练模式
            self.model.train()
            self.attention_layer.train()
            
            # 记录当前轮的损失
            train_epoch_loss = 0.0
            train_epoch_item_loss = 0.0
            train_epoch_total_loss = 0.0
            train_batches = 0
            
            # 遍历训练集
            for X_batch, y_batch in train_loader:
                # 移动数据到设备
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                
                # 基本项目预测
                y_pred_base = self.model(X_batch)
                
                # 注意力加权项目预测
                features = self.model.get_features(X_batch)
                
                # 创建项目索引，确保与输出维度匹配
                item_indices = torch.arange(self.num_items, device=self.device)
                item_indices = item_indices.expand(X_batch.size(0), -1)  # [batch_size, num_items]
                
                # 使用项目索引调用注意力层
                y_pred_attn = self.attention_layer(features, item_indices)
                
                # 确保维度匹配
                if y_pred_base.shape != y_pred_attn.shape:
                    self.logger.warning(f"维度不匹配: y_pred_base={y_pred_base.shape}, y_pred_attn={y_pred_attn.shape}")
                    # 如果维度不匹配，只使用基本预测
                    y_pred = y_pred_base
                else:
                    # 融合两种预测（简单平均）
                    y_pred = (y_pred_base + y_pred_attn) / 2.0
                
                # 计算项目级别损失（使用项目权重）
                item_loss = mse_loss(y_pred, y_batch)
                weighted_item_loss = (item_loss * item_weights_tensor).mean()
                
                # 计算总分损失
                y_total_pred = y_pred.sum(dim=1)
                y_total_true = y_batch.sum(dim=1)
                total_loss = F.mse_loss(y_total_pred, y_total_true)
                
                # 综合损失
                loss = weighted_item_loss + total_loss
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 记录损失
                train_epoch_loss += loss.item()
                train_epoch_item_loss += weighted_item_loss.item()
                train_epoch_total_loss += total_loss.item()
                train_batches += 1
            
            # 计算平均损失
            train_epoch_loss /= train_batches
            train_epoch_item_loss /= train_batches
            train_epoch_total_loss /= train_batches
            
            # 验证模式
            self.model.eval()
            self.attention_layer.eval()
            
            # 记录验证集损失
            val_epoch_loss = 0.0
            val_epoch_item_loss = 0.0
            val_epoch_total_loss = 0.0
            val_batches = 0
            
            # 禁用梯度计算
            with torch.no_grad():
                # 遍历验证集
                for X_batch, y_batch in val_loader:
                    # 移动数据到设备
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # 基本项目预测
                    y_pred_base = self.model(X_batch)
                    
                    # 注意力加权项目预测
                    features = self.model.get_features(X_batch)
                    y_pred_attn = self.attention_layer(features)
                    
                    # 融合两种预测
                    y_pred = (y_pred_base + y_pred_attn) / 2.0
                    
                    # 计算项目级别损失
                    item_loss = mse_loss(y_pred, y_batch)
                    weighted_item_loss = (item_loss * item_weights_tensor).mean()
                    
                    # 计算总分损失
                    y_total_pred = y_pred.sum(dim=1)
                    y_total_true = y_batch.sum(dim=1)
                    total_loss = F.mse_loss(y_total_pred, y_total_true)
                    
                    # 综合损失
                    loss = weighted_item_loss + total_loss
                    
                    # 记录损失
                    val_epoch_loss += loss.item()
                    val_epoch_item_loss += weighted_item_loss.item()
                    val_epoch_total_loss += total_loss.item()
                    val_batches += 1
            
            # 计算平均验证损失
            val_epoch_loss /= val_batches
            val_epoch_item_loss /= val_batches
            val_epoch_total_loss /= val_batches
            
            # 更新学习率
            self.scheduler.step(val_epoch_loss)
            
            # 记录历史
            history['train_loss'].append(train_epoch_loss)
            history['val_loss'].append(val_epoch_loss)
            history['train_item_loss'].append(train_epoch_item_loss)
            history['val_item_loss'].append(val_epoch_item_loss)
            history['train_total_loss'].append(train_epoch_total_loss)
            history['val_total_loss'].append(val_epoch_total_loss)
            
            # 输出进度
            self.logger.info(
                f"轮次 {epoch+1}/{num_epochs} - "
                f"训练损失: {train_epoch_loss:.4f}, "
                f"验证损失: {val_epoch_loss:.4f}, "
                f"项目损失: {val_epoch_item_loss:.4f}, "
                f"总分损失: {val_epoch_total_loss:.4f}"
            )
            
            # 检查是否为最佳模型
            if val_epoch_loss < self.best_val_loss:
                self.best_val_loss = val_epoch_loss
                self.best_model_state = self.model.state_dict().copy()
                self.best_attention_state = self.attention_layer.state_dict().copy()
                no_improve_count = 0
                self.logger.info(f"发现新的最佳模型，保存状态。")
            else:
                no_improve_count += 1
                self.logger.info(f"未改进: {no_improve_count}/{early_stopping}")
            
            # 早停检查
            if no_improve_count >= early_stopping:
                self.logger.info(f"早停在轮次 {epoch+1}")
                break
        
        # 加载最佳模型
        if self.best_model_state is not None and self.best_attention_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.attention_layer.load_state_dict(self.best_attention_state)
            self.logger.info("已加载最佳模型状态。")
        
        # 计算最佳模型指标
        best_metrics = self.evaluate(X, y, batch_size=batch_size)
        
        return history, best_metrics
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame, batch_size: int = 32) -> Dict[str, float]:
        """
        评估模型性能
        
        :param X: 特征矩阵
        :param y: 项目矩阵
        :param batch_size: 批量大小
        :return: 评估指标字典
        """
        # 准备数据加载器
        test_loader, _ = self._prepare_data_loader(
            X, y, batch_size, is_train=False
        )
        
        # 将模型设置为评估模式
        self.model.eval()
        self.attention_layer.eval()
        
        # 创建预测结果和真实值存储
        all_y_pred = []
        all_y_true = []
        all_y_total_pred = []
        all_y_total_true = []
        
        # 禁用梯度计算
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # 移动数据到设备
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 基本项目预测
                y_pred_base = self.model(X_batch)
                
                # 注意力加权项目预测
                features = self.model.get_features(X_batch)
                y_pred_attn = self.attention_layer(features)
                
                # 融合两种预测
                y_pred = (y_pred_base + y_pred_attn) / 2.0
                
                # 保存预测结果和真实值
                all_y_pred.append(y_pred.cpu().numpy())
                all_y_true.append(y_batch.cpu().numpy())
                
                # 计算总分
                y_total_pred = y_pred.sum(dim=1)
                y_total_true = y_batch.sum(dim=1)
                
                all_y_total_pred.append(y_total_pred.cpu().numpy())
                all_y_total_true.append(y_total_true.cpu().numpy())
        
        # 合并批次结果
        all_y_pred = np.vstack(all_y_pred)
        all_y_true = np.vstack(all_y_true)
        all_y_total_pred = np.concatenate(all_y_total_pred)
        all_y_total_true = np.concatenate(all_y_total_true)
        
        # 计算项目级别指标
        item_mse = mean_squared_error(all_y_true, all_y_pred)
        item_rmse = np.sqrt(item_mse)
        
        # 计算各项目的 R² 值
        item_r2_scores = []
        for j in range(self.num_items):
            item_r2 = r2_score(all_y_true[:, j], all_y_pred[:, j])
            item_r2_scores.append(item_r2)
        
        # 计算平均 R² 值
        mean_item_r2 = np.mean(item_r2_scores)
        
        # 计算总分指标
        total_mse = mean_squared_error(all_y_total_true, all_y_total_pred)
        total_rmse = np.sqrt(total_mse)
        total_r2 = r2_score(all_y_total_true, all_y_total_pred)
        
        # 汇总指标
        metrics = {
            'item_mse': float(item_mse),
            'item_rmse': float(item_rmse),
            'mean_item_r2': float(mean_item_r2),
            'total_mse': float(total_mse),
            'total_rmse': float(total_rmse),
            'total_r2': float(total_r2),
            'item_r2_scores': item_r2_scores
        }
        
        self.logger.info(f"评估指标:\n" + 
                       f"项目RMSE: {item_rmse:.4f}\n" +
                       f"项目平均R²: {mean_item_r2:.4f}\n" +
                       f"总分RMSE: {total_rmse:.4f}\n" +
                       f"总分R²: {total_r2:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用模型进行预测
        
        :param X: 特征矩阵
        :param batch_size: 批量大小
        :return: 项目预测结果和总分预测结果
        """
        # 转换数据类型
        X_np = X.values.astype(np.float32)
        
        # 创建数据集和加载器
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_np, dtype=torch.float32)
        )
        
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        # 设置为评估模式
        self.model.eval()
        self.attention_layer.eval()
        
        # 存储预测结果
        all_item_preds = []
        all_total_preds = []
        
        # 禁用梯度计算
        with torch.no_grad():
            for (X_batch,) in loader:
                # 移动数据到设备
                X_batch = X_batch.to(self.device)
                
                # 基本项目预测
                y_pred_base = self.model(X_batch)
                
                # 注意力加权项目预测
                features = self.model.get_features(X_batch)
                y_pred_attn = self.attention_layer(features)
                
                # 融合两种预测
                y_pred = (y_pred_base + y_pred_attn) / 2.0
                
                # 计算总分预测
                y_total_pred = y_pred.sum(dim=1)
                
                # 收集预测结果
                all_item_preds.append(y_pred.cpu().numpy())
                all_total_preds.append(y_total_pred.cpu().numpy())
        
        # 合并批次结果
        item_predictions = np.vstack(all_item_preds)
        total_predictions = np.concatenate(all_total_preds)
        
        return item_predictions, total_predictions
    
    def save_model(self, model_path: str, attention_path: str):
        """
        保存模型
        
        :param model_path: 基础模型保存路径
        :param attention_path: 注意力模型保存路径
        """
        # 如果存在最佳模型状态，则保存最佳状态
        if self.best_model_state is not None and self.best_attention_state is not None:
            torch.save(self.best_model_state, model_path)
            torch.save(self.best_attention_state, attention_path)
            self.logger.info(f"已保存最佳模型状态至 {model_path} 和 {attention_path}")
        else:
            # 否则保存当前状态
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.attention_layer.state_dict(), attention_path)
            self.logger.info(f"已保存当前模型状态至 {model_path} 和 {attention_path}")
    
    def load_model(self, model_path: str, attention_path: str):
        """
        加载模型
        
        :param model_path: 基础模型路径
        :param attention_path: 注意力模型路径
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.attention_layer.load_state_dict(torch.load(attention_path, map_location=self.device))
        self.logger.info(f"已从 {model_path} 和 {attention_path} 加载模型")