# morl.py
"""
多目标强化学习(MORL)模块：
- 优化项目子集选择
- 平衡预测性能与子集大小
- 使用PPO算法进行策略学习
"""

import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Set, Optional
from sklearn.metrics import r2_score
import random
import math


class PolicyNetwork(nn.Module):
    """强化学习策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化策略网络
        
        :param state_dim: 状态维度
        :param action_dim: 动作维度
        :param hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        :param x: 输入状态张量
        :return: 动作概率的logits值
        """
        return self.network(x)


class PPO:
    """近端策略优化算法(PPO)"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001, gamma: float = 0.99, 
                 clip_ratio: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01,
                 device: str = "cpu"):
        """
        初始化PPO算法
        
        :param state_dim: 状态维度
        :param action_dim: 动作维度
        :param lr: 学习率
        :param gamma: 折扣因子
        :param clip_ratio: PPO截断比例
        :param value_coef: 价值损失系数
        :param entropy_coef: 熵正则化系数
        :param device: 计算设备
        """
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device
        
        # 创建策略网络和价值网络
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.value = PolicyNetwork(state_dim, 1).to(device)
        
        # 创建优化器
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
    
    def get_action(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        根据状态选择动作
        
        :param state: 状态向量
        :return: 选择的动作和动作概率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy(state_tensor)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            
        return action, probs.squeeze(0).cpu().numpy()
    
    def evaluate(self, states: np.ndarray, actions: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作的log概率、熵和价值
        
        :param states: 状态向量批次
        :param actions: 动作向量批次
        :return: log概率、熵和价值
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        logits = self.policy(states_tensor)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1)
        
        selected_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        values = self.value(states_tensor).squeeze(1)
        
        return selected_log_probs, entropy, values
    
    def update(self, states: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray, 
               rewards: np.ndarray, dones: np.ndarray) -> Tuple[float, float]:
        """
        更新策略和价值网络
        
        :param states: 状态向量批次
        :param actions: 动作向量批次
        :param old_log_probs: 旧策略下的log概率
        :param rewards: 奖励向量批次
        :param dones: 终止标志向量批次
        :return: 策略损失和价值损失
        """
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # 计算目标值和优势函数
        with torch.no_grad():
            values = self.value(states_tensor).squeeze(1)
            returns = torch.zeros_like(rewards_tensor)
            
            # 计算折扣累计回报
            R = 0
            for i in reversed(range(len(rewards_tensor))):
                R = rewards_tensor[i] + self.gamma * R * (1 - dones_tensor[i])
                returns[i] = R
            
            advantages = returns - values
        
        # 更新策略网络
        for _ in range(10):  # 多次更新以提高样本效率
            log_probs, entropy, values_pred = self.evaluate(states, actions)
            
            # 计算比率
            ratios = torch.exp(log_probs - old_log_probs_tensor)
            
            # 计算裁剪后的目标函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 添加熵正则化
            policy_loss = policy_loss - self.entropy_coef * entropy.mean()
            
            # 更新策略
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        # 更新价值网络
        for _ in range(10):
            _, _, values_pred = self.evaluate(states, actions)
            value_loss = F.mse_loss(values_pred, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # 返回损失值供调试使用
        return policy_loss.item(), value_loss.item()


class MORL:
    """多目标强化学习用于项目子集选择
    
    优化版本特点：
    1. 减少搜索空间 - 使用MCW预先筛选重要项目
    2. 并行计算 - 利用GPU并行处理多个项目子集的评估
    3. 早停机制 - 当性能指标连续几次迭代没有显著提升时提前终止搜索
    """
    
    def __init__(self, interaction_matrix: np.ndarray, item_names: List[str], 
                 w1: float = 0.7, w2: float = 0.3, gamma: float = 0.9, 
                 device: str = "cpu", 
                 important_items: List[int] = None, 
                 early_stopping_patience: int = 10):
        """
        初始化MORL模型
        
        :param interaction_matrix: 交互矩阵 C [n_latent, n_items]
        :param item_names: 项目名称列表
        :param w1: 性能增益权重
        :param w2: 子集大小惩罚权重
        :param gamma: 折扣因子
        :param device: 计算设备
        """
        self.logger = logging.getLogger(__name__)
        self.interaction_matrix = interaction_matrix  # [n_latent, n_items]
        self.n_latent = interaction_matrix.shape[0]
        self.n_items = interaction_matrix.shape[1]
        self.item_names = item_names
        self.w1 = w1
        self.w2 = w2
        self.gamma = gamma
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        
        # 如果提供了重要项目列表，则使用它来减少搜索空间
        self.important_items = important_items
        
        # 计算每个项目的贡献
        self.item_contrib = np.linalg.norm(interaction_matrix, axis=0, ord=1)
        
        # 如果有重要项目列表，则只考虑这些项目
        if self.important_items is not None:
            self.search_items = self.important_items
            self.logger.info(f"使用预定义的重要项目列表，共{len(self.search_items)}个项目")
        else:
            # 如果没有提供重要项目列表，则根据贡献度排序
            # 选择贡献度最高的70%的项目，或者至少选择10个项目
            n_search_items = max(10, int(self.n_items * 0.7))
            sorted_indices = np.argsort(-self.item_contrib)  # 降序排序
            self.search_items = sorted_indices[:n_search_items].tolist()
            self.logger.info(f"根据贡献度选择前{len(self.search_items)}个项目进行搜索")
        
        # 更新状态和动作维度
        self.n_search_items = len(self.search_items)
        self.state_dim = self.n_search_items + 2  # 当前子集表示+R^2+特征覆盖率
        self.action_dim = 2 * self.n_search_items  # 添加或移除项目
        
        # 初始化PPO算法
        self.ppo = PPO(
            state_dim=self.state_dim, 
            action_dim=self.action_dim,
            lr=0.001,
            gamma=gamma,
            device=device
        )
        
        self.logger.info(f"初始化MORL: 项目数={self.n_items}, 潜在维度={self.n_latent}")
    
    def _compute_reward(self, r2_old: float, r2_new: float, subset_size_old: int, 
                        subset_size_new: int) -> float:
        """
        计算奖励
        
        :param r2_old: 旧子集的R^2
        :param r2_new: 新子集的R^2
        :param subset_size_old: 旧子集大小
        :param subset_size_new: 新子集大小
        :return: 奖励值
        """
        # 性能增益奖励
        r1 = r2_new - r2_old
        
        # 增加基础奖励，鼓励更高的R^2
        base_reward = 0.2 * r2_new
        
        # 大小变化惩罚，但使用非线性函数使其更平滑
        size_diff = subset_size_new - subset_size_old
        if size_diff > 0:
            # 增加尺寸时的惩罚
            r2 = -self.w2 * (size_diff ** 1.5) / 10
        else:
            # 减少尺寸时的奖励（如果R^2没有显著下降）
            if r1 >= -0.05:  # 允许小幅度下降
                r2 = -self.w2 * size_diff / 5  # 正值奖励
            else:
                r2 = 0  # 如果R^2显著下降，不给予奖励
        
        # 最终奖励
        reward = self.w1 * r1 + r2 + base_reward
        
        self.logger.debug(f"Reward: {reward:.4f} (R2 change: {r1:.4f}, Size effect: {r2:.4f}, Base: {base_reward:.4f})")
        return reward
    
    def _compute_r2_batch(self, subsets: List[Set[int]], X: pd.DataFrame, y: pd.Series) -> List[float]:
        """
        并行计算多个子集的预测性能
        
        :param subsets: 项目索引子集列表
        :param X: 特征矩阵
        :param y: 目标变量
        :return: R^2值列表
        """
        r2_values = []
        
        # 如果可以使用GPU并行计算
        if self.device != "cpu" and torch.cuda.is_available() and len(subsets) > 1:
            # 将特征矩阵转换为张量
            if X.shape[1] != self.interaction_matrix.shape[0]:
                # 创建特征适配层
                feature_dim = X.shape[1]
                latent_dim = self.interaction_matrix.shape[0]
                
                if feature_dim < latent_dim:
                    padding = np.zeros((X.shape[0], latent_dim - feature_dim))
                    X_adapted = np.hstack([X.values, padding])
                else:
                    try:
                        from sklearn.decomposition import TruncatedSVD
                        svd = TruncatedSVD(n_components=latent_dim)
                        X_adapted = svd.fit_transform(X.values)
                    except Exception as e:
                        self.logger.warning(f"SVD降维失败: {str(e)}，使用简单截断")
                        X_adapted = X.values[:, :latent_dim]
                        
                X_tensor = torch.tensor(X_adapted, dtype=torch.float32, device=self.device)
            else:
                X_tensor = torch.tensor(X.values, dtype=torch.float32, device=self.device)
            
            # 并行计算所有子集的预测
            batch_size = min(10, len(subsets))  # 每批处理的子集数量
            
            for i in range(0, len(subsets), batch_size):
                batch_subsets = subsets[i:i+batch_size]
                batch_r2 = []
                
                for subset in batch_subsets:
                    if not subset:
                        batch_r2.append(0.0)
                        continue
                        
                    # 使用子集中的项目进行预测
                    subset_list = list(subset)
                    subset_contrib = self.item_contrib[subset_list]
                    
                    # 创建交互矩阵子集
                    C_tensor = torch.tensor(self.interaction_matrix[:, subset_list], dtype=torch.float32, device=self.device)
                    
                    # 执行矩阵乘法
                    X_pred_tensor = torch.matmul(X_tensor, C_tensor)
                    
                    # 计算加权预测
                    weights = subset_contrib / subset_contrib.sum() if subset_contrib.sum() > 0 else np.ones_like(subset_contrib) / len(subset_contrib)
                    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
                    y_pred_tensor = torch.matmul(X_pred_tensor, weights_tensor)
                    
                    # 移回GPU并计算R^2
                    y_pred = y_pred_tensor.cpu().numpy()
                    r2 = r2_score(y, y_pred)
                    batch_r2.append(max(0, r2))  # 确保R^2非负
                
                r2_values.extend(batch_r2)
        else:
            # 如果不能使用GPU并行计算，则顺序计算
            for subset in subsets:
                r2 = self._compute_r2(subset, X, y)
                r2_values.append(r2)
                
        return r2_values
    
    def _compute_r2(self, subset: Set[int], X: pd.DataFrame, y: pd.Series) -> float:
        """
        计算子集的预测性能
        
        :param subset: 项目索引子集
        :param X: 特征矩阵
        :param y: 目标变量
        :return: R^2值
        """
        if not subset:
            return 0.0
        
        # 使用子集中的项目进行预测
        subset_list = list(subset)
        subset_contrib = self.item_contrib[subset_list]
        
        # 检查特征维度与交互矩阵维度是否匹配
        if X.shape[1] != self.interaction_matrix.shape[0]:
            self.logger.debug(f"特征维度({X.shape[1]})与交互矩阵维度({self.interaction_matrix.shape[0]})不匹配，进行适配")
            # 创建特征适配层 - 将X映射到交互矩阵的维度
            feature_dim = X.shape[1]
            latent_dim = self.interaction_matrix.shape[0]
            
            # 使用更高级的适配方法
            if feature_dim < latent_dim:
                # 如果特征维度小于交互矩阵维度，使用零填充
                padding = np.zeros((X.shape[0], latent_dim - feature_dim))
                X_adapted = np.hstack([X.values, padding])
            else:
                # 如果特征维度大于交互矩阵维度，使用SVD降维
                # 这比简单截断更好，因为它保留了更多的信息
                try:
                    # 尝试使用SVD降维
                    from sklearn.decomposition import TruncatedSVD
                    svd = TruncatedSVD(n_components=latent_dim)
                    X_adapted = svd.fit_transform(X.values)
                    self.logger.debug(f"使用SVD降维，解释方差比例: {svd.explained_variance_ratio_.sum():.4f}")
                except Exception as e:
                    # 如果SVD失败，回退到简单截断
                    self.logger.warning(f"SVD降维失败: {str(e)}，使用简单截断")
                    X_adapted = X.values[:, :latent_dim]
            
            # 检查是否需要使用GPU加速
            if self.device != "cpu" and torch.cuda.is_available():
                # 转换为PyTorch张量并移动到GPU
                X_tensor = torch.tensor(X_adapted, dtype=torch.float32, device=self.device)
                C_tensor = torch.tensor(self.interaction_matrix[:, subset_list], dtype=torch.float32, device=self.device)
                # 执行矩阵乘法
                X_pred_tensor = torch.matmul(X_tensor, C_tensor)
                # 移回CPU并转换为NumPy数组
                X_pred = X_pred_tensor.cpu().numpy()
            else:
                # 在CPU上执行矩阵乘法
                X_pred = X_adapted.dot(self.interaction_matrix[:, subset_list])
        else:
            # 如果维度匹配，则使用原始方法
            # 检查是否需要使用GPU加速
            if self.device != "cpu" and torch.cuda.is_available():
                # 转换为PyTorch张量并移动到GPU
                X_tensor = torch.tensor(X.values, dtype=torch.float32, device=self.device)
                C_tensor = torch.tensor(self.interaction_matrix[:, subset_list], dtype=torch.float32, device=self.device)
                # 执行矩阵乘法
                X_pred_tensor = torch.matmul(X_tensor, C_tensor)
                # 移回CPU并转换为NumPy数组
                X_pred = X_pred_tensor.cpu().numpy()
            else:
                # 在CPU上执行矩阵乘法
                X_pred = X.values.dot(self.interaction_matrix[:, subset_list])
        
        # 计算加权预测
        weights = subset_contrib / subset_contrib.sum() if subset_contrib.sum() > 0 else np.ones_like(subset_contrib) / len(subset_contrib)
        
        # 检查是否需要使用GPU加速
        if self.device != "cpu" and torch.cuda.is_available():
            # 转换为PyTorch张量并移动到GPU
            X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32, device=self.device)
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
            # 执行矩阵乘法
            y_pred_tensor = torch.matmul(X_pred_tensor, weights_tensor)
            # 移回CPU并转换为NumPy数组
            y_pred = y_pred_tensor.cpu().numpy()
        else:
            # 在CPU上执行矩阵乘法
            y_pred = X_pred.dot(weights)
        
        # 计算R^2
        r2 = r2_score(y, y_pred)
        return max(0, r2)  # 确保R^2非负
    
    def _compute_feature_coverage(self, subset: Set[int]) -> float:
        """
        计算特征覆盖率
        
        :param subset: 项目索引子集
        :return: 特征覆盖率（0到1之间）
        """
        if not subset:
            return 0.0
        
        subset_list = list(subset)
        subset_contrib = np.sum(np.abs(self.interaction_matrix[:, subset_list]), axis=1)
        total_contrib = np.sum(np.abs(self.interaction_matrix), axis=1)
        
        # 避免除零错误
        coverage = np.mean(subset_contrib / (total_contrib + 1e-10))
        return coverage
    
    def _get_state(self, subset: Set[int], r2: float, coverage: float) -> np.ndarray:
        """
        获取当前状态表示
        
        :param subset: 当前项目子集
        :param r2: 当前R^2值
        :param coverage: 当前特征覆盖率
        :return: 状态向量
        """
        # 二进制表示子集（只考虑搜索项目）
        subset_vector = np.zeros(self.n_search_items)
        
        # 将全局索引转换为搜索项目的局部索引
        search_items_set = set(self.search_items)
        for idx in subset:
            if idx in search_items_set:
                local_idx = self.search_items.index(idx)
                subset_vector[local_idx] = 1
        
        # 拼接R^2和覆盖率
        state = np.concatenate([subset_vector, [r2, coverage]])
        return state
    
    def _get_pruned_actions(self, subset: Set[int], k: int) -> List[int]:
        """
        获取修剪后的动作空间
        
        :param subset: 当前项目子集
        :param k: 每类动作的最大数量
        :return: 可执行动作的索引列表
        """
        # 只考虑搜索项目列表中的项目
        search_items_set = set(self.search_items)
        
        # 添加动作：考虑贡献最大的k个未选择项目（仅限搜索项目）
        unselected = list(search_items_set - subset)
        add_candidates = sorted(unselected, 
                                key=lambda i: self.item_contrib[i], 
                                reverse=True)[:k]
        
        # 移除动作：考虑贡献最小的k个已选择项目（仅限搜索项目）
        selected = list(subset & search_items_set)
        remove_candidates = sorted(selected, 
                                  key=lambda i: self.item_contrib[i])[:k]
        
        # 将全局索引转换为搜索项目的局部索引
        actions = []
        for idx in add_candidates:
            local_idx = self.search_items.index(idx)
            actions.append(local_idx)  # 添加动作
        for idx in remove_candidates:
            local_idx = self.search_items.index(idx)
            actions.append(local_idx + self.n_search_items)  # 移除动作
        
        return actions
    
    def _execute_action(self, action: int, subset: Set[int]) -> Set[int]:
        """
        执行选择的动作
        
        :param action: 动作索引（局部索引）
        :param subset: 当前项目子集（全局索引）
        :return: 更新后的项目子集（全局索引）
        """
        new_subset = subset.copy()
        
        if action < self.n_search_items:  # 添加项目
            # 将局部索引转换为全局索引
            item_idx = self.search_items[action]
            if item_idx not in new_subset:
                new_subset.add(item_idx)
        else:  # 移除项目
            # 将局部索引转换为全局索引
            local_idx = action - self.n_search_items
            item_idx = self.search_items[local_idx]
            if item_idx in new_subset:
                new_subset.remove(item_idx)
        
        return new_subset
    
    def select_optimal_subset(self, X: pd.DataFrame, y: pd.Series, 
                             max_episodes: int = 50, min_subset_size: int = 3) -> Tuple[Set[int], Dict[str, float]]:
        """
        选择最优项目子集
        
        :param X: 特征矩阵
        :param y: 目标总分
        :param max_episodes: 最大训练轮数
        :param min_subset_size: 最小子集大小
        :return: 最优项目子集索引和对应的指标
        """
        self.logger.info(f"开始选择最优项目子集: episodes={max_episodes}, 搜索项目数={self.n_search_items}")
        
        # 初始状态：包含所有搜索项目
        best_subset = set(self.search_items)
        best_r2 = self._compute_r2(best_subset, X, y)
        best_coverage = self._compute_feature_coverage(best_subset)
        best_metrics = {'r2': best_r2, 'size': len(best_subset), 'coverage': best_coverage}
        
        # 早停相关变量
        no_improvement_count = 0
        best_reward = float('-inf')
        
        # 训练循环
        for episode in range(max_episodes):
            # 检查是否应该早停
            if no_improvement_count >= self.early_stopping_patience:
                self.logger.info(f"连续{self.early_stopping_patience}轮没有改进，提前终止搜索")
                break
                
            # 每轮随机初始化子集（探索多样的起点）
            if episode < max_episodes // 2:
                # 前半部分从全量开始
                subset = set(self.search_items)
            else:
                # 后半部分从随机子集开始
                subset_size = random.randint(min_subset_size, self.n_search_items - 1)
                subset = set(random.sample(self.search_items, subset_size))
            
            # 计算初始状态的指标
            r2 = self._compute_r2(subset, X, y)
            coverage = self._compute_feature_coverage(subset)
            
            # 环境交互
            states, actions, log_probs, rewards, dones = [], [], [], [], []
            episode_best_reward = float('-inf')
            
            for step in range(50):  # 每轮最多50步
                # 获取状态
                state = self._get_state(subset, r2, coverage)
                
                # 获取修剪后的动作空间（提高效率）
                k = max(3, int(math.log(self.n_search_items)))
                valid_actions = self._get_pruned_actions(subset, k)
                
                # 采样动作
                action_idx, action_probs = self.ppo.get_action(state)
                action = valid_actions[action_idx % len(valid_actions)]
                
                # 执行动作
                new_subset = self._execute_action(action, subset)
                
                # 确保子集大小不小于最小值
                if len(new_subset) < min_subset_size:
                    # 如果太小，随机添加项目
                    candidates = list(set(range(self.n_items)) - new_subset)
                    to_add = min_subset_size - len(new_subset)
                    if candidates:
                        add_items = random.sample(candidates, min(to_add, len(candidates)))
                        new_subset.update(add_items)
                
                # 并行计算新状态的指标
                new_r2 = self._compute_r2(new_subset, X, y)
                new_coverage = self._compute_feature_coverage(new_subset)
                
                # 计算奖励
                reward = self._compute_reward(r2, new_r2, len(subset), len(new_subset))
                
                # 记录本轮最佳奖励
                if reward > episode_best_reward:
                    episode_best_reward = reward
                
                # 存储轨迹
                states.append(state)
                actions.append(action_idx)
                log_probs.append(math.log(action_probs[action_idx] + 1e-10))
                rewards.append(reward)
                dones.append(0)  # 非终止状态
                
                # 更新状态
                subset = new_subset
                r2 = new_r2
                coverage = new_coverage
                
                # 更新最优子集
                if r2 > best_r2 and len(subset) >= min_subset_size:
                    best_subset = subset.copy()
                    best_r2 = r2
                    best_coverage = coverage
                    best_metrics = {'r2': r2, 'size': len(subset), 'coverage': coverage}
                    self.logger.info(f"Episode {episode}, Step {step}: 发现更优子集 R²={r2:.4f}, 大小={len(subset)}, 覆盖率={coverage:.4f}")
            
            # 标记最后一步为终止状态
            dones[-1] = 1
            
            # 更新策略
            policy_loss, value_loss = self.ppo.update(
                states=np.array(states),
                actions=np.array(actions),
                old_log_probs=np.array(log_probs),
                rewards=np.array(rewards),
                dones=np.array(dones)
            )
            
            # 检查是否有改进
            if episode_best_reward > best_reward:
                best_reward = episode_best_reward
                no_improvement_count = 0
                self.logger.info(f"Episode {episode}: 发现更好的奖励 {best_reward:.4f}")
            else:
                no_improvement_count += 1
                if episode % 5 == 0:  # 每5轮输出一次进度
                    self.logger.info(f"Episode {episode}: 无改进计数 {no_improvement_count}/{self.early_stopping_patience}")
            
            # 输出训练信息
            if episode % 5 == 0:  # 每5轮输出一次详细信息
                self.logger.info(f"Episode {episode}: Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}, Avg Reward={np.mean(rewards):.4f}")
            
            if episode % 10 == 0:
                self.logger.info(f"Episode {episode}/{max_episodes}: 当前最优子集 R²={best_r2:.4f}, 大小={len(best_subset)}, 覆盖率={best_coverage:.4f}")
        
        # 将索引转换为项目名称
        best_items = {self.item_names[idx] for idx in best_subset}
        self.logger.info(f"最终选择的项目子集: {best_items}")
        self.logger.info(f"最优子集指标: R²={best_r2:.4f}, 大小={len(best_subset)}, 覆盖率={best_coverage:.4f}")
        
        return best_subset, best_metrics
    
    def get_weights_from_subset(self, subset: Set[int]) -> Dict[str, float]:
        """
        从子集生成权重字典
        
        :param subset: 项目索引子集
        :return: 项目权重字典
        """
        # 根据交互矩阵中的贡献计算权重
        weights = {}
        subset_list = list(subset)
        
        # 获取子集项目的贡献
        subset_contrib = self.item_contrib[subset_list]
        total_contrib = subset_contrib.sum()
        
        # 归一化权重
        normalized_weights = subset_contrib / total_contrib if total_contrib > 0 else np.ones_like(subset_contrib) / len(subset_contrib)
        
        # 构建权重字典
        for i, idx in enumerate(subset_list):
            weights[self.item_names[idx]] = normalized_weights[i]
        
        # 为未选择的项目分配零权重
        for i in range(self.n_items):
            if i not in subset:
                weights[self.item_names[i]] = 0.0
        
        return weights
