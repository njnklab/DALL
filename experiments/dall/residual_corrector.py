#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
结构化残差校正模块
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from config import RANDOM_SEED, BASE_ESTIMATOR, RESIDUAL_CORRECTOR, CV_FOLDS

class ResidualCorrector:
    """
    结构化残差校正类
    """

    def __init__(self, X_train, Y_sub_oof, y_total_train, subscale_weights, X_test=None, Y_sub_test=None):
        """
        初始化残差校正器

        Parameters
        ----------
        X_train : pandas.DataFrame
            训练集特征矩阵
        Y_sub_oof : pandas.DataFrame
            训练集OOF子量表预测矩阵
        y_total_train : pandas.Series
            训练集真实总分
        subscale_weights : pandas.Series
            子量表权重
        X_test : pandas.DataFrame, optional
            测试集特征矩阵
        Y_sub_test : pandas.DataFrame, optional
            测试集子量表预测矩阵
        """
        self.X_train = X_train
        self.Y_sub_oof = Y_sub_oof
        self.y_total_train = y_total_train
        self.subscale_weights = subscale_weights
        self.X_test = X_test
        self.Y_sub_test = Y_sub_test
        self.base_estimator = None
        self.residual_corrector = None

    def train(self):
        """
        训练残差校正模型

        Returns
        -------
        tuple
            (base_estimator, residual_corrector)
        """
        print("训练残差校正模型...")

        # 训练基估计器
        print("训练基估计器...")
        self.base_estimator = self._train_base_estimator()

        # 生成基预测
        y_total_base_oof = self._predict_base(self.Y_sub_oof)

        # 计算残差
        residuals = self.y_total_train - y_total_base_oof

        # 训练残差校正模型
        print("训练残差校正模型...")
        self.residual_corrector = self._train_residual_corrector(residuals)

        print("残差校正模型训练完成")
        return self.base_estimator, self.residual_corrector

    def _train_base_estimator(self):
        """
        训练基估计器

        Returns
        -------
        object
            训练好的基估计器
        """
        # 构建特征矩阵
        X = pd.DataFrame(index=self.Y_sub_oof.index)
        for subscale in self.Y_sub_oof.columns:
            if subscale in self.subscale_weights.index:
                X[f"weighted_{subscale}"] = self.Y_sub_oof[subscale] * self.subscale_weights[subscale]

        # 创建基估计器
        if BASE_ESTIMATOR == 'ridge':
            base_model = Ridge(random_state=RANDOM_SEED)
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
        elif BASE_ESTIMATOR == 'lasso':
            base_model = Lasso(random_state=RANDOM_SEED)
            param_grid = {'alpha': [0.01, 0.1, 1.0]}
        elif BASE_ESTIMATOR == 'elasticnet':
            base_model = ElasticNet(random_state=RANDOM_SEED)
            param_grid = {
                'alpha': [0.1, 1.0],
                'l1_ratio': [0.2, 0.5, 0.8]
            }
        else:
            raise ValueError(f"不支持的基估计器类型: {BASE_ESTIMATOR}")

        # 网格搜索
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=CV_FOLDS,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X, self.y_total_train)

        return grid_search.best_estimator_

    def _train_residual_corrector(self, residuals):
        """
        训练残差校正模型

        Parameters
        ----------
        residuals : pandas.Series
            残差

        Returns
        -------
        object
            训练好的残差校正模型
        """
        # 构建特征矩阵
        Z_train = pd.DataFrame(index=self.X_train.index)

        # 添加原始特征
        for col in self.X_train.columns:
            Z_train[f"acoustic_{col}"] = self.X_train[col]

        # 添加子量表预测
        for subscale in self.Y_sub_oof.columns:
            Z_train[f"subscale_{subscale}"] = self.Y_sub_oof[subscale]

        # 添加子量表权重
        for subscale in self.subscale_weights.index:
            if subscale in self.Y_sub_oof.columns:
                Z_train[f"weight_{subscale}"] = self.subscale_weights[subscale]

        # 创建残差校正模型
        if RESIDUAL_CORRECTOR == 'ridge':
            corrector_model = Ridge(random_state=RANDOM_SEED)
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
        elif RESIDUAL_CORRECTOR == 'lasso':
            corrector_model = Lasso(random_state=RANDOM_SEED)
            param_grid = {'alpha': [0.01, 0.1, 1.0]}
        elif RESIDUAL_CORRECTOR == 'elasticnet':
            corrector_model = ElasticNet(random_state=RANDOM_SEED)
            param_grid = {
                'alpha': [0.1, 1.0],
                'l1_ratio': [0.2, 0.5, 0.8]
            }
        else:
            raise ValueError(f"不支持的残差校正模型类型: {RESIDUAL_CORRECTOR}")

        # 网格搜索
        grid_search = GridSearchCV(
            corrector_model,
            param_grid,
            cv=CV_FOLDS,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(Z_train, residuals)

        return grid_search.best_estimator_

    def _predict_base(self, Y_sub):
        """
        使用基估计器生成预测

        Parameters
        ----------
        Y_sub : pandas.DataFrame
            子量表预测矩阵

        Returns
        -------
        numpy.ndarray
            基预测
        """
        # 构建特征矩阵
        X = pd.DataFrame(index=Y_sub.index)
        for subscale in Y_sub.columns:
            if subscale in self.subscale_weights.index:
                X[f"weighted_{subscale}"] = Y_sub[subscale] * self.subscale_weights[subscale]

        # 生成预测
        return self.base_estimator.predict(X)

    def _predict_residual(self, X, Y_sub):
        """
        使用残差校正模型生成预测

        Parameters
        ----------
        X : pandas.DataFrame
            特征矩阵
        Y_sub : pandas.DataFrame
            子量表预测矩阵

        Returns
        -------
        numpy.ndarray
            残差预测
        """
        # 构建特征矩阵
        Z = pd.DataFrame(index=X.index)

        # 添加原始特征
        for col in X.columns:
            Z[f"acoustic_{col}"] = X[col]

        # 添加子量表预测
        for subscale in Y_sub.columns:
            Z[f"subscale_{subscale}"] = Y_sub[subscale]

        # 添加子量表权重
        for subscale in self.subscale_weights.index:
            if subscale in Y_sub.columns:
                Z[f"weight_{subscale}"] = self.subscale_weights[subscale]

        # 生成预测
        return self.residual_corrector.predict(Z)

    def predict(self, X=None, Y_sub=None):
        """
        生成总分预测

        Parameters
        ----------
        X : pandas.DataFrame, optional
            特征矩阵，如果为None则使用测试集特征矩阵
        Y_sub : pandas.DataFrame, optional
            子量表预测矩阵，如果为None则使用测试集子量表预测矩阵

        Returns
        -------
        numpy.ndarray
            总分预测
        """
        # 如果未指定特征矩阵和子量表预测矩阵，则使用测试集
        if X is None:
            if self.X_test is None:
                raise ValueError("未指定特征矩阵，且测试集特征矩阵为None")
            X = self.X_test

        if Y_sub is None:
            if self.Y_sub_test is None:
                raise ValueError("未指定子量表预测矩阵，且测试集子量表预测矩阵为None")
            Y_sub = self.Y_sub_test

        print(f"生成总分预测 (样本数: {len(X)})...")

        # 生成基预测
        y_total_base = self._predict_base(Y_sub)

        # 生成残差预测
        residual = self._predict_residual(X, Y_sub)

        # 生成最终预测
        y_total_pred = y_total_base + residual

        print("总分预测生成完成")
        return y_total_pred
