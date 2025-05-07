#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目级预测与子量表聚合模块
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from config import RANDOM_SEED, SUBSCALES, ITEM_PREDICTOR_MODEL, CV_FOLDS, OOF_FOLDS

class ItemPredictor:
    """
    项目级预测类
    """

    def __init__(self, X_train, Y_train, X_val=None, Y_val=None, X_test=None, model_type=ITEM_PREDICTOR_MODEL):
        """
        初始化项目级预测器

        Parameters
        ----------
        X_train : pandas.DataFrame
            训练集特征矩阵
        Y_train : pandas.DataFrame
            训练集项目得分矩阵
        X_val : pandas.DataFrame, optional
            验证集特征矩阵
        Y_val : pandas.DataFrame, optional
            验证集项目得分矩阵
        X_test : pandas.DataFrame, optional
            测试集特征矩阵
        model_type : str
            模型类型，'rf', 'svr', 或 'xgb'
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.model_type = model_type
        self.models = {}

    def train_models(self, items=None):
        """
        训练项目级模型

        Parameters
        ----------
        items : list, optional
            要训练的项目列表，如果为None则训练所有项目

        Returns
        -------
        dict
            训练好的模型
        """
        print(f"训练项目级模型 (模型类型: {self.model_type})...")

        # 如果未指定项目，则使用所有有效项目
        if items is None:
            items = []
            for subscale_items in SUBSCALES.values():
                items.extend([item for item in subscale_items if item in self.Y_train.columns])

        # 训练每个项目的模型
        for item in items:
            print(f"训练{item}模型...")

            # 获取目标变量
            y_train = self.Y_train[item]

            # 初始化最佳模型和最佳得分
            best_model = None
            best_score = float('inf')
            best_model_type = None

            # 尝试每种模型类型
            models = [self.model_type] if isinstance(self.model_type, str) else self.model_type

            for model_type in models:
                # 创建基础模型
                if model_type == 'rf':
                    base_model = RandomForestRegressor(random_state=RANDOM_SEED)
                    param_grid = {
                        'n_estimators': [50, 100],
                        'max_depth': [5, 10]
                    }
                elif model_type == 'svr':
                    base_model = SVR()
                    param_grid = {
                        'C': [0.1, 1.0, 10.0],
                        'gamma': ['scale', 'auto']
                    }
                elif model_type == 'xgb':
                    base_model = XGBRegressor(random_state=RANDOM_SEED)
                    param_grid = {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5]
                    }
                else:
                    raise ValueError(f"不支持的模型类型: {model_type}")

                # 创建网格搜索
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=CV_FOLDS,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )

                # 执行网格搜索
                grid_search.fit(self.X_train, y_train)

                # 更新最佳模型
                if -grid_search.best_score_ < best_score:
                    best_score = -grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    best_model_type = model_type

            # 保存最佳模型
            self.models[item] = {
                'model': best_model,
                'model_type': best_model_type,
                'score': best_score
            }

        print(f"项目级模型训练完成，共训练了{len(self.models)}个模型")
        return self.models

    def generate_oof_predictions(self, items=None):
        """
        生成训练集的Out-of-Fold预测

        Parameters
        ----------
        items : list, optional
            要生成预测的项目列表，如果为None则使用所有已训练的项目

        Returns
        -------
        pandas.DataFrame
            OOF预测矩阵
        """
        print(f"生成训练集OOF预测 (折数: {OOF_FOLDS})...")

        # 如果未指定项目，则使用所有已训练的项目
        if items is None:
            items = list(self.models.keys())

        # 初始化OOF预测矩阵
        Y_oof = pd.DataFrame(index=self.X_train.index, columns=items)

        # 为每个项目生成OOF预测
        for item in items:
            print(f"生成{item}的OOF预测...")

            # 获取目标变量
            y_train = self.Y_train[item]

            # 创建K折交叉验证
            kf = KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=RANDOM_SEED)

            # 获取最佳模型类型
            model_type = self.models[item]['model_type']

            # 为每一折生成预测
            for train_idx, val_idx in kf.split(self.X_train):
                # 划分数据
                X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]

                # 创建模型
                if model_type == 'rf':
                    model = RandomForestRegressor(**{k: v for k, v in self.models[item]['model'].get_params().items()
                                                   if k in ['n_estimators', 'max_depth', 'random_state']})
                elif model_type == 'svr':
                    model = SVR(**{k: v for k, v in self.models[item]['model'].get_params().items()
                                 if k in ['C', 'gamma', 'kernel']})
                elif model_type == 'xgb':
                    model = XGBRegressor(**{k: v for k, v in self.models[item]['model'].get_params().items()
                                          if k in ['n_estimators', 'learning_rate', 'max_depth', 'random_state']})

                # 训练模型
                model.fit(X_fold_train, y_fold_train)

                # 生成预测
                Y_oof.loc[X_fold_val.index, item] = model.predict(X_fold_val)

        print("训练集OOF预测生成完成")
        return Y_oof

    def predict(self, X=None, items=None):
        """
        生成预测

        Parameters
        ----------
        X : pandas.DataFrame, optional
            特征矩阵，如果为None则使用测试集
        items : list, optional
            要生成预测的项目列表，如果为None则使用所有已训练的项目

        Returns
        -------
        pandas.DataFrame
            预测矩阵
        """
        # 如果未指定特征矩阵，则使用测试集
        if X is None:
            if self.X_test is None:
                raise ValueError("未指定特征矩阵，且测试集为None")
            X = self.X_test

        # 如果未指定项目，则使用所有已训练的项目
        if items is None:
            items = list(self.models.keys())

        print(f"生成预测 (样本数: {len(X)}, 项目数: {len(items)})...")

        # 初始化预测矩阵
        Y_pred = pd.DataFrame(index=X.index, columns=items)

        # 为每个项目生成预测
        for item in items:
            # 使用训练好的模型生成预测
            Y_pred[item] = self.models[item]['model'].predict(X)

        print("预测生成完成")
        return Y_pred

    def aggregate_subscales(self, Y_pred, subscale_mapping=SUBSCALES):
        """
        聚合子量表预测

        Parameters
        ----------
        Y_pred : pandas.DataFrame
            项目预测矩阵
        subscale_mapping : dict
            子量表映射

        Returns
        -------
        pandas.DataFrame
            子量表预测矩阵
        """
        print("聚合子量表预测...")

        # 初始化子量表预测矩阵
        Y_sub_pred = pd.DataFrame(index=Y_pred.index)

        # 聚合每个子量表的预测
        for subscale, items in subscale_mapping.items():
            # 确保所有项目都在预测矩阵中
            valid_items = [item for item in items if item in Y_pred.columns]
            if len(valid_items) != len(items):
                missing_items = set(items) - set(valid_items)
                print(f"警告: 子量表 {subscale} 缺少以下项目的预测: {missing_items}")

            # 聚合子量表预测
            if valid_items:
                Y_sub_pred[subscale] = Y_pred[valid_items].sum(axis=1)
            else:
                Y_sub_pred[subscale] = 0.0

        print("子量表预测聚合完成")
        return Y_sub_pred
