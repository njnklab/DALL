#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DALL模型配置文件
"""

# 随机种子
RANDOM_SEED = 42

# 数据路径
ACOUSTIC_FEATURES_PATH = './dataset/CS-NRAC-E.csv'
QUESTIONNAIRE_PATH = './dataset/raw_info.csv'

# 子量表映射
SUBSCALES = {
    'PHQ': ['PHQ1', 'PHQ2', 'PHQ3', 'PHQ4', 'PHQ5', 'PHQ6', 'PHQ7', 'PHQ8', 'PHQ9'],
    'GAD': ['GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7'],
    'ISI': ['ISI1', 'ISI2', 'ISI3', 'ISI4', 'ISI5', 'ISI6', 'ISI7'],
    'PSS': ['PSS1', 'PSS2', 'PSS3', 'PSS4', 'PSS5', 'PSS6', 'PSS7', 'PSS8', 'PSS9', 'PSS10', 'PSS11', 'PSS12', 'PSS13', 'PSS14']
}

# 数据集划分
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# 网络分析参数
GRAPHICAL_LASSO_CV = True
GRAPHICAL_LASSO_ALPHA = 0.1
MDS_ITERATIONS = 1000

# 模型参数
ITEM_PREDICTOR_MODEL = 'rf'  # 'rf', 'svr', 'xgb'
CV_FOLDS = 2
OOF_FOLDS = 2

# 残差校正参数
BASE_ESTIMATOR = 'ridge'  # 'ridge', 'lasso', 'elasticnet'
RESIDUAL_CORRECTOR = 'ridge'  # 'ridge', 'lasso', 'elasticnet'

# SCMA参数
PERTURBATION_SIZE = 1.0
