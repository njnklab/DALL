import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel
from sklearn.multioutput import MultiOutputRegressor
import warnings
import joblib
import shap

# 忽略警告
warnings.filterwarnings('ignore')

# 设置全局字体为Arial
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14

# 数据加载
features_df = pd.read_csv('/home/user/xuxiao/MTL4SDD/features/CS-NRAC/features.csv')
scale_df = pd.read_csv('/home/user/xuxiao/MTL4SDD/dataset/CS-NRAC/scale.csv')

# 确保数据通过ID列正确对应
data = pd.merge(features_df, scale_df, on='cust_id')

# 从数据中删除所有量表列
scale_columns = ['ISI1', 'ISI2', 'ISI3', 'ISI4', 'ISI5', 'ISI6', 'ISI7',
                 'GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7',
                 'PHQ1', 'PHQ2', 'PHQ3', 'PHQ4', 'PHQ5', 'PHQ6', 'PHQ7', 'PHQ8', 'PHQ9',
                 'PSS1', 'PSS2', 'PSS3', 'PSS4', 'PSS5', 'PSS6', 'PSS7', 'PSS8', 'PSS9', 'PSS10', 'PSS11', 'PSS12', 'PSS13', 'PSS14']

X = data.drop(columns=['cust_id'] + scale_columns)

# 定义特征和各题目得分（多任务）
y_tasks = data[['PSS12','ISI7','PHQ9','GAD7','PHQ3','PSS3','PSS1','PHQ5','PHQ1','ISI3','GAD6','ISI4','PHQ6','PHQ2','ISI2']]

# 总得分
y_total = data[['PHQ1', 'PHQ2', 'PHQ3', 'PHQ4', 'PHQ5', 'PHQ6', 'PHQ7', 'PHQ8', 'PHQ9']].sum(axis=1)

# 预处理特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 特征选择：配对样本 t 检验
selected_features = []
for i in range(X_scaled.shape[1]):
    p_vals = []
    for j in range(y_tasks.shape[1]):
        _, p_val = ttest_rel(X_scaled[:, i], y_tasks.iloc[:, j])
        p_vals.append(p_val)
    if any(p < 0.01 for p in p_vals):
        selected_features.append(i)

X_selected = X_scaled[:, selected_features]
print(f'Selected {X_selected.shape[1]} features out of {X_scaled.shape[1]}')

# PCA降维
pca = PCA(n_components=0.99, random_state=42)
X_pca = pca.fit_transform(X_selected)
print(f'PCA reduced features to {X_pca.shape[1]} components')

# 定义基础模型
rf = RandomForestRegressor(random_state=42)
svm = SVR()
lr = LinearRegression()

# 定义堆叠模型和提升模型
stacking_rf = StackingRegressor(
    estimators=[('rf', rf)],
    final_estimator=LinearRegression(),
    cv=5,
    passthrough=False
)

stacking_svm = StackingRegressor(
    estimators=[('svm', svm)],
    final_estimator=LinearRegression(),
    cv=5,
    passthrough=False
)

stacking_lr = StackingRegressor(
    estimators=[('lr', lr)],
    final_estimator=LinearRegression(),
    cv=5,
    passthrough=False
)

boosting_rf = GradientBoostingRegressor(random_state=42)
boosting_svm = GradientBoostingRegressor(random_state=42)
boosting_lr = GradientBoostingRegressor(random_state=42)

# 组合模型
combined_models = {
    'RF-Stacking': stacking_rf,
    'RF-Boosting': boosting_rf,
    'SVM-Stacking': stacking_svm,
    'SVM-Boosting': boosting_svm,
    'LR-Stacking': stacking_lr,
    'LR-Boosting': boosting_lr
}

# 定义多任务模型
multi_rf = MultiOutputRegressor(RandomForestRegressor(random_state=42))
multi_svm = MultiOutputRegressor(SVR())
multi_lr = MultiOutputRegressor(LinearRegression())

# 定义堆叠模型
estimators = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('svm', SVR()),
    ('lr', LinearRegression())
]
stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=5,
    passthrough=False
)

# 提升模型
boosting = GradientBoostingRegressor(random_state=42)

# 参数网格
param_grids_multi = {
    'MultiRF': {
        'estimator__n_estimators': [100, 200],
        'estimator__max_depth': [None, 10, 20],
        'estimator__min_samples_split': [2, 5]
    },
    'MultiSVR': {
        'estimator__C': [0.1, 1, 10],
        'estimator__kernel': ['linear', 'rbf']
    },
    'MultiLR': {
        'estimator__fit_intercept': [True, False]
    }
}

param_grids_stack_boost = {
    'StackingRegressor': {
        'final_estimator__fit_intercept': [True, False]
    },
    'GradientBoostingRegressor': {
        'n_estimators': [100],
        'learning_rate': [0.01],
        'max_depth': [5]
    }
}

# 交叉验证设置
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 训练和评估多任务模型
results_tasks = []
multi_models = {
    # 'MultiRF': multi_rf,
    'MultiSVR': multi_svm,
    'MultiLR': multi_lr
}

for name, model in multi_models.items():
    print(f'训练和评估多任务模型: {name}')
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids_multi[name],
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    try:
        # 执行网格搜索
        grid_search.fit(X_pca, y_tasks)
        
        # 检查是否成功完成
        if hasattr(grid_search, 'best_estimator_'):
            best_model = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")
        else:
            print("网格搜索未能找到最佳估计器。使用默认模型。")
            best_model = model
        
        # 使用最佳模型进行预测
        y_pred = best_model.predict(X_pca)
        
        # 计算并打印评估指标
        rmse = np.sqrt(mean_squared_error(y_tasks, y_pred, multioutput='raw_values'))
        mae = mean_absolute_error(y_tasks, y_pred, multioutput='raw_values')
        r2 = r2_score(y_tasks, y_pred, multioutput='raw_values')
        
        print(f"{name} 模型性能:")
        print(f"RMSE: {rmse.mean():.4f}")
        print(f"MAE: {mae.mean():.4f}")
        print(f"R2: {r2.mean():.4f}")
        
    except Exception as e:
        print(f"训练 {name} 模型时发生错误: {str(e)}")
        continue  # 继续下一个模型
    
    # 保存结果
    results_tasks.append({
        'Model': name,
        'RMSE': rmse.mean(),
        'MAE': mae.mean()
    })
    
    # 保存各题目得分的预测结果
    best_model.fit(X_pca, y_tasks)
    data[f'y_pred_{name}'] = best_model.predict(X_pca)

# 展示多任务模型性能
results_tasks_df = pd.DataFrame(results_tasks)
print('\n===== Multi-Task Model Performance =====')
print(results_tasks_df)

# 准备总得分预测的特征
X_total = data[[f'y_pred_{name}' for name in multi_models.keys()]].values

# 训练和评估总得分预测模型
results_total = []
total_models = {
    'StackingRegressor': stacking,
    'GradientBoostingRegressor': boosting
}

for name, model in total_models.items():
    print(f'Training and evaluating total score model: {name}')
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids_stack_boost[name],
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # 交叉验证预测
    y_pred_total = cross_val_predict(grid_search, X_total, y_total, cv=kf, n_jobs=-1)
    
    # 计算RMSE和MAE
    rmse = np.sqrt(mean_squared_error(y_total, y_pred_total))
    mae = mean_absolute_error(y_total, y_pred_total)
    
    # 保存结果
    results_total.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae
    })
    
    # 绘制预测值 vs ���际值散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_total, y_pred_total, alpha=0.5, edgecolors='k', linewidth=0.5)
    plt.plot([y_total.min(), y_total.max()], [y_total.min(), y_total.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Total Scores', fontsize=18, fontweight='bold')
    plt.ylabel('Predicted Total Scores', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'/home/user/xuxiao/MTL4SDD/draw/ML-NoWeight/{name}_TotalScore_plot.jpg', dpi=300, format='jpg')
    plt.close()

# 展示总得分预测模型性能
results_total_df = pd.DataFrame(results_total)
print('\n===== Total Score Model Performance =====')
print(results_total_df)

# 训练和评估组合模型
results_combined = []

for name, model in combined_models.items():
    print(f'训练和评估组合模型: {name}')
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids_stack_boost.get(name, {}),
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # 启用并行处理
        verbose=1
    )
    
    try:
        # 执行网格搜索
        grid_search.fit(X_pca, y_tasks)
        
        # 检查是否成功完成
        if hasattr(grid_search, 'best_estimator_'):
            best_model = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")
        else:
            print("网格搜索未能找到最佳估计器。使用默认模型。")
            best_model = model
        
        # 使用最佳模型进行预测
        y_pred = best_model.predict(X_pca)
        
        # 计算并打印评估指标
        rmse = np.sqrt(mean_squared_error(y_tasks, y_pred, multioutput='raw_values'))
        mae = mean_absolute_error(y_tasks, y_pred, multioutput='raw_values')
        r2 = r2_score(y_tasks, y_pred, multioutput='raw_values')
        
        print(f"{name} 模型性能:")
        print(f"RMSE: {rmse.mean():.4f}")
        print(f"MAE: {mae.mean():.4f}")
        print(f"R2: {r2.mean():.4f}")
        
    except Exception as e:
        print(f"训练 {name} 模型时发生错误: {str(e)}")
        continue  # 继续下一个模型

# 展示组合模型性能
results_combined_df = pd.DataFrame(results_combined)
print('\n===== Combined Model Performance =====')
print(results_combined_df)
