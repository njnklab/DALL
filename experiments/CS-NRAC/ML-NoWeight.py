import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel
import warnings
import matplotlib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set global font to Arial
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14

# 1. 数据加载与预处理

# 加载特征和量表数据
features_df = pd.read_csv('/home/user/xuxiao/MTL4SDD/features/CS-NRAC/features.csv')
scale_df = pd.read_csv('/home/user/xuxiao/MTL4SDD/dataset/CS-NRAC/scale.csv')

# 确保数据通过ID列正确对应
data = pd.merge(features_df, scale_df, on='cust_id')

# 定义特征和标签
# 排除不需要的列（包括个别题目得分）
X = data.drop(columns=['cust_id', 'PSS12','ISI7','PHQ9','GAD7','PHQ3','PSS3','PSS1','PHQ5','PHQ1','ISI3','GAD6','ISI4','PHQ6','PHQ2','ISI2'])
y = data[['PHQ1', 'PHQ2', 'PHQ3', 'PHQ4', 'PHQ5', 'PHQ6', 'PHQ7', 'PHQ8', 'PHQ9']].sum(axis=1)

# 预处理特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 特征选择：配对样本 t 检验筛选特征
selected_features = []
for i in range(X_scaled.shape[1]):
    t_stat, p_val = ttest_rel(X_scaled[:, i], y)
    if p_val < 0.01:
        selected_features.append(i)

X_selected = X_scaled[:, selected_features]
print(f'Selected {X_selected.shape[1]} features out of {X_scaled.shape[1]}')

# 3. PCA降维
pca = PCA(n_components=0.99, random_state=42)
X_pca = pca.fit_transform(X_selected)
print(f'PCA reduced features to {X_pca.shape[1]} components')

# 4. 定义模型及参数网格

# 基础模型
rf = RandomForestRegressor(random_state=42)
svm = SVR()
lr = LinearRegression()

# 堆叠模型
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

# 模型列表
models = {
    'Random Forest': rf,
    'Support Vector Machine': svm,
    'Linear Regression': lr,
    'Stacking Regressor': stacking,
    'Gradient Boosting': boosting
}

# 参数网格
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'Support Vector Machine': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'Linear Regression': {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    },
    'Stacking Regressor': {
        'final_estimator__fit_intercept': [True, False],
        'final_estimator__normalize': [True, False]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
}

# 5. 交叉验证设置
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 6. 结果存储
results = []

# 7. 模型训练与评估
for name, model in models.items():
    print(f'Training and evaluating: {name}')
    
    # 获取参数网格
    param_grid = param_grids[name]
    
    # 定义GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # 交叉验证预测
    y_pred = cross_val_predict(grid_search, X_pca, y, cv=kf, n_jobs=-1)
    
    # 计算RMSE和MAE
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # 保存结果
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae
    })
    
    # 8. 预测值 vs 实际值散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Values', fontsize=18, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'/home/user/xuxiao/MTL4SDD/draw/ML-NoWeight/{name.replace(" ", "_")}_plots.jpg', dpi=300, format='jpg')
    plt.close()

# 9. 结果展示
results_df = pd.DataFrame(results)
print('\nModel Performance:')
print(results_df)