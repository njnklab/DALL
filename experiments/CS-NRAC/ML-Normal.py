import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
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

# 特征选择
X = data.drop(columns=['cust_id', 'ISI1', 'ISI2', 'ISI3', 'ISI4', 'ISI5', 'ISI6', 'ISI7',
                 'GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7',
                 'PHQ1', 'PHQ2', 'PHQ3', 'PHQ4', 'PHQ5', 'PHQ6', 'PHQ7', 'PHQ8', 'PHQ9',
                 'PSS1', 'PSS2', 'PSS3', 'PSS4', 'PSS5', 'PSS6', 'PSS7', 'PSS8', 'PSS9', 'PSS10', 'PSS11', 'PSS12', 'PSS13', 'PSS14'])

# 总得分
y_total = data[['PHQ1', 'PHQ2', 'PHQ3', 'PHQ4', 'PHQ5', 'PHQ6', 'PHQ7', 'PHQ8', 'PHQ9']].sum(axis=1)

# 预处理特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义模型
rf = RandomForestRegressor(random_state=42)
svm = SVR()

# 定义线性回归模型
lr = LinearRegression()

# 交叉验证设置
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化结果存储列表
results = []

# # 评估 RandomForestRegressor
# y_pred_rf = cross_val_predict(rf, X_scaled, y_total, cv=kf, n_jobs=-1)
# rmse_rf = np.sqrt(mean_squared_error(y_total, y_pred_rf))
# mae_rf = mean_absolute_error(y_total, y_pred_rf)
# r2_rf = r2_score(y_total, y_pred_rf)

# # 保存随机森林模型结果
# results.append({
#     'Model': 'RandomForest',
#     'RMSE': rmse_rf,
#     'MAE': mae_rf,
#     'R2': r2_rf
# })

# # 评估 SVR
# y_pred_svm = cross_val_predict(svm, X_scaled, y_total, cv=kf, n_jobs=-1)
# rmse_svm = np.sqrt(mean_squared_error(y_total, y_pred_svm))
# mae_svm = mean_absolute_error(y_total, y_pred_svm)
# r2_svm = r2_score(y_total, y_pred_svm)

# # 保存 SVM 模型结果
# results.append({
#     'Model': 'SVM',
#     'RMSE': rmse_svm,
#     'MAE': mae_svm,
#     'R2': r2_svm
# })

# 评估 LinearRegression
y_pred_lr = cross_val_predict(lr, X_scaled, y_total, cv=kf, n_jobs=-1)
rmse_lr = np.sqrt(mean_squared_error(y_total, y_pred_lr))
mae_lr = mean_absolute_error(y_total, y_pred_lr)
r2_lr = r2_score(y_total, y_pred_lr)

# 保存线性回归模型结果
results.append({
    'Model': 'LinearRegression',
    'RMSE': rmse_lr,
    'MAE': mae_lr,
    'R2': r2_lr
})

# 显示结果
results_df = pd.DataFrame(results)
print(results_df)
