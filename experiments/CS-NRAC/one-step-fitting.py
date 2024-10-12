import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel

# 加载特征和量表数据
features_df = pd.read_csv('/home/user/xuxiao/Anxiety/features/NJMU/features.csv')
scale_df = pd.read_csv('/home/user/xuxiao/Anxiety/dataset/NJMU/scales.csv')

# 确保数据通过ID列正确对应
data = pd.merge(features_df, scale_df, on='id')

# 定义特征和标签
X = data.drop(columns=['id', 'GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7', 'PSS2', 'PSS7', 'ISI5'])
y = data[['GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7']].sum(axis=1)

# 预处理特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 配对样本 t 检验筛选特征
selected_features = []
for i in range(X_scaled.shape[1]):
    t_values, p_values = ttest_rel(X_scaled[:, i], y)
    if p_values < 0.01:
        selected_features.append(i)

X_selected = X_scaled[:, selected_features]

# PCA降维
pca = PCA(n_components=0.99)
X_pca = pca.fit_transform(X_selected)

# 定义交叉验证和模型
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 预测和评估
total_predictions = cross_val_predict(model, X_pca, y, cv=kf)
model.fit(X_pca, y)

rmse = np.sqrt(mean_squared_error(y, total_predictions))
mae = mean_absolute_error(y, total_predictions)

print(f'Total Score RMSE: {rmse}')
print(f'Total Score MAE: {mae}')