import pandas as pd
import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
from joblib import Parallel, delayed

# Load feature and scale data
features_df = pd.read_csv('/home/user/xuxiao/Anxiety/features/NJMU/features.csv')
scale_df = pd.read_csv('/home/user/xuxiao/Anxiety/dataset/NJMU/scales.csv')

# Ensure data is correctly merged by ID
data = pd.merge(features_df, scale_df, on='id')

# Define features and labels
X = data.drop(columns=['id', 'GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7'])
y = data[['GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7']]

# Preprocess features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection using t-test
selected_features = []
for i in range(y.shape[1]):
    t_values, p_values = ttest_ind(X_scaled, y.iloc[:, i].values)
    selected_features.append(np.where(p_values < 0.01)[0])
selected_features = np.unique(np.concatenate(selected_features))

X_selected = X_scaled[:, selected_features]
print(X_selected)

# PCA for dimensionality reduction
pca = PCA(n_components=0.99)
X_pca = pca.fit_transform(X_selected)

# Define cross-validation and grid search
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')

# Train sub-models and extract features
def train_and_extract_features(column, y_sub):
    grid_search.fit(X_pca, y_sub)
    best_rf = grid_search.best_estimator_
    
    fold_features = []
    for train_idx, test_idx in kf.split(X_pca):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y_sub.iloc[train_idx], y_sub.iloc[test_idx]
        
        best_rf.fit(X_train, y_train)
        sub_features = best_rf.predict(X_test).reshape(-1, 1)  # Use X_test to match sample size with y_test
        fold_features.append(sub_features)
        
        # Print each sub-model's feature matrix
        print(f'Features for {column} in fold:')
        print(sub_features)
    
    return np.vstack(fold_features)

sub_model_features = Parallel(n_jobs=-1)(
    delayed(train_and_extract_features)(column, y[column]) for column in y.columns
)

# Check if all sub_model_features have the same number of samples
for sub_features in sub_model_features:
    print(sub_features.shape)

# Weighted combination of features
weights = {
    'GAD1': 0.1, 'GAD2': 0.1, 'GAD3': 0.1, 'GAD4': 0.1,
    'GAD5': 0.1, 'GAD6': 0.1, 'GAD7': 0.1  # Replace with actual weights
}
weighted_features = np.zeros(sub_model_features[0].shape)

for sub_features, column in zip(sub_model_features, y.columns):
    weighted_features += weights[column] * sub_features

# Ensure y_total has the correct number of samples
y_total = y.sum(axis=1).iloc[:weighted_features.shape[0]]  # Trim y_total to match the feature shape

# Fit final model to total scores
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(weighted_features, y_total)

# Predict and evaluate
total_predictions = cross_val_predict(final_model, weighted_features, y_total, cv=kf)

rmse = np.sqrt(mean_squared_error(y_total, total_predictions))
mae = mean_absolute_error(y_total, total_predictions)

print(f'Total Score RMSE: {rmse}')
print(f'Total Score MAE: {mae}')