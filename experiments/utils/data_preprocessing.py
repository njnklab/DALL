# data_preprocessing.py
"""
数据预处理模块：
- 加载语音抑郁量表数据 (scale)
- 加载语音特征数据 (feature)
- 数据清洗与归一化
- 合并数据并进行预处理
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_scale_data(scale_csv_path: str) -> pd.DataFrame:
    """
    读取语音抑郁量表数据:
    cust_id, PHQ1, PHQ2, ..., PHQ9, GAD1, GAD2, ...
    
    :param scale_csv_path: 量表数据的csv路径
    :return: 量表数据的DataFrame, index=cust_id
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading depression scale data from: {scale_csv_path}")
    
    df = pd.read_csv(scale_csv_path)
    # standard_id or cust_id
    id_col = 'cust_id' if 'cust_id' in df.columns else 'ID'
    df.set_index(id_col, inplace=True)
    
    logger.info(f"Depression scale data shape: {df.shape}")
    return df


def load_feature_data(feature_csv_path: str) -> pd.DataFrame:
    """
    读取语音特征数据:
    cust_id, feature1, feature2, ...
    包括声学特征、韵律特征、频谱特征等
    
    :param feature_csv_path: 特征数据的csv路径
    :return: 特征数据的DataFrame, index=cust_id
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading speech features from: {feature_csv_path}")
    
    df = pd.read_csv(feature_csv_path)
    # ID or cust_id
    id_col = 'cust_id' if 'cust_id' in df.columns else 'ID'
    df.set_index(id_col, inplace=True)
    
    # 检测并移除常量特征
    constant_features = [col for col in df.columns if df[col].nunique() == 1]
    if constant_features:
        logger.warning(f"移除 {len(constant_features)} 个常量特征")
        df.drop(columns=constant_features, inplace=True)
    
    # 标准化语音特征
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df), 
        index=df.index, 
        columns=df.columns
    )
    
    logger.info(f"Speech feature data shape: {df_scaled.shape}")
    return df_scaled


def merge_data(scale_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    将量表数据与语音特征数据(按 ID/index)合并
    
    :param scale_df: 量表数据
    :param feature_df: 语音特征数据
    :return: 合并后的 DataFrame (inner join)，行索引是二者的交集
    """
    logger = logging.getLogger(__name__)
    logger.info("Merging scale_df and feature_df on index (inner join).")
    merged_df = scale_df.join(feature_df, how='inner')

    # 修正列名，去除一些模型不允许的字符
    merged_df.columns = [
        str(c).replace('[', '_')
              .replace(']', '_')
              .replace('<', '_')
              .replace('>', '_')
              .replace('(', '_')
              .replace(')', '_')
              .replace('/', '_')
              .replace('\\', '_')
              .replace('-', '_')
              for c in merged_df.columns
    ]
    
    # 检查缺失值
    missing_values = merged_df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"数据包含 {missing_values.sum()} 个缺失值")
        # 对于量表项，用中位数填充缺失值
        scale_cols = [col for col in merged_df.columns if col.startswith('PHQ') or 
                                                          col.startswith('GAD') or 
                                                          col.startswith('PSS') or 
                                                          col.startswith('ISI')]
        merged_df[scale_cols] = merged_df[scale_cols].fillna(merged_df[scale_cols].median())
        
        # 对于特征，用均值填充缺失值
        feature_cols = [col for col in merged_df.columns if col not in scale_cols]
        merged_df[feature_cols] = merged_df[feature_cols].fillna(merged_df[feature_cols].mean())
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    return merged_df


def compute_total_score(data: pd.DataFrame, item_prefix: str) -> pd.Series:
    """
    计算指定量表项的总分
    
    :param data: 包含量表项的DataFrame
    :param item_prefix: 量表项前缀，如'PHQ'
    :return: 总分的Series
    """
    logger = logging.getLogger(__name__)
    
    # 找出所有匹配前缀的列
    item_cols = [col for col in data.columns if col.startswith(item_prefix)]
    
    if not item_cols:
        logger.warning(f"未找到前缀为 {item_prefix} 的列")
        return pd.Series(index=data.index)
    
    logger.info(f"计算前缀为 {item_prefix} 的总分，使用 {len(item_cols)} 个条目")
    total_score = data[item_cols].sum(axis=1)
    
    return total_score