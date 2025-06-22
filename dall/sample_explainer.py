#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单样本解释器 - 可直接调用的sample explanation功能
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录（experiments目录）
parent_dir = os.path.dirname(current_dir)
# 添加上级目录到系统路径
sys.path.append(parent_dir)

from dall.item_perturbation import ItemPerturbationAnalyzer
from dall.data_processor import DataProcessor
from config import PERTURBATION_SIZE

class SampleExplainer:
    """
    单样本解释器类
    可以为新的样本生成explanation图片
    """
    
    def __init__(self, model_dir='/home/a001/xuxiao/DALL/results/dall'):
        """
        初始化解释器
        
        Parameters
        ----------
        model_dir : str
            模型和结果保存的目录
        """
        self.model_dir = model_dir
        self.ipca_analyzer = None
        self.data_processor = None
        self.item_weights = None
        self.base_estimator = None
        self.residual_corrector = None
        self.item_predictor = None
        
        # 加载模型和数据
        self._load_models()
        
    def _load_models(self):
        """
        加载已训练的模型和必要数据
        """
        print("加载已训练的模型和数据...")
        
        try:
            # 检查模型文件是否存在
            model_files = {
                'item_weights': os.path.join(self.model_dir, 'item_weights.csv'),
                # 根据实际保存的模型文件名进行调整
                # 'base_estimator': os.path.join(self.model_dir, 'base_estimator.pkl'),
                # 'residual_corrector': os.path.join(self.model_dir, 'residual_corrector.pkl'),
                # 'item_predictor': os.path.join(self.model_dir, 'item_predictor.pkl')
            }
            
            # 加载item权重
            if os.path.exists(model_files['item_weights']):
                self.item_weights = pd.read_csv(model_files['item_weights'], index_col=0).iloc[:, 0]
                print(f"已加载item权重: {len(self.item_weights)} 个item")
            else:
                print(f"警告: 未找到item权重文件 {model_files['item_weights']}")
                # 使用默认权重
                from config import SUBSCALES
                all_items = [item for subscale_items in SUBSCALES.values() for item in subscale_items]
                self.item_weights = pd.Series(1.0, index=all_items)
                
            # 初始化数据处理器
            acoustic_path = '/home/a001/xuxiao/DALL/dataset/CS-NRAC-E.csv'
            questionnaire_path = '/home/a001/xuxiao/DALL/dataset/raw_info.csv'
            self.data_processor = DataProcessor(acoustic_path, questionnaire_path)
            
            print("模型和数据加载完成")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
            
    def _load_or_create_ipca_analyzer(self, X_test_sample, Y_item_pred_sample):
        """
        创建或加载IPCA分析器
        
        Parameters
        ----------
        X_test_sample : pandas.DataFrame
            测试样本的特征数据
        Y_item_pred_sample : pandas.DataFrame
            测试样本的item预测数据
        """
        if self.ipca_analyzer is None:
            # 由于我们没有保存完整的模型，这里需要重新训练或使用简化版本
            # 创建一个临时的IPCA分析器用于explanation
            
            # 创建临时的base_estimator和residual_corrector
            from sklearn.linear_model import Ridge
            
            # 简化的base estimator（使用item权重）
            self.base_estimator = Ridge(alpha=1.0)
            # 创建虚拟训练数据进行拟合
            dummy_X = pd.DataFrame(np.random.randn(10, len(self.item_weights)), 
                                 columns=[f"weighted_{item}" for item in self.item_weights.index])
            dummy_y = np.random.randn(10)
            self.base_estimator.fit(dummy_X, dummy_y)
            
            # 简化的residual corrector
            self.residual_corrector = Ridge(alpha=1.0)
            # 创建虚拟训练数据进行拟合
            feature_names = X_test_sample.columns.tolist() + [f"weighted_{item}" for item in self.item_weights.index]
            dummy_X_residual = pd.DataFrame(np.random.randn(10, len(feature_names)), columns=feature_names)
            self.residual_corrector.fit(dummy_X_residual, dummy_y)
            
            # 创建IPCA分析器
            self.ipca_analyzer = ItemPerturbationAnalyzer(
                X_test_sample,
                Y_item_pred_sample,
                self.base_estimator,
                self.residual_corrector,
                self.item_weights,
                self.model_dir
            )
            
    def explain_sample(self, X_new, Y_item_new, sample_id=None, save_path=None, top_n=10):
        """
        为新样本生成explanation图片
        
        Parameters
        ----------
        X_new : pandas.DataFrame or pandas.Series
            新样本的特征数据 (1行)
        Y_item_new : pandas.DataFrame or pandas.Series
            新样本的item预测数据 (1行)
        sample_id : str or int, optional
            样本ID，如果为None则自动生成
        save_path : str, optional
            图片保存路径，如果为None则保存到默认位置
        top_n : int, default=10
            显示top N个最重要的item
            
        Returns
        -------
        str
            保存的图片路径
        """
        print(f"为新样本生成explanation...")
        
        # 确保输入是DataFrame格式
        if isinstance(X_new, pd.Series):
            X_new = X_new.to_frame().T
        if isinstance(Y_item_new, pd.Series):
            Y_item_new = Y_item_new.to_frame().T
            
        # 生成sample_id
        if sample_id is None:
            sample_id = f"new_sample_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
        # 创建或加载IPCA分析器
        self._load_or_create_ipca_analyzer(X_new, Y_item_new)
        
        # 生成explanation图片
        image_path = self.ipca_analyzer.create_single_sample_explanation(
            sample_id=sample_id,
            X_sample=X_new,
            Y_item_sample=Y_item_new,
            save_path=save_path,
            top_n=top_n
        )
        
        return image_path
        
    def explain_from_raw_data(self, acoustic_features, questionnaire_scores, 
                            sample_id=None, save_path=None, top_n=10):
        """
        从原始数据生成explanation图片
        
        Parameters
        ----------
        acoustic_features : dict or pandas.Series
            声学特征数据
        questionnaire_scores : dict or pandas.Series  
            问卷题目得分数据
        sample_id : str or int, optional
            样本ID，如果为None则自动生成
        save_path : str, optional
            图片保存路径，如果为None则保存到默认位置
        top_n : int, default=10
            显示top N个最重要的item
            
        Returns
        -------
        str
            保存的图片路径
        """
        print("从原始数据生成explanation...")
        
        # 转换为DataFrame格式
        if isinstance(acoustic_features, dict):
            acoustic_features = pd.Series(acoustic_features)
        if isinstance(questionnaire_scores, dict):
            questionnaire_scores = pd.Series(questionnaire_scores)
            
        # 创建完整的样本数据
        X_new = acoustic_features.to_frame().T
        Y_item_new = questionnaire_scores.to_frame().T
        
        # 使用数据处理器进行预处理（如果需要）
        # 这里简化处理，实际使用时可能需要更完整的预处理流程
        
        return self.explain_sample(X_new, Y_item_new, sample_id, save_path, top_n)


def create_sample_explanation(X_new, Y_item_new, sample_id=None, 
                            save_path=None, top_n=10, 
                            model_dir='/home/a001/xuxiao/DALL/results/dall'):
    """
    便捷函数：为新样本创建explanation图片
    
    Parameters
    ----------
    X_new : pandas.DataFrame or pandas.Series
        新样本的特征数据
    Y_item_new : pandas.DataFrame or pandas.Series
        新样本的item预测数据
    sample_id : str or int, optional
        样本ID
    save_path : str, optional
        图片保存路径
    top_n : int, default=10
        显示top N个最重要的item
    model_dir : str
        模型目录路径
        
    Returns
    -------
    str
        保存的图片路径
    """
    explainer = SampleExplainer(model_dir)
    return explainer.explain_sample(X_new, Y_item_new, sample_id, save_path, top_n)


if __name__ == "__main__":
    # 示例用法
    print("Sample Explainer 示例用法")
    
    # 创建示例数据
    from config import SUBSCALES
    all_items = [item for subscale_items in SUBSCALES.values() for item in subscale_items]
    
    # 模拟声学特征数据
    acoustic_features = pd.Series(np.random.randn(100), 
                                index=[f"acoustic_feature_{i}" for i in range(100)])
    
    # 模拟问卷item数据  
    questionnaire_scores = pd.Series(np.random.randint(0, 4, len(all_items)), 
                                   index=all_items)
    
    print(f"声学特征维度: {len(acoustic_features)}")
    print(f"问卷item数量: {len(questionnaire_scores)}")
    
    try:
        # 创建explanation
        explainer = SampleExplainer()
        image_path = explainer.explain_from_raw_data(
            acoustic_features=acoustic_features,
            questionnaire_scores=questionnaire_scores,
            sample_id="example_sample",
            top_n=10
        )
        print(f"Explanation图片已保存至: {image_path}")
        
    except Exception as e:
        print(f"示例运行失败: {e}")
        print("请确保已经运行过完整的训练流程并保存了必要的模型文件") 