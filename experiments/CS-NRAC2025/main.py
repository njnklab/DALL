# main.py
"""
DALL框架主程序入口：
1. 数据读取 & 合并
2. 模型控制权重 (MCW) - 计算项目重要性权重
3. 注意力增强特征-项目协作分解 (A-FICD) - 学习特征与项目的关系
4. 多目标强化学习 (MORL) - 优化项目子集选择
5. 模型评估与结果输出

注意：此版本已优化支持GPU加速
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import time

# 添加utils目录到路径
sys.path.append('../utils/')

# 导入DALL框架组件
from data_preprocessing import load_scale_data, load_feature_data, merge_data
from model_training import DALLTrainer
from item_prediction import ItemPredictor, evaluate_predictions
from model_control_weighting import ModelControlWeighting
from attention_ficd import AttentionFICD
from morl import MORL


def main():
    """
    DALL框架主程序入口
    """
    # 检查GPU可用性
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
    # 1. 配置参数
    config = {
        "logging_level": "INFO",
        "log_file_path": "dall_results.log",
        "scale_csv_path": "/home/user/xuxiao/DALL/dataset/CS-NRAC2025/scales.csv",
        "feature_csv_path": "/home/user/xuxiao/DALL/features/CS-NRAC/features.csv",
        "item_cols": [
            "PHQ1","PHQ2","PHQ3","PHQ4","PHQ5","PHQ6","PHQ7","PHQ8","PHQ9",
            "GAD1","GAD2","GAD3","GAD4","GAD5","GAD6","GAD7",
            "PSS1","PSS2","PSS3","PSS5","PSS6","PSS7","PSS8","PSS9","PSS10",
            "PSS11","PSS12","PSS13","PSS14",
            "ISI1","ISI2","ISI3","ISI4","ISI5","ISI6","ISI7"
        ],
        "target_col": "Depression_Total",
        "model_params": {
            "hidden_dim": 128,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "num_epochs": 100,
            "batch_size": 32,
            "early_stopping": 10,
            "scheduler_patience": 5,  # 添加学习率调度器的耐心参数
            "scheduler_factor": 0.5   # 添加学习率调度器的缩减因子
        },
        "item_predictor_params": {
            "d_model": 128,
            "num_heads": 4,
            "n_latent": 16,
            "morl_w1": 0.7,
            "morl_w2": 0.3,
            "correlation_threshold": 0.1,
            "alpha": 0.5
        },
        "save_model": True,
        "model_dir": "./models",
        "random_state": 42
    }

    # 2. 初始化日志
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, config["logging_level"]))
    
    # 记录GPU信息
    if use_gpu:
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
    else:
        logger.info("未检测到GPU，使用CPU运行")

    # 清空日志文件并设置格式
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    # 移除之前的处理器（如果有的话）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config["logging_level"]))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 创建文件处理器，模式设置为 'w' 以清空日志文件
    file_handler = logging.FileHandler(config["log_file_path"], mode='w')
    file_handler.setLevel(getattr(logging, config["logging_level"]))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("=== 启动DALL框架主程序 ===")

    # 3. 加载数据
    logger.info("加载量表数据...")
    scale_df = load_scale_data(config["scale_csv_path"])

    logger.info("加载特征数据...")
    feature_df = load_feature_data(config["feature_csv_path"])
    
    logger.info("合并数据...")
    df = merge_data(scale_df, feature_df)
    
    # 计算Depression_Total（PHQ-9总分）
    phq_items = [col for col in df.columns if col.startswith('PHQ')]
    df['Depression_Total'] = df[phq_items].sum(axis=1)

    # 检查 df 是否有重复的索引
    duplicate_indices = df.index.duplicated().sum()
    if duplicate_indices > 0:
        logger.warning(f"df 中存在 {duplicate_indices} 个重复的索引。")
        # 根据需求处理重复索引，例如保持第一个出现的记录
        df = df[~df.index.duplicated(keep='first')]
        logger.info(f"去重后 df 的形状: {df.shape}")
    else:
        logger.info("df 的索引唯一。")

    # 若 df 中还没有 target_col，就创建一个
    if config["target_col"] not in df.columns:
        logger.warning(f"df 中没有 {config['target_col']} 列，无法继续。")
        raise ValueError(f"df 中没有 {config['target_col']} 列，无法继续。")

    # 4. 构建 X / Y / target
    item_cols = config["item_cols"]
    exclude_cols = set(item_cols + [config["target_col"]])
    X_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[X_cols].copy()
    Y = df[item_cols].copy()
    total_score = df[config["target_col"]].copy()

    logger.info(f"X shape={X.shape}, Y shape={Y.shape}, total_score shape={total_score.shape}")

    # 5. 划分训练集和测试集
    X_train, X_test, Y_train, Y_test, total_train, total_test = train_test_split(
        X, Y, total_score, test_size=0.2, random_state=config["random_state"]
    )
    
    logger.info(f"训练集: X={X_train.shape}, Y={Y_train.shape}, total={total_train.shape}")
    logger.info(f"测试集: X={X_test.shape}, Y={Y_test.shape}, total={total_test.shape}")

    # 6. DALL框架组件1: 模型控制权重 (MCW)
    logger.info("DALL框架组件1: 模型控制权重 (MCW)")
    mcw = ModelControlWeighting(
        alpha=config["item_predictor_params"]["alpha"],
        correlation_threshold=config["item_predictor_params"]["correlation_threshold"],
        device=device
    )
    item_weights = mcw.compute_weights(X_train, Y_train)
    top_items = mcw.get_top_items(n=5)
    logger.info(f"前5个重要项目: {top_items}")
    
    # 获取特征-项目相关性矩阵
    feature_item_corr = mcw.get_feature_item_correlations(X_train, Y_train)
    logger.info(f"特征-项目相关性矩阵形状: {feature_item_corr.shape}")
    
    # 7. DALL框架组件2: 注意力增强特征-项目协作分解 (A-FICD)
    logger.info("DALL框架组件2: 注意力增强特征-项目协作分解 (A-FICD)")
    ficd = AttentionFICD(
        d_model=config["item_predictor_params"]["d_model"],
        num_heads=config["item_predictor_params"]["num_heads"],
        n_latent=config["item_predictor_params"]["n_latent"],
        device=device,
        item_names=list(Y_train.columns)
    )
    ficd_start_time = time.time()
    ficd.fit(X_train, Y_train, weights=item_weights, epochs=config["model_params"]["num_epochs"], lr=config["model_params"]["learning_rate"])
    ficd_time = time.time() - ficd_start_time
    logger.info(f"A-FICD训练完成，耗时: {ficd_time:.2f}秒")
    
    # 8. DALL框架组件3: 多目标强化学习 (MORL)
    logger.info("DALL框架组件3: 多目标强化学习 (MORL)")
    if ficd.interaction_matrix is not None:
        morl = MORL(
            interaction_matrix=ficd.interaction_matrix,
            item_names=list(Y_train.columns),
            w1=config["item_predictor_params"]["morl_w1"],
            w2=config["item_predictor_params"]["morl_w2"],
            device=device
        )
        morl_start_time = time.time()
        optimal_subset, subset_metrics = morl.select_optimal_subset(
            X=X_train, 
            y=total_train,
            max_episodes=50,
            min_subset_size=3
        )
        morl_time = time.time() - morl_start_time
        logger.info(f"MORL优化完成，耗时: {morl_time:.2f}秒")
        logger.info(f"最优子集大小: {len(optimal_subset)}/{Y_train.shape[1]}")
        logger.info(f"最优子集指标: R²={subset_metrics['r2']:.4f}, 覆盖率={subset_metrics['coverage']:.4f}")
        optimal_items = [Y_train.columns[i] for i in optimal_subset]
        logger.info(f"最优子集项目: {optimal_items}")
    
    # 9. 方法1: 使用DALLTrainer训练双层学习器模型
    logger.info("方法1: 使用DALLTrainer训练双层学习器模型")
    
    # 初始化DALLTrainer
    trainer = DALLTrainer(
        input_dim=X_train.shape[1],
        hidden_dim=config["model_params"]["hidden_dim"],
        num_items=Y_train.shape[1],  # 确保与实际项目数量匹配
        learning_rate=config["model_params"]["learning_rate"],
        weight_decay=config["model_params"]["weight_decay"],
        device=device
    )
    
    # 打印模型参数信息
    logger.info(f"DALLTrainer参数: input_dim={X_train.shape[1]}, hidden_dim={config['model_params']['hidden_dim']}, num_items={Y_train.shape[1]}")
    
    # 训练模型
    start_time = time.time()
    history = trainer.train(
        X=X_train,
        y=Y_train,
        num_epochs=config["model_params"]["num_epochs"],
        batch_size=config["model_params"]["batch_size"],
        early_stopping=config["model_params"]["early_stopping"]
    )
    train_time = time.time() - start_time
    logger.info(f"模型训练完成，耗时: {train_time:.2f}秒")
    
    # 评估模型
    eval_metrics = trainer.evaluate(X_test, Y_test)
    logger.info(f"DALLTrainer测试集评估: {eval_metrics}")
    
    # 保存模型
    if config["save_model"]:
        os.makedirs(config["model_dir"], exist_ok=True)
        trainer.save_model(
            f"{config['model_dir']}/dall_model.pt",
            f"{config['model_dir']}/dall_attention.pt"
        )
        logger.info(f"模型已保存到 {config['model_dir']}")
    
    # 10. 方法2: 使用ItemPredictor进行项目预测（集成三个组件）
    logger.info("方法2: 使用ItemPredictor进行项目预测和优化（集成DALL框架的三个组件）")
    
    # 初始化ItemPredictor
    predictor = ItemPredictor(
        d_model=config["item_predictor_params"]["d_model"],
        num_heads=config["item_predictor_params"]["num_heads"],
        n_latent=config["item_predictor_params"]["n_latent"],
        morl_w1=config["item_predictor_params"]["morl_w1"],
        morl_w2=config["item_predictor_params"]["morl_w2"],
        device="cuda" if use_gpu else "cpu"
    )
    
    # 训练ItemPredictor
    start_time = time.time()
    fit_metrics = predictor.fit(
        X=X_train,
        Y=Y_train,
        total_score=total_train,
        epochs=config["model_params"]["num_epochs"],
        lr=config["model_params"]["learning_rate"],
        correlation_threshold=config["item_predictor_params"]["correlation_threshold"],
        alpha=config["item_predictor_params"]["alpha"],
        optimize_subset=True
    )
    fit_time = time.time() - start_time
    logger.info(f"ItemPredictor训练完成，耗时: {fit_time:.2f}秒")
    
    logger.info(f"ItemPredictor训练指标: {fit_metrics}")
    
    # 获取最优子集和权重
    optimal_subset = predictor.get_optimal_subset(item_names=list(Y.columns))
    item_weights = predictor.get_item_weights()
    
    logger.info(f"最优子集: {optimal_subset}")
    logger.info(f"项目权重: {item_weights}")
    
    # 在测试集上预测
    item_preds = predictor.predict_items(X_test)
    total_pred = predictor.predict_total(X_test)
    
    # 评估预测结果
    item_metrics = evaluate_predictions(Y_test, item_preds)
    total_r2 = r2_score(total_test, total_pred)
    total_mae = mean_absolute_error(total_test, total_pred)
    total_rmse = np.sqrt(mean_squared_error(total_test, total_pred))
    
    logger.info(f"项目预测平均指标: R²={item_metrics['avg_r2']:.4f}, MAE={item_metrics['avg_mae']:.4f}, RMSE={item_metrics['avg_rmse']:.4f}")
    logger.info(f"总分预测指标: R²={total_r2:.4f}, MAE={total_mae:.4f}, RMSE={total_rmse:.4f}")
    
    # 11. 比较两种方法
    logger.info("比较两种方法的性能")
    logger.info(f"方法1 (DALLTrainer) - 测试集R²: {eval_metrics['r2']:.4f}")
    logger.info(f"方法2 (ItemPredictor) - 测试集R²: {total_r2:.4f}")
    
    # 比较两种方法的运行时间
    logger.info(f"方法1 (DALLTrainer) - 训练时间: {train_time:.2f}秒")
    logger.info(f"方法2 (ItemPredictor) - 训练时间: {fit_time:.2f}秒")
    
    # 12. 输出最终结果
    logger.info("=== 最终结果 ===")
    logger.info(f"最优子集大小: {len(optimal_subset)}/{len(item_cols)}")
    logger.info(f"最优子集项目: {optimal_subset}")
    
    # 输出项目权重
    item_weights = predictor.get_item_weights()
    logger.info(f"最终项目权重: {json.dumps(item_weights, indent=2)}")
    
    # 输出预测指标
    logger.info(f"总分预测R²: {total_r2:.4f}")
    logger.info(f"总分预测MAE: {total_mae:.4f}")
    logger.info(f"总分预测RMSE: {total_rmse:.4f}")
    
    # 输出各项目的预测性能
    logger.info(f"各项目预测性能:")
    for item in Y_test.columns:
        r2 = item_metrics[f"{item}_r2"]
        mae = item_metrics[f"{item}_mae"]
        rmse = item_metrics[f"{item}_rmse"]
        logger.info(f"  {item}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    logger.info("=== 完成 ===")


if __name__ == "__main__":
    main()
