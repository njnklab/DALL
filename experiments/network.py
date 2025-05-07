import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLasso
import random

def construct_network(data, columns):
    # 使用Graphical Lasso构建逆协方差矩阵
    model = GraphicalLasso()
    model.fit(data)
    precision_matrix = model.precision_

    # 构建NetworkX图
    G = nx.Graph()
    n_nodes = precision_matrix.shape[0]

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            weight = precision_matrix[i, j]
            if weight != 0:
                G.add_edge(columns[i], columns[j], weight=abs(weight))  # 使用列名作为节点标签

    return G

def approximate_mds(G, max_samples=1000):
    """
    使用启发式方法近似找到最小支配集合（MDSet）。
    max_samples 控制采样的次数。
    """
    nodes = list(G.nodes())
    mds_sets = []

    for _ in range(max_samples):
        remaining_nodes = set(nodes)
        mds = set()
        while remaining_nodes:
            # 随机选择一个节点作为控制节点
            node = random.choice(list(remaining_nodes))
            mds.add(node)
            # 移除该节点及其邻居节点（因为它们已被支配）
            neighbors = set(G.neighbors(node))
            remaining_nodes -= neighbors
            remaining_nodes.discard(node)

        mds_sets.append(mds)

    return mds_sets

def calculate_control_frequency(G, mds_sets):
    cf = {node: 0 for node in G.nodes()}

    for mds in mds_sets:
        for node in mds:
            cf[node] += 1

    total_mds_sets = len(mds_sets)
    cf = {node: freq / total_mds_sets for node, freq in cf.items()}  # 归一化CF

    return cf

def identify_arm_nodes(G, cf):
    """
    识别与焦虑相关的模块（ARM）中的节点。
    通过分析与GAD相关的节点及其邻居节点。
    """
    gad_nodes = [node for node in G.nodes() if 'gad' in node]  # 假设GAD量表的节点名称包含"GAD"
    arm_nodes = set(gad_nodes)
    
    for node in gad_nodes:
        arm_nodes.update(G.neighbors(node))  # 包含所有与GAD节点直接相连的节点
    
    # 输出ARM节点及其CF
    arm_cf = {node: cf[node] for node in arm_nodes}

    return arm_cf

# 示例主程序
def main(data):
    columns = data.columns
    data_values = data.values
    
    # 步骤1: 构建网络
    G = construct_network(data_values, columns)

    # 步骤2: 近似计算最小支配集合（MDSets）
    mds_sets = approximate_mds(G, max_samples=100000)

    # 步骤3: 计算控制频率（CF）
    control_frequency = calculate_control_frequency(G, mds_sets)

    # 步骤4: 识别ARM模块的节点及其CF
    arm_cf = identify_arm_nodes(G, control_frequency)

    # 输出ARM节点及其CF
    print("ARM节点及其CF:")
    for node, cf in arm_cf.items():
        print(f"{node}: {cf:.3f}")

if __name__ == "__main__":
    data = pd.read_csv('/home/user/xuxiao/Anxiety/dataset/CS-NRAC/scale.csv').drop(columns=['cust_id'])
    main(data)