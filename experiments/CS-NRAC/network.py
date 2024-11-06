import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
import community.community_louvain as louvain
import random
import itertools
from scipy import stats
import datetime

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

def greedy_minimum_dominating_set(nxG, times):
    min_dominating_set = []

    for time in range(times):
        nxG_copy = nxG.copy()
        dominating_set = []

        while nxG_copy.nodes():
            node = random.choice(list(nxG_copy.nodes()))
            dominating_set.append(node)
            remove_list = [node] + list(nxG_copy.neighbors(node))

            for node in remove_list:
                nxG_copy.remove_node(node)

        dominating_set = set(dominating_set)
        if len(min_dominating_set) == 0:
            min_dominating_set.append(dominating_set)
        elif len(min_dominating_set[0]) == len(dominating_set) and dominating_set not in min_dominating_set:
            min_dominating_set.append(dominating_set)
        elif len(min_dominating_set[0]) > len(dominating_set):
            min_dominating_set.clear()
            min_dominating_set.append(dominating_set)

        print(f"times: {time + 1} MDSet size: {len(min_dominating_set[0])} MDSet number: {len(min_dominating_set)} MDSet: {min_dominating_set}")

    return min_dominating_set

def calculate_control_frequency(G, mds_sets):
    cf = {node: 0 for node in G.nodes()}

    for mds in mds_sets:
        for node in mds:
            cf[node] += 1

    total_mds_sets = len(mds_sets)
    cf = {node: freq / total_mds_sets for node, freq in cf.items()}  # 归一化CF

    return cf

def module_controllability(nxG, all_dom_set, louvain_communities):
    number_of_communities = max(louvain_communities.values()) + 1
    print(f"module number: {number_of_communities}")

    module = {i: [] for i in range(number_of_communities)}
    for node, community_index in louvain_communities.items():
        module[community_index].append(node)

    for i in range(number_of_communities):
        print(f"module {i} has {len(module[i])} nodes")

    average_module_controllability_result = {f"{source}_{target}": 0 for source in module for target in module}

    for min_dom_set in all_dom_set:
        dominated_area = {dom_node: set(nxG.neighbors(dom_node)).union({dom_node}) for dom_node in min_dom_set}

        modules_control_area = {module_index: set().union(*(dominated_area[node] for node in node_in_module if node in min_dom_set)) 
                                for module_index, node_in_module in module.items()}

        temp_module_controllability_result = {}
        for module_source, control_area in modules_control_area.items():
            for module_target, target_module_area in module.items():
                inter = control_area.intersection(set(target_module_area))
                controllability = len(inter) / len(target_module_area)
                temp_module_controllability_result[f"{module_source}_{module_target}"] = controllability
                average_module_controllability_result[f"{module_source}_{module_target}"] += controllability

        print(f"dom_set: {min_dom_set} module_controllability: {temp_module_controllability_result}")

    average_module_controllability_result = {k: v / len(all_dom_set) for k, v in average_module_controllability_result.items()}
    print(f"average_module_controllability: {average_module_controllability_result}")
    return average_module_controllability_result

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
    mds_sets = greedy_minimum_dominating_set(G, times=1000)

    # 步骤3: 计算控制频率（CF）
    control_frequency = calculate_control_frequency(G, mds_sets)

    # 步骤4: 识别ARM模块的节点及其CF
    arm_cf = identify_arm_nodes(G, control_frequency)

    # 计算模块可控性
    louvain_communities = louvain.best_partition(G)
    average_module_controllability = module_controllability(G, mds_sets, louvain_communities)

    # 输出结果
    print("ARM节点及其CF:")
    for node, cf in arm_cf.items():
        print(f"{node}: {cf:.3f}")

    print("平均模块可控性:")
    for key, value in average_module_controllability.items():
        print(f"{key}: {value:.3f}")

if __name__ == "__main__":
    data = pd.read_csv('/home/user/xuxiao/DALL/dataset/CS-NRAC/scale.csv').drop(columns=['cust_id'])
    main(data)
