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

def get_scale_columns(scale_type):
    """
    获取不同量表的列名
    """
    scale_prefixes = {
        'ISI': ['ISI'],
        'GAD': ['GAD'],
        'PHQ': ['PHQ'],
        'PSS': ['PSS'],
        'ALL': ['ISI', 'GAD', 'PHQ', 'PSS']  # 包含所有量表
    }
    return scale_prefixes[scale_type]

def identify_arm_nodes(G, cf, scale_type='ALL'):
    """
    仅在处理所有量表时识别以PHQ为中心的arm nodes
    """
    if scale_type == 'ALL':
        # 只在处理所有量表时，找出PHQ节点及其邻居
        phq_nodes = [node for node in G.nodes() if 'PHQ' in node]
        arm_nodes = set(phq_nodes)
        for node in phq_nodes:
            arm_nodes.update(G.neighbors(node))
        arm_cf = {node: cf[node] for node in arm_nodes}
    else:
        # 对于单个量表，返回所有节点的CF
        arm_cf = cf
    
    return arm_cf

# 示例主程序
def main(data, scale_type):
    columns = data.columns
    data_values = data.values
    
    # 步骤1: 构建网络
    G = construct_network(data_values, columns)

    # 步骤2: 近似计算最小支配集合（MDSets）
    mds_sets = greedy_minimum_dominating_set(G, times=1000)

    # 步骤3: 计算控制频率（CF）
    control_frequency = calculate_control_frequency(G, mds_sets)

    # 步骤4: 根据scale_type决定是否需要识别arm nodes
    arm_cf = identify_arm_nodes(G, control_frequency, scale_type)

    # 计算模块可控性
    louvain_communities = louvain.best_partition(G)
    average_module_controllability = module_controllability(G, mds_sets, louvain_communities)

    # 输出结果
    if scale_type == 'ALL':
        print("PHQ相关节点及其CF:")
    else:
        print(f"{scale_type}量表节点及其CF:")
    for node, cf in arm_cf.items():
        print(f"{node}: {cf:.3f}")

    print("平均模块可控性:")
    for key, value in average_module_controllability.items():
        print(f"{key}: {value:.3f}")

    # 修改返回值，同时返回arm_cf和average_module_controllability
    return arm_cf, average_module_controllability

if __name__ == "__main__":
    scale_types = ['ISI', 'GAD', 'PHQ', 'PSS', 'ALL']
    data_path = '/home/user/xuxiao/DALL/dataset/CS-NRAC/scales.csv'
    
    all_results = {}  # 存储所有量表的CF结果
    
    for scale_type in scale_types:
        print(f"\n=== 分析 {scale_type} 量表 ===")
        df = pd.read_csv(data_path)
        prefixes = get_scale_columns(scale_type)
        
        selected_cols = ['cust_id'] + [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]
        df_scale = df[selected_cols].drop(columns=['cust_id'])
        
        # 收集每个量表的结果
        cf_results, module_results = main(df_scale, scale_type)
        all_results[scale_type] = cf_results

    # 最后统一打印所有CF值
    print("\n=== 所有量表的CF值 ===")
    for scale_type, cf_dict in all_results.items():
        print(f"\n{scale_type}量表:")
        for node, cf in cf_dict.items():
            print(f"{node}: {cf:.3f}")
