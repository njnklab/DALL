import pandas as pd

# 读取数据文件
scales_df = pd.read_csv('./dataset/CNRAC/scales.csv')
label_df = pd.read_csv('./dataset/CNRAC/label.csv')

# 合并两个数据框，只保留匹配上的数据
merged_df = pd.merge(scales_df, label_df, on='standard_id', how='inner')

# 提取所需的列并筛选年龄不超过25岁的数据
result_df = merged_df[['standard_id', 'sex', 'age', 'label']]
result_df = result_df[result_df['age'] <= 20]

# 计算性别分布
sex_distribution = result_df['sex'].value_counts()

# 计算年龄统计
age_stats = result_df['age'].describe()

# 计算诊断分布
diagnosis_distribution = result_df['label'].value_counts()

# 输出结果
print("性别分布:")
print(sex_distribution)
print("\n年龄统计:")
print(age_stats)
print("\n诊断分布:")
print(diagnosis_distribution)

# 保存结果到CSV文件
result_df.to_csv('./results/CNRAC_distribution.csv', index=False)
print("\n匹配上的详细数据已保存到 CNRAC_distribution.csv")

# 输出匹配上的数据数量
print(f"\n成功匹配的数据数量: {len(result_df)}")
print(f"原始scales数据数量: {len(scales_df)}")
print(f"原始label数据数量: {len(label_df)}")
print(f"未匹配数据数量: {len(scales_df) + len(label_df) - len(result_df)}")