import pandas as pd

# 读取CSV文件
scale_df = pd.read_csv('../../dataset/CS-NRAC/scale.csv')
base_info_df = pd.read_csv('../../dataset/CS-NRAC/base_info.csv')

# 将cust_id列重命名为standard_id
scale_df = scale_df.rename(columns={'cust_id': 'standard_id'})
base_info_df = base_info_df.rename(columns={'cust_id': 'standard_id'})

# 合并两个数据框
merged_df = pd.merge(scale_df, base_info_df, on='standard_id')

# 计算性别分布
gender_distribution = merged_df['gender'].value_counts()

# 计算性别百分比
gender_percentage = merged_df['gender'].value_counts(normalize=True) * 100

# 创建结果数据框
result_df = pd.DataFrame({
    'Count': gender_distribution,
    'Percentage': gender_percentage
})

# 添加总计行
total_count = result_df['Count'].sum()
total_percentage = result_df['Percentage'].sum()
result_df.loc['Total'] = [total_count, total_percentage]

# 格式化百分比列
result_df['Percentage'] = result_df['Percentage'].apply(lambda x: f'{x:.2f}%')

# 打印结果表格
print(result_df)
