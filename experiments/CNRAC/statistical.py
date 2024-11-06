import pandas as pd

# 读取数据文件
df = pd.read_csv('/home/user/xuxiao/DALL/dataset/CNRAC/scales.csv')


# 提取所需的列并筛选年龄不超过25岁的数据
result_df = df[['standard_id', 'sex', 'age']]
result_df = result_df[result_df['age'] <= 30]

# 计算性别分布
sex_distribution = result_df['sex'].value_counts()

# 计算年龄统计
age_stats = result_df['age'].describe()

# 输出结果
print("性别分布:")
print(sex_distribution)
print("\n年龄统计:")
print(age_stats)