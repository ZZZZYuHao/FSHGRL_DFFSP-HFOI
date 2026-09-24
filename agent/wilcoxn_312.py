"""
读取csv数据并执行非参数检验
"""
import scipy.stats as stats, csv

# 读取csv数据
file_path_name = '../result/wilcoxn.csv'
algorithms_list = ['FGPPO', 'noFluid-HGNN-PPO', 'Fluid-MLP-PPO', 'A2C', 'DQN', 'DDQN']

with open(file_path_name, 'r') as file:
    reader = csv.reader(file)
    # 创建空字典列表存储每一列的数据
    algorithm_dict = {name: [] for name in algorithms_list}
    next(reader)  # 跳过标题行
    # 逐行读取CSV文件并将每一列的数据添加到对应的列表中
    for row in reader:

        if not any(cell.strip() for cell in row):
            continue

        for i in range(len(algorithms_list)):
            val = row[i].strip()
            if val == '':
                break
            algorithm_dict[algorithms_list[i]].append(float(val))

algorithm_A = 'FGPPO'
algorithm_B_list = ['noFluid-HGNN-PPO', 'Fluid-MLP-PPO', 'A2C', 'DQN', 'DDQN']

for algorithm in algorithm_B_list:
    statistic, p = stats.wilcoxon(algorithm_dict[algorithm_A], algorithm_dict[algorithm], correction=True, alternative='two-sided')
    R_plus = sum(rank for rank in range(1, len(algorithm_dict[algorithm_A]) + 1) if algorithm_dict[algorithm_A][rank - 1] <= algorithm_dict[algorithm][rank - 1])
    R_minus = sum(rank for rank in range(1, len(algorithm_dict[algorithm_A]) + 1) if algorithm_dict[algorithm_A][rank - 1] >= algorithm_dict[algorithm][rank - 1])
    print("_______________{}_________________".format(algorithm))
    print("R-:", R_minus)
    print("R+:", R_plus)
    print("p-value:", p)