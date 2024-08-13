import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import rankdata

def match_indices(df_1, df_2):
    # 获取df_acc的索引
    target_indices = df_1.index
    
    # 过滤df_time，使其只保留在df_acc中存在的索引
    df_2_new = df_2[df_2.index.isin(target_indices)]
    
    return df_2_new

# 读取xlsx到df
def read_file(file_path, drop_datasets=[]):
    df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
    # 将第一列设置为索引
    df.set_index(df.columns[0], inplace=True)
    # 删除指定索引的行
    df.drop(drop_datasets, errors='ignore', inplace=True)
    return df


def collect_info(df):
    df_index = df.columns
# avg.acc/f1
    avg_acc = df.mean(axis=0)

# avg.rank
    all_ranks = []
    for data_name in df.index:
        accuracies = df.loc[data_name]
        ranks = rankdata(-accuracies, method='average')
        all_ranks.append(ranks)
    avg_ranks = pd.DataFrame(all_ranks).mean().tolist()
    # max_avg_rank = min(avg_ranks)  # 由于排名越小越好，取最小值

# p_value
    num_algorithms = df.shape[1]
    p_values_matrix = np.zeros((num_algorithms, num_algorithms))
    for i in range(num_algorithms):
        for j in range(num_algorithms):
            if i != j:
                stat, p_value = wilcoxon(df.iloc[:, i], df.iloc[:, j])
                p_values_matrix[i, j] = p_value
            else:
                p_values_matrix[i, j] = np.nan  # 自己与自己比较，用NaN表示
    p_value_ours = p_values_matrix[-1, :].tolist()


    ours_index = df.shape[1]
    wins = [0] * ours_index
    ties = [0] * ours_index
    losses = [0] * ours_index
    for i in range(0, len(df)):
        ours_accuracy = df.iloc[i, ours_index - 1]
        for j in range(ours_index - 1):
            if df.iloc[i, j] > ours_accuracy:
                losses[j] += 1
            elif df.iloc[i, j] < ours_accuracy:
                wins[j] += 1
            else:
                ties[j] += 1
    wins[ours_index - 1] = "-"
    ties[ours_index - 1] = "-"
    losses[ours_index - 1] = "-"
    conbine = []
    for a, b, c in zip(wins, ties, losses):
        conbine.append(f'{c}/{b}/{a}') # {a}/{b}/{c}

    # 创建一个新的 DataFrame
    new_df = pd.DataFrame(avg_acc, index=df_index, columns=['Average'])
    # 添加 avg_ranks 到 new_df
    new_df['Avg Rank'] = avg_ranks
    new_df['p-Value'] = p_value_ours
    new_df['wins/ties/losses'] = conbine

    return new_df


def print_latex_table(df):
    print("\\begin{tabular}{l" + "c" * len(df.columns) + "}")  # 根据列的数量调整格式
    print("\\hline")
    print(f"Column Names & {' & '.join(list(df.columns))}\\\\")
    for index, row in df.iterrows():
        # 使用 str.join 生成每一行的 LaTeX 字符串
        row_values = []
        for col, val in row.items():
            if col == 'p-Value' and isinstance(val, float):  # 特定检查 p-Value 列且确保数据类型为浮点数
                row_values.append(f"{val:.2e}")  # 以科学计数法格式化
            elif isinstance(val, float):
                row_values.append(f"{val:.4f}")  # 其他浮点数以 .4f 格式化
            else:
                row_values.append(str(val))  # 非浮点数直接转换为字符串
        row_str = " & ".join(row_values)
        print(f"{index} & {row_str} \\\\")
    print("\\hline")
    print("\\end{tabular}")

if __name__ == "__main__":
    acc_file_path = r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\Comparison\results_total.xlsx'
    f1_file_path = r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\Comparison\total_f1score.xlsx'
    indices_to_remove=[]
    # indices_to_remove = ['SyntheticControl', 'TwoPatterns', 'SmoothSubspace', 
    #                     'GestureMidAirD1', 'GestureMidAirD3', 'UWaveGestureLibraryX', 
    #                     'UWaveGestureLibraryZ']
    df_acc = read_file(acc_file_path, indices_to_remove)
    df_f1 = read_file(f1_file_path, indices_to_remove)
    df_f1 = match_indices(df_acc, df_f1)
    print(len(df_acc.index), len(df_f1.index))

    acc_info = collect_info(df_acc)
    f1_info = collect_info(df_f1)
    total_info = pd.concat([acc_info, f1_info], axis=1)
    total_info_path = r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\Comparison\result_analysis\total_info.xlsx'
    total_info.to_csv(total_info_path)
    print_latex_table(total_info)




