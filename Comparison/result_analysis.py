import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import rankdata

# 读取xlsx到df
def read_file(file_path, drop_datasets):
    df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
    # 将第一列设置为索引
    df.set_index(df.columns[0], inplace=True)
    # 删除指定索引的行
    df.drop(drop_datasets, errors='ignore', inplace=True)
    return df

# 打印全部的a
def print_total_df(df):
    methods = [f"\\textbf{{{method}}}" for method in df.columns]
    methods_str = ' & '.join(methods)
    print(methods_str)
    for dataset_name in df.index:
        accuracies = df.loc[dataset_name, :]
        max_acc = accuracies.max()
        formatted_accuracies = [f"\\textbf{{{acc:.4f}}}" if acc == max_acc else "%.4f" % acc for acc in accuracies]
        row_str = f"{dataset_name} & {' & '.join(formatted_accuracies)} \\\\"
        print(row_str)


def average_accuracy(df):
    avg_accuracies = df.mean()
    max_avg_acc = avg_accuracies.max()
    formatted_avg_accuracies = [f"\\textbf{{{acc:.3f}}}" if acc == max_avg_acc else f"{acc:.3f}" for acc in
                                avg_accuracies]
    avg_str = f"Avg. Acc. & {' & '.join(formatted_avg_accuracies)} \\\\"
    print(f"Column Names & {' & '.join(list(df.columns))}")
    print(avg_str)


def average_rank(df):
    all_ranks = []
    for data_name in df.index:
        accuracies = df.loc[data_name, :]
        # ranks = rankdata(-accuracies, method='min')
        ranks = rankdata(-accuracies, method='average')
        all_ranks.append(ranks)

    avg_ranks = pd.DataFrame(all_ranks).mean().tolist()
    max_avg_rank = min(avg_ranks)  # 由于排名越小越好，取最小值
    formatted_avg_ranks = [f"\\textbf{{{rank:.3f}}}" if rank == max_avg_rank else f"{rank:.3f}" for rank in avg_ranks]
    avg_rank_str = f"Avg. Rank & {' & '.join(formatted_avg_ranks)} \\\\"
    print(avg_rank_str)


def rank_counts(df, k):
    all_ranks = []
    for dataset_name in df.index:
        accuracies = df.loc[dataset_name, :] # 取某一行的全部值 type:series
        ranks = rankdata(-accuracies, method='min') # 对数据进行排名，将较小（-较大）的排在前头
        all_ranks.append(ranks)

    all_ranks_df = pd.DataFrame(all_ranks)
    counts = (all_ranks_df <= k).sum(axis=0).tolist()  # 计算排名第一的次数
    count_str = f"Num. of Top-{k} & {' & '.join(map(str, counts))} \\\\"
    # print(f"Column Names & {' & '.join(list(df.columns))}")
    print(count_str)


def one_to_one_comparison(df):
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

    wins_str = f"Ours 1-to-1 Wins & {' & '.join(map(str, wins))} \\\\"
    ties_str = f"Ours 1-to-1 Ties & {' & '.join(map(str, ties))} \\\\"
    losses_str = f"Ours 1-to-1 Losses & {' & '.join(map(str, losses))} \\\\"
    # print(f"Column Names & {' & '.join(list(df.columns))}")
    print(wins_str)
    print(ties_str)
    print(losses_str)


def print_p_values(df):
    algorithms = df.columns

    methods = [f"{method}" for method in algorithms]
    methods_str = ' & '.join(methods)
    print(methods_str)

    accuracies = df
    num_algorithms = accuracies.shape[1]
    p_values_matrix = np.zeros((num_algorithms, num_algorithms))

    for i in range(num_algorithms):
        for j in range(num_algorithms):
            if i != j:
                stat, p_value = wilcoxon(accuracies.iloc[:, i], accuracies.iloc[:, j])
                p_values_matrix[i, j] = p_value
            else:
                p_values_matrix[i, j] = np.nan  # 自己与自己比较，用NaN表示

    latex_str = ""

    # p_value_ours = p_values_matrix[-1, :].tolist()
    # p_value_ours_str = f"p_value & {' & '.join(map(str, p_value_ours))} \\\\"
    # print(p_value_ours_str)

    for i in range(len(algorithms)):
        row = [algorithms[i]]
        for j in range(len(algorithms)):
            if i == j:
                row.append("-")
            else:
                p_value_str = f"{p_values_matrix[i, j]:.2E}"
                if p_values_matrix[i, j] > 0.05:
                    p_value_str = f"\\cellcolor{{gray!25}}{p_value_str}"
                row.append(p_value_str)
        latex_str += " & ".join(row) + " \\\\\n"

    print(latex_str)


if __name__ == "__main__":
    acc_file_path = r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\Comparison\results_total.xlsx'
    indices_to_remove = []
    # ['SyntheticControl', 'TwoPatterns', 'SmoothSubspace', 
    #                     'GestureMidAirD1', 'GestureMidAirD3', 'UWaveGestureLibraryX', 
    #                     'UWaveGestureLibraryZ']
    df_acc = read_file(acc_file_path, indices_to_remove)
    # print_total_df(df_acc)
    # average_accuracy(df_acc)
    # average_rank(df_acc)
    # # rank_counts(df_acc, 1)
    # one_to_one_comparison(df_acc)
    print_p_values(df_acc)