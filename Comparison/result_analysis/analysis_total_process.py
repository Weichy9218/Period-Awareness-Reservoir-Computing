import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import rankdata

class_counts = [37, 3, 5, 2, 3, 4, 3, 2, 2, 4, 3, 2, 6, 2, 5, 2, 2, 2, 2, 2, 5, 2, 7, 2, 8, 3, 6, 42, 42, 4, 7, 3,
                3, 3, 2, 4, 2, 2, 2, 5, 2, 2]


# 读取Excel文件的第一个sheet
def read_excel(file_path):
    df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
    return df

# 打印全部的acc，保留4位有效数字
def print_total_accuracy(df):
    for i in range(0, len(df)):
        dataset_name = df.iloc[i, 0]
        accuracies = df.iloc[i, 1:]
        max_acc = accuracies.max()
        formatted_accuracies = [f"\\textbf{{{acc:.4f}}}" if acc == max_acc else "%.4f" % acc for acc in accuracies]
        row_str = f"{dataset_name} & {' & '.join(formatted_accuracies)} \\\\"
        print(row_str)


def average_accuracy(df):
    avg_accuracies = df.iloc[:, 1:].mean()
    max_avg_acc = avg_accuracies.max()
    formatted_avg_accuracies = [f"\\textbf{{{acc:.3f}}}" if acc == max_avg_acc else f"{acc:.3f}" for acc in
                                avg_accuracies]
    avg_str = f"Avg. Acc. & {' & '.join(formatted_avg_accuracies)} \\\\"
    print(avg_str)


def average_rank(df):
    all_ranks = []
    for i in range(0, len(df)):
        accuracies = df.iloc[i, 1:]
        # ranks = rankdata(-accuracies, method='min')
        ranks = rankdata(-accuracies, method='average')
        all_ranks.append(ranks)

    avg_ranks = pd.DataFrame(all_ranks).mean().tolist()
    max_avg_rank = min(avg_ranks)  # 由于排名越小越好，取最小值
    formatted_avg_ranks = [f"\\textbf{{{rank:.3f}}}" if rank == max_avg_rank else f"{rank:.3f}" for rank in avg_ranks]
    avg_rank_str = f"Avg. Rank & {' & '.join(formatted_avg_ranks)} \\\\"
    print(avg_rank_str)

# Acc top1的次数
def rank_counts(df, k):
    all_ranks = []
    for i in range(0, len(df)):
        accuracies = df.iloc[i, 1:] # 取某一行的全部值 type:series
        ranks = rankdata(-accuracies, method='min') # 对数据进行排名，将较小（-较大）的排在前头
        all_ranks.append(ranks)

    all_ranks_df = pd.DataFrame(all_ranks)
    counts = (all_ranks_df <= k).sum(axis=0).tolist()  # 计算排名第一的次数
    count_str = f"Num. of Top-{k} & {' & '.join(map(str, counts))} \\\\"
    print(f"Column Names & {' & '.join(list(df.columns[1:]))}")
    print(count_str)

# 1对1比较
def one_to_one_comparison(df):
    ours_index = df.shape[1] - 1
    wins = [0] * ours_index
    ties = [0] * ours_index
    losses = [0] * ours_index

    for i in range(0, len(df)):
        ours_accuracy = df.iloc[i, ours_index]
        for j in range(1, ours_index):
            if df.iloc[i, j] > ours_accuracy:
                losses[j-1] += 1
            elif df.iloc[i, j] < ours_accuracy:
                wins[j-1] += 1
            else:
                ties[j-1] += 1

    wins[ours_index - 1] = "-"
    ties[ours_index - 1] = "-"
    losses[ours_index - 1] = "-"

    wins_str = f"Ours 1-to-1 Wins & {' & '.join(map(str, wins))} \\\\"
    ties_str = f"Ours 1-to-1 Ties & {' & '.join(map(str, ties))} \\\\"
    losses_str = f"Ours 1-to-1 Losses & {' & '.join(map(str, losses))} \\\\"
    print(f"Column Names & {' & '.join(list(df.columns[1:]))}")
    print(wins_str)
    print(ties_str)
    print(losses_str)


def print_p_values(df):
    algorithms = df.columns[1:]
    accuracies = df.iloc[:, 1:]
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


def print_mcba(df):
    accuracies = df.iloc[:, 1:]
    num_datasets = len(class_counts)
    num_algorithms = accuracies.shape[1]

    mcba_values = []

    for i in range(num_algorithms):
        pce_values = []
        for k in range(num_datasets):
            accuracy = accuracies.iloc[k, i]
            pce = accuracy * class_counts[k]
            pce_values.append(pce)
        mcba = round(sum(pce_values) / sum(class_counts), 3)
        mcba_values.append(mcba)

    max_mcba = max(mcba_values)
    formatted_mcba = [f"\\textbf{{{mcba:.3f}}}" if mcba == max_mcba else f"{mcba:.3f}" for mcba in mcba_values]

    result_str = "MCBA & " + " & ".join(formatted_mcba) + " \\\\"
    print(result_str)


def pairwise_plot(baseline, ours, baseline_name):
    win, tie, loose = 0, 0, 0

    for b, o in zip(baseline, ours):
        if o > b:
            win += 1
        elif o == b:
            tie += 1
        else:
            loose += 1

    font_size = 37
    # plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.unicode_minus"] = False
    # plt.rcParams["figure.figsize"] = (12.8 * 3.3, 9.6 * 3.3)

    # 0.895
    x = baseline
    y = ours

    # Define the line from (0,0) to (1,1)
    line_x = np.linspace(0, 1, 100)
    line_y = line_x

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Fill the area above the line in blue
    ax.fill_between(line_x, line_y, 1, color='lightblue')  # label added for legend  lightblue

    # Plot all points in black
    ax.scatter(x, y, color='midnightblue', s=150)

    # Plot the line
    ax.plot(line_x, line_y, color='black', linewidth=2)

    # Set axis limits, labels, and title
    ax.set_xlim(0.3, 1)
    ax.set_ylim(0.3, 1)
    # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.set_xticks(np.arange(0.3, 1.05, 0.1))
    ax.set_yticks(np.arange(0.3, 1.05, 0.1))
    # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.set_xlabel(baseline_name)
    ax.set_ylabel('MsDL')
    # ax.set_title('Scatter plot with area above line filled')
    ax.grid(True, which='both', linestyle='--', linewidth=0.1)

    # Add black border around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # Add black border on the right and top side
    ax.axhline(1, color='black', linewidth=2)
    ax.axvline(1, color='black', linewidth=2)

    ax.tick_params(axis='both', which='major', size=7, labelsize=font_size-5, width=2)

    textbox_props = dict(boxstyle='round,pad=0.3', facecolor='none', edgecolor='none')
    ax.text(0.06, 0.95, 'Ours is better here', color='black', transform=ax.transAxes, verticalalignment='top',
            bbox=textbox_props, fontsize=font_size-3)

    textbox_props = dict(boxstyle='square,pad=0.3', facecolor='none', edgecolor='black', linewidth=1.5)
    ax.text(0.07, 0.80, f'Win: {win}\nTie: {tie}\nLoose: {loose}', color='black',
            transform=ax.transAxes, verticalalignment='top', bbox=textbox_props, fontsize=font_size-6)

    plt.savefig("./pairwise_%s.pdf" % baseline_name, dpi=600, format="pdf", bbox_inches='tight')
    # plt.show()


def pairwise_plot_total(df):
    algorithms = df.columns[1:df.shape[1]-1]
    ours = df.columns[-1]
    ours_results = df[ours].tolist()
    for name in algorithms:
        pairwise_plot(df[name].tolist(), ours_results, name)
        # break


if __name__ == "__main__":
    # 设置文件路径
    file_path = r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\result_analysis\results_total.xlsx' # r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\npy_result\total_result.xlsx'
    df = read_excel(file_path)
    
    # pairwise_plot_total(df)
    print_total_accuracy(df)
    rank_counts(df, 1)
    one_to_one_comparison(df)
    average_rank(df)
    average_accuracy(df)
    # print_mcba(df)
    print_p_values(df)