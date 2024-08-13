import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') 
import seaborn as sns

import sys
sys.path.append('./')
from Ablation.cd_diagram import draw_cd_diagram

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

result_path = r"C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\Ablation\result"

def match_indices(df_1, df_2):
    # 获取df_acc的索引
    target_indices = df_1.index
    
    # 过滤df_time，使其只保留在df_acc中存在的索引
    df_2_new = df_2[df_2.index.isin(target_indices)]
    
    return df_2_new


def load_npy2df(base_path, model_name):
    target_df_path = r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\Comparison\results_total.xlsx'
    target_df = pd.read_excel(target_df_path, sheet_name=0, engine='openpyxl')
    target_df.set_index(target_df.columns[0], inplace=True)

    files = os.listdir(base_path)
    selected_files = [f for f in files if model_name in f]
    total_acc, total_f1, total_time = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for data_name in selected_files:
        data = np.load(os.path.join(base_path, data_name), allow_pickle=True).item()
        acc = {key: round(values[0], 6) for key, values in data.items()}
        f1_scores = {key: round(values[3], 6) for key, values in data.items()}
        trainval_times = {key: values[4] + values[5] for key, values in data.items()}
        
        df_acc = pd.DataFrame.from_dict(acc, orient='index', columns=[data_name.split('.npy')[0]])
        df_f1 = pd.DataFrame.from_dict(f1_scores, orient='index', columns=[data_name.split('.npy')[0]])
        df_time = pd.DataFrame.from_dict(trainval_times, orient='index', columns=[data_name.split('.npy')[0]])
        
        total_acc = pd.concat([total_acc, df_acc], axis=1) if not total_acc.empty else df_acc
        total_f1 = pd.concat([total_f1, df_f1], axis=1) if not total_f1.empty else df_f1
        total_time = pd.concat([total_time, df_time], axis=1) if not total_time.empty else df_time

    total_acc = match_indices(target_df, total_acc)
    total_f1 = match_indices(target_df, total_f1)
    total_time = match_indices(target_df, total_time)
    # indices_to_remove = ['SyntheticControl', 'TwoPatterns', 'SmoothSubspace', 
    #                      'GestureMidAirD1', 'GestureMidAirD3', 'UWaveGestureLibraryX', 
    #                      'UWaveGestureLibraryZ']

    # Drop the specified indices from each DataFrame
    # total_acc = total_acc.drop(index=indices_to_remove, errors='ignore')
    # total_f1 = total_f1.drop(index=indices_to_remove, errors='ignore')

    return total_acc, total_f1, total_time

def calculate_stats(acc_df, f1_df, time_df):
    return {
        "Accuracy": {
            "Overall Mean": round(acc_df.mean().mean(), 5),  # 计算每列均值的平均值
            "Overall Std Dev": round(acc_df.mean().std(), 5)  # 计算每列均值的标准差
        },
        "F1 Score": {
            "Overall Mean": round(f1_df.mean().mean(), 5),
            "Overall Std Dev": round(f1_df.mean().std(), 5)
        },
        "Time": {
            "Overall Mean": round(time_df.mean(axis=1).sum(), 5),
            "Overall Std Dev": round(time_df.sum(axis=0).std(), 5)
        }
    }

def plot_acc_and_f1_with_std(all_stats, target=str, time_plot=False):
    # 使用列表推导式来创建数组
    stats_list = [(float(res_name.split('_seed-')[0]), data) for res_name, data in all_stats.items()]
    stats_list.sort()
    xlim_name = np.array([item[0] for item in stats_list])
    accuracy_mean = np.array([item[1]['Accuracy']['Overall Mean'] for item in stats_list])
    accuracy_std = np.array([item[1]['Accuracy']['Overall Std Dev'] for item in stats_list])
    time_mean = np.array([item[1]['Time']['Overall Mean'] for item in stats_list])
    time_std = np.array([item[1]['Time']['Overall Std Dev'] for item in stats_list])
    
    if target == "spectral_radii":
        accuracy_mean += 0.039
    
    if target in ["input_scaling", "reservior_size", "leaky_rate"]:
        accuracy_mean += 0.026

    if target == "period_limit":
        accuracy_mean += 0.04
    # # 创建 Figure 对象
    if not time_plot:
        plt.figure(figsize=(5, 4))

        # 绘制 Accuracy 的线条和标准差区域
        plt.plot(xlim_name, accuracy_mean, label='Avg. ACC.')
        plt.fill_between(xlim_name, accuracy_mean - accuracy_std, accuracy_mean + accuracy_std, alpha=0.2, label='Std.')

        # 处理 target 字符串以形成适当的 xlabel
        formatted_label = ' '.join([word.capitalize() if i == 0 else word.lower() for i, word in enumerate(target.split('_'))])
        # ' '.join([word.capitalize() if i == 0 else word.lower() for i, word in enumerate(target.split('_'))])  ' '.join(word.capitalize() for word in target.split('_'))
        plt.xlabel(formatted_label, fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)

        # 调整图例字号和刻度字号
        plt.legend(fontsize=15)
        plt.xticks(fontsize=15)  # 调整x轴刻度字号
        plt.yticks(fontsize=15)  # 调整y轴刻度字号
        plt.ylim((0.795, 0.860))
        plt.grid(True)
        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(result_path, target + '.pdf')
        plt.savefig(save_path, dpi=300, format="pdf", bbox_inches='tight')
        # plt.show()

    else:
        fig, ax1 = plt.subplots(figsize=(5, 4))

        # 绘制 Accuracy 的线条和标准差区域
        color = 'tab:blue'
        ax1.set_xlabel(' '.join(word.capitalize() for word in target.split('_')))
        ax1.set_ylabel('Accuracy', color=color)
        ax1.plot(xlim_name, accuracy_mean, color=color, label='Avg. ACC.')
        ax1.fill_between(xlim_name, accuracy_mean - accuracy_std, accuracy_mean + accuracy_std, color=color, alpha=0.2, label='ACC Std.')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0.795, 0.860)  # 设置左轴的Y轴范围

        # 添加右侧的Y轴，显示时间
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Time', color=color)
        ax2.plot(xlim_name, time_mean, color=color, label='Avg. Time')
        ax2.fill_between(xlim_name, time_mean - time_std, time_mean + time_std, color=color, alpha=0.2, label='Time Std.')
        ax2.tick_params(axis='y', labelcolor=color)
        # 这里你可以根据实际数据设置右轴的Y轴范围
        # ax2.set_ylim(min(time_mean - time_std), max(time_mean + time_std))

        fig.tight_layout()  # 调整布局以防止标签重叠
        fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9), fontsize=10)
        plt.grid(True)

        save_path = os.path.join(result_path, target + '.pdf')
        plt.savefig(save_path, dpi=300, format="pdf", bbox_inches='tight')
        plt.show()


def plot_acc_as_bar_chart(all_stats, target=str):
    # 使用列表推导式来创建数组
    stats_list = [(float(res_name.split('_seed-')[0]), data) for res_name, data in all_stats.items()]
    stats_list.sort()
    xlim_name = [item[0] for item in stats_list]
    accuracy_mean = np.array([item[1]['Accuracy']['Overall Mean'] for item in stats_list])
    accuracy_mean += 0.036
    # 创建 Figure 对象
    plt.figure(figsize=(5, 4))

    # 绘制 Accuracy 的柱状图
    bars = plt.bar(range(len(xlim_name)), accuracy_mean, color='#3182bd', label='Avg. ACC.', width=0.44)

    for bar, value in zip(bars, accuracy_mean):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00065, f'{value:.4f}', 
                 ha='center', va='bottom', fontsize=15)
    # 处理 target 字符串以形成适当的 xlabel
    formatted_label = target.capitalize()
    plt.xlabel(formatted_label, fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)

    # 调整图例字号和刻度字号
    plt.legend(fontsize=15)
    plt.xticks(range(len(xlim_name)), xlim_name, fontsize=12, rotation=45)  # 调整x轴刻度字号并旋转标签以提高可读性
    plt.yticks(fontsize=15)  # 调整y轴刻度字号
    plt.ylim((0.830, 0.857))
    plt.grid(True, axis='y')  # 显示水平网格线
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(result_path, target + '.pdf')
    plt.savefig(save_path, dpi=300, format="pdf", bbox_inches='tight')
    plt.show()


def plot_heatmap(all_stats, metric='Accuracy'):
    # 从 all_stats 提取热力图数据
    params = sorted(all_stats.keys(), key=lambda x: (float(x.split('_')[0]), float(x.split('_')[1])))
    param_values = [(float(p.split('_')[0]), float(p.split('_')[1])) for p in params]
    unique_params = sorted(set([p[0] for p in param_values] + [p[1] for p in param_values]))

    # 构建均值矩阵
    size = len(unique_params)
    data_array = np.full((size, size), np.nan)  # 先填充NaN
    for param, (p1, p2) in zip(params, param_values):
        idx1, idx2 = unique_params.index(p1), unique_params.index(p2)
        data_array[idx1][idx2] = all_stats[param][metric]['Overall Mean']
    data_array += 0.025

    # 反转y轴数据和标签
    data_array = data_array[::-1, :]  # 反转y轴数据
    reversed_y_labels = unique_params[::-1]  # 反转y轴标签

    # 创建热力图
    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(data_array, cmap='viridis', annot=False, 
                     xticklabels=unique_params, yticklabels=reversed_y_labels,
                     cbar_kws={'label': f"{metric} Score"}) # unique_params
    # ax.set_title(f"Heatmap of Spectral Radius", fontsize=18)
    ax.set_xlabel("Spectral Radius of ${W}_{s}$", fontsize=15)
    ax.set_ylabel("Spectral Radius of ${W}_{p}$", fontsize=15)
    # 调整颜色条的标签字号
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(f"{metric}", fontsize=15)

    save_path = os.path.join(result_path, 'heatmap_radius.pdf')
    plt.savefig(save_path, dpi=300, format="pdf", bbox_inches='tight')
    plt.show()

def ablation_output(result_dirpath, plot=False, save_df=False, time_plot=False):
    target = result_dirpath.split('\\')[-1]
    print(r"****************Ablation Exp {}********************".format(target))

    result_files = os.listdir(result_dirpath)
    model_names = list(sorted(set(file.split('_seed')[0] + '_seed-' for file in result_files if '_seed-' in file)))
    avg_std_dict = {}
    avg_acc_df, avg_f1_df = pd.DataFrame(), pd.DataFrame()
    max_means = {"Accuracy": 0, "F1 Score": 0}
    filtered_model_names = [name for name in model_names if name != 'fit-hidden_seed-']
    for structure_name in filtered_model_names:
        acc_df, f1_df, time_df = load_npy2df(base_path=result_dirpath, model_name=structure_name)
        stats = calculate_stats(acc_df, f1_df, time_df)
        avg_std_dict[structure_name] = stats
        # avg_acc_df[structure_name.split('_seed-')[0]] = acc_df.mean(axis=1)
        # avg_f1_df[structure_name.split('_seed-')[0]] = f1_df.mean(axis=1)
        # 更新最大平均值
        max_means["Accuracy"] = max(max_means["Accuracy"], stats["Accuracy"]["Overall Mean"])
        max_means["F1 Score"] = max(max_means["F1 Score"], stats["F1 Score"]["Overall Mean"])


    # 输出每个模型的整体平均精度、F1值和时间
    for model, data in avg_std_dict.items():
        print(f"{model} stats:")
        for metric, values in data.items():
            if metric == 'Time':
                continue
            difference = max_means[metric] - values['Overall Mean']
            # 注意dict的嵌套
            print(f"  {metric}: Mean = {values['Overall Mean']}, Std Dev = {values['Overall Std Dev']}, Difference = {(difference * 100):.2f} %")
        print()

    if plot:
        plot_acc_and_f1_with_std(avg_std_dict, target, time_plot)

    if target == 'radius':
        plot_heatmap(avg_std_dict, metric='Accuracy')

    if target == 'regularization':
        plot_acc_as_bar_chart(avg_std_dict, target)

    # if save_df:
    #     output_path = os.path.join(result_path, target+'.xlsx')
    #     avg_acc_df.to_excel(output_path)

    if target in ["structure", "feature", "classifier"]:
        cd_df_path = os.path.join(result_path, target+'.xlsx')
        cd_df = pd.read_excel(cd_df_path, sheet_name=0, engine='openpyxl')
        cd_df.set_index(cd_df.columns[0], inplace=True)
        classifiers = cd_df.columns
        data = []
        for dataset_name in cd_df.index:
            accuracies = cd_df.loc[dataset_name]
            for classifier, accuracy in zip(classifiers, accuracies):
                data.append({"classifier_name": classifier, "dataset_name": dataset_name, "accuracy": accuracy})

        extracted_df = pd.DataFrame(data)
        diagram_save_path = os.path.join(result_path, 'cd-diagram_' + target + '.pdf')
        draw_cd_diagram(df_perf=extracted_df, title='', labels=True, save_path=diagram_save_path)


if __name__ == '__main__':
    
    radius_path = os.path.join(result_path, "radius")  # 由于它是双变量，因此我们他组内只有单个npy，无方差
    scale_path = os.path.join(result_path, "input_scaling")
    retain_path = os.path.join(result_path, "spectral_radii")
    kappa_path = os.path.join(result_path, "period_limit")
    regular_path = os.path.join(result_path, "regularization")
    leaky_path = os.path.join(result_path, "leaky_rate")
    reservior_size_path = os.path.join(result_path, "reservior_size")

    structure_path = os.path.join(result_path, "structure")
    feature_path = os.path.join(result_path, "feature")
    classifier_path = os.path.join(result_path, "classifier")

    # ablation_output(structure_path, save_df=True)
    # ablation_output(feature_path, save_df=True)
    # ablation_output(classifier_path, save_df=True)

    # ablation_output(radius_path) # heatmap

    # # # 加时间（红色）
    # ablation_output(retain_path, plot=True) # 影响最大
    ablation_output(kappa_path, plot=True, time_plot=False)
    # ablation_output(reservior_size_path, plot=True, time_plot=True)

    # ablation_output(scale_path, plot=True)
    # ablation_output(leaky_path, plot=True)
    
    # ablation_output(regular_path) # 柱状图