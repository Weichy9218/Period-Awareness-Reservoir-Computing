import numpy as np
import pandas as pd
import os 

data_path = r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\Comparison\npy_result'

def acc2elsx(npy_list, output_path):
    
    total_df = pd.DataFrame()

    for npy_name in npy_list:
        data = np.load(os.path.join(data_path, npy_name), allow_pickle=True).item()
        if len(data[list(data.keys())[0]]) < 8:
            acc = {key: round(values[0], 5) for key, values in data.items()}
        else:
            acc = {key: round(values[6], 5) for key, values in data.items()}
        df = pd.DataFrame(list(acc.values()), index=acc.keys(), columns=[npy_name.split('_')[0].split('.npy')[0]])
        if total_df.empty:
            total_df = df
        else:
            total_df = total_df.join(df, how='outer')
    total_df.to_excel(output_path)

    print(f'Excel file has been saved to {output_path}')


def f12elsx(npy_list, output_path):
    
    total_df = pd.DataFrame()

    for npy_name in npy_list:
        data = np.load(os.path.join(data_path, npy_name), allow_pickle=True).item()
        if len(data[list(data.keys())[0]]) < 8:
            f1_scores = {key: round(values[3], 5) for key, values in data.items()}
        else:
            f1_scores = {key: round(values[9], 5) for key, values in data.items()}
        df = pd.DataFrame(list(f1_scores.values()), index=f1_scores.keys(), columns=[npy_name.split('_')[0].split('.npy')[0]])
        if total_df.empty:
            total_df = df
        else:
            total_df = total_df.join(df, how='outer')
    total_df.to_excel(output_path)
    print(f'Excel file has been saved to {output_path}')

def time2elsx(npy_list, output_path):
    
    total_df = pd.DataFrame()

    for npy_name in npy_list:
        data = np.load(f"./Comparison/npy_result/{npy_name}", allow_pickle=True).item()
        if len(data[list(data.keys())[0]]) < 8:
            trianval_times =  {key: values[4] + values[5] for key, values in data.items()}
        else:
            trianval_times =  {key: values[10] + values[11] for key, values in data.items()}
        df = pd.DataFrame(list(trianval_times.values()), index=trianval_times.keys(), columns=[npy_name.split('_')[0].split('.npy')[0]])
        if total_df.empty:
            total_df = df
        else:
            total_df = total_df.join(df, how='outer')
    total_df.to_excel(output_path)

    print(f'Excel file has been saved to {output_path}')

if __name__ == "__main__":
    directory_path = r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\Comparison\npy_result'
    npy_list = ['rmESN_seed-0.npy', 'ConvMESN_seed-0.npy','FEDformer_seed-0.npy',  'LSTNet_seed-0.npy', 'TimesNet_seed-0.npy', 'InceptionTime_seed-0.npy', 'COTE_seed-0.npy', 'Hydra_seed-0.npy', 'Rocket_seed-0.npy', 'MiniRocket_seed-0.npy', 'PARC_seed-0.npy' ]
    # npy_list = os.listdir(directory_path)# [file for file in os.listdir(directory_path) if file.endswith('.npy')], 
    output_acc_path = r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\Comparison\total_acc.xlsx'
    output_time_path = r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\Comparison\total_time.xlsx'
    output_f1score_path = r'C:\Users\xycy1\Desktop\PeriodRes_AAAI\code\Comparison\total_f1score.xlsx'
    acc2elsx(npy_list, output_acc_path)  # acc手动删去了SyntheticControl, TwoPatterns, SmoothSubspace, GestureMidAirD1, GestureMidAirD3, UWaveGestureLibraryX, UWaveGestureLibraryZ
    time2elsx(npy_list, output_time_path)
    f12elsx(npy_list, output_f1score_path)