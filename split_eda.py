import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from datetime import datetime, timedelta
import biosppy.signals as signals
import neurokit2 as nk

# 设置数据路径
base_path = "D:\\sccj\\E4\\"
raw_data_path = os.path.join(base_path, "E4_rawdata")
marker_data_path = os.path.join(base_path, "E4Markers4Driving")
output_path_bvp = os.path.join(base_path, "Visulization_Raw_BVP_delay8s")
output_path_eda = os.path.join(base_path, "Visulization_Raw_EDA_delay8s")
hr_output_path = os.path.join(base_path, "Visulization_Raw_HR_delay8s")


def eda_custom_process(eda_signal, sampling_rate=4, method="neurokit"):
    eda_signal = nk.signal_sanitize(eda_signal)

    # Series check for non-default index
    if type(eda_signal) is pd.Series and type(eda_signal.index) != pd.RangeIndex:
        eda_signal = eda_signal.reset_index(drop=True)

    # Preprocess
    eda_cleaned = eda_signal  # Add your custom cleaning module here or skip cleaning
    eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate)

    # Find peaks
    peak_signal, info = nk.eda_peaks(
        eda_decomposed["EDA_Phasic"].values,
        sampling_rate=sampling_rate,
        method=method,
        amplitude_min=0.1,
    )
    info['sampling_rate'] = sampling_rate  # Add sampling rate in dict info

    # Store
    signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned})

    signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)

    return signals, info


# 被试信息对应关系
subject_mapping = {
    "A04A07": "01",
    "A042AE": "02",
    "A03E19": "03"
}

# 获取所有EDA raw数据文件路径
eda_files = glob.glob(os.path.join(raw_data_path, "*", "*/*EDA_addtime.csv"))


def process_bvp_file(data2write, output_path):
    # 保存处理后的文件
    data2write.to_csv(output_path, index=False, float_format='%.2f')


# 遍历每个BVP raw数据文件
for eda_file in eda_files:

    # 提取场次信息
    session = os.path.basename(os.path.dirname(os.path.dirname(eda_file)))

    # 提取被试信息
    subject_folder = os.path.basename(os.path.dirname(eda_file))
    subject_code = None
    subject_name = None

    for code, name in subject_mapping.items():
        if code in subject_folder:
            subject_code = code
            subject_name = name
            break

    if subject_name is None:
        continue

    # 提取工况marker文件路径
    marker_file = os.path.join(marker_data_path, f"{session}_{subject_name}_e4marker_driving.csv")

    # 检查marker文件是否存在
    if not os.path.exists(marker_file):
        continue

    # 读取BVP数据
    eda_data = pd.read_csv(eda_file, skiprows=1, names=["Amplitude", "Timestamp"])
    # 读取marker数据
    marker_data = pd.read_csv(marker_file)

    start_timestamp = marker_data.loc[marker_data["event"] == 'baseline', "timestamp"].values[
                          0] + 30.00  # cut 30s if is baseline, because baseline is too long
    end_timestamp = marker_data.loc[marker_data["event"] == 'end_L3', "timestamp"].values[
                        0] + 8.00  # 8 seconds for BVP performance delay
    full_eda_data = eda_data.loc[
        (eda_data["Timestamp"] >= start_timestamp) & (eda_data["Timestamp"] <= end_timestamp)]
    ppg_data = full_eda_data.iloc[0:, 0].values.astype(float)
    timestamp = full_eda_data.iloc[0:, 1].values

    # signals, info = nk.eda_process(ppg_data, sampling_rate=4)
    signals, info = eda_custom_process(ppg_data)
    try:
        nk.eda_plot(signals)
        img_filename = os.path.join(output_path_eda + '\\figure\\', f"{session}_{subject_name}_full_EDA.png")
        plt.savefig(img_filename, dpi=300, bbox_inches="tight")
        plt.clf()
        plt.close()
    except:
        print(session, subject_name, 'full')

    signals.insert(signals.shape[1], 'Timestamp', timestamp)

    csv_filename = os.path.join(output_path_eda + '\\process\\full\\', f"{session}_{subject_name}_full.csv")
    signals.to_csv(csv_filename, index=False)

    # 遍历每个工况
    for condition in marker_data["event"].unique():
        # 检查工况是否以"end_"开头
        if not condition.startswith("end_"):
            # 提取起始和结束时间戳
            if (condition == 'baseline'):
                start_time_cut = 30.00  # 如果是baseline，不要前30秒，留下30秒作为baseline
            else:
                start_time_cut = 0.00
            start_timestamp1 = marker_data.loc[marker_data["event"] == condition, "timestamp"].values[
                                   0] + start_time_cut  # cut 30s if is baseline, because baseline is too long
            end_timestamp1 = marker_data.loc[marker_data["event"] == f"end_{condition}", "timestamp"].values[
                                 0] + 8.00  # 8 seconds for BVP performance delay

            # -----------------------BVP--------------------------------
            # 根据起始和结束时间戳筛选BVP数据
            condition_eda_data_with_timestamp = full_eda_data.loc[
                (full_eda_data["Timestamp"] >= start_timestamp1) & (full_eda_data["Timestamp"] <= end_timestamp1)]
            condition_eda_data = condition_eda_data_with_timestamp.iloc[:, 0]

            signals, info = eda_custom_process(condition_eda_data)
            try:
                nk.eda_plot(signals)
                img_filename = os.path.join(output_path_eda + '\\figure\\',
                                            f"{session}_{subject_name}_{condition}_EDA.png")
                plt.savefig(img_filename, dpi=300, bbox_inches="tight")
                plt.clf()
                plt.close()
            except:
                print(session, subject_name, condition)

            analyze_signals = nk.eda_analyze(signals, sampling_rate=4, method="interval-related")
            analyze_signals.insert(0, 'Session', session)
            analyze_signals.insert(1, 'Subject Name', subject_name)
            analyze_signals.insert(2, 'Condition', condition)

            # 保存数据
            csv_filename1 = os.path.join(output_path_eda + '\\process\\'+condition,
                                         f"{session}_{subject_name}_{condition}.csv")
            signals.to_csv(csv_filename1, index=False)
            csv_filename2 = os.path.join(output_path_eda + '\\analyze\\' + condition,
                                         f"{session}_{subject_name}_{condition}_analyze.csv")
            analyze_signals.to_csv(csv_filename2, index=False)
