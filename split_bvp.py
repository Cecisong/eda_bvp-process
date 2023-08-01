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
# output_path_bvp = os.path.join(base_path, "Visulization_Raw_BVP_delay8s")
output_path_bvp = os.path.join(base_path, "BVP")
output_path_eda = os.path.join(base_path, "Visulization_Raw_EDA_delay8s")
hr_output_path = os.path.join(base_path, "Visulization_Raw_HR_delay8s")

# 被试信息对应关系
subject_mapping = {
    "A04A07": "01",
    "A042AE": "02",
    "A03E19": "03"
}

# 获取所有BVP raw数据文件路径
bvp_files = glob.glob(os.path.join(raw_data_path, "*", "*/*BVP_addtime.csv"))
eda_files = glob.glob(os.path.join(raw_data_path, "*", "*/*EDA_addtime.csv"))


def process_bvp_file(data2write, output_path):
    # 保存处理后的文件
    data2write.to_csv(output_path, index=False, float_format='%.2f')


# 遍历每个BVP raw数据文件
for bvp_file in bvp_files:

    # 提取场次信息
    session = os.path.basename(os.path.dirname(os.path.dirname(bvp_file)))

    # 提取被试信息
    subject_folder = os.path.basename(os.path.dirname(bvp_file))
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
    bvp_data = pd.read_csv(bvp_file, skiprows=1, names=["Amplitude", "Timestamp"])
    # 读取marker数据
    marker_data = pd.read_csv(marker_file)

    start_timestamp = marker_data.loc[marker_data["event"] == 'baseline', "timestamp"].values[
                          0] + 30.00  # cut 30s if is baseline, because baseline is too long
    end_timestamp = marker_data.loc[marker_data["event"] == 'end_L3', "timestamp"].values[
                        0] + 8.00  # 8 seconds for BVP performance delay
    full_bvp_data = bvp_data.loc[
        (bvp_data["Timestamp"] >= start_timestamp) & (bvp_data["Timestamp"] <= end_timestamp)]
    ppg_data = full_bvp_data.iloc[0:, 0].values.astype(float)
    timestamp = full_bvp_data.iloc[0:, 1].values
    # ppg_clean = nk.ppg_clean(ppg_data, method='elgendi')
    # clean_bvp_data = full_bvp_data
    # clean_bvp_data.loc[:, 'Amplitude'] = ppg_clean
    signals, info = nk.ppg_process(ppg_data, sampling_rate=64)
    ppg_peaks = signals["PPG_Peaks"]
    signals.insert(signals.shape[1], 'Timestamp', timestamp)

    csv_filename = os.path.join(output_path_bvp + '\\process\\full\\', f"{session}_{subject_name}_full.csv")
    signals.to_csv(csv_filename, index=False)
    # clean_csv_filename = os.path.join(output_path_bvp + '\\clean\\full\\',
    #                                   f"{session}_{subject_name}_full_clean.csv")
    # clean_bvp_data.to_csv(clean_csv_filename, index=False)

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
            condition_bvp_data_with_timestamp = signals.loc[
                (signals["Timestamp"] >= start_timestamp1) & (signals["Timestamp"] <= end_timestamp1)]
            condition_bvp_data = condition_bvp_data_with_timestamp.iloc[:, 0:-1]
            # 保存数据
            csv_filename1 = os.path.join(output_path_bvp + '\\process\\' + condition,
                                         f"{session}_{subject_name}_{condition}.csv")
            condition_bvp_data.to_csv(csv_filename1, index=False)
            try:
                analyze_signals = nk.ppg_analyze(condition_bvp_data, sampling_rate=64, method="interval-related")
                analyze_signals.insert(0, 'Session', session)
                analyze_signals.insert(1, 'Subject Name', subject_name)
                analyze_signals.insert(2, 'Condition', condition)


                csv_filename2 = os.path.join(output_path_bvp + '\\analyze\\' + condition,
                                             f"{session}_{subject_name}_{condition}_analyze.csv")
                analyze_signals.to_csv(csv_filename2, index=False)

            except:
                print(session, subject_name, condition)

#             # ------------------------HR-------------------------------
#             # 处理HR from raw BVP，但是不使用E4原有HR值
#             # 使用原始BVP计算HR，64Hz
#             out = signals.bvp.bvp(signal=condition_bvp_data["Amplitude"], sampling_rate=64, show=False)
#             # 获取HR结果
#             hr = out['heart_rate']
#             hr_ts = out['heart_rate_ts']
#             # hr_timestamps = [start_timestamp + i / 1 for i in range(len(hr))]
#             hr_with_time = pd.DataFrame({'hr': hr, 'time': hr_ts})
#             # 保存数据
#             csv_filename = os.path.join(hr_output_path + '\\'+condition, f"{session}_{subject_name}_{condition}_hr.csv")
#             hr_with_time.to_csv(csv_filename, index=False)
#
# # 遍历每个BVP raw数据文件
# for eda_file in eda_files:
#
#     # 提取场次信息
#     session = os.path.basename(os.path.dirname(os.path.dirname(eda_file)))
#
#     # 提取被试信息
#     subject_folder = os.path.basename(os.path.dirname(eda_file))
#     subject_code = None
#     subject_name = None
#
#     for code, name in subject_mapping.items():
#         if code in subject_folder:
#             subject_code = code
#             subject_name = name
#             break
#
#     if subject_name is None:
#         continue
#
#     # 提取工况marker文件路径
#     marker_file = os.path.join(marker_data_path, f"{session}_{subject_name}_e4marker_driving.csv")
#
#     # 检查marker文件是否存在
#     if not os.path.exists(marker_file):
#         continue
#
#     # 读取BVP数据
#     eda_data = pd.read_csv(eda_file, skiprows=1, names=["Amplitude", "Timestamp"])
#
#     # 读取marker数据
#     marker_data = pd.read_csv(marker_file)
#
#     # 遍历每个工况
#     for condition in marker_data["event"].unique():
#         # 检查工况是否以"end_"开头
#         if not condition.startswith("end_"):
#             # 提取起始和结束时间戳
#             if (condition == 'baseline'):
#                 start_time_cut = 30.00  # 如果是baseline，不要前30秒，留下30秒作为baseline
#             else:
#                 start_time_cut = 0.00
#             start_timestamp = marker_data.loc[marker_data["event"] == condition, "timestamp"].values[
#                                   0] + start_time_cut  # cut 30s if is baseline, because baseline is too long
#             end_timestamp = marker_data.loc[marker_data["event"] == f"end_{condition}", "timestamp"].values[
#                                 0] + 8.00  # 8 seconds for BVP performance delay
#
#             # -----------------------EDA--------------------------------
#             # 根据起始和结束时间戳筛选EDA数据
#             condition_eda_data = eda_data.loc[
#                 (eda_data["Timestamp"] >= start_timestamp) & (eda_data["Timestamp"] <= end_timestamp)]
#
#             # 保存数据
#             csv_filename = os.path.join(output_path_eda, f"{session}_{subject_name}_{condition}.csv")
#             condition_eda_data.to_csv(csv_filename, index=False)
