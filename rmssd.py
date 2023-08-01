import numpy as np
import os
import glob
import pandas as pd
import neurokit2 as nk
import datetime
import matplotlib.pyplot as plt

# 设置数据路径
base_path = "D:\\sccj\\E4\\"
raw_data_path = os.path.join(base_path, "E4_rawdata")
marker_data_path = os.path.join(base_path, "E4Markers4Driving")
# output_path_bvp = os.path.join(base_path, "Visulization_Raw_BVP_delay8s")
output_path_rmssd = os.path.join(base_path, "RMSSD")

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


def time_transform(t):
    real = []
    for x in t:
        d = datetime.datetime.fromtimestamp(x)
        str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
        real.append(str1)
    return real


def time_interval(t1, t2):
    itv = []
    start_time = time_transform(t1)
    finish_time = time_transform(t2)
    for i in range(len(finish_time)):
        time1 = datetime.datetime.strptime(start_time[i], "%Y-%m-%d %H:%M:%S.%f")
        time2 = datetime.datetime.strptime(finish_time[i], "%Y-%m-%d %H:%M:%S.%f")
        duration = (time2 - time1).total_seconds() * 1000
        itv.append(duration)
    return itv


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

    signals, info = nk.ppg_process(ppg_data, sampling_rate=64)

    signals.insert(signals.shape[1], 'Timestamp', timestamp)
    signals_array = np.array(signals)
    peaks = signals_array[:, 3]
    full_timestamp = signals_array[:, 4]
    windows = len(peaks) - 640 + 1  # 窗口个数 长度10s
    windows_x = np.arange(0, windows)
    full_rmssd = []

    for i in range(windows):
        part_timestamp = full_timestamp[i:i + 640]
        part_peaks = peaks[i:i + 640]
        index = np.where(part_peaks == 1)
        peak_timestamp = part_timestamp[index]
        x0 = peak_timestamp[:-1]
        x1 = peak_timestamp[1:]
        # 所有peak之间的间隔时间
        hr_intervals = time_interval(x0, x1)
        # interval之间差异
        diff_intervals = np.diff(hr_intervals)
        # 计算差异序列的平方
        squared_diff = np.square(diff_intervals)
        # 计算平方序列的均值
        mean_squared_diff = np.mean(squared_diff)
        # 计算均值的平方根，即RMSSD
        rmssd = np.sqrt(mean_squared_diff)

        full_rmssd.append(rmssd)

    array_rmssd = np.array(full_rmssd)
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.plot(windows_x, array_rmssd,label='full trail')

    for condition in marker_data["event"].unique():
        # 检查工况是否以"end_"开头
        if not condition.startswith("end_"):
            # 提取起始和结束时间戳
            if (condition == 'fulltrail'):
                continue
            if (condition == 'baseline'):
                start_time_cut = 30.00  # 如果是baseline，不要前30秒，留下30秒作为baseline
            else:
                start_time_cut = 0.00
            start_timestamp = marker_data.loc[marker_data["event"] == condition, "timestamp"].values[
                                  0] + start_time_cut  # cut 30s if is baseline, because baseline is too long
            end_timestamp = marker_data.loc[marker_data["event"] == f"end_{condition}", "timestamp"].values[
                                0] + 8.00  # 8 seconds for BVP performance delay

            condition1 = full_timestamp >= start_timestamp
            condition2 = full_timestamp < end_timestamp
            indices = np.where(condition1 & condition2)[0]
            indices = indices[0:-640]
            condition_x = windows_x[indices]
            condition_y = array_rmssd[indices]
            t0 = condition_x[0]
            t1 = condition_x[-1]
            p_xt = 0.5
            plt.plot(condition_x, condition_y, label=condition)
            plt.axvspan(t0, t1, facecolor='gray', alpha=0.1)
            plt.text((t0 + t1) / 2, p_xt, condition, fontsize=12, verticalalignment="top",
                     horizontalalignment="center")
    ax.axes.xaxis.set_ticks([])
    plt.ylim(0, 800)
    # plt.legend()
    plt.title(f"{session}_{subject_name}_RMSSD", fontsize=20)
    img_filename = os.path.join(output_path_rmssd + '\\figure\\',
                                f"{session}_{subject_name}_RMSSD.png")
    plt.savefig(img_filename, dpi=300, bbox_inches="tight")
    # plt.show()
    
