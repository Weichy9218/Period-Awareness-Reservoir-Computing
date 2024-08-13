import numpy as np
from PyEMD import EMD
import pywt
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import periodogram
from scipy.fft import fft, fftfreq


def imfs_plot(original_series, imfs):
    # 可视化输入序列和分解得到的每个成分
    plt.figure(figsize=(12, 8))

    # 绘制输入时间序列
    plt.subplot(len(imfs) + 1, 1, 1)
    plt.plot(original_series)
    plt.title('Original Time Series')

    # 绘制每个小波成分
    for i, imf in enumerate(imfs):
        plt.subplot(len(imfs) + 1, 1, i + 2)
        plt.plot(imf)
        plt.title(f'IMF {i + 1}')

    plt.tight_layout()
    plt.show()


def emd_decompose(time_series, spline_kind='cubic'):
    emd = EMD()
    emd.spline_kind = spline_kind
    imfs = emd.emd(time_series)
    return imfs


def wavelet_decompose(time_series, wavelet="db4", lamb=1600):
    cycle, trend = hpfilter(time_series, lamb=lamb)
    coeffs = pywt.wavedec(cycle, wavelet)
    imfs = [trend]
    for i in range(len(coeffs)):
        coeffs_temp = [np.zeros_like(c) for c in coeffs]
        coeffs_temp[i] = coeffs[i]
        imf = pywt.waverec(coeffs_temp, wavelet)
        imfs.append(imf[:len(time_series)])
    imfs = np.array(imfs)
    return imfs


def from_fft(imfs, k=1, fs=1.0):
    """
    从每个IMF成分中提取可能的k个周期。

    参数：
    - imfs: 包含多个IMF成分的二维数组 [num, length]。
    - k: 需要提取的前k个周期。
    - fs: 采样频率（每秒采样次数）。

    返回值：
    - periods: 每个IMF成分中提取的周期列表。
    """
    all_periods = set()
    for imf in imfs:
        # 计算快速傅里叶变换 (FFT)
        N = len(imf)
        yf = fft(imf)
        xf = fftfreq(N, 1 / fs)[:N // 2]  # 只取正频部分

        power_spectrum = 2.0 / N * np.abs(yf[:N // 2])

        top_k_indices = np.argsort(power_spectrum)[-k:][::-1]  # 从大到小排序
        top_k_freqs = xf[top_k_indices]  # 获取对应的频率

        top_k_periods_imf = [round(1 / freq) for freq in top_k_freqs if freq >= 1/(N - 1)]
        all_periods.update(top_k_periods_imf)  # 添加到集合中，自动去重
    return list(all_periods)


def from_periodogram(imfs, k=1, fs=1.0):
    """
    从每个IMF成分中提取可能的k个周期。

    参数：
    - imfs: 包含多个IMF成分的数组 [num, length]。
    - fs: 采样频率（每秒采样次数）。

    返回值：
    - periods: 每个IMF成分中提取的周期列表。
    """
    all_periods = set()
    length = imfs.shape[1]
    for imf in imfs:
        f, Pxx = periodogram(imf, fs)  # 计算周期图
        # 获取前k个最大功率的频率索引
        top_k_indices = np.argsort(Pxx)[-k:][::-1]  # 从大到小排序
        top_k_freqs = f[top_k_indices]  # 获取对应的频率
        # 计算前k个频率对应的周期
        top_k_periods_imf = [round(1 / freq) for freq in top_k_freqs if freq >= 1/(length - 1)]
        all_periods.update(top_k_periods_imf)
    return list(all_periods)


def extend(original_periods, length, upper=200):
    extended_periods = set(original_periods)
    for period in original_periods:
        extended_periods.update(list(range(period, min(length, upper), period)))
    return list(extended_periods)


def extract_period(sample, k=1, decompose="wavelet", wave="db4", method="periodogram", mode="extended"):
    """
    sample: ndarray [length, dim]
    k: 对每个成分需要提取的最可能的k个周期。
    decompose: ["wavelet", "emd"]
    wave: ["db4", "sym", "coif"]
    method: ["periodogram", "fft"]
    mode: ["original", "extended", "all"]
    results: list, 每个元素是一个dim的潜在周期值list
    """
    length, dim = sample.shape
    if mode.split("-")[0] == "all":
        periods = list(range(1, min(int(mode.split("-")[1]) + 1, length)))
        return [periods for _ in range(dim)]

    results = []

    for j in range(dim):
        single_dim_series = sample[:, j]
        if decompose == "wavelet":
            imfs = wavelet_decompose(single_dim_series, wavelet=wave)
        else:
            imfs = emd_decompose(single_dim_series)
        # imfs_plot(single_dim_series, imfs)
        if method == "fft":
            periods = from_fft(imfs, k)
        else:
            periods = from_periodogram(imfs, k)
        if mode == "extended":
            results.append(sorted(extend(periods, length)))
        else:
            results.append(sorted(periods))

    return results

